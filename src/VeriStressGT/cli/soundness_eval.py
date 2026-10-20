"""Soundness evaluation: run a verifier pool on a ground-truth benchmark and score bug detection.

1. Uses (or rebuilds) a ground-truth-labelled benchmark: analytic-UNSAT constructions plus
   witness-certified SAT instances (see VeriStressGT.soundness.ground_truth). The default is the
   committed benchmarks/soundness_gt.
2. Runs a pool of verifiers on it:
     - sound controls: the in-house reference verifier and its IBP-only variant
     - planted buggy verifiers, tier (b): reference-verifier mutants, one injected bug each
     - real verifiers (abcrown, pyrat, nnenum, neuralsat, marabou)
     - planted buggy verifiers, tier (a): output-level fault wrappers around each real verifier
3. Scores every verdict against the benchmark's labels. For benchmarks without a label on every
   instance it also scores the benchmark's labels plus majority vote on the unlabelled instances,
   and counts how many verdicts can be judged at all (summary["coverage"]).

Primary metric: the fraction of exercised planted bugs flagged by the ground-truth labels (any
definitive verdict contradicting ground truth), with no sound control flagged.

Writes results.json, verifiers.csv, evidence.json and verdict_matrix.png to --out_dir.

Example:
  VeriStressGT-soundness --out_dir runs/soundness --timeout 60
  VeriStressGT-soundness --real_verifiers abcrown pyrat --allow_missing_verifiers
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from VeriStressGT.soundness.bugs import MUTANT_BUGS, SOUND_CONTROLS, WRAPPER_BUGS, apply_wrapper
from VeriStressGT.soundness.scoring import score

PKG = Path(__file__).resolve().parents[1]       # src/VeriStressGT
REPO_ROOT = PKG.parents[1]                        # checkout root (scripts/ lives here)

TWIN_BASES = ["milp_s0_a", "milp_s1_a", "milp_s2_a", "milp_s3_a", "corners_03"]   # id prefixes
N_RANDOM_CNN = 3
N_HARD_CNN = 4
SCALE_SEED_STRIDE = 1000   # seed offset of each scaled copy, clear of every seed in the base spec

# pyrat in counterexample-search mode; --check/--attack must come last (passed via --pyrat_extra)
PYRAT_ARGS = ["--pyrat_domains", "con_z", "--pyrat_device", "cpu", "--pyrat_library", "torch",
              "--pyrat_split_relu", "--no-pyrat_split", "--pyrat_split_heuristic", "better",
              "--pyrat_extra", "--check", "both", "--attack", "pgd"]

_UNSUPPORTED_KEYWORDS = ("unsupported", "not supported", "not implement", "notimplementederror",
                         "unsupportedop", "layer type", "no support")


def _env() -> Dict[str, str]:
    """Verifier subprocess environment. Make subprocesses import this checkout even if another copy
    of VeriStressGT is installed, and default the verifier locations to this checkout so a fresh
    clone runs without sourcing .env. Values already set in the environment win."""
    defaults = {
        "ABCROWN_VNNCOMP2024_DIR": str(PKG / "verifiers" / "alpha-beta-CROWN"),
        "ABCROWN_CONDA_ENV": "alpha-beta-crown",
        "NEURALSAT_CONDA_ENV": "neuralsat",
    }
    marabou_env = REPO_ROOT / "scripts" / "marabou_env.sh"
    if marabou_env.exists():
        defaults["MARABOU_BIN"] = str(marabou_env)
    src = str(PKG.parent)
    pp = os.environ.get("PYTHONPATH", "")
    return {**defaults, **os.environ, "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": src + (os.pathsep + pp if pp else "")}


def _scaled_spec(spec_path: Path, scale: int, out: Path) -> Path:
    """The base spec plus ``scale - 1`` re-seeded copies of every instance (ids suffixed ``_x<k>``).

    Each copy is the same construction and arguments with a fresh seed, so it is a new network
    with the same analytic UNSAT certificate. SAT twins follow automatically (TWIN_BASES are id
    prefixes)."""
    import yaml
    spec = yaml.safe_load(spec_path.read_text())
    seed0 = int((spec.get("defaults") or {}).get("seed", 0))
    base = spec["instances"]
    spec["name"] = f"{spec.get('name', 'soundness_gt')}_x{scale}"
    spec["instances"] = base + [
        {**inst, "id": f"{inst['id']}_x{k}", "seed": int(inst.get("seed", seed0)) + SCALE_SEED_STRIDE * k}
        for k in range(1, scale) for inst in base
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    return out


def _run(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env=_env())


def _load_statuses(results_jsonl: Path, instance_ids: List[str]) -> Dict[str, str]:
    recs: Dict[str, Dict[str, Any]] = {}
    if results_jsonl.exists():
        for line in results_jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["instance_id"]] = r
    out = {}
    for iid in instance_ids:
        r = recs.get(iid)
        if r is None:
            out[iid] = "missing"
            continue
        s = str(r["status"]).lower()
        if s == "error":
            preview = ((r.get("error_preview") or "") + " " + (r.get("stdout_preview") or "")).lower()
            s = "unsupported" if any(k in preview for k in _UNSUPPORTED_KEYWORDS) else "error"
        out[iid] = s
    return out


def _certify_sat_labels(bench: Path, instances: Dict[str, Dict]) -> int:
    """Re-check every SAT label's witness (float64 and onnxruntime float32) before scoring."""
    import numpy as np
    from VeriStressGT.soundness.ground_truth import certify_witness
    from VeriStressGT.soundness.refverify import load_problem

    n = 0
    for iid, inst in instances.items():
        gtl = inst["ground_truth"]
        if gtl["label"] != "sat":
            continue
        onnx_p = str(bench / inst["paths"]["onnx"])
        graph, spec = load_problem(onnx_p, str(bench / inst["paths"]["vnnlib"]))
        x = np.asarray(json.loads((bench / gtl["witness"]).read_text())["x"])
        ok, m64, m32 = certify_witness(onnx_p, graph, spec.C, x, spec.lo, spec.hi)
        if not ok:
            raise RuntimeError(f"SAT label of {iid} no longer certifies (f64={m64}, f32={m32})")
        n += 1
    return n


def _plot_matrix(verdicts: Dict[str, Dict[str, str]], gt: Dict[str, str], order: List[str],
                 roles: Dict[str, str], flagged: Dict[str, bool], out_path: Path) -> None:
    try:
        import matplotlib
    except ImportError:
        print("  (matplotlib not installed: skipping the verdict matrix plot)", flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    order_key = {"unsat": 0, "unknown": 1, "sat": 2}
    iids = sorted(gt, key=lambda i: (order_key.get(gt[i], 1), i))
    code = {"correct": 0, "wrong": 1, "undecided": 2, "unsupported": 3, "unlabelled": 4}
    colors = ["#4C9F70", "#D1495B", "#C9C9C9", "#FFFFFF", "#8FB3D9"]
    M = []
    for v in order:
        row = []
        for i in iids:
            s = verdicts[v].get(i, "missing")
            if s in ("sat", "unsat") and gt[i] not in ("sat", "unsat"):
                row.append(code["unlabelled"])      # decided, but nothing to check it against
            elif s in ("sat", "unsat"):
                row.append(code["correct"] if s == gt[i] else code["wrong"])
            elif s == "unsupported":
                row.append(code["unsupported"])
            else:
                row.append(code["undecided"])
        M.append(row)
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(iids) + 4), max(4, 0.32 * len(order) + 2)))
    ax.imshow(M, cmap=ListedColormap(colors), vmin=0, vmax=4, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(iids)))
    ax.set_xticklabels(iids, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{v}  [{roles[v]}]{'  FLAGGED' if flagged[v] else ''}" for v in order], fontsize=8)
    x0 = 0
    for lab, title in (("unsat", "ground truth UNSAT"), ("unknown", "no label"), ("sat", "ground truth SAT (witness)")):
        n = sum(1 for i in iids if gt[i] == lab)
        if n:
            ax.text(x0 + (n - 1) / 2, -1.0, title, ha="center", fontsize=9, fontweight="bold")
            x0 += n
            if x0 < len(iids):
                ax.axvline(x0 - 0.5, color="black", lw=1.5)
    ax.set_xticks([x - 0.5 for x in range(1, len(iids))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(order))], minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="gray", label=l) for c, l in
                        zip(colors, ["correct verdict", "wrong verdict", "timeout / unknown / error", "unsupported"])],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench_dir", default=str(PKG / "benchmarks" / "soundness_gt"),
                    help="Ground-truth benchmark directory.")
    ap.add_argument("--spec_path", default=str(PKG / "configs" / "soundness_gt.yaml"),
                    help="Base-instance spec, used only with --rebuild / --scale.")
    ap.add_argument("--out_dir", default="runs/soundness",
                    help="Per-verifier runs and the scored results go here.")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-instance wall-clock timeout (s).")
    ap.add_argument("--jobs", type=int, default=4, help="Parallel instances for the in-house verifiers.")
    ap.add_argument("--real_jobs", type=int, default=2, help="Parallel instances for each real verifier.")
    ap.add_argument("--real_verifiers", nargs="*", default=["abcrown", "pyrat", "nnenum", "neuralsat", "marabou"],
                    help="Real verifiers to run.")
    ap.add_argument("--mutant_bugs", nargs="*", default=list(MUTANT_BUGS), help="Reference-verifier mutants to run.")
    ap.add_argument("--wrapper_bugs", nargs="*", default=list(WRAPPER_BUGS),
                    help="Output-level faults applied to each real verifier.")
    ap.add_argument("--abcrown_config", default=str(PKG / "configs" / "abcrown_attack.yaml"),
                    help="alpha-beta-CROWN config (attack enabled so it can report SAT).")
    ap.add_argument("--rebuild", action="store_true", help="Regenerate the benchmark and its SAT twins.")
    ap.add_argument("--scale", type=int, default=1,
                    help="Benchmark size multiplier. scale > 1 builds (once) a benchmark with scale re-seeded "
                         "copies of every base instance and scale times the random / attack-hard CNNs, in "
                         "<out_dir>/bench_x<scale>. Needs gurobipy.")
    ap.add_argument("--allow_missing_verifiers", action="store_true",
                    help="Continue (reporting them as skipped) when a requested real verifier is not installed. "
                         "Default: fail, so a partial environment cannot silently shrink the verifier pool.")
    ap.add_argument("--reuse_runs", action="store_true",
                    help="Skip verifiers whose results.jsonl already exists in out_dir (re-score only).")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.scale < 1:
        raise ValueError(f"scale must be >= 1, got {args.scale}")
    out_dir = Path(args.out_dir).resolve()
    bench = Path(args.bench_dir).resolve() if args.scale == 1 else out_dir / f"bench_x{args.scale}"
    timeout = float(args.timeout)

    # ── 1. benchmark ────────────────────────────────────────────────────────────────
    if args.rebuild or not (bench / "manifest.json").exists():
        print(f"\n=== Building ground-truth benchmark (scale {args.scale}) in {bench} ===", flush=True)
        spec = Path(args.spec_path).resolve()
        if args.scale > 1:
            spec = _scaled_spec(spec, args.scale, out_dir / f"soundness_gt_x{args.scale}.yaml")
        _run([sys.executable, "-m", "VeriStressGT.cli.create_benchmark", "--spec", str(spec),
              "--out_dir", str(bench), "--overwrite"])
        _run([sys.executable, "-m", "VeriStressGT.soundness.ground_truth", "--bench", str(bench),
              "--twins", *TWIN_BASES, "--random_cnn", str(N_RANDOM_CNN * args.scale),
              "--hard_cnn", str(N_HARD_CNN * args.scale)])
    manifest = json.loads((bench / "manifest.json").read_text())
    instances = {i["id"]: i for i in manifest["instances"]}
    # labels other than sat/unsat (e.g. "unknown" on an external benchmark) are left unjudged
    gt = {iid: i["ground_truth"]["label"] for iid, i in instances.items()}
    gt_known = {iid: (lab if lab in ("sat", "unsat") else None) for iid, lab in gt.items()}
    n_witness_ok = _certify_sat_labels(bench, instances)
    iids = sorted(gt)
    print(f"Benchmark: {len(iids)} instances ({sum(v == 'unsat' for v in gt.values())} UNSAT, "
          f"{sum(v == 'sat' for v in gt.values())} SAT, {sum(v is None for v in gt_known.values())} unlabelled)",
          flush=True)

    # ── 2. verifier pool ────────────────────────────────────────────────────────────
    verdicts: Dict[str, Dict[str, str]] = {}
    for bug in list(SOUND_CONTROLS) + args.mutant_bugs:
        name = "reference" if bug == "none" else (bug if bug in SOUND_CONTROLS else f"mutant:{bug}")
        print(f"\n=== {name} ===", flush=True)
        out = out_dir / f"refverify_{bug}"
        if not (args.reuse_runs and (out / "results.jsonl").exists()):
            _run([sys.executable, "-u", "-m", "VeriStressGT.cli.verify_benchmark", "--benchmark", str(bench),
                  "--verifier", "refverify", "--out_dir", str(out), "--timeout", str(timeout),
                  "--jobs", str(args.jobs), "--overwrite",
                  "--refverify_bug", bug, "--refverify_budget", str(max(5.0, timeout - 5.0))])
        verdicts[name] = _load_statuses(out / "results.jsonl", iids)

    real_extra = {"abcrown": ["--abcrown_config", str(Path(args.abcrown_config).resolve())], "pyrat": PYRAT_ARGS}
    real_run, real_skipped = [], []
    for v in args.real_verifiers:
        print(f"\n=== real verifier: {v} ===", flush=True)
        out = out_dir / v
        try:
            if not (args.reuse_runs and (out / "results.jsonl").exists()):
                _run([sys.executable, "-u", "-m", "VeriStressGT.cli.verify_benchmark", "--benchmark",
                      str(bench), "--verifier", v, "--out_dir", str(out), "--timeout", str(timeout),
                      "--overwrite", "--jobs", str(args.real_jobs), *real_extra.get(v, [])])
        except subprocess.CalledProcessError as exc:
            print(f"  [SKIP] {v} exited with {exc.returncode}", flush=True)
            real_skipped.append(v)
            continue
        st = _load_statuses(out / "results.jsonl", iids)
        if all(s in ("error", "missing") for s in st.values()):
            print(f"  [SKIP] {v}: every instance errored (not installed?)", flush=True)
            real_skipped.append(v)
            continue
        verdicts[v] = st
        real_run.append(v)
        for wb in args.wrapper_bugs:
            verdicts[f"{v}+{wb}"] = apply_wrapper(wb, st, instances)

    if real_skipped and not args.allow_missing_verifiers:
        raise SystemExit(
            f"Real verifier(s) not working in this environment: {real_skipped}. Their logs are in "
            f"{out_dir}/<verifier>/logs. Run `VeriStressGT-doctor` to diagnose, or pass "
            f"--allow_missing_verifiers to score without them.")

    # ── 3. scoring ──────────────────────────────────────────────────────────────────
    planted: Dict[str, str] = {}
    base_of: Dict[str, str] = {}
    for bug in args.mutant_bugs:
        planted[f"mutant:{bug}"] = MUTANT_BUGS[bug]["fails_as"]
        base_of[f"mutant:{bug}"] = "reference"
    for v in real_run:
        for wb in args.wrapper_bugs:
            planted[f"{v}+{wb}"] = WRAPPER_BUGS[wb]["fails_as"]
            base_of[f"{v}+{wb}"] = v
    controls = ["reference", "ibp_only"]
    res = score(verdicts, gt_known, planted, controls, real_run, base_of=base_of)
    per, summ = res["per_verifier"], res["summary"]
    exercised = [v for v in planted if per[v]["exercised"]]
    gt_s, lmv, cov = summ["ground_truth"], summ["labels_plus_majority"], summ["coverage"]
    # Majority vote on its own is not reported: it is only an ingredient of labels_plus_majority.
    per_out = {v: {k: x for k, x in d.items() if not k.startswith(("mv_one_buggy", "mv_full"))}
               for v, d in per.items()}
    result = {
        "summary": {"ground_truth": gt_s, "labels_plus_majority": lmv, "coverage": cov},
        "per_verifier": per_out,
        "verdicts": verdicts,
        "ground_truth": gt,
        "scale": args.scale,
        "n_instances": len(iids),
        "n_unsat": sum(v == "unsat" for v in gt.values()),
        "n_sat": sum(v == "sat" for v in gt.values()),
        "n_unlabelled": sum(v is None for v in gt_known.values()),
        "n_planted": len(planted),
        "n_planted_exercised": len(exercised),
        "real_verifiers": {"run": real_run, "skipped": real_skipped},
        "sat_witnesses_recertified": n_witness_ok,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_dir / 'results.json'}", flush=True)

    # ── 4. side outputs ─────────────────────────────────────────────────────────────
    with open(out_dir / "verifiers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["verifier", "role", "fails_as", "exercised", "definitive", "gt_flagged", "gt_wrong",
                    "labels_mv_flagged"])
        for v, d in per.items():
            w.writerow([v, d["role"], d["fails_as"] or "", d.get("exercised", ""), d["definitive"],
                        d["gt_flagged"], len(d["gt_evidence"]), d["labels_mv_flagged"]])
    (out_dir / "evidence.json").write_text(json.dumps(
        {v: {"gt": d["gt_evidence"], "labels_mv": d["labels_mv_evidence"]} for v, d in per.items()}, indent=2))
    order = controls + real_run + sorted(planted)
    roles = {v: per[v]["role"] for v in order}
    _plot_matrix(verdicts, gt, order, roles, {v: per[v]["gt_flagged"] for v in order},
                 out_dir / "verdict_matrix.png")

    print("\n── 1. Buggy verifiers caught (exercised planted bugs) ─────────", flush=True)
    for label, s in (("benchmark labels", gt_s), ("labels + majority vote", lmv)):
        dr = s["detection_rate"]
        print(f"  {label:<22s} detected {len(s['detected'])}/{len(exercised)}"
              f" ({dr:.0%})  controls flagged: {s['controls_flagged'] or 'none'}"
              if dr is not None else f"  {label}: n/a", flush=True)
    print("\n── 2. Scoring accuracy (every definitive verdict judged correctly?) ──", flush=True)
    sc = gt_s["scoring"]
    if sc["judgments"]:
        print(f"  {'benchmark labels':<22s} {sc['correctly_scored']}/{sc['judgments']} ({sc['scoring_accuracy']:.1%})"
              f"  wrongful accusations: {sc['wrongful_accusations']}"
              f"  wrongful acquittals: {sc['wrongful_acquittals']}  unscored (ties): {sc['unscored']}"
              + (f"  [{sc['unknown_truth']} verdicts on unlabelled instances left out]" if sc.get("unknown_truth") else ""),
              flush=True)
    print(f"  (ground-truth SAT labels re-certified from witnesses: {n_witness_ok})", flush=True)
    n = cov["verdicts"] or 1
    print(f"\n── 3. How the {cov['verdicts']} definitive verdicts can be judged ──", flush=True)
    print(f"  by the benchmark's labels (certain)     {cov['judged_by_labels']} ({cov['judged_by_labels'] / n:.1%})\n"
          f"  only by majority vote (unverifiable)    {cov['judged_by_majority_only']} ({cov['judged_by_majority_only'] / n:.1%})\n"
          f"  not at all (majority tie)               {cov['unjudged']} ({cov['unjudged'] / n:.1%})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
