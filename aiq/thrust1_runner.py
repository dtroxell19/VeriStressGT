"""MAGNET pipeline-node runner for the Thrust 1 (soundness / bug detection) card.

1. Uses (or rebuilds) a ground-truth-labelled benchmark: analytic-UNSAT constructions plus
   witness-certified SAT instances (see VeriStressGT.soundness.ground_truth).
2. Runs a pool of verifiers on it:
     - sound controls: the in-house reference verifier and its IBP-only variant
     - planted buggy verifiers, tier (b): reference-verifier mutants, one injected bug each
     - real verifiers (abcrown, pyrat; skipped if not installed)
     - planted buggy verifiers, tier (a): output-level fault wrappers around each real verifier
3. Scores every verdict against the benchmark's labels. For benchmarks without full labels (external
   ones, aiq/build_external_bench.py) it also scores the benchmark's labels plus majority vote on
   its unlabelled instances, the most such a benchmark can do, and counts how many verdicts each
   benchmark can judge at all (summary["coverage"]).

Primary metric: the fraction of planted buggy verifiers flagged by the ground-truth labels
(any definitive verdict contradicting ground truth), with no sound control flagged.

Writes {"result": {...}} to --results_fpath (MAGNET lifts each key into a card symbol), plus
thrust1_verifiers.csv, thrust1_evidence.json and thrust1_verdict_matrix.png alongside it.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import scriptconfig as scfg
import ubelt as ub

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from VeriStressGT.soundness.bugs import MUTANT_BUGS, SOUND_CONTROLS, WRAPPER_BUGS, apply_wrapper  # noqa: E402
from VeriStressGT.soundness.scoring import score  # noqa: E402
from verifier_args import verifier_env_defaults, verifier_extra  # noqa: E402

TWIN_BASES = ["milp_s0_a", "milp_s1_a", "milp_s2_a", "milp_s3_a", "corners_03"]   # id prefixes
N_RANDOM_CNN = 3
N_HARD_CNN = 4
SCALE_SEED_STRIDE = 1000   # seed offset of each scaled copy, clear of every seed in the base spec

_UNSUPPORTED_KEYWORDS = ("unsupported", "not supported", "not implement", "notimplementederror",
                         "unsupportedop", "layer type", "no support")


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def _env() -> Dict[str, str]:
    # Make subprocesses import this checkout even if another copy of VeriStressGT is installed.
    src = str(REPO_ROOT / "src")
    pp = os.environ.get("PYTHONPATH", "")
    return {**os.environ, **verifier_env_defaults(), "PYTHONUNBUFFERED": "1",
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
    spec["name"] = f"{spec.get('name', 'thrust1')}_x{scale}"
    spec["instances"] = base + [
        {**inst, "id": f"{inst['id']}_x{k}", "seed": int(inst.get("seed", seed0)) + SCALE_SEED_STRIDE * k}
        for k in range(1, scale) for inst in base
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    return out


def _run(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env=_env(), cwd=str(REPO_ROOT))


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
    import matplotlib
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


class Thrust1RunnerCLI(scfg.DataConfig):
    """Run the Thrust 1 verifier pool on a ground-truth benchmark and score bug detection."""

    bench_dir = scfg.Value("thrust1_bench", help="Ground-truth benchmark directory.", tags=["algo_param"])
    spec_path = scfg.Value("src/VeriStressGT/configs/thrust1.yaml",
                           help="Base-instance spec, used only with --rebuild.", tags=["algo_param"])
    run_dir = scfg.Value("thrust1_run", help="Per-verifier output root.", tags=["algo_param"])
    timeout = scfg.Value(60.0, type=float, help="Per-instance wall-clock timeout (s).", tags=["algo_param"])
    jobs = scfg.Value(4, type=int, help="Parallel instances for the in-house verifiers.", tags=["algo_param"])
    real_jobs = scfg.Value(2, type=int, help="Parallel instances for each real verifier.", tags=["algo_param"])
    real_verifiers = scfg.Value(["abcrown", "pyrat", "nnenum", "neuralsat", "marabou"], help="Real verifiers to run (skipped if missing).",
                                tags=["algo_param"])
    mutant_bugs = scfg.Value(list(MUTANT_BUGS), help="Reference-verifier mutants to run.", tags=["algo_param"])
    wrapper_bugs = scfg.Value(list(WRAPPER_BUGS), help="Output-level faults applied to each real verifier.",
                              tags=["algo_param"])
    abcrown_config = scfg.Value("src/VeriStressGT/configs/abcrown_thrust1.yaml",
                                help="alpha-beta-CROWN config (attack enabled so it can report SAT).",
                                tags=["algo_param"])
    rebuild = scfg.Value(False, help="Regenerate the benchmark and its SAT twins.", tags=["algo_param"])
    scale = scfg.Value(1, type=int, help="Benchmark size multiplier for scaling runs. scale > 1 builds (once) a "
                                         "benchmark with scale re-seeded copies of every base instance and "
                                         "scale times the random / attack-hard CNNs, in <bench_dir>_x<scale> "
                                         "(verifier outputs in <run_dir>_x<scale>). Needs gurobipy.",
                       tags=["algo_param"])
    allow_missing_verifiers = scfg.Value(False, help="Continue (and report them as skipped) when a requested real "
                                                     "verifier is not installed. Default: fail, so a partial "
                                                     "environment cannot silently shrink the verifier pool.",
                                         tags=["algo_param"])
    reuse_runs = scfg.Value(False, help="Skip verifiers whose results.jsonl already exists in run_dir "
                                        "(re-score only).", tags=["algo_param"])
    results_fpath = scfg.Value("results.json", help="Output JSON consumed by MAGNET.", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        as_list = lambda v: v if isinstance(v, list) else [x.strip() for x in str(v).split(",") if x.strip()]
        scale = int(config.scale)
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        suffix = f"_x{scale}" if scale > 1 else ""
        bench = _resolve(str(config.bench_dir) + suffix)
        run_dir = _resolve(str(config.run_dir) + suffix)
        timeout = float(config.timeout)

        # ── 1. benchmark ────────────────────────────────────────────────────────────────
        if config.rebuild or not (bench / "manifest.json").exists():
            print(f"\n=== Building ground-truth benchmark (scale {scale}) in {bench} ===", flush=True)
            spec = _resolve(config.spec_path)
            if scale > 1:
                spec = _scaled_spec(spec, scale, run_dir / f"thrust1_x{scale}.yaml")
            _run([sys.executable, "-m", "VeriStressGT.cli.create_benchmark", "--spec", str(spec),
                  "--out_dir", str(bench), "--overwrite"])
            _run([sys.executable, "-m", "VeriStressGT.soundness.ground_truth", "--bench", str(bench),
                  "--twins", *TWIN_BASES, "--random_cnn", str(N_RANDOM_CNN * scale),
                  "--hard_cnn", str(N_HARD_CNN * scale)])
        manifest = json.loads((bench / "manifest.json").read_text())
        instances = {i["id"]: i for i in manifest["instances"]}
        # "unknown": external benchmarks (aiq/build_external_bench.py) can only certify SAT instances
        gt = {iid: i["ground_truth"]["label"] for iid, i in instances.items()}
        gt_known = {iid: (lab if lab in ("sat", "unsat") else None) for iid, lab in gt.items()}
        n_witness_ok = _certify_sat_labels(bench, instances)
        iids = sorted(gt)
        print(f"Benchmark: {len(iids)} instances ({sum(v == 'unsat' for v in gt.values())} UNSAT, "
              f"{sum(v == 'sat' for v in gt.values())} SAT, {sum(v is None for v in gt_known.values())} unlabelled)",
              flush=True)

        # ── 2. verifier pool ────────────────────────────────────────────────────────────
        verdicts: Dict[str, Dict[str, str]] = {}
        variants = list(SOUND_CONTROLS) + as_list(config.mutant_bugs)
        for bug in variants:
            name = "reference" if bug == "none" else (bug if bug in SOUND_CONTROLS else f"mutant:{bug}")
            print(f"\n=== {name} ===", flush=True)
            out = run_dir / f"synthetic_{bug}"
            if not (config.reuse_runs and (out / "results.jsonl").exists()):
                _run([sys.executable, "-u", "-m", "VeriStressGT.cli.verify_benchmark", "--benchmark", str(bench),
                      "--verifier", "synthetic", "--out_dir", str(out), "--timeout", str(timeout),
                      "--jobs", str(config.jobs), "--overwrite",
                      "--synthetic_bug", bug, "--synthetic_budget", str(max(5.0, timeout - 5.0))])
            verdicts[name] = _load_statuses(out / "results.jsonl", iids)

        real_extra = verifier_extra(_resolve(config.abcrown_config), falsify=True)
        real_run, real_skipped = [], []
        for v in as_list(config.real_verifiers):
            print(f"\n=== real verifier: {v} ===", flush=True)
            out = run_dir / v
            try:
                if not (config.reuse_runs and (out / "results.jsonl").exists()):
                    _run([sys.executable, "-u", "-m", "VeriStressGT.cli.verify_benchmark", "--benchmark",
                          str(bench), "--verifier", v, "--out_dir", str(out), "--timeout", str(timeout),
                          "--overwrite", "--jobs", str(config.real_jobs), *real_extra.get(v, [])])
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
            for wb in as_list(config.wrapper_bugs):
                verdicts[f"{v}+{wb}"] = apply_wrapper(wb, st, instances)

        if real_skipped and not config.allow_missing_verifiers:
            raise SystemExit(
                f"Real verifier(s) not working in this environment: {real_skipped}. Their logs are in "
                f"{run_dir}/<verifier>/logs. Run `python scripts/check_aiq_setup.py` to diagnose, or pass "
                f"allow_missing_verifiers=True to score without them.")

        # ── 3. scoring ──────────────────────────────────────────────────────────────────
        planted: Dict[str, str] = {}
        base_of: Dict[str, str] = {}
        for bug in as_list(config.mutant_bugs):
            planted[f"mutant:{bug}"] = MUTANT_BUGS[bug]["fails_as"]
            base_of[f"mutant:{bug}"] = "reference"
        for v in real_run:
            for wb in as_list(config.wrapper_bugs):
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
            "scale": scale,
            "n_instances": len(iids),
            "n_unsat": sum(v == "unsat" for v in gt.values()),
            "n_sat": sum(v == "sat" for v in gt.values()),
            "n_unlabelled": sum(v is None for v in gt_known.values()),
            "n_planted": len(planted),
            "n_planted_exercised": len(exercised),
            # nested: MAGNET 0.1.0 crashes on a top-level empty list (Symbols.simple_view)
            "real_verifiers": {"run": real_run, "skipped": real_skipped},
            # flat scalars for the MAGNET dashboard
            "gt_detection_rate": gt_s["detection_rate"],
            "gt_control_false_flags": len(gt_s["controls_flagged"]),
            "sat_witnesses_recertified": n_witness_ok,
            "gt_scoring_accuracy": gt_s["scoring"]["scoring_accuracy"],
            "gt_detected": len(gt_s["detected"]),
            # the best an external benchmark can do: its own labels, majority vote where it has none
            "labels_mv_detection_rate": lmv["detection_rate"],
            "labels_mv_detected": len(lmv["detected"]),
            # how each definitive verdict can be judged (dict: MAGNET lifts only top-level keys)
            "coverage": cov,
            "coverage_judged_by_labels": cov["judged_by_labels"] / cov["verdicts"] if cov["verdicts"] else None,
        }
        out_fpath = ub.Path(config.results_fpath)
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps({"result": result}, indent=2))
        print(f"\nWrote {out_fpath}", flush=True)

        # ── 4. side outputs ─────────────────────────────────────────────────────────────
        out_dir = Path(out_fpath.parent)
        with open(out_dir / "thrust1_verifiers.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["verifier", "role", "fails_as", "exercised", "definitive", "gt_flagged", "gt_wrong",
                        "labels_mv_flagged"])
            for v, d in per.items():
                w.writerow([v, d["role"], d["fails_as"] or "", d.get("exercised", ""), d["definitive"],
                            d["gt_flagged"], len(d["gt_evidence"]), d["labels_mv_flagged"]])
        (out_dir / "thrust1_evidence.json").write_text(json.dumps(
            {v: {"gt": d["gt_evidence"], "labels_mv": d["labels_mv_evidence"]} for v, d in per.items()}, indent=2))
        order = controls + real_run + sorted(planted)
        roles = {v: per[v]["role"] for v in order}
        _plot_matrix(verdicts, gt, order, roles, {v: per[v]["gt_flagged"] for v in order},
                     out_dir / "thrust1_verdict_matrix.png")

        print("\n── 1. Buggy verifiers caught (exercised planted bugs) ─────────", flush=True)
        for label, s in (("benchmark labels", gt_s), ("labels + majority vote", lmv)):
            dr = s["detection_rate"]
            print(f"  {label:<22s} detected {len(s['detected'])}/{len(exercised)}"
                  f" ({dr:.0%})  controls flagged: {s['controls_flagged'] or 'none'}"
                  if dr is not None else f"  {label}: n/a", flush=True)
        print("\n── 2. Scoring accuracy (every definitive verdict judged correctly?) ──", flush=True)
        for label, s in (("benchmark labels", gt_s),):
            sc = s["scoring"]
            print(f"  {label:<22s} {sc['correctly_scored']}/{sc['judgments']} ({sc['scoring_accuracy']:.1%})"
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


__cli__ = Thrust1RunnerCLI

if __name__ == "__main__":
    __cli__.main()
