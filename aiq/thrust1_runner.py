"""MAGNET pipeline-node runner for the Thrust 1 (soundness / bug detection) card.

1. Uses (or rebuilds) a ground-truth-labelled benchmark: analytic-UNSAT constructions plus
   witness-certified SAT instances (see VeriStressGT.soundness.ground_truth).
2. Runs a pool of verifiers on it:
     - sound controls: the in-house reference verifier and its IBP-only variant
     - planted buggy verifiers, tier (b): reference-verifier mutants, one injected bug each
     - real verifiers (abcrown, pyrat; skipped if not installed)
     - planted buggy verifiers, tier (a): output-level fault wrappers around each real verifier
3. Scores bug detection twice: against ground truth, and against VNN-COMP-style majority vote.

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

TWIN_BASES = ["milp_s0_a", "milp_s1_a", "milp_s2_a", "milp_s3_a", "corners_03"]
N_RANDOM_CNN = 3

_UNSUPPORTED_KEYWORDS = ("unsupported", "not supported", "not implement", "notimplementederror",
                         "unsupportedop", "layer type", "no support")


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def _env() -> Dict[str, str]:
    # Make subprocesses import this checkout even if another copy of VeriStressGT is installed.
    src = str(REPO_ROOT / "src")
    pp = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": src + (os.pathsep + pp if pp else "")}


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


def _plot_matrix(verdicts: Dict[str, Dict[str, str]], gt: Dict[str, str], order: List[str],
                 roles: Dict[str, str], flagged: Dict[str, bool], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    iids = sorted(gt, key=lambda i: (gt[i] != "unsat", i))
    code = {"correct": 0, "wrong": 1, "undecided": 2, "unsupported": 3}
    colors = ["#4C9F70", "#D1495B", "#C9C9C9", "#FFFFFF"]
    M = []
    for v in order:
        row = []
        for i in iids:
            s = verdicts[v].get(i, "missing")
            if s in ("sat", "unsat"):
                row.append(code["correct"] if s == gt[i] else code["wrong"])
            elif s == "unsupported":
                row.append(code["unsupported"])
            else:
                row.append(code["undecided"])
        M.append(row)
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(iids) + 4), max(4, 0.32 * len(order) + 2)))
    ax.imshow(M, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(iids)))
    ax.set_xticklabels(iids, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{v}  [{roles[v]}]{'  FLAGGED' if flagged[v] else ''}" for v in order], fontsize=8)
    n_unsat = sum(1 for i in iids if gt[i] == "unsat")
    ax.axvline(n_unsat - 0.5, color="black", lw=1.5)
    ax.text((n_unsat - 1) / 2, -1.0, "ground truth UNSAT", ha="center", fontsize=9, fontweight="bold")
    ax.text(n_unsat + (len(iids) - n_unsat - 1) / 2, -1.0, "ground truth SAT (witness)", ha="center",
            fontsize=9, fontweight="bold")
    ax.set_xticks([x - 0.5 for x in range(1, len(iids))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(order))], minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="gray", label=l) for c, l in
                       zip(colors, ["correct verdict", "wrong verdict", "timeout / unknown / error", "unsupported"])],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8)
    fig.tight_layout()
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
    real_verifiers = scfg.Value(["abcrown", "pyrat"], help="Real verifiers to run (skipped if missing).",
                                tags=["algo_param"])
    mutant_bugs = scfg.Value(list(MUTANT_BUGS), help="Reference-verifier mutants to run.", tags=["algo_param"])
    wrapper_bugs = scfg.Value(list(WRAPPER_BUGS), help="Output-level faults applied to each real verifier.",
                              tags=["algo_param"])
    abcrown_config = scfg.Value("src/VeriStressGT/configs/abcrown_thrust1.yaml",
                                help="alpha-beta-CROWN config (attack enabled so it can report SAT).",
                                tags=["algo_param"])
    rebuild = scfg.Value(False, help="Regenerate the benchmark and its SAT twins.", tags=["algo_param"])
    reuse_runs = scfg.Value(False, help="Skip verifiers whose results.jsonl already exists in run_dir "
                                        "(re-score only).", tags=["algo_param"])
    results_fpath = scfg.Value("results.json", help="Output JSON consumed by MAGNET.", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        as_list = lambda v: v if isinstance(v, list) else [x.strip() for x in str(v).split(",") if x.strip()]
        bench = _resolve(config.bench_dir)
        run_dir = _resolve(config.run_dir)
        timeout = float(config.timeout)

        # ── 1. benchmark ────────────────────────────────────────────────────────────────
        if config.rebuild or not (bench / "manifest.json").exists():
            print("\n=== Building ground-truth benchmark ===", flush=True)
            _run([sys.executable, "-m", "VeriStressGT.cli.create_benchmark", "--spec", str(_resolve(config.spec_path)),
                  "--out_dir", str(bench), "--overwrite"])
            _run([sys.executable, "-m", "VeriStressGT.soundness.ground_truth", "--bench", str(bench),
                  "--twins", *TWIN_BASES, "--random_cnn", str(N_RANDOM_CNN)])
        manifest = json.loads((bench / "manifest.json").read_text())
        instances = {i["id"]: i for i in manifest["instances"]}
        gt = {iid: i["ground_truth"]["label"] for iid, i in instances.items()}
        iids = sorted(gt)
        print(f"Benchmark: {len(iids)} instances ({sum(v == 'unsat' for v in gt.values())} UNSAT, "
              f"{sum(v == 'sat' for v in gt.values())} SAT)", flush=True)

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

        real_extra = {
            "abcrown": ["--abcrown_config", str(_resolve(config.abcrown_config))],
            # --check/--attack enable pyrat's counterexample search (off by default); must come last
            "pyrat": ["--pyrat_domains", "con_z", "--pyrat_device", "cpu", "--pyrat_library", "torch",
                      "--pyrat_split_relu", "--no-pyrat_split", "--pyrat_split_heuristic", "better",
                      "--pyrat_extra", "--check", "both", "--attack", "pgd"],
        }
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
        res = score(verdicts, gt, planted, controls, real_run, base_of=base_of)
        per, summ = res["per_verifier"], res["summary"]
        for v, base in base_of.items():   # did the bug change any verdict at all on this benchmark?
            per[v]["manifested"] = any(verdicts[v][i] != verdicts[base][i] for i in iids)

        manifested = [v for v in planted if per[v]["manifested"]]
        gt_s, mv1, mvf = summ["ground_truth"], summ["majority_one_buggy"], summ["majority_full"]
        result = {
            "summary": summ,
            "per_verifier": {v: {k: x for k, x in d.items()} for v, d in per.items()},
            "verdicts": verdicts,
            "ground_truth": gt,
            "n_instances": len(iids),
            "n_unsat": sum(v == "unsat" for v in gt.values()),
            "n_sat": sum(v == "sat" for v in gt.values()),
            "n_planted": len(planted),
            "n_planted_manifested": len(manifested),
            "real_verifiers_run": real_run,
            "real_verifiers_skipped": real_skipped,
            # flat scalars for the MAGNET dashboard
            "gt_detection_rate": gt_s["detection_rate"],
            "gt_detection_rate_manifested": (sum(per[v]["gt_flagged"] for v in manifested) / len(manifested))
                                            if manifested else None,
            "gt_control_false_flags": len(gt_s["controls_flagged"]),
            "mv_one_buggy_detection_rate": mv1["detection_rate"],
            "mv_full_detection_rate": mvf["detection_rate"],
            "mv_full_control_false_flags": len(mvf["controls_flagged"]),
            "mv_full_sound_verifiers_penalized": len(mvf["sound_verifiers_penalized"]),
            "mv_full_label_errors": len(mvf["label_errors"]),
        }
        out_fpath = ub.Path(config.results_fpath)
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps({"result": result}, indent=2))
        print(f"\nWrote {out_fpath}", flush=True)

        # ── 4. side outputs ─────────────────────────────────────────────────────────────
        out_dir = Path(out_fpath.parent)
        with open(out_dir / "thrust1_verifiers.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["verifier", "role", "fails_as", "manifested", "definitive", "gt_flagged", "gt_wrong",
                        "mv_one_buggy_flagged", "mv_full_flagged", "mv_full_wrongful"])
            for v, d in per.items():
                w.writerow([v, d["role"], d["fails_as"] or "", d.get("manifested", ""), d["definitive"],
                            d["gt_flagged"], len(d["gt_evidence"]), d["mv_one_buggy_flagged"],
                            d["mv_full_flagged"], len(d["mv_full_wrongful"])])
        (out_dir / "thrust1_evidence.json").write_text(json.dumps(
            {v: {"gt": d["gt_evidence"], "mv_full": d["mv_full_evidence"]} for v, d in per.items()}, indent=2))
        order = controls + real_run + sorted(planted)
        roles = {v: per[v]["role"] for v in order}
        _plot_matrix(verdicts, gt, order, roles, {v: per[v]["gt_flagged"] for v in order},
                     out_dir / "thrust1_verdict_matrix.png")

        print("\n── Bug detection ──────────────────────────────────────────────", flush=True)
        for label, s in (("ground truth", gt_s), ("majority (one buggy)", mv1), ("majority (full pool)", mvf)):
            dr = s["detection_rate"]
            print(f"  {label:<22s} detected {len(s['detected'])}/{len(planted)}"
                  f" ({dr:.0%})  controls flagged: {s['controls_flagged'] or 'none'}"
                  if dr is not None else f"  {label}: n/a", flush=True)
        print(f"  majority (full pool) label errors: {mvf['label_errors'] or 'none'}", flush=True)
        print(f"  sound verifiers penalized by majority (full): {mvf['sound_verifiers_penalized'] or 'none'}", flush=True)


__cli__ = Thrust1RunnerCLI

if __name__ == "__main__":
    __cli__.main()
