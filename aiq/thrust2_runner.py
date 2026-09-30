"""MAGNET pipeline-node runner for the Thrust 2 (timeout prediction) card.

1. Recorded evaluation. Per-verifier timeout predictors (elastic-net logistic on size/type +
   Difficulty-Profile features) are trained on the paper's server runs (345 instances x 5
   verifiers) and scored on network-grouped held-out splits, repeated over several seeds. Labels
   use a common horizon equal to the live timeout, so the two parts of the card agree.
2. Live out-of-distribution check. For a pre-built benchmark of fresh networks (new seeds,
   off-grid parameters), features are computed on the fly, abcrown and pyrat are run live, and
   models fitted on all recorded data predict which instances time out.

Primary metric: held-out AUC per verifier (size+profile), target >= 0.7 for every verifier.

Writes {"result": {...}} to --results_fpath plus thrust2_auc.csv, thrust2_ood_predictions.csv
and thrust2_auc.png alongside it.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scriptconfig as scfg
import ubelt as ub

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from VeriStressGT.prediction.features import instance_features  # noqa: E402
from VeriStressGT.prediction.model import (FEATURE_SETS, VERIFIERS, anchor_time_scale, auc,  # noqa: E402
                                           design, evaluate_recorded, fit_full, load_training_rows,
                                           select_anchors)

from verifier_args import verifier_env_defaults, verifier_extra  # noqa: E402

ANCHOR_BENCH = "src/VeriStressGT/benchmarks/sweep_all"


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def _env() -> Dict[str, str]:
    src = str(REPO_ROOT / "src")
    pp = os.environ.get("PYTHONPATH", "")
    return {**os.environ, **verifier_env_defaults(), "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": src + (os.pathsep + pp if pp else "")}


def _run(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env=_env(), cwd=str(REPO_ROOT))


def _summ(xs: List[float]) -> Dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "min": None, "n": 0}
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "min": float(np.min(xs)), "n": len(xs)}


def _plot(recorded: Dict[str, Dict], ood: Dict[str, Dict], out_path: Path, target: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = list(FEATURE_SETS)
    # Categorical slots 1-3 of the validated default palette, in fixed order (light surface).
    colors = {"size": "#2a78d6", "profile": "#eb6834", "size+profile": "#1baf7a"}
    ink, muted, surface = "#0b0b0b", "#6b6b66", "#fcfcfb"
    x = np.arange(len(VERIFIERS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=surface)
    ax.set_facecolor(surface)
    for k, name in enumerate(sets):
        m = [recorded[v][name]["mean"] or 0 for v in VERIFIERS]
        s = [recorded[v][name]["std"] or 0 for v in VERIFIERS]
        ax.bar(x + (k - 1) * w, m, w * 0.92, yerr=s, capsize=2, color=colors[name], edgecolor=surface,
               linewidth=1, error_kw={"elinewidth": 1, "ecolor": muted}, label=f"held-out: {name}")
    labels = {"auc_anchored": "live OOD, anchored", "auc_uncalibrated": "live OOD, uncalibrated"}
    for i, v in enumerate(VERIFIERS):
        for key, filled in (("auc_uncalibrated", False), ("auc_anchored", True)):
            a = ood.get(v, {}).get(key)
            if a is None:
                continue
            dx = 0.07 if filled else -0.07  # side by side, so equal values do not hide each other
            ax.plot(x[i] + w + dx, a, marker="D", ms=7, ls="none", zorder=5,
                    color=ink if filled else surface, mec=ink if not filled else surface,
                    mew=1.5 if not filled else 2, label=labels.pop(key, None))
    ax.axhline(target, color=ink, ls="--", lw=1)
    ax.text(len(VERIFIERS) - 0.45, target, f"target {target}", color=ink, ha="left", va="center", fontsize=8,
            bbox={"facecolor": surface, "edgecolor": "none", "pad": 1.5})
    ax.axhline(0.5, color=muted, ls=":", lw=1)
    ax.text(len(VERIFIERS) - 0.45, 0.5, "chance", color=muted, ha="left", va="center", fontsize=8,
            bbox={"facecolor": surface, "edgecolor": "none", "pad": 1.5})
    ax.set_xlim(-0.5, len(VERIFIERS) + 0.1)  # right margin holds the reference-line labels
    ax.yaxis.grid(True, color="#e6e6e2", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink)
    ax.set_xticks(x)
    ax.set_xticklabels(VERIFIERS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Timeout AUC", color=ink)
    ax.set_title("Per-verifier timeout prediction (mean ± sd over grouped held-out splits)", color=ink,
                 fontsize=10, pad=24)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    leg = ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                    ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved {out_path}", flush=True)


class Thrust2RunnerCLI(scfg.DataConfig):
    """Evaluate per-verifier timeout predictors on recorded runs and on live out-of-distribution runs."""

    horizon = scfg.Value(120.0, type=float, help="Timeout horizon (s): live timeout and label horizon.",
                         tags=["algo_param"])
    repeats = scfg.Value(10, type=int, help="Grouped held-out splits (seeds 0..repeats-1).", tags=["algo_param"])
    ood_bench_dir = scfg.Value("thrust2_ood_bench", help="Pre-built out-of-distribution benchmark.",
                               tags=["algo_param"])
    ood_spec_path = scfg.Value("src/VeriStressGT/configs/thrust2_ood.yaml",
                               help="Spec used to (re)build the OOD benchmark with --rebuild.", tags=["algo_param"])
    run_dir = scfg.Value("thrust2_run", help="Output root for live verifier runs.", tags=["algo_param"])
    live_verifiers = scfg.Value(["abcrown", "pyrat", "nnenum", "neuralsat", "marabou"], help="Verifiers run live (skipped if missing).",
                                tags=["algo_param"])
    real_jobs = scfg.Value(2, type=int, help="Parallel instances per live verifier.", tags=["algo_param"])
    abcrown_config = scfg.Value("src/VeriStressGT/configs/abcrown_basic.yaml", help="alpha-beta-CROWN config.",
                                tags=["algo_param"])
    rebuild = scfg.Value(False, help="Regenerate the OOD benchmark (slow: exact-radius MILPs).",
                         tags=["algo_param"])
    reuse_runs = scfg.Value(False, help="Reuse live results.jsonl already in run_dir.", tags=["algo_param"])
    anchors_per_arch = scfg.Value(4, type=int, help="Anchor instances per architecture: re-run recorded "
                                  "instances here to measure the environment speed shift and fit the "
                                  "(exploratory) anchored model; 0 disables.", tags=["algo_param"])
    target_auc = scfg.Value(0.7, type=float, help="Per-verifier AUC target (reporting only).", tags=["algo_param"])
    results_fpath = scfg.Value("results.json", help="Output JSON consumed by MAGNET.", tags=["out_path", "primary"])

    @classmethod
    def main(cls, argv=None, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose=True)
        as_list = lambda v: v if isinstance(v, list) else [x.strip() for x in str(v).split(",") if x.strip()]
        H = float(config.horizon)
        df = load_training_rows()

        # ── 1. recorded, grouped held-out evaluation ────────────────────────────────────
        print(f"\n=== Recorded held-out evaluation (horizon {H:g}s, {config.repeats} grouped splits) ===",
              flush=True)
        runs = [evaluate_recorded(df, H, seed=s) for s in range(int(config.repeats))]
        recorded: Dict[str, Dict] = {}
        for v in VERIFIERS:
            recorded[v] = {name: _summ([r[v]["auc"][name] for r in runs]) for name in FEATURE_SETS}
            recorded[v]["transfer_synthetic_to_established"] = {
                "size+profile": _summ([r[v]["transfer_synthetic_to_established_auc"] for r in runs[:1]]),
                "size": _summ([r[v]["transfer_size_only_auc"] for r in runs[:1]]),
            }
            recorded[v]["test_timeout_rate"] = _summ([r[v]["test_timeout_rate"] for r in runs])
            recorded[v]["n_test"] = _summ([r[v]["n_test"] for r in runs])
            m = recorded[v]["size+profile"]
            print(f"  {v:9s} size+profile AUC {m['mean']:.3f} ± {m['std']:.3f} (min {m['min']:.3f})"
                  f"   size {recorded[v]['size']['mean']:.3f}   profile {recorded[v]['profile']['mean']:.3f}",
                  flush=True)

        # ── 2. live out-of-distribution check ───────────────────────────────────────────
        bench = _resolve(config.ood_bench_dir)
        if config.rebuild or not (bench / "manifest.json").exists():
            _run([sys.executable, "-m", "VeriStressGT.cli.create_benchmark", "--spec",
                  str(_resolve(config.ood_spec_path)), "--out_dir", str(bench), "--overwrite"])
        manifest = json.loads((bench / "manifest.json").read_text())
        insts = {i["id"]: i for i in manifest["instances"]}
        print(f"\n=== Live OOD: featurizing {len(insts)} fresh instances ===", flush=True)
        feats: Dict[str, Dict] = {}
        for iid, inst in insts.items():
            feats[iid] = instance_features(str(bench / inst["paths"]["onnx"]), str(bench / inst["paths"]["vnnlib"]))

        run_dir = _resolve(config.run_dir)
        real_extra = verifier_extra(_resolve(config.abcrown_config), falsify=False)
        import pandas as pd
        ood: Dict[str, Dict] = {}
        pred_rows = []
        for v in as_list(config.live_verifiers):
            print(f"\n=== live verifier: {v} (timeout {H:g}s) ===", flush=True)
            out = run_dir / v
            try:
                if not (config.reuse_runs and (out / "results.jsonl").exists()):
                    _run([sys.executable, "-u", "-m", "VeriStressGT.cli.verify_benchmark", "--benchmark", str(bench),
                          "--verifier", v, "--out_dir", str(out), "--timeout", str(H), "--overwrite",
                          "--jobs", str(config.real_jobs), *real_extra.get(v, [])])
            except subprocess.CalledProcessError as exc:
                print(f"  [SKIP] {v} exited with {exc.returncode}", flush=True)
                continue
            recs = {json.loads(l)["instance_id"]: json.loads(l)
                    for l in (out / "results.jsonl").read_text().splitlines() if l.strip()}
            rows = []
            for iid in insts:
                r = recs.get(iid)
                st = str(r["status"]).upper() if r else "MISSING"
                t = float(r.get("wall_time_s") or 0.0) if r else np.nan
                y = 1 if st == "TIMEOUT" else (0 if st in ("UNSAT", "SAT") and t < H else (1 if st in ("UNSAT", "SAT") else np.nan))
                rows.append({"instance_id": iid, "status": st, "wall_time_s": t, "y": y, **feats[iid]})
            live = pd.DataFrame(rows)
            if v not in VERIFIERS:
                continue
            cols = FEATURE_SETS["size+profile"]
            scored = live[live.y.notna()].copy()

            # Environment anchoring: re-run recorded anchor instances here, rescale recorded runtimes.
            anchor = None
            if int(config.anchors_per_arch) > 0:
                anchors = select_anchors(df, v, int(config.anchors_per_arch))
                aout = run_dir / f"{v}_anchors"
                if not (config.reuse_runs and (aout / "results.jsonl").exists()):
                    _run([sys.executable, "-u", "-m", "VeriStressGT.cli.verify_benchmark", "--benchmark",
                          str(_resolve(ANCHOR_BENCH)), "--verifier", v, "--out_dir", str(aout), "--timeout", str(H),
                          "--overwrite", "--jobs", str(config.real_jobs), "--instances", *anchors,
                          *real_extra.get(v, [])])
                arecs = [json.loads(l) for l in (aout / "results.jsonl").read_text().splitlines() if l.strip()]
                anchor = anchor_time_scale(df, v, {r["instance_id"]: (r["status"], float(r["wall_time_s"] or 0))
                                                   for r in arecs})
                if anchor["identifiable"]:
                    print(f"  {v}: anchors -> local_s = {anchor['overhead_s']:.1f} + slope * recorded_s, slopes "
                          f"{ {k: round(x, 3) for k, x in anchor['slope_by_arch'].items()} }", flush=True)
                else:
                    print(f"  {v}: anchor calibration not identifiable ({anchor['reason']})", flush=True)

            res_v = {"n": int(len(scored)), "n_excluded": int(live.y.isna().sum()),
                     "timeout_rate": float(scored.y.mean()) if len(scored) else None,
                     "anchor_calibration": anchor}
            for tag, calib in (("uncalibrated", None), ("anchored", anchor)):
                if tag == "anchored" and not (calib and calib.get("identifiable")):
                    continue
                from VeriStressGT.prediction.model import rows_for
                if rows_for(df, v, H, calib).y.nunique() < 2:
                    res_v[f"note_{tag}"] = "recorded runs yield a single label class at this horizon"
                    continue
                model = fit_full(df, v, H, calibration=calib)
                live[f"p_timeout_{tag}"] = model.predict_proba(design(live, cols))[:, 1]
                res_v[f"auc_{tag}"] = auc(model, scored, cols) if len(scored) else None
            # Headline = uncalibrated: with a handful of anchors the affine correction was not reliably
            # better (it can be unidentifiable, or noisier than the shift it corrects). The anchors are
            # still reported as a diagnostic of how the verifier's speed differs between environments.
            res_v["auc"] = res_v.get("auc_uncalibrated")
            ood[v] = res_v
            fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
            print(f"  {v}: OOD AUC uncalibrated {fmt(res_v['auc_uncalibrated'])}, anchored "
                  f"{fmt(res_v.get('auc_anchored'))} on {len(scored)} instances "
                  f"(timeout rate {res_v['timeout_rate']:.2f}; excluded errors: {res_v['n_excluded']})", flush=True)
            for _, r in live.iterrows():
                pred_rows.append({"verifier": v, "instance_id": r.instance_id, "arch": r.get("_arch", ""),
                                  "status": r.status, "wall_time_s": r.wall_time_s, "timed_out": r.y,
                                  "p_timeout_uncalibrated": r.get("p_timeout_uncalibrated", np.nan),
                                  "p_timeout_anchored": r.get("p_timeout_anchored", np.nan)})

        # ── 3. results ──────────────────────────────────────────────────────────────────
        per_v_auc = {v: recorded[v]["size+profile"]["mean"] for v in VERIFIERS}
        result = {
            "horizon_s": H,
            "repeats": int(config.repeats),
            "recorded": recorded,
            "ood": ood,
            "per_verifier_auc": per_v_auc,
            "min_verifier_auc": min(a for a in per_v_auc.values() if a is not None),
            # nested: MAGNET 0.1.0 crashes on a top-level empty list (Symbols.simple_view)
            "target_met": {"verifiers": sorted(v for v, a in per_v_auc.items()
                                               if a is not None and a >= config.target_auc)},
            "ood_auc": {v: d["auc"] for v, d in ood.items()},
            "ood_auc_uncalibrated": {v: d["auc_uncalibrated"] for v, d in ood.items()},
            "n_ood_instances": len(insts),
        }
        live = {v: d["auc_uncalibrated"] for v, d in ood.items() if d.get("auc_uncalibrated") is not None}
        result["n_verifiers_meeting_target"] = len(result["target_met"]["verifiers"])
        result["min_ood_auc"] = min(live.values()) if live else None   # headline live number (uncalibrated)
        for v in VERIFIERS:  # flat scalars for the MAGNET dashboard
            result[f"{v}_auc"] = per_v_auc[v]
            result[f"{v}_auc_size_only"] = recorded[v]["size"]["mean"]
            result[f"{v}_ood_auc"] = live.get(v)
        out_fpath = ub.Path(config.results_fpath)
        out_fpath.parent.ensuredir()
        out_fpath.write_text(json.dumps({"result": result}, indent=2))
        out_dir = Path(out_fpath.parent)
        with open(out_dir / "thrust2_auc.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["verifier", "feature_set", "auc_mean", "auc_std", "auc_min", "n_splits"])
            for v in VERIFIERS:
                for name in FEATURE_SETS:
                    s = recorded[v][name]
                    w.writerow([v, name, s["mean"], s["std"], s["min"], s["n"]])
            for v, d in ood.items():
                w.writerow([v, "size+profile (live OOD)", d["auc"], "", "", 1])
        if pred_rows:
            with open(out_dir / "thrust2_ood_predictions.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(pred_rows[0]))
                w.writeheader()
                w.writerows(pred_rows)
        _plot(recorded, ood, out_dir / "thrust2_auc.png", config.target_auc)
        print(f"\nWrote {out_fpath}", flush=True)
        print(f"Verifiers with held-out AUC >= {config.target_auc}: "
              f"{result['n_verifiers_meeting_target']}/{len(VERIFIERS)} (min {result['min_verifier_auc']:.3f})",
              flush=True)


__cli__ = Thrust2RunnerCLI

if __name__ == "__main__":
    __cli__.main()
