"""Difficulty-Profile violins: synthetic vs real (all) vs real (UNSAT-only).

Our constructed instances are all PROVABLY ROBUST (UNSAT). The fair real comparison is real instances
that are *also* verified robust (UNSAT by >=1 verifier), not the full real set (which mixes SAT /
counterexample and timeout instances). This adds the "real (UNSAT-only)" group.

For each real benchmark we read the verifier results (results.jsonl) and keep profile instances whose
status is UNSAT.

Usage:
    PYTHONPATH=src python -m VeriStressGT.realism.profile_unsat \
        --synthetic benchmarks/sweep_all/difficulty_profiles.json \
        --real NAME=benchmarks/<b>/difficulty_profiles.json:rebuttal_materials/real_verify/<b>/results.jsonl ... \
        --out-dir rebuttal_materials/realism_sweep_analytic/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMPONENTS = [
    ("margin_sample_min", "M̂_min (min margin)"),
    ("ibp_relative_gap", "G_IBP (IBP rel. gap)"),
    ("unstable_frac", "U (unstable frac)"),
    ("A_tau_local_log", "A_τ (local regions)"),
    ("effective_grad_dim_mean", "d_eff / input_dim"),
]


def _instances(path):
    d = json.loads(Path(path).read_text())
    return d["instances"] if isinstance(d, dict) and "instances" in d else [d]


def _unsat_ids(results_jsonl):
    ids = set()
    for line in Path(results_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("status", "")).upper() == "UNSAT":
            ids.add(str(r.get("instance_id")))
    return ids


def _vals(insts, col):
    if col == "effective_grad_dim_mean":
        a = [i[col] / i["input_dim"] for i in insts
             if isinstance(i.get(col), (int, float)) and i.get("input_dim") and np.isfinite(i[col])]
    else:
        a = [i.get(col) for i in insts
             if isinstance(i.get(col), (int, float)) and np.isfinite(i.get(col))]
    return np.asarray(a, float)


def _slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--real", nargs="+", required=True,
                    help="NAME=profiles.json:results.jsonl entries.")
    ap.add_argument("--real-cap", type=int, default=30)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    synth = _instances(args.synthetic)
    real_all, real_unsat = [], []
    per_bench = {}
    for entry in args.real:
        name, rest = entry.split("=", 1)
        prof_path, res_path = rest.split(":", 1)
        insts = _instances(prof_path)
        unsat = _unsat_ids(res_path)
        keep = [i for i in insts if str(i.get("instance_id")) in unsat]
        # equal-size cap on BOTH pools for balanced per-benchmark weighting
        allp = insts
        if args.real_cap and len(allp) > args.real_cap:
            idx = np.linspace(0, len(allp) - 1, args.real_cap).astype(int)
            allp = [allp[i] for i in idx]
        if args.real_cap and len(keep) > args.real_cap:
            idx = np.linspace(0, len(keep) - 1, args.real_cap).astype(int)
            keep = [keep[i] for i in idx]
        real_all += allp
        real_unsat += keep
        per_bench[name] = {"n_total": len(insts), "n_unsat": len(keep)}
    print("per-benchmark UNSAT counts:")
    for k, v in per_bench.items():
        print(f"  {k:22s} UNSAT {v['n_unsat']:3d} / {v['n_total']:3d}")
    print(f"pooled: synth={len(synth)}  real_all={len(real_all)}  real_unsat={len(real_unsat)}")

    groups = [("synthetic constructors", synth, "#DD8452"),
              ("real (all)", real_all, "#9AC7A8"),
              ("real (UNSAT-only)", real_unsat, "#2E7D32")]

    # closeness of synthetic to real-UNSAT
    print(f"\n  {'component':22s} {'med(synth)':>11s} {'med(real_all)':>13s} {'med(real_UNSAT)':>15s}")
    summary = {"per_bench": per_bench, "components": {}}
    for col, label in COMPONENTS:
        vs, va, vu = _vals(synth, col), _vals(real_all, col), _vals(real_unsat, col)
        ms = np.median(vs) if vs.size else np.nan
        ma = np.median(va) if va.size else np.nan
        mu = np.median(vu) if vu.size else np.nan
        summary["components"][col] = {"med_synth": float(ms), "med_real_all": float(ma),
                                      "med_real_unsat": float(mu),
                                      "dist_synth_to_unsat": float(abs(_slog(ms) - _slog(mu)))}
        print(f"  {label:22s} {ms:11.3g} {ma:13.3g} {mu:15.3g}")
    (out / "unsat_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(3.7 * len(COMPONENTS), 3.8))
    for ax, (col, label) in zip(axes, COMPONENTS):
        if col == "effective_grad_dim_mean":
            xs = [_vals(g[1], col) for g in groups]; yl = "value"
        else:
            xs = [_slog(_vals(g[1], col)) for g in groups]; yl = "signed log1p"
        xs = [x if x.size else np.array([np.nan]) for x in xs]
        parts = ax.violinplot(xs, showmedians=True, showextrema=False)
        for pc, g in zip(parts["bodies"], groups):
            pc.set_facecolor(g[2]); pc.set_alpha(0.65)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["synth", "real\n(all)", "real\n(UNSAT)"], fontsize=8)
        ax.set_title(label, fontsize=10); ax.set_ylabel(yl, fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Difficulty-Profile: synthetic vs real (all) vs real verified-robust (UNSAT by ≥1 verifier)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / "profile_synth_vs_real_unsat.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}/profile_synth_vs_real_unsat.png + unsat_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
