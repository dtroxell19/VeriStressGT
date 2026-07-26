"""Plot Difficulty-Profile component distributions for three groups:
  (1) TRAINED realistic constructors (Lipschitz prefix + structured head, real-world data),
  (2) SYNTHETIC bare constructors (the original sweep_all),
  (3) REAL standard networks (MNIST_fc + oval21).

Goal (rebuttal): show the trained-realistic nets match the REAL networks' difficulty profiles MORE
closely than the bare synthetic constructors do. Produced BEFORE any verifier is run.

Usage:
    PYTHONPATH=src python -m VeriStressGT.realism.profile_distributions \
        --trained rebuttal_materials/realism_sweep_analytic/difficulty_profiles.json \
        --synthetic benchmarks/sweep_all/difficulty_profiles.json \
        --real benchmarks/vnncomp_mnist_fc/difficulty_profiles.json benchmarks/oval21/difficulty_profiles.json \
        --out-dir rebuttal_materials/realism_sweep_analytic/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

COMPONENTS = [
    ("margin_sample_min", "M̂_min (min margin)"),
    ("ibp_relative_gap", "G_IBP (IBP rel. gap)"),
    ("unstable_frac", "U (unstable frac)"),
    ("A_tau_local_log", "A_τ (local regions)"),
    ("effective_grad_dim_mean", "d_eff (eff. grad dim)"),
]


def _instances(path: str) -> List[dict]:
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict) and "instances" in d:
        return d["instances"]
    return [d]  # single-instance flat dict


def _vals(insts, col):
    a = [i.get(col) for i in insts if isinstance(i.get(col), (int, float)) and np.isfinite(i.get(col))]
    return np.asarray(a, dtype=np.float64)


def _deff_over_dim(insts):
    out = []
    for i in insts:
        de, dim = i.get("effective_grad_dim_mean"), i.get("input_dim")
        if isinstance(de, (int, float)) and dim:
            out.append(de / dim)
    return np.asarray(out, dtype=np.float64)


def _slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", default=None,
                    help="Optional. Omit to plot synthetic-vs-real only (2 groups, no trained).")
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--real-cap", type=int, default=30,
                    help="Subsample each real benchmark to this many instances (equal size).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    with_trained = args.trained is not None
    trained = _instances(args.trained) if with_trained else []
    synth = _instances(args.synthetic)
    real = []; real_sources = {}
    for r in args.real:
        ins = _instances(r)
        if args.real_cap and len(ins) > args.real_cap:      # equal-size, deterministic even spacing
            idx = np.linspace(0, len(ins) - 1, args.real_cap).astype(int)
            ins = [ins[i] for i in idx]
        real_sources[Path(r).parent.name] = len(ins)
        real += ins
    print("real benchmarks:", real_sources)
    groups = ([("trained realistic", trained, "#4C72B0")] if with_trained else []) + \
             [("synthetic constructors", synth, "#DD8452"),
              ("real networks", real, "#55A868")]

    # closeness-to-real table (signed-log median distance; d_eff uses d_eff/dim)
    summary = {"n": {g: len(insts) for g, insts, _ in groups}, "components": {}}
    print("=" * 84)
    print(f"PROFILE DISTRIBUTIONS  trained={len(trained) if with_trained else 'omitted'} "
          f"synthetic={len(synth)} real={len(real)}")
    if with_trained:
        print(f"  {'component':22s} {'med(trained)':>12s} {'med(synth)':>11s} {'med(real)':>10s}  {'closer to real':>14s}")
    else:
        print(f"  {'component':22s} {'med(synth)':>11s} {'med(real)':>10s}  {'|slog diff|':>11s}")
    for col, label in COMPONENTS:
        if col == "effective_grad_dim_mean":
            vt = _deff_over_dim(trained) if with_trained else np.array([])
            vs, vr = _deff_over_dim(synth), _deff_over_dim(real)
            label = "d_eff / input_dim"
        else:
            vt = _vals(trained, col) if with_trained else np.array([])
            vs, vr = _vals(synth, col), _vals(real, col)
        if vr.size == 0 or vs.size == 0:
            continue
        ms, mr = np.median(vs), np.median(vr)
        dsn = abs(_slog(ms) - _slog(mr))
        rec = {"med_synth": float(ms), "med_real": float(mr), "dist_synth": float(dsn)}
        if with_trained and vt.size:
            mt = np.median(vt); dt = abs(_slog(mt) - _slog(mr))
            rec.update({"med_trained": float(mt), "dist_trained": float(dt),
                        "closer_to_real": "TRAINED" if dt < dsn else "synthetic"})
            print(f"  {label:22s} {mt:12.3g} {ms:11.3g} {mr:10.3g}  {rec['closer_to_real']:>14s}")
        else:
            print(f"  {label:22s} {ms:11.3g} {mr:10.3g}  {dsn:11.3f}")
        summary["components"][col] = rec
    (out / "distribution_summary.json").write_text(json.dumps(summary, indent=2))

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tick_labels = [{"trained realistic": "trained", "synthetic constructors": "synth",
                    "real networks": "real"}[g[0]] for g in groups]
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(4 * len(COMPONENTS), 3.6))
    for ax, (col, label) in zip(axes, COMPONENTS):
        if col == "effective_grad_dim_mean":
            xs = [_deff_over_dim(g[1]) for g in groups]; label = "d_eff / input_dim"
        else:
            xs = [_slog(_vals(g[1], col)) for g in groups]
        parts = ax.violinplot(xs, showmedians=True, showextrema=False)
        for pc, g in zip(parts["bodies"], groups):
            pc.set_facecolor(g[2]); pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("value" if col == "effective_grad_dim_mean" else "signed log1p", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    title = ("Difficulty-Profile component distributions: trained realistic vs synthetic vs real"
             if with_trained else
             "Difficulty-Profile component distributions: synthetic constructors vs real networks")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = ("profile_distributions_trained_vs_synth_vs_real.png" if with_trained
             else "profile_synth_vs_real.png")
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out}/{fname} + distribution_summary.json")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
