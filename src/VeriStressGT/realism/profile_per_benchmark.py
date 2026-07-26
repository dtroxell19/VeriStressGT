"""Per-real-benchmark Difficulty-Profile breakdown.

Instead of pooling all real benchmarks into one wide distribution, compare the TRAINED realistic
constructors and the SYNTHETIC bare constructors against EACH real benchmark separately, per component.

Rebuttal claim this supports: real networks are HETEROGENEOUS in difficulty; the synthetic constructors
reproduce the hard end (e.g. MNIST_fc), the trained constructors reproduce the lower-difficulty end
(e.g. oval21, rl_benchmarks), and together they span the real spectrum ("complementary coverage").

Outputs (to --out-dir):
  - profile_per_benchmark_points.png : 5 component panels; trained/synth as violins, each real
    benchmark as a labelled median point -> shows which real benchmarks fall in which generator's range.
  - profile_per_benchmark_winner.png : heatmap rows=real benchmarks, cols=components, coloured by which
    generator (trained=blue / synth=orange) is closer, annotated with the real median.
  - per_benchmark_summary.json : medians + per-cell distances + winner + per-benchmark win counts.

Usage:
    PYTHONPATH=src python -m VeriStressGT.realism.profile_per_benchmark \
        --trained rebuttal_materials/realism_sweep_analytic/difficulty_profiles.json \
        --synthetic benchmarks/sweep_all/difficulty_profiles.json \
        --real benchmarks/real_*/difficulty_profiles.json benchmarks/vnncomp_mnist_fc/difficulty_profiles.json \
        --out-dir rebuttal_materials/realism_sweep_analytic/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

COMPONENTS = [
    ("margin_sample_min", "M̂_min"),
    ("ibp_relative_gap", "G_IBP"),
    ("unstable_frac", "U"),
    ("A_tau_local_log", "A_τ"),
    ("effective_grad_dim_mean", "d_eff/dim"),
]


def _instances(path: str) -> List[dict]:
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict) and "instances" in d:
        return d["instances"]
    return [d]


def _vals(insts, col):
    if col == "effective_grad_dim_mean":
        a = [i["effective_grad_dim_mean"] / i["input_dim"] for i in insts
             if isinstance(i.get("effective_grad_dim_mean"), (int, float)) and i.get("input_dim")
             and np.isfinite(i["effective_grad_dim_mean"])]
    else:
        a = [i.get(col) for i in insts
             if isinstance(i.get(col), (int, float)) and np.isfinite(i.get(col))]
    return np.asarray(a, dtype=np.float64)


def _slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def _name(path: str) -> str:
    return Path(path).parent.name.replace("real_", "").replace("vnncomp_", "").replace("vnncomp22_", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--min-vals", type=int, default=3,
                    help="Skip a (benchmark, component) cell with fewer than this many finite values.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    trained = _instances(args.trained)
    synth = _instances(args.synthetic)
    reals = {}
    for r in args.real:
        try:
            reals[_name(r)] = _instances(r)
        except FileNotFoundError:
            print(f"  (skip missing {r})")
    names = list(reals.keys())

    tmed = {c: (np.median(_vals(trained, c)) if _vals(trained, c).size else np.nan) for c, _ in COMPONENTS}
    smed = {c: (np.median(_vals(synth, c)) if _vals(synth, c).size else np.nan) for c, _ in COMPONENTS}

    summary = {"trained_median": {c: float(tmed[c]) for c, _ in COMPONENTS},
               "synth_median": {c: float(smed[c]) for c, _ in COMPONENTS},
               "benchmarks": {}}
    # winner matrix: +1 trained closer, -1 synth closer, 0 = no data; magnitude = |dS|-|dT| in slog units
    winner = np.zeros((len(names), len(COMPONENTS)))
    realmed = np.full((len(names), len(COMPONENTS)), np.nan)
    print("=" * 100)
    print("PER-REAL-BENCHMARK breakdown (T=trained closer, S=synthetic closer; median shown)")
    hdr = f"  {'benchmark':16s} " + " ".join(f"{lab:>14s}" for _, lab in COMPONENTS) + "   T/S"
    print(hdr)
    print(f"  {'[trained med]':16s} " + " ".join(f"{tmed[c]:14.3g}" for c, _ in COMPONENTS))
    print(f"  {'[synth med]':16s} " + " ".join(f"{smed[c]:14.3g}" for c, _ in COMPONENTS))
    print("-" * 100)
    for bi, nm in enumerate(names):
        ins = reals[nm]; wt = ws = 0; cells = []; bsum = {}
        for ci, (c, _) in enumerate(COMPONENTS):
            rv = _vals(ins, c)
            if rv.size < args.min_vals or not np.isfinite(tmed[c]):
                cells.append(f"{'n/a':>12s}"); bsum[c] = None
                continue
            mr = float(np.median(rv)); realmed[bi, ci] = mr
            dt = abs(_slog(tmed[c]) - _slog(mr))
            ds = abs(_slog(smed[c]) - _slog(mr)) if np.isfinite(smed[c]) else np.inf
            win = "T" if dt < ds else "S"; wt += win == "T"; ws += win == "S"
            winner[bi, ci] = (ds - dt)  # >0 trained closer
            cells.append(f"{mr:10.3g}[{win}]")
            bsum[c] = {"median": mr, "dist_trained": float(dt), "dist_synth": float(ds), "closer": win}
        summary["benchmarks"][nm] = {"n": len(ins), "components": bsum, "wins_trained": wt, "wins_synth": ws}
        print(f"  {nm:16s} " + " ".join(f"{x:>14s}" for x in cells) + f"   {wt}/{ws}")
    (out / "per_benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- Figure 1: per-component panels, real benchmarks as labelled points ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(4.1 * len(COMPONENTS), 4.2))
    for ci, (ax, (c, lab)) in enumerate(zip(axes, COMPONENTS)):
        vt, vs = _slog(_vals(trained, c)), _slog(_vals(synth, c))
        parts = ax.violinplot([vt, vs], positions=[0, 1], showmedians=True, showextrema=False, widths=0.8)
        for pc, col in zip(parts["bodies"], ["#4C72B0", "#DD8452"]):
            pc.set_facecolor(col); pc.set_alpha(0.55)
        for bi, nm in enumerate(names):
            mr = realmed[bi, ci]
            if not np.isfinite(mr):
                continue
            ax.scatter([2 + (bi - len(names) / 2) * 0.04], [_slog(mr)], color=palette(bi % 10),
                       s=42, zorder=5, edgecolor="k", linewidth=0.4,
                       label=nm if ci == 0 else None)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["trained", "synth", "real\nbenchmarks"], fontsize=8)
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel("signed log1p(median)", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.legend(loc="lower center", ncol=min(len(names), 7), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Difficulty-Profile components: trained & synthetic ranges vs each real benchmark", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out / "profile_per_benchmark_points.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: winner heatmap ----------
    fig, ax = plt.subplots(figsize=(1.5 * len(COMPONENTS) + 2, 0.6 * len(names) + 1.6))
    vmax = np.nanmax(np.abs(winner)) or 1.0
    im = ax.imshow(winner, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(COMPONENTS))); ax.set_xticklabels([l for _, l in COMPONENTS])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for bi in range(len(names)):
        for ci in range(len(COMPONENTS)):
            mr = realmed[bi, ci]
            if not np.isfinite(mr):
                ax.text(ci, bi, "n/a", ha="center", va="center", fontsize=7, color="gray")
            else:
                w = "T" if winner[bi, ci] > 0 else "S"
                ax.text(ci, bi, f"{mr:.2g}\n{w}", ha="center", va="center", fontsize=7)
    ax.set_title("Closer generator per (benchmark, component)\nblue = trained closer, red = synthetic closer", fontsize=10)
    fig.colorbar(im, ax=ax, label="dist(synth) - dist(trained)   [signed-log]")
    fig.tight_layout()
    fig.savefig(out / "profile_per_benchmark_winner.png", dpi=150)
    plt.close(fig)

    print("-" * 100)
    tt = sum(v["wins_trained"] for v in summary["benchmarks"].values())
    ss = sum(v["wins_synth"] for v in summary["benchmarks"].values())
    print(f"total cells: trained-closer={tt}  synth-closer={ss}")
    print(f"Wrote {out}/profile_per_benchmark_points.png, profile_per_benchmark_winner.png, per_benchmark_summary.json")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
