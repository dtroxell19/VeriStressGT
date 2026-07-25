"""Figures for the actionable-feature analysis (analogs of prior-work
Figs 2, 3, 7): bootstrap null vs observed, per-instance delta forest by family,
profile-value density split by delta sign, partial-dependence, and permutation-
importance stability. Consumes results/actionable/ and refits a GBM for PDP/importance.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

import common as C, models as M
import actionable_analysis as A

COL_S = "#4C78A8"; COL_SD = "#F58518"; COL_POS = "#E4572E"; COL_NEG = "#3A86A8"
INK = "#2B2B2B"; MUTED = "#8A8A8A"; GRID = "#E6E6E6"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})
PROF = C.PROFILE_FEATURES
OUT = C.PLOTS


def fig_htest(ht):
    ht = ht.copy(); ht["cell"] = ht.scenario.str[:4] + "·" + ht.verifier
    ht = ht.sort_values(["scenario", "verifier"])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(ht) + 1))
    y = np.arange(len(ht))
    ax.barh(y, ht.null_dAUC_p95, color=GRID, height=0.7, label="null 95% (shuffled feature-set value)")
    ax.plot(ht.null_dAUC_mean, y, "|", color=MUTED, ms=12, label="null mean")
    for i, (_, r) in enumerate(ht.iterrows()):
        col = COL_SD if r.p_AUC < 0.05 else MUTED
        ax.plot(r.obs_dAUC, i, "o", color=col, ms=7)
        ax.text(r.obs_dAUC + 0.004, i, f"Δ={r.obs_dAUC:+.3f}  p={r.p_AUC:.3f}", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(ht.cell, fontsize=8)
    ax.set_xlabel("Held-out ΔAUC (S+Profile − S), GBM   ·   dot = observed, bar = null 95%")
    ax.set_title("Bootstrap hypothesis test: profile adds predictive value beyond adding variables", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "actionable_htest.png", bbox_inches="tight"); plt.close(fig)


def fig_delta_forest(scenario="combined", verifier="marabou"):
    p = C.RESULTS / "actionable" / f"delta_{scenario}_{verifier}.csv"
    if not p.exists(): return
    d = pd.read_csv(p)
    g = d.groupby("constructor").agg(delta=("delta", "mean"),
                                     lo=("delta_lo", "mean"), hi=("delta_hi", "mean"),
                                     n=("delta", "size")).reset_index().sort_values("delta")
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(g) + 1.5))
    yy = np.arange(len(g))
    for i, (_, r) in enumerate(g.iterrows()):
        col = COL_POS if r.delta > 0 else COL_NEG
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.delta, i, "o", color=col, ms=6)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([f"{r.constructor} (n={r.n})" for _, r in g.iterrows()], fontsize=8)
    ax.set_xlabel("δ = P̂(timeout | size+profile) − P̂(timeout | size)   (mean per family, 95% grouped-bootstrap CI)")
    ax.set_title(f"Effect of adding the Difficulty Profile — {verifier}, {scenario}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"actionable_delta_forest_{scenario}_{verifier}.png", bbox_inches="tight"); plt.close(fig)


def fig_density_by_sign(scenario="combined", verifier="marabou"):
    p = C.RESULTS / "actionable" / f"delta_{scenario}_{verifier}.csv"
    if not p.exists(): return
    d = pd.read_csv(p)
    feats = [("g_ibp", "G_IBP (signed-log1p)"), ("unstable_fraction", "unstable fraction")]
    fig, axes = plt.subplots(1, len(feats), figsize=(5.2 * len(feats), 3.6))
    for ax, (f, lab) in zip(axes, feats):
        pos = d[d.delta > 0][f].dropna(); neg = d[d.delta <= 0][f].dropna()
        x = C.apply_transforms(d, [f])[f]
        xp = C.apply_transforms(d[d.delta > 0], [f])[f].dropna()
        xn = C.apply_transforms(d[d.delta <= 0], [f])[f].dropna()
        for vals, col, lb in [(xp, COL_POS, "δ>0 (profile ↑ timeout prob)"),
                              (xn, COL_NEG, "δ≤0 (profile ↓ timeout prob)")]:
            if len(vals) > 3:
                ax.hist(vals, bins=15, density=True, alpha=0.45, color=col, label=lb)
                ax.axvline(np.median(vals), color=col, ls="--", lw=1.2)
        ax.set_xlabel(lab); ax.set_ylabel("density")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Profile values by direction of effect — {verifier}, {scenario}", y=1.02, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"actionable_density_{scenario}_{verifier}.png", bbox_inches="tight"); plt.close(fig)


def _fit_gbm_cell(df, v, dom):
    d, y = A._cell(df, v, dom)
    cols = M.feature_sets()["SD"]
    X = pd.DataFrame(M.build_design_matrix(d, cols), columns=cols)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=0).fit(X, y)
    return m, X, y, d.network_group.to_numpy(), cols


def fig_pdp(df, v="marabou", dom="combined"):
    m, X, y, g, cols = _fit_gbm_cell(df, v, dom)
    fig, axes = plt.subplots(1, len(PROF), figsize=(3.0 * len(PROF), 3.0))
    for ax, f in zip(axes, PROF):
        pd_ = partial_dependence(m, X, [cols.index(f)], grid_resolution=30, kind="average")
        gx = pd_["grid_values"][0]; yv = pd_["average"][0]
        ax.plot(gx, yv, color=COL_SD, lw=2)
        ax.set_title(f, fontsize=8); ax.set_xlabel("transformed value", fontsize=7)
    axes[0].set_ylabel("partial dependence\n(P timeout)")
    fig.suptitle(f"Partial dependence of profile components — {v}, {dom}", y=1.04, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"actionable_pdp_{dom}_{v}.png", bbox_inches="tight"); plt.close(fig)
    # 2D interaction: g_ibp x n_param_total
    if "n_param_total" in cols:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        pdp2 = partial_dependence(m, X, [(cols.index("g_ibp"), cols.index("n_param_total"))],
                                  grid_resolution=20, kind="average")
        Z = pdp2["average"][0]; gx, gy = pdp2["grid_values"]
        im = ax.contourf(gx, gy, Z.T, levels=12, cmap="magma")
        ax.set_xlabel("g_ibp (transformed)"); ax.set_ylabel("n_param_total (transformed)")
        ax.set_title(f"2-D PDP: G_IBP × params — {v}, {dom}", fontsize=9)
        fig.colorbar(im, label="P(timeout)"); fig.tight_layout()
        fig.savefig(OUT / f"actionable_pdp2d_{dom}_{v}.png", bbox_inches="tight"); plt.close(fig)


def fig_importance(df, v="marabou", dom="combined", n_repeats=30):
    m, X, y, g, cols = _fit_gbm_cell(df, v, dom)
    r = permutation_importance(m, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=0)
    imp = pd.DataFrame({"feature": cols, "mean": r.importances_mean, "std": r.importances_std})
    imp["is_prof"] = imp.feature.isin(PROF)
    imp = imp.sort_values("mean", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(6.5, 0.34 * len(imp) + 1))
    cols_c = [COL_SD if p else COL_S for p in imp.is_prof]
    ax.barh(np.arange(len(imp)), imp["mean"], xerr=imp["std"], color=cols_c, height=0.7)
    ax.set_yticks(np.arange(len(imp))); ax.set_yticklabels(imp.feature, fontsize=8)
    ax.set_xlabel("permutation importance (AUC drop)")
    ax.set_title(f"Feature importance (profile = orange) — {v}, {dom}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"actionable_importance_{dom}_{v}.png", bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ht_path = C.RESULTS / "actionable" / "bootstrap_htest_h240.csv"
    if ht_path.exists():
        fig_htest(pd.read_csv(ht_path))
    df = C.load_rows()
    for v, dom in [("marabou", "combined"), ("abcrown", "combined"), ("nnenum", "combined")]:
        fig_delta_forest(dom, v); fig_density_by_sign(dom, v)
        try: fig_pdp(df, v, dom)
        except Exception as e: print("pdp skip", v, e)
        try: fig_importance(df, v, dom)
        except Exception as e: print("imp skip", v, e)
    print("actionable plots ->", OUT)


if __name__ == "__main__":
    main()
