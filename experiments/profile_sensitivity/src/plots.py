"""Figures for the Difficulty-Profile sensitivity study.

Consumes results/*.csv and writes plots/*.png.  Visual style matches the
predictive-modeling figures (recessive grid, RdBu_r correlation heatmaps).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

COMP = C.COMPONENTS
LABEL = {"margin_hat_min": "M̂_min", "g_ibp": "G_IBP",
         "unstable_fraction": "U", "a_tau": "A_τ", "d_eff": "d_eff"}
COL_PROF = "#F58518"; COL_S = "#4C78A8"; INK = "#2B2B2B"; MUTED = "#8A8A8A"; GRID = "#E6E6E6"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = C.PLOTS


def _exists(name):
    return (C.RESULTS / name).exists()


def fig_stability_vs_N():
    if not _exists("stability_vs_N.csv"):
        return
    df = pd.read_csv(C.RESULTS / "stability_vs_N.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for comp in df.component.unique():
        d = df[df.component == comp].sort_values("n_samples")
        ax.plot(d.n_samples, d.norm_drift, "-o", ms=4, label=LABEL.get(comp, comp))
    ax.axhline(0.05, color=MUTED, ls="--", lw=1)
    ax.text(df.n_samples.min(), 0.052, "5% tolerance", fontsize=7, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sample count N"); ax.set_ylabel("normalized drift vs N=1600 estimate")
    ax.set_title("Component stability vs sample count", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "stability_vs_N.png", bbox_inches="tight"); plt.close(fig)


def fig_cov_bar():
    if not _exists("cov_table.csv"):
        return
    df = pd.read_csv(C.RESULTS / "cov_table.csv")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    y = np.arange(len(df))
    ax.barh(y, df.median_cov, color=COL_PROF, height=0.65)
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(c, c) for c in df.component])
    ax.set_xlabel("run-to-run coefficient of variation (median over instances)")
    ax.set_title("Seed stability of each component", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "cov_bar.png", bbox_inches="tight"); plt.close(fig)


def fig_tornado():
    if not _exists("tornado.csv"):
        return
    df = pd.read_csv(C.RESULTS / "tornado.csv").dropna(subset=["swing"])
    comps = [c for c in COMP if c in df.component.unique()]
    fig, axes = plt.subplots(1, len(comps), figsize=(2.6 * len(comps), 3.2), squeeze=False)
    for ax, comp in zip(axes[0], comps):
        d = df[df.component == comp].sort_values("swing")
        ax.barh(np.arange(len(d)), d.swing, color=COL_S, height=0.6)
        ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d.knob, fontsize=7)
        ax.set_title(LABEL.get(comp, comp), fontsize=9)
        ax.set_xlabel("norm. swing", fontsize=8)
    fig.suptitle("Knob sensitivity per component (normalized swing across each knob's grid)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "tornado.png", bbox_inches="tight"); plt.close(fig)


def _heat(ax, M, title):
    im = ax.imshow(M.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels([LABEL.get(c, c) for c in M.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels([LABEL.get(c, c) for c in M.index], fontsize=8)
    for i in range(len(M.index)):
        for j in range(len(M.columns)):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=INK if abs(v) < 0.6 else "white")
    ax.set_title(title, fontsize=9)
    return im


def fig_dependence():
    for kind, vlab in [("spearman", "Spearman ρ"), ("dcor", "distance corr")]:
        doms = [d for d in ["all", "synthetic", "established"]
                if (C.RESULTS / "correlation" / f"{kind}_{d}.csv").exists()]
        if not doms:
            continue
        fig, axes = plt.subplots(1, len(doms), figsize=(3.6 * len(doms), 3.4), squeeze=False)
        for ax, dom in zip(axes[0], doms):
            M = pd.read_csv(C.RESULTS / "correlation" / f"{kind}_{dom}.csv", index_col=0)
            im = _heat(ax, M, f"{vlab} — {dom}")
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025)
        fig.savefig(OUT / f"{kind}_5x5.png", bbox_inches="tight"); plt.close(fig)


def fig_atau_grid():
    if not _exists("atau_grid.csv"):
        return
    df = pd.read_csv(C.RESULTS / "atau_grid.csv")
    piv = df.pivot(index="proj_dim", columns="atau_width", values="value")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    im = ax.imshow(piv.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("A_τ grid width τ"); ax.set_ylabel("projection dim")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title("A_τ vs grid width × projection (mean over subset)", fontsize=9)
    fig.colorbar(im, label="A_τ"); fig.tight_layout()
    fig.savefig(OUT / "atau_grid.png", bbox_inches="tight"); plt.close(fig)


def fig_u_tau():
    if not _exists("u_tau_overall.csv"):
        return
    df = pd.read_csv(C.RESULTS / "u_tau_overall.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for mode, d in df.groupby("u_mode"):
        d = d.sort_values("u_tau")
        ax.plot(d.u_tau, d.value, "-o", ms=4, label=f"{mode}")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("u_tau"); ax.set_ylabel("mean unstable fraction U")
    ax.set_title("U vs τ (width = legacy, omega = paper eq.12)\nReLU is τ-invariant; τ only bites smooth activations", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "u_tau_curve.png", bbox_inches="tight"); plt.close(fig)


def fig_eta():
    if not _exists("eta_effect.csv"):
        return
    df = pd.read_csv(C.RESULTS / "eta_effect.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for comp, d in df.groupby("component"):
        d = d.sort_values("eta")
        ax.plot(d.eta, d.value, "-o", ms=4, label=LABEL.get(comp, comp))
    ax.set_xscale("log")
    ax.set_xlabel("η (numerical-stability constant)"); ax.set_ylabel("mean component value")
    ax.set_title("G_IBP and d_eff vs η (near-flat: η is a numerical constant, not a difficulty knob)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "eta_effect.png", bbox_inches="tight"); plt.close(fig)


def fig_index_validation():
    if not _exists("difficulty_index.csv"):
        return
    di = pd.read_csv(C.RESULTS / "difficulty_index.csv")
    val = pd.read_csv(C.RESULTS / "index_validation.csv") if _exists("index_validation.csv") else None
    params = json.load(open(C.RESULTS / "difficulty_index_params.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for col, color in [("DI_pred", COL_PROF), ("DI_PCA", COL_S)]:
        d = di[[col, "p_timeout"]].dropna()
        dec = pd.qcut(d[col].rank(method="first"), 10, labels=False)
        dm = d.groupby(dec).agg(x=(col, "mean"), y=("p_timeout", "mean"))
        r = ""
        if val is not None and col in set(val["index"]):
            rr = val[val["index"] == col].iloc[0]
            r = f"  (ρ={rr.spearman:.2f}, isoR²={rr.isotonic_r2:.2f})"
        ax.plot(dm.x, dm.y, "-o", ms=5, color=color, label=col + r)
    ax.set_xlabel("Difficulty Index (decile mean)"); ax.set_ylabel("empirical P(timeout)")
    ax.set_title("Unified Difficulty Index vs verifier timeout rate", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    # component loadings of the PCA index
    ax2 = axes[1]
    comps = list(params["pca_components"].keys())
    load = [params["pca_components"][c] * params.get("pca_sign", 1.0) for c in comps]
    ax2.bar(range(len(comps)), load, color=COL_S)
    ax2.set_xticks(range(len(comps))); ax2.set_xticklabels([LABEL.get(c, c) for c in comps])
    ax2.set_ylabel("PC1 loading (oriented)")
    ax2.set_title(f"DI_PCA loadings (EVR={params['pca_explained_variance_ratio']:.0%})", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "difficulty_index_validation.png", bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_stability_vs_N(); fig_cov_bar(); fig_tornado()
    fig_dependence(); fig_atau_grid(); fig_u_tau(); fig_eta()
    fig_index_validation()
    print("plots ->", OUT, flush=True)


if __name__ == "__main__":
    main()
