"""Rebuttal-quality figures. Reads results/{cv,transfer,coefficients} and writes
PNGs under plots/. Every figure states verifier, scenario, eligible n, timeout
base rate, grouping unit, and the repeated-CV design. Guards missing inputs so it
can run on a partial results tree.
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

import common as C

# CVD-safe pair (blue / orange) + neutral
COL_S = "#4C78A8"      # size/type baseline
COL_SD = "#F58518"     # size + profile
COL_D = "#54A24B"      # profile only
INK = "#2B2B2B"; MUTED = "#8A8A8A"; GRID = "#E6E6E6"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})

SCEN_ORDER = ["A_synthetic", "B_established", "C_combined"]
SCEN_LABEL = {"A_synthetic": "Synthetic (225 nets)", "B_established": "Established (6 nets, exploratory)",
              "C_combined": "Combined (231 nets)"}


def load_summaries():
    out = {}
    for f in glob.glob(str(C.RESULTS / "cv" / "*.summary.json")):
        tag = os.path.basename(f).replace(".summary.json", "")
        out[tag] = json.load(open(f))
    return out


def load_oof(tag):
    p = C.RESULTS / "cv" / f"{tag}.oof.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _annot(ax, meta):
    ax.text(0.5, -0.32, f"n={meta['n']}  base={meta['base_rate']:.2f}  groups={meta['n_groups']}  "
            f"H={meta['horizon_s']}s  |  5×20 grouped CV",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color=MUTED)


def fig_auc_comparison(summ):
    scns = [s for s in SCEN_ORDER if any(k.startswith(s) for k in summ)]
    fig, axes = plt.subplots(1, len(scns), figsize=(4.3 * len(scns), 4.0), squeeze=False)
    for ci, scn in enumerate(scns):
        ax = axes[0][ci]
        vs, s_au, sd_au, sd_lo, sd_hi, s_lo, s_hi = [], [], [], [], [], [], []
        for v in C.VERIFIERS:
            tag = f"{scn}__{v}"
            if tag not in summ or not summ[tag]["summary"]:
                continue
            fsd = summ[tag]["summary"]["feature_sets"]
            vs.append(v)
            s_au.append(fsd["S"]["auc"]["median"]); sd_au.append(fsd["SD"]["auc"]["median"])
            s_lo.append(fsd["S"]["auc"]["median"] - fsd["S"]["auc"]["p10"])
            s_hi.append(fsd["S"]["auc"]["p90"] - fsd["S"]["auc"]["median"])
            sd_lo.append(fsd["SD"]["auc"]["median"] - fsd["SD"]["auc"]["p10"])
            sd_hi.append(fsd["SD"]["auc"]["p90"] - fsd["SD"]["auc"]["median"])
        x = np.arange(len(vs)); w = 0.38
        ax.bar(x - w/2, s_au, w, yerr=[s_lo, s_hi], color=COL_S, label="Size/type", capsize=2)
        ax.bar(x + w/2, sd_au, w, yerr=[sd_lo, sd_hi], color=COL_SD, label="Size+Profile", capsize=2)
        for xi in range(len(vs)):
            ax.text(xi - w/2, s_au[xi] + s_hi[xi] + 0.02, f"{s_au[xi]:.2f}", ha="center", fontsize=7, color=INK)
            ax.text(xi + w/2, sd_au[xi] + sd_hi[xi] + 0.02, f"{sd_au[xi]:.2f}", ha="center", fontsize=7, color=INK)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(vs, rotation=0, fontsize=9)
        ax.set_ylim(0.4, 1.0); ax.set_title(SCEN_LABEL.get(scn, scn), fontsize=10)
        if ci == 0: ax.set_ylabel("Held-out ROC-AUC (median over 20 repeats)")
    axes[0][0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Timeout prediction: size/type vs size/type+Difficulty-Profile", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(C.PLOTS / "auc_comparison.png", bbox_inches="tight"); plt.close(fig)


def fig_delta_auc(summ):
    rows = []
    for tag, s in summ.items():
        if not s["summary"] or "bootstrap_dauc" not in s["summary"]:
            continue
        scn, v = tag.split("__")
        b = s["summary"]["bootstrap_dauc"]
        rows.append((scn, v, b["point"], b["lo"], b["hi"], s["meta"]["eligible"]))
    if not rows:
        return
    rows.sort(key=lambda r: (SCEN_ORDER.index(r[0]) if r[0] in SCEN_ORDER else 9, r[1]))
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.42 * len(rows))))
    for i, (scn, v, pt, lo, hi, elig) in enumerate(rows):
        col = COL_SD if (lo is not None and lo > 0) else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(pt, i, "o", color=col, ms=6)
        ax.text(hi + 0.005, i, f"{pt:+.3f}", va="center", fontsize=7, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{SCEN_LABEL.get(s,s).split(' (')[0]} · {v}" + ("" if e else " *expl")
                        for s, v, _, _, _, e in rows], fontsize=8)
    ax.set_xlabel("Aggregate ΔAUC (Size+Profile − Size)   with 95% group-bootstrap CI")
    ax.set_title("Does the Difficulty Profile add held-out predictive value?", fontsize=11)
    fig.tight_layout(); fig.savefig(C.PLOTS / "delta_auc.png", bbox_inches="tight"); plt.close(fig)


def fig_brier(summ):
    scns = [s for s in SCEN_ORDER if any(k.startswith(s) for k in summ)]
    fig, axes = plt.subplots(1, len(scns), figsize=(4.3 * len(scns), 3.6), squeeze=False)
    for ci, scn in enumerate(scns):
        ax = axes[0][ci]; vs, s_b, sd_b = [], [], []
        for v in C.VERIFIERS:
            tag = f"{scn}__{v}"
            if tag not in summ or not summ[tag]["summary"]: continue
            fsd = summ[tag]["summary"]["feature_sets"]; vs.append(v)
            s_b.append(fsd["S"]["brier"]["median"]); sd_b.append(fsd["SD"]["brier"]["median"])
        x = np.arange(len(vs)); w = 0.38
        ax.bar(x - w/2, s_b, w, color=COL_S, label="Size/type")
        ax.bar(x + w/2, sd_b, w, color=COL_SD, label="Size+Profile")
        ax.set_xticks(x); ax.set_xticklabels(vs); ax.set_title(SCEN_LABEL.get(scn, scn), fontsize=9)
        if ci == 0: ax.set_ylabel("Brier score (lower better)")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Calibration (Brier) — size/type vs size/type+profile", y=1.02)
    fig.tight_layout(); fig.savefig(C.PLOTS / "brier_comparison.png", bbox_inches="tight"); plt.close(fig)


def fig_roc_pr_calib(summ):
    for scn in [s for s in SCEN_ORDER if any(k.startswith(s) for k in summ)]:
        for v in C.VERIFIERS:
            tag = f"{scn}__{v}"
            oof = load_oof(tag)
            if oof is None: continue
            agg = (oof.groupby(["instance_id", "feature_set"]).agg(y=("y","first"), p=("p","mean")).reset_index())
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
            for fs, col in [("S", COL_S), ("SD", COL_SD)]:
                d = agg[agg.feature_set == fs]
                if d.y.nunique() < 2: continue
                fpr, tpr, _ = roc_curve(d.y, d.p); axs[0].plot(fpr, tpr, color=col, label=fs)
                pr, rc, _ = precision_recall_curve(d.y, d.p); axs[1].plot(rc, pr, color=col, label=fs)
                try:
                    ft, mp = calibration_curve(d.y, d.p, n_bins=8, strategy="quantile")
                    axs[2].plot(mp, ft, "o-", color=col, label=fs, ms=4)
                except Exception: pass
            axs[0].plot([0,1],[0,1], ":", color=MUTED); axs[0].set_title("ROC"); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR")
            axs[1].set_title("Precision-Recall"); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
            axs[2].plot([0,1],[0,1], ":", color=MUTED); axs[2].set_title("Calibration"); axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("Observed")
            for a in axs: a.legend(fontsize=8, frameon=False)
            m = summ[tag]["meta"]
            fig.suptitle(f"{v} · {SCEN_LABEL.get(scn,scn)} · n={m['n']} base={m['base_rate']:.2f} (aggregated OOF)", y=1.03, fontsize=10)
            fig.tight_layout(); fig.savefig(C.PLOTS / f"roc_pr_calib_{tag}.png", bbox_inches="tight"); plt.close(fig)


def fig_ablation(summ):
    """drop-one and add-one ΔAUC vs SD / S for the combined scenario."""
    for scn in [s for s in SCEN_ORDER if any(k.startswith(s) for k in summ)]:
        rows = []
        for v in C.VERIFIERS:
            tag = f"{scn}__{v}"
            if tag not in summ or not summ[tag]["summary"]: continue
            fss = summ[tag]["summary"]["feature_sets"]
            base_sd = fss["SD"]["auc"]["median"]; base_s = fss["S"]["auc"]["median"]
            for p in C.PROFILE_FEATURES:
                if f"SD_minus_{p}" in fss:
                    rows.append((v, p, "drop_from_SD", base_sd - fss[f"SD_minus_{p}"]["auc"]["median"]))
                if f"S_plus_{p}" in fss:
                    rows.append((v, p, "add_to_S", fss[f"S_plus_{p}"]["auc"]["median"] - base_s))
        if not rows: continue
        d = pd.DataFrame(rows, columns=["verifier","comp","kind","delta"])
        piv = d[d.kind=="add_to_S"].pivot_table(index="comp", columns="verifier", values="delta")
        fig, ax = plt.subplots(figsize=(1.1*len(C.VERIFIERS)+2, 3.4))
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i,j]:+.02f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Add-one-profile ΔAUC over size baseline — {SCEN_LABEL.get(scn,scn)}", fontsize=9)
        fig.colorbar(im, fraction=0.046); fig.tight_layout()
        fig.savefig(C.PLOTS / f"ablation_addone_{scn}.png", bbox_inches="tight"); plt.close(fig)


def fig_transfer():
    p = C.RESULTS / "transfer" / "transfer_summary.json"
    if not p.exists(): return
    t = json.load(open(p)); rows = []
    for tag, e in t.items():
        if "S_auc" not in e: continue
        v, direction = tag.split("__")
        rows.append((v, direction, e["S_auc"], e["SD_auc"]))
    if not rows: return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, direction, title in [(axes[0], "synth_to_estab", "Train synthetic → test established"),
                                 (axes[1], "estab_to_synth", "Train established → test synthetic")]:
        sub = [r for r in rows if r[1] == direction]; vs=[r[0] for r in sub]
        x=np.arange(len(vs)); w=0.38
        ax.bar(x-w/2, [r[2] for r in sub], w, color=COL_S, label="Size/type")
        ax.bar(x+w/2, [r[3] for r in sub], w, color=COL_SD, label="Size+Profile")
        for xi,r in enumerate(sub):
            ax.text(xi-w/2, r[2]+0.01, f"{r[2]:.2f}", ha="center", fontsize=7)
            ax.text(xi+w/2, r[3]+0.01, f"{r[3]:.2f}", ha="center", fontsize=7)
        ax.axhline(0.5, color=MUTED, ls=":"); ax.set_xticks(x); ax.set_xticklabels(vs)
        ax.set_ylim(0,1); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Target-domain AUC (one-shot transfer)"); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Cross-domain transfer: profile carries architecture-agnostic difficulty", y=1.02)
    fig.tight_layout(); fig.savefig(C.PLOTS / "transfer.png", bbox_inches="tight"); plt.close(fig)


def fig_coefficients():
    p = C.RESULTS / "coefficients" / "final_coefficients_SD.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    for scn in df.scenario.unique():
        sub = df[(df.scenario == scn) & (df.feature.isin(C.PROFILE_FEATURES))]
        if sub.empty: continue
        piv = sub.pivot_table(index="feature", columns="verifier", values="std_coef")
        fig, ax = plt.subplots(figsize=(1.1*piv.shape[1]+2, 3.2))
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                if np.isfinite(piv.values[i,j]):
                    ax.text(j, i, f"{piv.values[i,j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Std. profile coefficients (SD model) — {scn}", fontsize=9)
        fig.colorbar(im, fraction=0.046, label="std coef (log-odds / SD)")
        fig.tight_layout(); fig.savefig(C.PLOTS / f"coefficients_{scn}.png", bbox_inches="tight"); plt.close(fig)


def fig_correlation():
    for dom in ["all", "synthetic", "established"]:
        p = C.RESULTS / "coefficients" / f"spearman_{dom}.csv"
        if not p.exists(): continue
        corr = pd.read_csv(p, index_col=0)
        keep = [c for c in C.PROFILE_FEATURES if c in corr.index] + \
               [c for c in ["input_dim","n_param_total","graph_depth","n_relu","max_hidden_width"] if c in corr.index]
        corr = corr.loc[keep, keep]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(keep))); ax.set_xticklabels(keep, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(keep))); ax.set_yticklabels(keep, fontsize=7)
        for i in range(len(keep)):
            for j in range(len(keep)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(f"Spearman correlation ({dom})", fontsize=10)
        fig.colorbar(im, fraction=0.046); fig.tight_layout()
        fig.savefig(C.PLOTS / f"correlation_{dom}.png", bbox_inches="tight"); plt.close(fig)


def main():
    C.PLOTS.mkdir(parents=True, exist_ok=True)
    summ = load_summaries()
    print(f"loaded {len(summ)} summaries")
    if summ:
        fig_auc_comparison(summ); fig_delta_auc(summ); fig_brier(summ)
        fig_roc_pr_calib(summ); fig_ablation(summ)
    fig_transfer(); fig_coefficients(); fig_correlation()
    print("plots written to", C.PLOTS)


if __name__ == "__main__":
    main()
