"""Assemble the primary results table, coefficient table, and per-cell verdicts
from the frozen CV outputs. Writes machine tables the human-written REPORT.md
pulls in, and computes the pre-registered incremental-value verdict (spec sec 9.3).
"""
from __future__ import annotations
import glob, os, json
import numpy as np
import pandas as pd

import common as C

SCEN_NAME = {"A_synthetic": "Synthetic", "B_established": "Established*", "C_combined": "Combined"}


def verdict(meta, summ):
    if summ is None:
        return meta.get("verdict_gate", "BLOCKED_INSUFFICIENT_CLASS_SUPPORT")
    fss = summ["feature_sets"]
    s_auc = fss["S"]["auc"]["median"]; sd_auc = fss["SD"]["auc"]["median"]
    s_br = fss["S"]["brier"]["median"]; sd_br = fss["SD"]["brier"]["median"]
    boot = summ.get("bootstrap_dauc", {})
    lo = boot.get("lo"); pt = boot.get("point")
    paired = summ.get("paired", {}).get("auc", {})
    fracpos = paired.get("frac_positive", 0.5)
    brier_ok = sd_br <= s_br + 0.02
    tag = "exploratory " if not meta.get("eligible") else ""
    if lo is not None and lo > 0 and sd_auc > s_auc and brier_ok:
        return tag + "PROFILE_ADDS_PREDICTIVE_VALUE"
    if (pt is not None and pt > 0) or fracpos > 0.5:
        return tag + "PROFILE_ADDS_PARTIAL_VALUE"
    return tag + "NO_INCREMENTAL_PROFILE_VALUE"


def primary_table():
    rows = []
    for f in sorted(glob.glob(str(C.RESULTS / "cv" / "*.summary.json"))):
        o = json.load(open(f)); meta = o["meta"]; summ = o["summary"]
        scn, v = os.path.basename(f).replace(".summary.json", "").split("__")
        r = dict(Verifier=v, Scenario=SCEN_NAME.get(scn, scn), n=meta["n"],
                 TO_rate=round(meta["base_rate"], 2), Groups=meta["n_groups"],
                 H=meta["horizon_s"])
        if summ:
            fss = summ["feature_sets"]; boot = summ.get("bootstrap_dauc", {})
            r["Size_AUC"] = round(fss["S"]["auc"]["median"], 3)
            r["SizeProfile_AUC"] = round(fss["SD"]["auc"]["median"], 3)
            r["dAUC"] = round(boot.get("point", np.nan), 3)
            r["CI95"] = f"[{boot.get('lo',float('nan')):+.3f}, {boot.get('hi',float('nan')):+.3f}]"
            r["Size_Brier"] = round(fss["S"]["brier"]["median"], 3)
            r["SizeProfile_Brier"] = round(fss["SD"]["brier"]["median"], 3)
        r["Verdict"] = verdict(meta, summ)
        rows.append(r)
    df = pd.DataFrame(rows)
    order = {"Synthetic": 0, "Combined": 1, "Established*": 2}
    df["__o"] = df.Scenario.map(order).fillna(9)
    df = df.sort_values(["__o", "Verifier"]).drop(columns="__o")
    df.to_csv(C.RESULTS / "primary_results_table.csv", index=False)
    return df


def to_markdown(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def coefficient_table():
    p = C.RESULTS / "coefficients" / "final_coefficients_SD.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df.feature.isin(C.PROFILE_FEATURES)].copy()
    df["odds_ratio"] = df.odds_ratio.round(3); df["std_coef"] = df.std_coef.round(3)
    df["nonzero_freq"] = df.nonzero_freq.round(2); df["sign_consistency"] = df.sign_consistency.round(2)
    df = df[["verifier", "scenario", "feature", "std_coef", "odds_ratio", "nonzero_freq", "sign_consistency"]]
    df.to_csv(C.RESULTS / "coefficients" / "profile_coefficient_table.csv", index=False)
    return df


def transfer_table():
    p = C.RESULTS / "transfer" / "transfer_summary.json"
    if not p.exists():
        return None
    t = json.load(open(p)); rows = []
    for tag, e in t.items():
        v, direction = tag.split("__")
        rows.append(dict(Verifier=v, Direction=direction.replace("_", "→"),
                         tgt_n=e.get("n_tgt"), tgt_base=round(e.get("tgt_base_rate") or 0, 2),
                         Size_AUC=round(e.get("S_auc", float("nan")), 3) if "S_auc" in e else None,
                         SizeProfile_AUC=round(e.get("SD_auc", float("nan")), 3) if "SD_auc" in e else None,
                         dAUC=round(e.get("dauc_point", float("nan")), 3) if "dauc_point" in e else None,
                         CI95=(f"[{e.get('dauc_lo'):+.3f}, {e.get('dauc_hi'):+.3f}]"
                               if e.get("dauc_lo") is not None else e.get("verdict", ""))))
    df = pd.DataFrame(rows).sort_values(["Direction", "Verifier"])
    df.to_csv(C.RESULTS / "transfer" / "transfer_table.csv", index=False)
    return df


def ablation_summary():
    """Best add-one-profile component per (verifier, scenario)."""
    rows = []
    for f in sorted(glob.glob(str(C.RESULTS / "cv" / "*.summary.json"))):
        o = json.load(open(f))
        if not o["summary"]:
            continue
        scn, v = os.path.basename(f).replace(".summary.json", "").split("__")
        fss = o["summary"]["feature_sets"]; base_s = fss["S"]["auc"]["median"]
        best_c, best_d = None, -9
        for p in C.PROFILE_FEATURES:
            k = f"S_plus_{p}"
            if k in fss:
                d = fss[k]["auc"]["median"] - base_s
                if d > best_d:
                    best_d, best_c = d, p
        rows.append(dict(Verifier=v, Scenario=SCEN_NAME.get(scn, scn),
                         best_addone=best_c, delta_auc=round(best_d, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "ablation" / "addone_best.csv", index=False)
    return df


def per_subgroup_breakdown():
    """Held-out AUC by constructor/benchmark subgroup, from aggregated OOF
    (spec: per-constructor / per-benchmark held-out performance)."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for f in glob.glob(str(C.RESULTS / "cv" / "*.oof.parquet")):
        tag = os.path.basename(f).replace(".oof.parquet", "")
        scn, v = tag.split("__")
        oof = pd.read_parquet(f)
        agg = (oof.groupby(["instance_id", "feature_set", "constructor", "benchmark", "domain"])
                  .agg(y=("y", "first"), p=("p", "mean")).reset_index())
        grpcol = "constructor" if scn.startswith("A") or scn.startswith("C") else "benchmark"
        for grp, gd in agg.groupby(grpcol):
            piv = gd.pivot_table(index="instance_id", columns="feature_set", values="p")
            ys = gd.groupby("instance_id")["y"].first()
            piv = piv.join(ys).dropna()
            if piv["y"].nunique() < 2 or len(piv) < 12:
                continue
            r = dict(scenario=scn, verifier=v, subgroup=grp, n=int(len(piv)),
                     to_rate=round(float(piv["y"].mean()), 2))
            for fs in ["S", "SD"]:
                if fs in piv:
                    try:
                        r[f"{fs}_auc"] = round(float(roc_auc_score(piv["y"], piv[fs])), 3)
                    except Exception:
                        r[f"{fs}_auc"] = None
            if r.get("S_auc") is not None and r.get("SD_auc") is not None:
                r["dAUC"] = round(r["SD_auc"] - r["S_auc"], 3)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "per_subgroup_auc.csv", index=False)
    return df


def verdict_summary():
    pt = primary_table()
    out = {"cells": [], "counts": {}}
    for _, r in pt.iterrows():
        out["cells"].append({k: (None if pd.isna(r[k]) else r[k]) for k in pt.columns})
    vc = pt.Verdict.value_counts().to_dict()
    out["counts"] = {str(k): int(v) for k, v in vc.items()}
    json.dump(out, open(C.RESULTS / "verdict_summary.json", "w"), indent=2, default=str)
    return out


if __name__ == "__main__":
    pt = primary_table()
    print(to_markdown(pt))
    (C.RESULTS / "primary_results_table.md").write_text(to_markdown(pt))
    tt = transfer_table()
    if tt is not None:
        (C.RESULTS / "transfer" / "transfer_table.md").write_text(to_markdown(tt))
        print("\nTRANSFER:\n", to_markdown(tt))
    ab = ablation_summary(); print("\nADD-ONE BEST:\n", ab.to_string(index=False))
    ct = coefficient_table()
    if ct is not None:
        (C.RESULTS / "coefficients" / "profile_coefficient_table.md").write_text(to_markdown(ct))
    sg = per_subgroup_breakdown()
    if not sg.empty:
        sg.to_csv(C.RESULTS / "per_subgroup_auc.csv", index=False)
        print(f"\nper-subgroup rows: {len(sg)}")
    vs = verdict_summary()
    print("\nVERDICT COUNTS:", vs["counts"])
