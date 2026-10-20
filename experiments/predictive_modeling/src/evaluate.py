"""Repeated grouped nested-CV evaluation and the paired size-only vs
size+profile comparison. This is the scientific core.

Discipline enforced here:
  * identical outer folds for every feature set within a cell (splits depend
    only on (y, groups), generated once per repeat);
  * all imputation/standardization/tuning happen inside the training fold;
  * decision thresholds are picked from inner training predictions only;
  * no network group spans train and test (asserted).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, GroupKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             balanced_accuracy_score)

import common as C
import models as M
import grouped_splits as GS

warnings.filterwarnings("ignore")


def _inner_cv(y, groups, k=4):
    try:
        s = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
        folds = list(s.split(np.zeros(len(y)), y, groups))
        if all(len(np.unique(y[tr])) == 2 for tr, _ in folds):
            return s
    except Exception:
        pass
    return GroupKFold(n_splits=min(k, len(np.unique(groups))))


def _tune_fit_predict(Xtr, ytr, gtr, Xte, grid_cfg, inner_k):
    inner = _inner_cv(ytr, gtr, inner_k)
    try:
        gs = GridSearchCV(M.make_pipeline(), M.param_grid(grid_cfg), scoring="roc_auc",
                          cv=inner, n_jobs=1, refit=True, error_score=np.nan)
        gs.fit(Xtr, ytr, groups=gtr)
        best = gs.best_estimator_
        best_params = gs.best_params_
    except Exception:
        # degenerate inner fold (e.g. established: very few groups) -> default, untuned model
        best = M.make_pipeline(class_weight="balanced").fit(Xtr, ytr)
        best_params = {"clf__C": 1.0, "clf__l1_ratio": 0.5, "clf__class_weight": "balanced"}
        try:
            oof = cross_val_predict(best, Xtr, ytr, cv=inner, groups=gtr,
                                    method="predict_proba", n_jobs=1)[:, 1]
            return best.predict_proba(Xte)[:, 1], _best_threshold(ytr, oof), best_params
        except Exception:
            return best.predict_proba(Xte)[:, 1], 0.5, best_params
    # threshold from inner OOF training predictions only
    try:
        oof = cross_val_predict(best, Xtr, ytr, cv=inner, groups=gtr,
                                method="predict_proba", n_jobs=1)[:, 1]
        thr = _best_threshold(ytr, oof)
    except Exception:
        thr = 0.5
    pte = best.predict_proba(Xte)[:, 1]
    return pte, thr, best_params


def _best_threshold(y, p):
    grid = np.unique(np.clip(p, 1e-4, 1 - 1e-4))
    if len(grid) > 200:
        grid = np.quantile(p, np.linspace(0.02, 0.98, 100))
    best_t, best_s = 0.5, -1
    for t in grid:
        s = balanced_accuracy_score(y, (p >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)


def run_cell(df_cell, feats_to_run, cfg, horizon=None, label="cell", n_jobs=-1):
    """Evaluate a (verifier, scenario) cell. Returns (oof_df, meta)."""
    cv = cfg["cv"]
    grid_cfg = cfg["grid"]
    df_cell = df_cell.reset_index(drop=True)

    if horizon is None:
        H, yfull, ndrop = GS.choose_horizon(df_cell)
    else:
        H = horizon
        yfull = np.array([GS.label_at_horizon(s, t, b, H) for s, t, b in
                          zip(df_cell["raw_status"], df_cell["raw_time"], df_cell["budget_s"])])
        ndrop = int((~np.isfinite(yfull)).sum())

    mask = np.isfinite(yfull)
    d = df_cell[mask].reset_index(drop=True)
    y = yfull[mask].astype(int)
    groups = d["network_group"].to_numpy()

    meta = {
        "label": label, "horizon_s": H, "n": int(len(d)), "n_dropped": int(ndrop),
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "base_rate": float(y.mean()), "n_groups": int(len(np.unique(groups))),
    }
    # eligibility gate
    elig = cfg["eligibility"]
    meta["eligible"] = (len(d) >= elig["min_rows"] and (y == 1).sum() >= elig["min_pos"]
                        and (y == 0).sum() >= elig["min_neg"]
                        and len(np.unique(groups)) >= elig["min_groups"])
    meta["exploratory"] = (not meta["eligible"]) and (
        (y == 1).sum() >= elig["exploratory_min_class"]
        and (y == 0).sum() >= elig["exploratory_min_class"])
    if not (meta["eligible"] or meta["exploratory"]):
        meta["verdict_gate"] = "BLOCKED_INSUFFICIENT_CLASS_SUPPORT"
        return pd.DataFrame(), meta

    # design matrices (same rows for every feature set)
    X = {fs: M.build_design_matrix(d, feats_to_run[fs]) for fs in feats_to_run}

    # splits generated once; reused across feature sets -> identical folds
    splits = list(GS.repeated_group_splits(y, groups, cv["outer_folds"], cv["outer_repeats"],
                                            cv["outer_seed_base"], tuple(cv["min_folds_fallback"])))
    meta["splitter"] = splits[0][5] if splits else None
    meta["k_folds"] = splits[0][4] if splits else None

    def one_task(rep, fold, tr, te):
        GS.assert_no_group_leakage(groups, tr, te)
        out = []
        for fs in feats_to_run:
            pte, thr, bp = _tune_fit_predict(X[fs][tr], y[tr], groups[tr], X[fs][te],
                                             grid_cfg, cv["inner_folds"])
            for j, idx in enumerate(te):
                out.append(dict(repeat=rep, fold=fold, feature_set=fs, row=int(idx),
                                instance_id=d.loc[idx, "instance_id"],
                                network_group=d.loc[idx, "network_group"],
                                benchmark=d.loc[idx, "benchmark"],
                                constructor=d.loc[idx, "constructor"],
                                domain=d.loc[idx, "domain"],
                                arch_type=d.loc[idx, "arch_type"],
                                y=int(y[idx]), p=float(pte[j]), thr=float(thr)))
        return out

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one_task)(rep, fold, tr, te) for (rep, fold, tr, te, k, sp) in splits)
    oof = pd.DataFrame([r for chunk in results for r in chunk])
    return oof, meta


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _metrics(y, p, thr):
    y = np.asarray(y); p = np.asarray(p)
    m = {}
    try: m["auc"] = roc_auc_score(y, p)
    except Exception: m["auc"] = np.nan
    try: m["ap"] = average_precision_score(y, p)
    except Exception: m["ap"] = np.nan
    try: m["brier"] = brier_score_loss(y, p)
    except Exception: m["brier"] = np.nan
    yhat = (p >= thr).astype(int)
    try: m["bal_acc"] = balanced_accuracy_score(y, yhat)
    except Exception: m["bal_acc"] = np.nan
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    m["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    m["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    return m


def per_repeat_metrics(oof, fs):
    sub = oof[oof.feature_set == fs]
    rows = []
    for rep, g in sub.groupby("repeat"):
        mm = _metrics(g.y.values, g.p.values, g.thr.iloc[0])
        mm["repeat"] = rep
        rows.append(mm)
    return pd.DataFrame(rows)


def summarize(oof, feats, primary=("SD", "S")):
    out = {"feature_sets": {}}
    per_fs = {}
    for fs in feats:
        pr = per_repeat_metrics(oof, fs)
        per_fs[fs] = pr
        summ = {}
        for metric in ["auc", "ap", "brier", "bal_acc", "sensitivity", "specificity"]:
            v = pr[metric].dropna().values
            summ[metric] = dict(median=float(np.median(v)) if len(v) else np.nan,
                                p10=float(np.percentile(v, 10)) if len(v) else np.nan,
                                p90=float(np.percentile(v, 90)) if len(v) else np.nan,
                                mean=float(np.mean(v)) if len(v) else np.nan,
                                std=float(np.std(v)) if len(v) else np.nan)
        out["feature_sets"][fs] = summ

    # paired repeat-level difference SD - S
    if primary[0] in per_fs and primary[1] in per_fs:
        a, b = per_fs[primary[0]], per_fs[primary[1]]
        merged = a.merge(b, on="repeat", suffixes=("_aug", "_base"))
        for metric in ["auc", "ap", "brier"]:
            d = (merged[f"{metric}_aug"] - merged[f"{metric}_base"]).dropna().values
            out.setdefault("paired", {})[metric] = dict(
                median=float(np.median(d)) if len(d) else np.nan,
                p10=float(np.percentile(d, 10)) if len(d) else np.nan,
                p90=float(np.percentile(d, 90)) if len(d) else np.nan,
                mean=float(np.mean(d)) if len(d) else np.nan,
                frac_positive=float(np.mean(d > 0)) if len(d) else np.nan)
        # group bootstrap CI on aggregate paired dAUC
        out["bootstrap_dauc"] = group_bootstrap_dauc(oof, primary)
    return out, per_fs


def group_bootstrap_dauc(oof, primary=("SD", "S"), n_boot=2000, seed=7):
    """Average each instance's OOF prob across repeats, then bootstrap over
    network groups for a 95% CI on aggregate AUC(SD) - AUC(S)."""
    rng = np.random.RandomState(seed)
    agg = (oof.groupby(["instance_id", "network_group", "feature_set"])
              .agg(y=("y", "first"), p=("p", "mean")).reset_index())
    piv = agg.pivot_table(index=["instance_id", "network_group"], columns="feature_set",
                          values="p")
    ys = agg.groupby(["instance_id", "network_group"])["y"].first()
    piv = piv.join(ys)
    piv = piv.dropna(subset=[primary[0], primary[1], "y"]).reset_index()
    groups = piv["network_group"].values
    uniq = np.unique(groups)

    def dauc(frame):
        try:
            return (roc_auc_score(frame["y"], frame[primary[0]])
                    - roc_auc_score(frame["y"], frame[primary[1]]))
        except Exception:
            return np.nan

    point = dauc(piv)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        fr = piv.iloc[idx]
        if fr["y"].nunique() < 2:
            continue
        boots.append(dauc(fr))
    boots = np.array([b for b in boots if np.isfinite(b)])
    return dict(point=float(point) if np.isfinite(point) else np.nan,
                lo=float(np.percentile(boots, 2.5)) if len(boots) else np.nan,
                hi=float(np.percentile(boots, 97.5)) if len(boots) else np.nan,
                n_boot=int(len(boots)))
