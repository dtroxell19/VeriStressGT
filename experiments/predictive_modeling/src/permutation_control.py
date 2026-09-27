"""Permutation negative-control for the profile boost.

For each (verifier, scenario) at the uniform 240 s horizon we compare, on identical
grouped folds and a fixed elastic-net (no tuning, so the only moving part is the data):
  * S      : size/type only
  * SD     : size/type + the real profile block
  * SD_shuf: size/type + the profile block, but the profile rows are PERMUTED in the
             training data of every fold (profile<->instance/target link destroyed,
             feature count and profile-internal distribution preserved).

We build a null distribution of SD_shuf out-of-fold AUC over many permutation seeds and
report a permutation p-value: P(SD_shuf AUC >= real SD AUC). If the boost is genuine
signal, SD_shuf collapses to ~S and the p-value is tiny.

Run:  python permutation_control.py
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import common as C, models as M, grouped_splits as GS

HORIZON = 240
REAL_REPEATS = 5
NULL_PERMS = 30
SEED_BASE = 1100


def _pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                                l1_ratio=0.5, C=1.0, class_weight="balanced",
                                                max_iter=3000, tol=1e-3))])


def _cell(df, v, dom):
    cell = (df[df.verifier == v] if dom == "combined"
            else df[(df.verifier == v) & (df.domain == dom)]).reset_index(drop=True)
    y = np.array([GS.label_at_horizon(s, t, b, HORIZON) for s, t, b in
                  zip(cell.raw_status, cell.raw_time, cell.budget_s)])
    m = np.isfinite(y)
    return cell[m].reset_index(drop=True), y[m].astype(int)


def _oof_auc(X, y, groups, n_repeats, perm_profile_idx=None, perm_seed=0):
    """Pooled out-of-fold AUC. If perm_profile_idx given, permute those columns'
    rows inside each TRAINING fold (test rows untouched)."""
    oof_y, oof_p = [], []
    for rep, fold, tr, te, k, sp in GS.repeated_group_splits(y, groups, 5, n_repeats, SEED_BASE):
        Xtr = X[tr]
        if perm_profile_idx is not None:
            Xtr = Xtr.copy()
            rng = np.random.RandomState(97 * perm_seed + 7 * rep + fold)
            order = rng.permutation(len(tr))
            Xtr[:, perm_profile_idx] = Xtr[order][:, perm_profile_idx]
        p = _pipe().fit(Xtr, y[tr]).predict_proba(X[te])[:, 1]
        oof_y.append(y[te]); oof_p.append(p)
    yy = np.concatenate(oof_y); pp = np.concatenate(oof_p)
    # average per-repeat pooled AUC would also work; single pooled AUC is fine here
    try:
        return roc_auc_score(yy, pp)
    except Exception:
        return np.nan


def run(df, cells):
    cols_s, cols_sd = M.feature_sets()["S"], M.feature_sets()["SD"]
    prof_idx = [cols_sd.index(p) for p in C.PROFILE_FEATURES]
    rows = []
    for v, dom in cells:
        d, y = _cell(df, v, dom)
        if len(np.unique(y)) < 2 or len(d) < 40:
            continue
        g = d.network_group.to_numpy()
        XS = M.build_design_matrix(d, cols_s); XSD = M.build_design_matrix(d, cols_sd)
        s_auc = _oof_auc(XS, y, g, REAL_REPEATS)
        sd_auc = _oof_auc(XSD, y, g, REAL_REPEATS)
        null = np.array([_oof_auc(XSD, y, g, 1, prof_idx, k) for k in range(NULL_PERMS)])
        null = null[np.isfinite(null)]
        pval = (1 + int((null >= sd_auc).sum())) / (len(null) + 1)
        rows.append(dict(verifier=v, scenario=dom, n=len(d), pos=int(y.sum()),
                         S_auc=round(s_auc, 3), SD_auc=round(sd_auc, 3),
                         SDshuf_mean=round(float(null.mean()), 3),
                         SDshuf_p95=round(float(np.percentile(null, 95)), 3),
                         real_boost=round(sd_auc - s_auc, 3),
                         shuf_boost=round(float(null.mean()) - s_auc, 3),
                         perm_p=round(pval, 4)))
        print(f"  {v:9} {dom:10} S={s_auc:.3f} SD={sd_auc:.3f} SDshuf={null.mean():.3f} "
              f"(p95={np.percentile(null,95):.3f})  real_boost={sd_auc-s_auc:+.3f}  perm_p={pval:.4f}", flush=True)
    out = pd.DataFrame(rows)
    (C.RESULTS / "permutation_control").mkdir(parents=True, exist_ok=True)
    out.to_csv(C.RESULTS / "permutation_control" / "profile_shuffle_h240.csv", index=False)
    return out


if __name__ == "__main__":
    df = C.load_rows()
    cells = [(v, dom) for dom in ["synthetic", "combined"] for v in C.VERIFIERS]
    print(f"Permutation control (H={HORIZON}s, {NULL_PERMS} shuffles/cell):")
    run(df, cells)
