"""Faithful adaptation of a prior cross-country excess-mortality study (Sci. Reports
2023): "Bootstrap hypothesis testing" + "Measuring the effect of actionable features",
to the VeriStress timeout-prediction setting.

Mapping: intrinsic features <- network size/type (S); actionable features <- the
Difficulty Profile (D); response <- timeout (binary, uniform 240 s horizon).
Model: gradient boosting (HistGradientBoostingClassifier), matching their GBM.
Grouping: network-grouped CV throughout (their rows were one-per-country; ours
share networks, so we keep the leakage-safe grouping and a grouped row-bootstrap).

Two analyses:
  (1) bootstrap_htest  -- does S+D beat S BEYOND simply adding variables?
      Null H0: A _|_ Y, built by a PARAMETRIC bootstrap (classification analog of
      their residual bootstrap y* = yhat_I + r*):  y*_i ~ Bernoulli(phat_I(x_i)),
      where phat_I are the intrinsic-only grouped-OOF probabilities. Refit S and
      S+D on y*, statistic = oof performance gain; repeat B times -> null;
      p = frac(null gain >= observed gain).
  (2) delta_analysis   -- per-instance delta_i = phat_SD(i) - phat_S(i), grouped
      bootstrap 95% CIs, summarized by constructor family, plus feature values
      split by delta sign (their Fig 7 analog).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, log_loss

import common as C, models as M, grouped_splits as GS

HORIZON = 240
B_NULL = 500
CV_K = 5


def _hgb(seed=0):
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def _cell(df, v, dom):
    cell = (df[df.verifier == v] if dom == "combined"
            else df[(df.verifier == v) & (df.domain == dom)]).reset_index(drop=True)
    y = np.array([GS.label_at_horizon(s, t, b, HORIZON) for s, t, b in
                  zip(cell.raw_status, cell.raw_time, cell.budget_s)])
    m = np.isfinite(y)
    return cell[m].reset_index(drop=True), y[m].astype(int)


def _grouped_oof(X, y, groups, seed=0, k=CV_K):
    try:
        sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(sp.split(X, y, groups))
        if not all(len(np.unique(y[tr])) == 2 for tr, _ in folds):
            raise ValueError
    except Exception:
        sp = GroupKFold(n_splits=min(k, len(np.unique(groups))))
        folds = list(sp.split(X, y, groups))
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        p[te] = _hgb(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def _gain(y, p_s, p_sd):
    """performance gain of S+D over S: (dAUC, d_deviance) — positive = SD better."""
    dauc = roc_auc_score(y, p_sd) - roc_auc_score(y, p_s)
    ddev = log_loss(y, p_s, labels=[0, 1]) - log_loss(y, p_sd, labels=[0, 1])
    return dauc, ddev


def bootstrap_htest(Xs, Xsd, y, groups, B=B_NULL, n_jobs=-1):
    # observed
    ps = _grouped_oof(Xs, y, groups, seed=0)
    psd = _grouped_oof(Xsd, y, groups, seed=0)
    obs_auc, obs_dev = _gain(y, ps, psd)
    # intrinsic OOF probabilities define the null signal (leakage-safe; avoids the
    # in-sample overfitting a fitted-value null would inject)
    pI = np.clip(ps, 0.02, 0.98)

    def one(b):
        rb = np.random.RandomState(9000 + b)
        ystar = (rb.uniform(size=len(y)) < pI).astype(int)
        if ystar.min() == ystar.max():
            return None
        qs = _grouped_oof(Xs, ystar, groups, seed=b)
        qsd = _grouped_oof(Xsd, ystar, groups, seed=b)
        try:
            return _gain(ystar, qs, qsd)
        except Exception:
            return None

    res = [r for r in Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(B)) if r]
    na = np.array([r[0] for r in res]); nd = np.array([r[1] for r in res])
    return dict(
        n=int(len(y)), pos=int(y.sum()),
        obs_dAUC=round(float(obs_auc), 4), obs_dDeviance=round(float(obs_dev), 4),
        null_dAUC_mean=round(float(na.mean()), 4), null_dAUC_p95=round(float(np.percentile(na, 95)), 4),
        null_dDev_mean=round(float(nd.mean()), 4), null_dDev_p95=round(float(np.percentile(nd, 95)), 4),
        p_AUC=round((1 + int((na >= obs_auc).sum())) / (len(na) + 1), 4),
        p_Deviance=round((1 + int((nd >= obs_dev).sum())) / (len(nd) + 1), 4),
        n_null=int(len(na)))


def delta_analysis(cell_df, y, groups, B=300, n_jobs=-1):
    """Per-instance delta = phat_SD - phat_S with grouped-bootstrap 95% CI, plus a
    by-constructor summary and feature values split by delta sign."""
    Xs = M.build_design_matrix(cell_df, M.feature_sets()["S"])
    Xsd = M.build_design_matrix(cell_df, M.feature_sets()["SD"])
    ps = _grouped_oof(Xs, y, groups, seed=0)
    psd = _grouped_oof(Xsd, y, groups, seed=0)
    delta = psd - ps
    uniq = np.unique(groups)

    def one(b):
        rb = np.random.RandomState(3000 + b)
        samp = rb.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in samp])
        yb = y[idx]
        if yb.min() == yb.max():
            return None
        a = _grouped_oof(Xs[idx], yb, groups[idx], seed=b)
        c = _grouped_oof(Xsd[idx], yb, groups[idx], seed=b)
        d = np.full(len(y), np.nan)
        # map each original row's bootstrap delta by averaging its resampled copies
        db = c - a
        acc = {}
        for j, orig in enumerate(idx):
            acc.setdefault(orig, []).append(db[j])
        for orig, vals in acc.items():
            d[orig] = np.mean(vals)
        return d

    boots = [r for r in Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(B)) if r is not None]
    Bmat = np.vstack(boots)                      # (B, n) with NaNs where group absent
    sd = np.nanstd(Bmat, axis=0)
    lo = np.nanpercentile(Bmat, 2.5, axis=0)
    hi = np.nanpercentile(Bmat, 97.5, axis=0)
    out = cell_df[["instance_id", "constructor", "benchmark", "domain",
                   "g_ibp", "unstable_fraction", "margin_hat_min", "a_tau", "d_eff"]].copy()
    out["y_timeout"] = y
    out["delta"] = delta
    out["delta_sd"] = sd
    out["delta_lo"] = lo
    out["delta_hi"] = hi
    return out


def run(df, cells, n_jobs=-1):
    outdir = C.RESULTS / "actionable"
    outdir.mkdir(parents=True, exist_ok=True)
    htest_rows = []
    for v, dom in cells:
        d, y = _cell(df, v, dom)
        if len(np.unique(y)) < 2 or len(d) < 40:
            continue
        g = d.network_group.to_numpy()
        Xs = M.build_design_matrix(d, M.feature_sets()["S"])
        Xsd = M.build_design_matrix(d, M.feature_sets()["SD"])
        r = bootstrap_htest(Xs, Xsd, y, g, n_jobs=n_jobs)
        r.update(verifier=v, scenario=dom)
        htest_rows.append(r)
        print(f"  {v:9} {dom:10} obs_dAUC={r['obs_dAUC']:+.3f} p_AUC={r['p_AUC']:.4f} | "
              f"obs_dDev={r['obs_dDeviance']:+.3f} p_Dev={r['p_Deviance']:.4f} "
              f"(null dAUC mean {r['null_dAUC_mean']:+.3f}, p95 {r['null_dAUC_p95']:+.3f}; B={r['n_null']})",
              flush=True)
        # per-instance delta for the powered scenarios
        da = delta_analysis(d, y, g, n_jobs=n_jobs)
        da.to_csv(outdir / f"delta_{dom}_{v}.csv", index=False)
    ht = pd.DataFrame(htest_rows)
    ht.to_csv(outdir / "bootstrap_htest_h240.csv", index=False)
    return ht


if __name__ == "__main__":
    df = C.load_rows()
    cells = [(v, dom) for dom in ["synthetic", "combined"] for v in C.VERIFIERS]
    print(f"Bootstrap hypothesis test (GBM, Bernoulli null, grouped CV, H={HORIZON}s, B={B_NULL}):")
    run(df, cells)
    print("done -> results/actionable/")
