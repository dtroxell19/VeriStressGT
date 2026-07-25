"""Single OVERALL bootstrap hypothesis test: pooling all instances x all verifiers
into one model (verifier identity as an intrinsic covariate, analogous to countries
in the prior study), does adding the Difficulty Profile improve held-out timeout
prediction beyond size/type + verifier identity? One p-value, one effect size.

Intrinsic  = size/type features + verifier one-hot
Full       = intrinsic + 5 profile components
Null       = y*_i ~ Bernoulli(p_hat_intrinsic(x_i))  (parametric bootstrap, H0: profile _|_ Y)
Grouping   = network identity (all 5 verifier-rows of a network stay together)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import common as C, models as M, grouped_splits as GS
import actionable_analysis as A

HORIZON = 240
B = 1000


def _pooled(df, domain):
    sub = df if domain == "all" else df[df.domain == domain]
    sub = sub.reset_index(drop=True)
    y = np.array([GS.label_at_horizon(s, t, b, HORIZON) for s, t, b in
                  zip(sub.raw_status, sub.raw_time, sub.budget_s)])
    m = np.isfinite(y)
    d = sub[m].reset_index(drop=True)
    yv = y[m].astype(int)
    # intrinsic = size features + verifier one-hot; full = + profile
    Xsize = M.build_design_matrix(d, M.feature_sets()["S"])
    voh = pd.get_dummies(d["verifier"]).to_numpy(dtype=float)
    Xprof = M.build_design_matrix(d, C.PROFILE_FEATURES)
    Xs = np.hstack([Xsize, voh])
    Xsd = np.hstack([Xsize, voh, Xprof])
    groups = d["network_group"].to_numpy()
    return Xs, Xsd, yv, groups, d


def main():
    df = C.load_rows()
    rows = []
    for dom in ["all", "synthetic", "established"]:
        Xs, Xsd, y, g, d = _pooled(df, dom)
        r = A.bootstrap_htest(Xs, Xsd, y, g, B=B, n_jobs=-1)
        r.update(pool=dom, n_rows=len(y), n_networks=int(len(np.unique(g))),
                 base_rate=round(float(y.mean()), 3))
        rows.append(r)
        print(f"[{dom:11}] N={len(y)} nets={len(np.unique(g))} base={y.mean():.2f}  "
              f"obs_dAUC={r['obs_dAUC']:+.3f} (null p95 {r['null_dAUC_p95']:+.3f}) p_AUC={r['p_AUC']:.4f} | "
              f"obs_dDev={r['obs_dDeviance']:+.3f} p_Dev={r['p_Deviance']:.4f}  [B={r['n_null']}]", flush=True)
    out = pd.DataFrame(rows)
    (C.RESULTS / "actionable").mkdir(parents=True, exist_ok=True)
    out.to_csv(C.RESULTS / "actionable" / "overall_htest_h240.csv", index=False)
    return out


if __name__ == "__main__":
    print(f"OVERALL bootstrap hypothesis test (pooled, GBM, grouped CV, H={HORIZON}s, B={B}):")
    main()
    print("-> results/actionable/overall_htest_h240.csv")
