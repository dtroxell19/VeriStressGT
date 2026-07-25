"""Interpretation artifacts (spec sec 8.1/8.3/8.4):
  * final descriptive elastic-net (standardized coefficients + odds ratios),
  * coefficient stability across group-bootstrap refits (nonzero freq, sign),
  * predeclared sparse interaction model (stable interaction terms),
  * shallow depth-3 tree (visual sanity check).
Everything here is interpretation-only; in-sample fit is not evidence.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import export_text, DecisionTreeClassifier

import common as C
import models as M
import grouped_splits as GS


def _cell_xyz(df, verifier, domain, feat_key):
    if domain == "combined":
        cell = df[df.verifier == verifier].reset_index(drop=True)
    else:
        cell = df[(df.verifier == verifier) & (df.domain == domain)].reset_index(drop=True)
    H, y, _ = GS.choose_horizon(cell)
    mask = np.isfinite(y)
    d = cell[mask].reset_index(drop=True)
    cols = M.feature_sets()[feat_key]
    X = M.build_design_matrix(d, cols)
    return d, X, y[mask].astype(int), d.network_group.to_numpy(), cols, H


def final_coefficients(df, cells, feat_key="SD"):
    grid = C.load_config()["grid"]
    rows = []
    for verifier, domain in cells:
        d, X, y, g, cols, H = _cell_xyz(df, verifier, domain, feat_key)
        if len(np.unique(y)) < 2 or len(d) < 40:
            continue
        pipe = M.make_pipeline()
        gs = GridSearchCV(pipe, M.param_grid(grid), scoring="roc_auc",
                          cv=GS.StratifiedGroupKFold(4, shuffle=True, random_state=0),
                          n_jobs=-1, error_score=np.nan)
        try:
            gs.fit(X, y, groups=g)
            clf = gs.best_estimator_.named_steps["clf"]
        except Exception:
            continue
        coef = clf.coef_.ravel()
        # stability via group-bootstrap refits at the tuned hyperparams
        stab = _stability(X, y, g, gs.best_params_, cols, n=60)
        for j, c in enumerate(cols):
            rows.append(dict(verifier=verifier, scenario=domain, feature=c,
                             std_coef=float(coef[j]), odds_ratio=float(np.exp(coef[j])),
                             nonzero_freq=stab["nonzero_freq"][j],
                             sign_consistency=stab["sign_consistency"][j],
                             median_coef=stab["median_coef"][j]))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "coefficients" / f"final_coefficients_{feat_key}.csv", index=False)
    return out


def _stability(X, y, g, best_params, cols, n=60, seed=11):
    rng = np.random.RandomState(seed)
    uniq = np.unique(g)
    coefs = []
    for _ in range(n):
        samp = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(g == gg)[0] for gg in samp])
        if len(np.unique(y[idx])) < 2:
            continue
        pipe = M.make_pipeline(class_weight=best_params.get("clf__class_weight"),
                               C_=best_params.get("clf__C", 1.0),
                               l1_ratio=best_params.get("clf__l1_ratio", 0.5))
        try:
            pipe.fit(X[idx], y[idx])
            coefs.append(pipe.named_steps["clf"].coef_.ravel())
        except Exception:
            continue
    coefs = np.array(coefs) if coefs else np.zeros((1, len(cols)))
    nz = np.abs(coefs) > 1e-8
    nonzero_freq = nz.mean(axis=0)
    sign = np.sign(coefs)
    sign_consistency = []
    for j in range(coefs.shape[1]):
        s = sign[nz[:, j], j]
        sign_consistency.append(float(max((s > 0).mean(), (s < 0).mean())) if len(s) else 0.0)
    return dict(nonzero_freq=[float(x) for x in nonzero_freq],
                sign_consistency=sign_consistency,
                median_coef=[float(x) for x in np.median(coefs, axis=0)])


def interaction_model(df, cells):
    """Predeclared sparse profile-interaction logistic. Reports stable terms
    (nonzero in >=50% of group-bootstrap fits, sign-consistent in >=75%)."""
    rows = []
    prof = C.PROFILE_FEATURES
    for verifier, domain in cells:
        d, _, y, g, _, H = _cell_xyz(df, verifier, domain, "SD")
        if len(np.unique(y)) < 2 or len(d) < 40:
            continue
        # base design: size + profile (transformed), then add predeclared interactions
        base_cols = M.feature_sets()["SD"]
        B = pd.DataFrame(M.build_design_matrix(d, base_cols), columns=base_cols)
        feats = {c: B[c].to_numpy() for c in base_cols}
        names = list(base_cols)
        # profile x profile
        for a, b in combinations(prof, 2):
            names.append(f"{a}*{b}"); feats[names[-1]] = B[a] * B[b]
        # profile x log-param, profile x depth
        for p in prof:
            for s in ["n_param_total", "graph_depth"]:
                names.append(f"{p}*{s}"); feats[names[-1]] = B[p] * B[s]
        # profile x arch onehot
        for p in prof:
            for a in [c for c in base_cols if c.startswith("arch_")]:
                names.append(f"{p}*{a}"); feats[names[-1]] = B[p] * B[a]
        X = np.column_stack([np.nan_to_num(feats[n], nan=0.0) for n in names])
        best = {"clf__C": 0.1, "clf__l1_ratio": 0.75, "clf__class_weight": "balanced"}
        stab = _stability(X, y, g, best, names, n=60)
        for j, nm in enumerate(names):
            if "*" not in nm:
                continue
            if stab["nonzero_freq"][j] >= 0.5 and stab["sign_consistency"][j] >= 0.75:
                rows.append(dict(verifier=verifier, scenario=domain, term=nm,
                                 nonzero_freq=stab["nonzero_freq"][j],
                                 sign_consistency=stab["sign_consistency"][j],
                                 median_coef=stab["median_coef"][j]))
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "coefficients" / "stable_interactions.csv", index=False)
    return out


def shallow_trees(df, cells):
    txt = {}
    for verifier, domain in cells:
        d, X, y, g, cols, H = _cell_xyz(df, verifier, domain, "SD")
        if len(np.unique(y)) < 2 or len(d) < 40:
            continue
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0))])
        pipe.fit(X, y)
        txt[f"{verifier}|{domain}"] = export_text(pipe.named_steps["clf"], feature_names=list(cols))
    (C.RESULTS / "coefficients" / "shallow_trees.txt").write_text(
        "\n\n".join(f"=== {k} ===\n{v}" for k, v in txt.items()))
    return txt


if __name__ == "__main__":
    df = C.load_rows()
    cells = [(v, dom) for v in C.VERIFIERS for dom in ["synthetic", "combined"]]
    print("final coefficients..."); print(final_coefficients(df, cells).head())
    print("interactions..."); print(interaction_model(df, cells).head())
    print("trees..."); shallow_trees(df, cells)
    print("interpret done")
