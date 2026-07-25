"""Secondary outcome: solved-only runtime regression (spec sec 4.3 / 9.2).

Target y = log(1 + runtime_seconds) on conclusively-solved instances only.
This is selection-biased (timed-out instances are censored) and labeled secondary.
Paired size-only vs size+profile under the same grouped repeated CV.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

import common as C
import models as M
import grouped_splits as GS


def _pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)),
    ])


def run_runtime(df, cfg, out_path):
    rows = []
    for scn, sub in [("A_synthetic", df[df.domain == "synthetic"]),
                     ("C_combined", df)]:
        for v in C.VERIFIERS:
            cell = sub[sub.verifier == v].reset_index(drop=True)
            solved = cell[cell.raw_status.isin(["UNSAT", "SAT"]) & cell.raw_time.notna()].reset_index(drop=True)
            if len(solved) < 40:
                rows.append(dict(scenario=scn, verifier=v, n=len(solved),
                                 verdict="BLOCKED_INSUFFICIENT")); continue
            y = np.log1p(solved.raw_time.to_numpy(dtype=float))
            groups = solved.network_group.to_numpy()
            res = {"scenario": scn, "verifier": v, "n": int(len(solved)),
                   "n_groups": int(len(np.unique(groups)))}
            for fk in ["S", "SD"]:
                cols = M.feature_sets()[fk]
                X = M.build_design_matrix(solved, cols)
                preds = np.full(len(y), np.nan)
                for rep, fold, tr, te, k, sp in GS.repeated_group_splits(
                        y > np.median(y), groups, cfg["cv"]["outer_folds"], 5, cfg["cv"]["outer_seed_base"]):
                    m = _pipe().fit(X[tr], y[tr])
                    preds[te] = m.predict(X[te])   # last-repeat OOF (deterministic enough for secondary)
                ok = np.isfinite(preds)
                res[f"{fk}_spearman"] = float(spearmanr(y[ok], preds[ok]).correlation)
                res[f"{fk}_r2"] = float(r2_score(y[ok], preds[ok]))
                res[f"{fk}_mae"] = float(mean_absolute_error(y[ok], preds[ok]))
            res["delta_spearman"] = res["SD_spearman"] - res["S_spearman"]
            rows.append(res)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    cfg = C.load_config()
    df = C.load_rows()
    out = run_runtime(df, cfg, C.RESULTS / "runtime" / "runtime_regression.csv")
    print(out.to_string(index=False))
