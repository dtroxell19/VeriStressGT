"""Per-verifier timeout predictors (Thrust 2).

Training data: ``data/training_rows.csv`` -- one row per (instance, verifier) from the server
runs used in the paper (345 instances: 225 synthetic constructor instances + 120 VNN-COMP
mnist_fc/oval21 instances; 5 verifiers; budgets 240-600 s), with size/type and profile features.

Target at horizon H (seconds): 1 = the verifier did not finish within H (TIMEOUT with budget >= H,
or a conclusive verdict that took >= H); 0 = a conclusive UNSAT/SAT in < H. ERROR/UNKNOWN rows
and TIMEOUTs whose budget was shorter than H are excluded. Recorded runtimes let the same data be
relabelled at any horizon up to its budget, so the predictor can match a live run's timeout.

Model: the rebuttal study's elastic-net logistic regression (impute -> standardize -> logistic),
tuned by ROC-AUC with grouped inner CV. Groups are networks, so no network appears on both sides
of any split (VNN-COMP benchmarks reuse a handful of networks across many properties).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import PROFILE_FEATURES, SIZE_FEATURES, apply_transforms

TRAINING_ROWS = Path(__file__).resolve().parent / "data" / "training_rows.csv"
VERIFIERS = ["abcrown", "neuralsat", "marabou", "nnenum", "pyrat"]
FEATURE_SETS = {
    "size": list(SIZE_FEATURES),
    "profile": list(PROFILE_FEATURES),
    "size+profile": list(SIZE_FEATURES) + list(PROFILE_FEATURES),
}
GRID = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__l1_ratio": [0.0, 0.5, 1.0],
        "clf__class_weight": [None, "balanced"]}


def load_training_rows(path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_csv(path or TRAINING_ROWS)


def label_at_horizon(status: pd.Series, runtime: pd.Series, budget: pd.Series, horizon: float) -> pd.Series:
    s = status.astype(str).str.upper()
    conclusive = s.isin(["UNSAT", "SAT"])
    y = pd.Series(np.nan, index=status.index)
    y[conclusive & (runtime < horizon)] = 0
    y[conclusive & (runtime >= horizon)] = 1
    y[(s == "TIMEOUT") & (budget >= horizon)] = 1
    return y


def rows_for(df: pd.DataFrame, verifier: str, horizon: float) -> pd.DataFrame:
    d = df[df.verifier == verifier].copy()
    d["y"] = label_at_horizon(d.raw_status, d.raw_time, d.budget_s, horizon)
    return d[d.y.notna()]


def instance_split(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0) -> Tuple[set, set]:
    """Network-grouped train/test split of instance ids, shared by every verifier.

    Stratified by domain so the 6 established networks land on both sides."""
    inst = df.drop_duplicates("instance_id")[["instance_id", "network_group", "domain"]].reset_index(drop=True)
    k = max(2, int(round(1 / test_frac)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(inst, inst.domain, inst.network_group))
    test = set(inst.instance_id.iloc[test_idx])
    return set(inst.instance_id) - test, test


def _pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=5000, tol=1e-3)),  # l1_ratio set by the grid
    ])


def design(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return apply_transforms(df, cols)[cols].to_numpy(dtype=float)


def fit(train: pd.DataFrame, cols: List[str], seed: int = 0):
    """Grouped-CV-tuned elastic-net logistic on one verifier's training rows."""
    X, y, g = design(train, cols), train.y.to_numpy(int), train.network_group.to_numpy()
    n_splits = min(4, len(np.unique(g)))
    search = GridSearchCV(_pipeline(), GRID, scoring="roc_auc", cv=GroupKFold(n_splits=n_splits),
                          n_jobs=1, error_score=np.nan)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X, y, groups=g)
    return search.best_estimator_


def auc(model, rows: pd.DataFrame, cols: List[str]) -> Optional[float]:
    y = rows.y.to_numpy(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, model.predict_proba(design(rows, cols))[:, 1]))


def evaluate_recorded(df: pd.DataFrame, horizon: float, seed: int = 0) -> Dict[str, Dict]:
    """Held-out AUC per verifier and feature set, plus synthetic -> established transfer."""
    train_ids, test_ids = instance_split(df, seed=seed)
    out: Dict[str, Dict] = {}
    for v in VERIFIERS:
        rows = rows_for(df, v, horizon)
        tr, te = rows[rows.instance_id.isin(train_ids)], rows[rows.instance_id.isin(test_ids)]
        res = {"n_train": int(len(tr)), "n_test": int(len(te)),
               "test_timeout_rate": float(te.y.mean()) if len(te) else None, "auc": {}}
        for name, cols in FEATURE_SETS.items():
            res["auc"][name] = auc(fit(tr, cols, seed), te, cols)
        syn, est = rows[rows.domain == "synthetic"], rows[rows.domain == "established"]
        cols = FEATURE_SETS["size+profile"]
        res["transfer_synthetic_to_established_auc"] = auc(fit(syn, cols, seed), est, cols)
        res["transfer_size_only_auc"] = auc(fit(syn, FEATURE_SETS["size"], seed), est, FEATURE_SETS["size"])
        out[v] = res
    return out


def fit_full(df: pd.DataFrame, verifier: str, horizon: float, cols: Optional[List[str]] = None, seed: int = 0):
    """Final model on all recorded rows (used for predicting fresh instances)."""
    cols = cols or FEATURE_SETS["size+profile"]
    return fit(rows_for(df, verifier, horizon), cols, seed)
