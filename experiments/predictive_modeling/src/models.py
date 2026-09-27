"""Feature-set definitions, design-matrix construction, and the interpretable
estimator (elastic-net logistic regression) with its tuning grid.

Leakage discipline: the stateless nonlinear transforms are applied here (they
use no data statistics); the *stateful* steps (median imputation, standardization)
live inside the sklearn Pipeline and are therefore fit only on training folds.
"""
from __future__ import annotations
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import common as C


# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #
def feature_sets():
    S = list(C.SIZE_FEATURES)
    D = list(C.PROFILE_FEATURES)
    S_noarch = [c for c in S if not c.startswith("arch_")]
    fs = {
        "S": S,               # size/type baseline
        "D": D,               # profile only (diagnostic)
        "SD": S + D,          # augmented (headline)
        "S_noarch": S_noarch,
        "SD_noarch": S_noarch + D,
    }
    # add-one-profile-component to size baseline
    for p in D:
        fs[f"S_plus_{p}"] = S + [p]
    # drop-one-profile-component from augmented
    for p in D:
        fs[f"SD_minus_{p}"] = S + [c for c in D if c != p]
    return fs


def build_design_matrix(df, cols):
    """Return X (np.ndarray) with declared stateless transforms applied and NaNs
    preserved for the in-fold imputer. Column order == `cols`."""
    tdf = C.apply_transforms(df, cols)
    return tdf[cols].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Estimator + grid
# --------------------------------------------------------------------------- #
def make_pipeline(class_weight=None, C_=1.0, l1_ratio=0.5, max_iter=5000):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=C_,
            class_weight=class_weight, max_iter=max_iter, tol=1e-3, n_jobs=1,
        )),
    ])


def param_grid(grid_cfg):
    return {
        "clf__C": grid_cfg["C"],
        "clf__l1_ratio": grid_cfg["l1_ratio"],
        "clf__class_weight": grid_cfg["class_weight"],
    }


def make_tree(max_depth=3):
    from sklearn.tree import DecisionTreeClassifier
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                       random_state=0)),
    ])


def make_spline_pipeline(n_knots=4, class_weight="balanced", C_=1.0):
    """Additive spline logistic (secondary). Splines on continuous columns only;
    binary arch/flag columns pass through."""
    from sklearn.preprocessing import SplineTransformer
    from sklearn.compose import ColumnTransformer
    return SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
