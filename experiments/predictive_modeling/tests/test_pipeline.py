"""Leakage & correctness tests (spec sec 13). Fail closed on any leakage.

Run:  PYTHONPATH=../src python test_pipeline.py     (from tests/)
  or: pytest test_pipeline.py
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

import common as C
import models as M
import grouped_splits as GS
import evaluate as E

DF = pd.read_parquet(C.DATA_PROCESSED / "instance_verifier_rows.parquet")

FORBIDDEN = {"raw_status", "raw_time", "raw_outcome", "budget_s", "instance_id",
             "benchmark", "constructor", "network_group", "domain", "verifier",
             "y", "target"}


def test_no_forbidden_features():
    fs = M.feature_sets()
    for name, cols in fs.items():
        bad = set(cols) & FORBIDDEN
        assert not bad, f"feature set {name} leaks {bad}"
    print("PASS test_no_forbidden_features")


def test_group_no_leakage_and_identical_folds():
    cell = DF[(DF.verifier == "abcrown") & (DF.domain == "synthetic")].reset_index(drop=True)
    H, y, _ = GS.choose_horizon(cell)
    mask = np.isfinite(y); y = y[mask].astype(int)
    groups = cell[mask].network_group.to_numpy()
    splits = list(GS.repeated_group_splits(y, groups, 5, 3, 1100))
    seen = {}
    for rep, fold, tr, te, k, sp in splits:
        GS.assert_no_group_leakage(groups, tr, te)          # (1) no leakage
        assert not (set(groups[tr]) & set(groups[te]))
        seen[(rep, fold)] = (tuple(tr), tuple(te))
    # (2) folds are a function of (y,groups,seed) only -> deterministic re-gen matches
    splits2 = list(GS.repeated_group_splits(y, groups, 5, 3, 1100))
    for (rep, fold, tr, te, k, sp) in splits2:
        assert (tuple(tr), tuple(te)) == seen[(rep, fold)]
    print("PASS test_group_no_leakage_and_identical_folds")


def test_label_boundaries():
    # budget below horizon -> excluded
    assert np.isnan(GS.label_at_horizon("UNSAT", 100, 240, 360))
    # solved before horizon -> 0
    assert GS.label_at_horizon("UNSAT", 100, 600, 360) == 0.0
    # solved exactly at horizon -> timeout (>=)
    assert GS.label_at_horizon("UNSAT", 360, 600, 360) == 1.0
    # solved after horizon -> timeout
    assert GS.label_at_horizon("UNSAT", 500, 600, 360) == 1.0
    # native TIMEOUT with budget>=H -> 1
    assert GS.label_at_horizon("TIMEOUT", 600, 600, 360) == 1.0
    # ERROR / UNKNOWN / missing excluded
    for s in ["ERROR", "UNKNOWN", "missing", "other"]:
        assert np.isnan(GS.label_at_horizon(s, 10, 600, 360))
    print("PASS test_label_boundaries")


def test_determinism_splits():
    y = np.array([0, 1] * 30); g = np.repeat(np.arange(20), 3)
    a = list(GS.repeated_group_splits(y, g, 5, 2, 1100))
    b = list(GS.repeated_group_splits(y, g, 5, 2, 1100))
    assert all((tuple(x[2]), tuple(x[3])) == (tuple(z[2]), tuple(z[3])) for x, z in zip(a, b))
    print("PASS test_determinism_splits")


def test_oof_covers_each_row_once_per_repeat():
    cfg = C.load_config(); cfg["cv"]["outer_repeats"] = 2
    cell = DF[(DF.verifier == "abcrown") & (DF.domain == "synthetic")]
    fs = {k: M.feature_sets()[k] for k in ["S", "SD"]}
    oof, meta = E.run_cell(cell, fs, cfg, n_jobs=1)
    for (rep, feat), g in oof.groupby(["repeat", "feature_set"]):
        assert g["row"].is_unique, "a row predicted twice in one repeat"
        assert len(g) == meta["n"], f"repeat {rep} {feat}: {len(g)} != {meta['n']}"
    print("PASS test_oof_covers_each_row_once_per_repeat")


def test_pipeline_fits_only_on_train():
    # structural: imputer/scaler are steps in the pipeline -> sklearn fits them per-fold.
    p = M.make_pipeline()
    steps = [s[0] for s in p.steps]
    assert steps[:2] == ["impute", "scale"] and steps[-1] == "clf"
    print("PASS test_pipeline_fits_only_on_train")


def test_serialization_roundtrip():
    import pickle
    cell = DF[(DF.verifier == "abcrown") & (DF.domain == "synthetic")].reset_index(drop=True)
    H, y, _ = GS.choose_horizon(cell); mask = np.isfinite(y)
    X = M.build_design_matrix(cell[mask], M.feature_sets()["SD"]); yv = y[mask].astype(int)
    p = M.make_pipeline(C_=1.0, l1_ratio=0.5).fit(X, yv)
    p2 = pickle.loads(pickle.dumps(p))
    assert np.allclose(p.predict_proba(X)[:, 1], p2.predict_proba(X)[:, 1])
    print("PASS test_serialization_roundtrip")


def _toy(n=400, profile_effect=True, seed=0):
    rng = np.random.RandomState(seed)
    size = rng.normal(size=(n, 3))
    prof = rng.normal(size=(n, 2))
    groups = np.repeat(np.arange(n // 4), 4)
    logit = 0.8 * size[:, 0]
    if profile_effect:
        logit = logit + 1.6 * prof[:, 0]     # profile adds signal beyond size
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return size, prof, y, groups


def _toy_cv_auc(X, y, groups, seed=1100):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for rep, fold, tr, te, k, sp in GS.repeated_group_splits(y, groups, 5, 3, seed):
        clf = M.make_pipeline(C_=1.0, l1_ratio=0.0).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def test_toy_positive_control():
    size, prof, y, groups = _toy(profile_effect=True)
    auc_s = _toy_cv_auc(size, y, groups)
    auc_sd = _toy_cv_auc(np.hstack([size, prof]), y, groups)
    assert auc_sd > auc_s + 0.03, f"profile increment not recovered: S={auc_s:.3f} SD={auc_sd:.3f}"
    print(f"PASS test_toy_positive_control  S={auc_s:.3f} SD={auc_sd:.3f}")


def test_toy_negative_control():
    size, prof, y, groups = _toy(profile_effect=False)
    auc_s = _toy_cv_auc(size, y, groups)
    auc_sd = _toy_cv_auc(np.hstack([size, prof]), y, groups)
    assert auc_sd <= auc_s + 0.03, f"spurious increment: S={auc_s:.3f} SD={auc_sd:.3f}"
    print(f"PASS test_toy_negative_control  S={auc_s:.3f} SD={auc_sd:.3f}")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    fails = 0
    for t in ALL:
        try:
            t()
        except AssertionError as e:
            fails += 1; print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            fails += 1; print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL)-fails}/{len(ALL)} tests passed")
    sys.exit(1 if fails else 0)
