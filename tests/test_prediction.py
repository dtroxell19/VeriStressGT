"""Tests for Thrust 2 timeout prediction (labels, leakage-free splits, feature consistency).

Run: PYTHONPATH=src python tests/test_prediction.py      (or: PYTHONPATH=src pytest tests/test_prediction.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from VeriStressGT.prediction.features import SIZE_FEATURES, instance_features
from VeriStressGT.prediction.model import instance_split, label_at_horizon, load_training_rows

SWEEP = Path(__file__).resolve().parents[1] / "src" / "VeriStressGT" / "benchmarks" / "sweep_all" / "instances"


def test_label_at_horizon():
    status = pd.Series(["UNSAT", "UNSAT", "TIMEOUT", "TIMEOUT", "ERROR", "SAT"])
    runtime = pd.Series([10.0, 90.0, 600.0, 240.0, 5.0, 30.0])
    budget = pd.Series([600.0, 600.0, 600.0, 30.0, 600.0, 600.0])
    y = label_at_horizon(status, runtime, budget, horizon=60.0)
    # solved fast -> 0; solved after H -> 1; timeout -> 1; timeout with budget < H -> excluded;
    # error -> excluded; conclusive SAT fast -> 0
    assert y.tolist()[:3] == [0, 1, 1] and np.isnan(y[3]) and np.isnan(y[4]) and y[5] == 0
    print("ok horizon labels")


def test_split_is_network_grouped():
    df = load_training_rows()
    train, test = instance_split(df, seed=3)
    g = df.drop_duplicates("instance_id").set_index("instance_id").network_group
    assert not (set(g[list(train)]) & set(g[list(test)])), "a network appears in both train and test"
    assert set(df[df.instance_id.isin(test)].domain) == {"synthetic", "established"}
    print(f"ok grouped split: {len(train)} train / {len(test)} test instances, no shared networks")


def test_features_match_training_table():
    """Features computed now reproduce the training table (size exactly, profile within sampling noise)."""
    df = load_training_rows()
    df = df[(df.verifier == "abcrown") & (df.benchmark == "sweep_all")].set_index("instance_id")
    for iid in ("pb_cnn_4", "fp2"):
        f = instance_features(str(SWEEP / iid / "model.onnx"), str(SWEEP / iid / "spec.vnnlib"))
        row = df.loc[iid]
        for c in SIZE_FEATURES:
            assert (pd.isna(row[c]) and pd.isna(f[c])) or row[c] == f[c], f"{iid}.{c}: {row[c]} != {f[c]}"
        for c in ("margin_hat_min", "g_ibp", "unstable_fraction", "a_tau"):
            assert abs(row[c] - f[c]) <= 0.02 * max(1.0, abs(row[c])), f"{iid}.{c}: {row[c]} vs {f[c]}"
    print("ok features reproduce the training table")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all prediction tests passed")
    sys.exit(0)
