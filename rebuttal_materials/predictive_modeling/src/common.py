"""Shared constants, paths, schema, and leak-free transforms for the
Difficulty-Profile predictive-modeling experiment.

Everything in this module is *stateless* configuration. All data-dependent
fitting (imputation, standardization) happens inside cross-validation folds
via sklearn Pipelines (see models.py). The nonlinear transforms declared here
are elementwise and stateless -> applying them globally introduces no leakage.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths (repo-root-relative; resolved from this file's location)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
PM_ROOT = REPO_ROOT / "rebuttal_materials" / "predictive_modeling"
CONFIG_PATH = PM_ROOT / "config" / "experiment.json"

MERGED_RECORDS = REPO_ROOT / "rebuttal_materials" / "merged_records_constructed_and_real.json"
BENCH_ROOT = REPO_ROOT / "src" / "VeriStressGT" / "benchmarks"

DATA_PROCESSED = PM_ROOT / "data" / "processed"
DATA_MANIFESTS = PM_ROOT / "data" / "manifests"
RESULTS = PM_ROOT / "results"
PLOTS = PM_ROOT / "plots"
LOGS = PM_ROOT / "logs"
MODELS = PM_ROOT / "models"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
VERIFIERS = ["abcrown", "neuralsat", "marabou", "nnenum", "pyrat"]

# merged-file field -> canonical profile name
PROFILE_MAP = {
    "M_hat_min": "margin_hat_min",
    "G_IBP": "g_ibp",
    "U": "unstable_fraction",
    "A_tau": "a_tau",
    "d_eff": "d_eff",
}
PROFILE_FEATURES = ["margin_hat_min", "g_ibp", "unstable_fraction", "a_tau", "d_eff"]

ESTABLISHED_BENCHES = {"vnncomp_mnist_fc", "oval21"}
SYNTHETIC_BENCHES = {"sweep_all", "polynomial_stress_22"}


def domain_of(benchmark: str) -> str:
    return "established" if benchmark in ESTABLISHED_BENCHES else "synthetic"


# Timeout-target semantics (spec sec 4.1)
POS_STATUS = "TIMEOUT"
# conclusive negative = solved (a robust UNSAT) or the rare unsound SAT/counterexample.
NEG_STATUSES = {"UNSAT", "SAT"}
# excluded from the strict timeout target
EXCLUDE_STATUSES = {"ERROR", "UNKNOWN", "missing", "other", None}

# --------------------------------------------------------------------------- #
# Network size/type feature schema
# --------------------------------------------------------------------------- #
# numeric size features extracted from ONNX (or reconstructed for poly)
SIZE_NUMERIC = [
    "input_dim",
    "output_dim",
    "n_param_total",
    "n_nodes",
    "n_gemm_matmul",
    "n_conv",
    "n_relu",
    "n_add",
    "n_mul",
    "n_softmax",
    "n_pow",
    "graph_depth",
    "n_hidden_layers",
    "max_hidden_width",
    "onnx_file_bytes",
]
# architecture-type one-hot categories (derived from graph structure)
ARCH_TYPES = ["MLP", "CNN", "ATTENTION", "POLYNOMIAL", "HYBRID", "OTHER"]
ARCH_ONEHOT = [f"arch_{a}" for a in ARCH_TYPES]
# operation-presence indicators
OP_FLAGS = ["has_conv", "has_softmax", "has_add_residual", "has_pow"]

SIZE_FEATURES = SIZE_NUMERIC + ARCH_ONEHOT + OP_FLAGS

# --------------------------------------------------------------------------- #
# Leak-free elementwise transforms (declared per column, applied globally).
# Stateful steps (impute/scale) live in the fold pipeline, never here.
# --------------------------------------------------------------------------- #
def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def log1p_nonneg(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, 0, None))


def logit_frac(x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def identity(x):
    return np.asarray(x, dtype=float)


# transform choice per feature (fixed, declared in advance).
TRANSFORM = {
    "margin_hat_min": signed_log1p,
    "g_ibp": signed_log1p,
    "unstable_fraction": logit_frac,
    "a_tau": identity,          # already A_tau_local_log (log-scaled)
    "d_eff": log1p_nonneg,
    # size numeric
    "input_dim": log1p_nonneg,
    "output_dim": log1p_nonneg,
    "n_param_total": log1p_nonneg,
    "n_nodes": log1p_nonneg,
    "n_gemm_matmul": log1p_nonneg,
    "n_conv": log1p_nonneg,
    "n_relu": log1p_nonneg,
    "n_add": log1p_nonneg,
    "n_mul": log1p_nonneg,
    "n_softmax": log1p_nonneg,
    "n_pow": log1p_nonneg,
    "graph_depth": log1p_nonneg,
    "n_hidden_layers": log1p_nonneg,
    "max_hidden_width": log1p_nonneg,
    "onnx_file_bytes": log1p_nonneg,
}
# one-hot + flags -> identity
for _c in ARCH_ONEHOT + OP_FLAGS:
    TRANSFORM[_c] = identity


def apply_transforms(df, cols):
    """Return a new DataFrame with each column in `cols` mapped through its
    declared stateless transform. Missing (NaN) values are preserved as NaN so
    the in-fold imputer can handle them."""
    import pandas as pd
    out = {}
    for c in cols:
        f = TRANSFORM.get(c, identity)
        v = df[c].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            t = f(v)
        t = np.where(np.isfinite(df[c].to_numpy(dtype=float)), t, np.nan)
        out[c] = t
    return pd.DataFrame(out, index=df.index)


def load_rows():
    """Load the frozen canonical row table; parquet preferred, CSV fallback
    (so the CV runs on hosts without pyarrow). Override the source with the
    PM_ROWS env var (e.g. the 600 s-rerun table `instance_verifier_rows_600.parquet`)."""
    import pandas as pd
    override = os.environ.get("PM_ROWS")
    if override:
        return pd.read_parquet(override) if override.endswith(".parquet") else pd.read_csv(override)
    pq = DATA_PROCESSED / "instance_verifier_rows.parquet"
    csv = DATA_PROCESSED / "instance_verifier_rows.csv"
    try:
        return pd.read_parquet(pq)
    except Exception:
        return pd.read_csv(csv)


def sha256_of(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
