"""Size/type and Difficulty-Profile features for per-verifier timeout prediction (Thrust 2).

The size/type schema, transforms and ONNX extractor are copied verbatim from
``experiments/predictive_modeling/src`` (common.py, extract_network_features.py), which built the
training table in ``data/training_rows.csv``; keep them identical so new instances are featurized
exactly like the training data.

Profile components map to ``difficulty_profile.estimate_profile`` outputs as below. ``a_tau`` is
the paper's log-count of distinct local fingerprints (Eq. 14) at 600 samples, which reproduces the
training table (checked on sweep_all instances).
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict

import numpy as np

PROFILE_FEATURES = ["margin_hat_min", "g_ibp", "unstable_fraction", "a_tau", "d_eff"]
PROFILE_SOURCE = {
    "margin_hat_min": "margin_sample_min",
    "g_ibp": "ibp_relative_gap",
    "unstable_fraction": "unstable_frac",
    "a_tau": "A_tau_local_log",
    "d_eff": "effective_grad_dim_mean",
}
ATAU_N_SAMPLES = 600

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


def _init_feat() -> dict:
    d = {k: np.nan for k in SIZE_NUMERIC}
    for a in ARCH_ONEHOT:
        d[a] = 0
    for f in OP_FLAGS:
        d[f] = 0
    return d


def _classify_arch(ops: dict) -> str:
    has_conv = ops.get("Conv", 0) > 0
    has_softmax = ops.get("Softmax", 0) > 0
    has_pow = ops.get("Pow", 0) > 0
    has_attn_shape = has_softmax or (ops.get("MatMul", 0) >= 3 and ops.get("Transpose", 0) >= 1)
    has_gemm = ops.get("Gemm", 0) + ops.get("MatMul", 0) > 0
    if has_pow:
        return "POLYNOMIAL"
    if has_attn_shape:
        return "ATTENTION"
    if has_conv:
        return "HYBRID" if has_softmax else "CNN"
    if has_gemm:
        return "MLP"
    return "OTHER"


def _longest_path(nodes) -> int:
    """Longest chain length over the node DAG (edges via tensor names)."""
    producer = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o] = i
    memo = {}

    def depth(i, stack):
        if i in memo:
            return memo[i]
        if i in stack:  # cycle guard (shouldn't happen in a valid ONNX)
            return 0
        stack.add(i)
        best = 0
        for inp in nodes[i].input:
            j = producer.get(inp)
            if j is not None:
                best = max(best, depth(j, stack))
        stack.discard(i)
        memo[i] = best + 1
        return memo[i]

    return max((depth(i, set()) for i in range(len(nodes))), default=0)


def features_from_onnx(onnx_path: str) -> dict:
    import onnx
    m = onnx.load(onnx_path)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:
        pass
    g = m.graph
    feat = _init_feat()

    # input / output dims (product of non-batch dims)
    def prod_dims(vi):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        dims = [d for d in dims if d and d > 1] or [d for d in dims if d]
        p = 1
        for d in dims:
            p *= d
        return int(p) if dims else np.nan

    feat["input_dim"] = prod_dims(g.input[0]) if g.input else np.nan
    feat["output_dim"] = prod_dims(g.output[0]) if g.output else np.nan

    # parameter count from initializers
    n_param = 0
    for init in g.initializer:
        sz = 1
        for d in init.dims:
            sz *= d
        n_param += sz
    feat["n_param_total"] = int(n_param)

    ops = collections.Counter(n.op_type for n in g.node)
    feat["n_nodes"] = int(sum(ops.values()))
    feat["n_gemm_matmul"] = int(ops.get("Gemm", 0) + ops.get("MatMul", 0))
    feat["n_conv"] = int(ops.get("Conv", 0))
    feat["n_relu"] = int(ops.get("Relu", 0))
    feat["n_add"] = int(ops.get("Add", 0))
    feat["n_mul"] = int(ops.get("Mul", 0))
    feat["n_softmax"] = int(ops.get("Softmax", 0))
    feat["n_pow"] = int(ops.get("Pow", 0))
    feat["graph_depth"] = _longest_path(list(g.node))
    # hidden layers ~ number of ReLU activation stages; width from value_info shapes
    feat["n_hidden_layers"] = int(ops.get("Relu", 0))

    # max hidden width: largest intermediate 2D/feature dim among value_info
    widths = []
    for vi in list(g.value_info) + list(g.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        dims = [d for d in dims if d and d > 1]
        if dims:
            widths.append(max(dims))
    feat["max_hidden_width"] = int(max(widths)) if widths else np.nan

    feat["onnx_file_bytes"] = int(Path(onnx_path).stat().st_size)

    arch = _classify_arch(ops)
    feat[f"arch_{arch}"] = 1
    feat["has_conv"] = int(ops.get("Conv", 0) > 0)
    feat["has_softmax"] = int(ops.get("Softmax", 0) > 0)
    feat["has_add_residual"] = int(ops.get("Add", 0) > 0)
    feat["has_pow"] = int(ops.get("Pow", 0) > 0)
    feat["_arch"] = arch
    return feat


def profile_features(onnx_path: str, vnnlib_path: str) -> Dict[str, float]:
    from VeriStressGT.difficulty_profile.profile import estimate_profile
    p = estimate_profile(onnx_path, vnnlib_path, verbose=False, atau_n_samples=ATAU_N_SAMPLES).to_dict()
    out = {}
    for k, src in PROFILE_SOURCE.items():
        v = p.get(src)
        out[k] = float(v) if isinstance(v, (int, float)) and v is not None else np.nan
    return out


def instance_features(onnx_path: str, vnnlib_path: str) -> Dict[str, float]:
    """All model features (size/type + profile) for one verification instance."""
    feat = features_from_onnx(onnx_path)
    feat.update(profile_features(onnx_path, vnnlib_path))
    return feat
