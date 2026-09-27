"""Deterministic network size/type feature extraction.

For each instance we prefer extraction from the local ONNX graph. The 22
polynomial_stress_22 instances have no committed ONNX, so their features are
reconstructed from the generator `args` (input_dim, hidden_dim, degree,
num_outputs). Every row carries a `feat_source` and per-feature missingness is
tracked via the `onnx_file_bytes` column (NaN for reconstructed rows).
"""
from __future__ import annotations
import collections
from pathlib import Path

import numpy as np

from common import SIZE_NUMERIC, ARCH_TYPES, ARCH_ONEHOT, OP_FLAGS, sha256_of


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
    feat["_onnx_sha256"] = sha256_of(onnx_path)
    return feat


def features_from_poly_args(args: dict) -> dict:
    """Reconstruct size/type features for polynomial nets lacking a local ONNX.
    Architecture: x -> Gemm(in->hidden) -> Pow(degree) -> Gemm(hidden->out)."""
    feat = _init_feat()
    din = int(args.get("input_dim", 0))
    h = int(args.get("hidden_dim", 0))
    out = int(args.get("num_outputs", 0))
    feat["input_dim"] = din
    feat["output_dim"] = out
    feat["n_param_total"] = din * h + h + h * out + out
    feat["n_gemm_matmul"] = 2
    feat["n_conv"] = 0
    feat["n_relu"] = 0
    feat["n_add"] = 0
    feat["n_mul"] = 0
    feat["n_softmax"] = 0
    feat["n_pow"] = 1
    feat["n_nodes"] = 5  # gemm, pow, gemm, (+2 reshape/const typical)
    feat["graph_depth"] = 4
    feat["n_hidden_layers"] = 1
    feat["max_hidden_width"] = h
    feat["onnx_file_bytes"] = np.nan  # not committed -> missing (tracked)
    feat["arch_POLYNOMIAL"] = 1
    feat["has_pow"] = 1
    feat["_arch"] = "POLYNOMIAL"
    feat["_onnx_sha256"] = None
    return feat
