"""Rigorous interval (mpmath.iv) forward evaluation of small ONNX graphs.

Propagates an exact input box through the graph with outward-rounded interval
arithmetic, so the returned logit intervals provably CONTAIN the true value set over
the box. Used to certify the attention families (Dominant-Key: matmul/relu/reshape;
Fixed-Order: + softmax) by direct enclosure — the margin's interval lower bound > 0
is a rigorous certificate over the continuum, needing no Lipschitz constant.

Only the ops appearing in those graphs are implemented. Float tensors are object
arrays of iv.mpf; integer shape tensors stay as numpy int arrays.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict

import numpy as np
import onnx
from onnx import numpy_helper
from mpmath import iv

from . import sound_backends as SB


def _to_iv_array(arr: np.ndarray) -> np.ndarray:
    out = np.empty(arr.size, dtype=object)
    out[:] = [iv.mpf(float(x)) for x in arr.reshape(-1)]
    return out.reshape(arr.shape)


def _is_int_tensor(a: np.ndarray) -> bool:
    return a.dtype.kind in ("i", "u")


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(a.t)
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
    return default


def _relu(a):
    return np.vectorize(SB.iv_relu, otypes=[object])(a)


def _softmax(a, axis):
    a = np.moveaxis(a, axis, -1)
    shape = a.shape
    flat = a.reshape(-1, shape[-1])
    out = np.empty_like(flat)
    for r in range(flat.shape[0]):
        out[r] = SB.iv_softmax(list(flat[r]))
    out = out.reshape(shape)
    return np.moveaxis(out, -1, axis)


def interval_logits(onnx_path: str, box_lo, box_hi, prec_bits: int = 200) -> np.ndarray:
    """Return object-array of iv.mpf logits enclosing f(x) for all x in the box."""
    iv.prec = prec_bits
    m = onnx.load(onnx_path)
    g = m.graph
    env: Dict[str, np.ndarray] = {}
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        env[init.name] = arr if _is_int_tensor(arr) else _to_iv_array(arr)

    # input box -> interval object array in the graph's declared input shape
    ishape = [d.dim_value if d.dim_value > 0 else 1
              for d in g.input[0].type.tensor_type.shape.dim]
    n = int(np.prod(ishape))
    xiv = np.array([SB.iv_interval(box_lo[i], box_hi[i]) for i in range(n)], dtype=object).reshape(ishape)
    env[g.input[0].name] = xiv

    for node in g.node:
        op = node.op_type
        ins = [env[i] for i in node.input if i in env]
        if op == "Constant":
            arr = _attr(node, "value")
            env[node.output[0]] = arr if _is_int_tensor(arr) else _to_iv_array(arr)
        elif op == "Identity":
            env[node.output[0]] = ins[0]
        elif op == "MatMul":
            env[node.output[0]] = np.matmul(ins[0], ins[1])
        elif op == "Gemm":
            A, B = ins[0], ins[1]
            C = ins[2] if len(ins) > 2 else None
            if _attr(node, "transA", 0):
                A = np.swapaxes(A, -1, -2)
            if _attr(node, "transB", 0):
                B = np.swapaxes(B, -1, -2)
            alpha = _attr(node, "alpha", 1.0); beta = _attr(node, "beta", 1.0)
            out = np.matmul(A, B)
            if alpha != 1.0:
                out = out * iv.mpf(float(alpha))
            if C is not None:
                out = out + (C * iv.mpf(float(beta)) if beta != 1.0 else C)
            env[node.output[0]] = out
        elif op == "Relu":
            env[node.output[0]] = _relu(ins[0])
        elif op == "Add":
            env[node.output[0]] = ins[0] + ins[1]
        elif op == "Sub":
            env[node.output[0]] = ins[0] - ins[1]
        elif op == "Mul":
            b = ins[1]
            if isinstance(b, np.ndarray) and b.size == 1:
                b = b.reshape(-1)[0]
            env[node.output[0]] = ins[0] * b
        elif op == "Transpose":
            perm = _attr(node, "perm")
            env[node.output[0]] = np.transpose(ins[0], perm)
        elif op == "Reshape":
            data, shape = ins[0], np.array(ins[1]).astype(int).reshape(-1).tolist()
            resolved = []
            for k, s in enumerate(shape):
                resolved.append(int(data.shape[k]) if s == 0 else int(s))
            env[node.output[0]] = data.reshape(resolved)
        elif op == "Flatten":
            ax = _attr(node, "axis", 1)
            sh = ins[0].shape
            env[node.output[0]] = ins[0].reshape(int(np.prod(sh[:ax])) or 1, -1)
        elif op == "Unsqueeze":
            axes = ins[1].tolist() if len(ins) > 1 else _attr(node, "axes")
            out = ins[0]
            for ax in sorted(axes):
                out = np.expand_dims(out, ax)
            env[node.output[0]] = out
        elif op == "Concat":
            env[node.output[0]] = np.concatenate(ins, axis=_attr(node, "axis", 0))
        elif op == "Pow":
            base, exp = ins[0], ins[1]
            e = exp.reshape(-1)[0]
            n_exp = int(round(float(e.a))) if hasattr(e, "a") else int(round(float(e)))
            env[node.output[0]] = np.vectorize(lambda z: z ** n_exp, otypes=[object])(base)
        elif op == "Softmax":
            env[node.output[0]] = _softmax(ins[0], _attr(node, "axis", -1))
        else:
            raise NotImplementedError(f"iv_onnx: unsupported op {op}")

    return env[g.output[0].name].reshape(-1)


def interval_margin_lb(onnx_path: str, box_lo, box_hi, label: int, prec_bits: int = 200):
    """Rigorous lower bound on min-margin over the box: logits[label].lower - max_{k!=label} logits[k].upper."""
    logits = interval_logits(onnx_path, box_lo, box_hi, prec_bits)
    lo_label = logits[label].a
    hi_others = max(logits[k].b for k in range(len(logits)) if k != label)
    margin = logits[label] - iv.mpf(hi_others)
    return float(margin.a), logits
