"""Strict ONNX loader producing a piecewise-linear DAG.

Every node is one of:
  input   -- the network input
  linear  -- y = L(x) + b for a single variable input x (Gemm, MatMul-with-const, Conv,
             Flatten, Reshape, Add/Sub/Mul with a constant, Identity)
  add     -- y = x1 + x2 (or x1 - x2) for two variable inputs
  relu    -- y = max(x, 0)

Any other op (Softmax, variable x variable MatMul, ...) raises ``UnsupportedOp``. Unlike the
difficulty-profile IBP (which passes unknown ops through), nothing here is approximated, so
bounds computed on this DAG are sound for the ONNX model it came from. ``load_graph`` also
checks the DAG's forward pass against onnxruntime before returning it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DTYPE = torch.float64


class UnsupportedOp(Exception):
    pass


@dataclass
class Node:
    name: str
    kind: str                                   # input | linear | add | relu
    inputs: List[str] = field(default_factory=list)
    shape: Tuple[int, ...] = ()                 # per-sample shape (no batch dim)
    # linear nodes
    lin: Optional[Callable] = None              # linear part L (batched)
    lin_abs: Optional[Callable] = None          # |L| applied to a nonnegative tensor (for IBP radius)
    bias: Optional[torch.Tensor] = None         # shape == self.shape
    op_type: str = ""
    # add nodes: y = x1 + sign * x2
    sign: float = 1.0


@dataclass
class Graph:
    nodes: List[Node]                           # topological order
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    op_types: List[str]

    def by_name(self) -> Dict[str, Node]:
        return {n.name: n for n in self.nodes}

    def forward(self, x: torch.Tensor, *, drop_bias_ops: Tuple[str, ...] = ()) -> torch.Tensor:
        """Batched forward. x: [B, *input_shape]. Returns [B, num_outputs]."""
        vals: Dict[str, torch.Tensor] = {self.input_name: x}
        for n in self.nodes:
            if n.kind == "input":
                continue
            if n.kind == "linear":
                y = n.lin(vals[n.inputs[0]])
                if n.op_type not in drop_bias_ops:
                    y = y + n.bias
                vals[n.name] = y
            elif n.kind == "add":
                vals[n.name] = vals[n.inputs[0]] + n.sign * vals[n.inputs[1]]
            elif n.kind == "relu":
                vals[n.name] = torch.relu(vals[n.inputs[0]])
        out = vals[self.output_name]
        return out.reshape(out.shape[0], -1)


def _attrs(node) -> dict:
    import onnx
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _dims(value_info) -> List[int]:
    return [d.dim_value for d in value_info.type.tensor_type.shape.dim]


def _gemm_node(W: np.ndarray, attrs: dict, C: Optional[np.ndarray], in_shape, name) -> Tuple[Callable, Callable, torch.Tensor, Tuple[int, ...]]:
    if attrs.get("transA", 0):
        raise UnsupportedOp("Gemm with transA=1 on a variable input")
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    Wt = torch.tensor(W, dtype=DTYPE)
    if attrs.get("transB", 0):
        Wt = Wt.T                               # now [in, out]
    Wt = alpha * Wt
    Wabs = Wt.abs()
    out_dim = Wt.shape[1]
    b = torch.zeros(out_dim, dtype=DTYPE)
    if C is not None:
        b = b + beta * torch.tensor(np.broadcast_to(C, (out_dim,)).copy(), dtype=DTYPE)
    lin = lambda x, W=Wt: x.reshape(x.shape[0], -1) @ W
    lin_abs = lambda x, W=Wabs: x.reshape(x.shape[0], -1) @ W
    return lin, lin_abs, b, (out_dim,)


def load_graph(onnx_path: str, *, check: bool = True, weight_dtype: Optional[str] = None) -> Graph:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    g = model.graph
    consts: Dict[str, np.ndarray] = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    if weight_dtype:  # [BUG weights_fp16] reason about a reduced-precision copy of the weights
        consts = {k: v.astype(weight_dtype).astype(v.dtype) if v.dtype.kind == "f" else v
                  for k, v in consts.items()}
        check = False
    real_inputs = [i for i in g.input if i.name not in consts]
    if len(real_inputs) != 1:
        raise UnsupportedOp(f"expected exactly one graph input, got {len(real_inputs)}")
    inp = real_inputs[0]
    in_dims = _dims(inp)
    input_shape = tuple(int(d) for d in in_dims[1:]) if len(in_dims) > 1 else (int(in_dims[0]),)

    nodes: List[Node] = [Node(name=inp.name, kind="input", shape=input_shape)]
    shapes: Dict[str, Tuple[int, ...]] = {inp.name: input_shape}
    op_types: List[str] = []

    def var(name: str) -> bool:
        return name in shapes

    for node in g.node:
        op = node.op_type
        op_types.append(op)
        a = _attrs(node)
        ins = list(node.input)
        out = node.output[0]

        # ---- constant folding --------------------------------------------------------------
        if op == "Constant":
            consts[out] = numpy_helper.to_array(a["value"])
            continue
        if all((i in consts) or i == "" for i in ins):
            x = [consts[i] for i in ins if i]
            if op == "Transpose":
                consts[out] = np.transpose(x[0], a.get("perm"))
            elif op == "Reshape":
                consts[out] = x[0].reshape(x[1].astype(np.int64))
            elif op in ("Identity", "Cast"):
                consts[out] = x[0]
            elif op == "Unsqueeze":
                axes = a.get("axes", x[1].tolist() if len(x) > 1 else None)
                consts[out] = np.expand_dims(x[0], tuple(axes))
            elif op == "Neg":
                consts[out] = -x[0]
            else:
                raise UnsupportedOp(f"constant folding for {op}")
            continue

        # ---- variable ops ------------------------------------------------------------------
        if op == "Relu":
            nodes.append(Node(out, "relu", [ins[0]], shapes[ins[0]], op_type=op))
            shapes[out] = shapes[ins[0]]
            continue

        if op == "Gemm":
            if not var(ins[0]) or ins[1] not in consts:
                raise UnsupportedOp("Gemm must be variable x constant")
            C = consts.get(ins[2]) if len(ins) > 2 and ins[2] else None
            lin, lin_abs, b, oshape = _gemm_node(consts[ins[1]], a, C, shapes[ins[0]], out)
            nodes.append(Node(out, "linear", [ins[0]], oshape, lin, lin_abs, b, op))
            shapes[out] = oshape
            continue

        if op == "MatMul":
            if var(ins[0]) and ins[1] in consts:
                W = consts[ins[1]]
                if W.ndim != 2 or len(shapes[ins[0]]) != 1:
                    raise UnsupportedOp("MatMul with non-2D weight or non-vector input")
                lin, lin_abs, b, oshape = _gemm_node(W, {}, None, shapes[ins[0]], out)
                nodes.append(Node(out, "linear", [ins[0]], oshape, lin, lin_abs, b, op))
                shapes[out] = oshape
                continue
            raise UnsupportedOp("MatMul between two variable tensors (bilinear, e.g. attention)")

        if op == "Conv":
            if ins[1] not in consts:
                raise UnsupportedOp("Conv with variable weight")
            W = torch.tensor(consts[ins[1]], dtype=DTYPE)
            bias = consts.get(ins[2]) if len(ins) > 2 and ins[2] else None
            pads = a.get("pads", [0, 0, 0, 0])
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise UnsupportedOp("asymmetric Conv padding")
            kw = dict(stride=tuple(a.get("strides", [1, 1])), padding=(pads[0], pads[1]),
                      dilation=tuple(a.get("dilations", [1, 1])), groups=int(a.get("group", 1)))
            lin = lambda x, W=W, kw=kw: F.conv2d(x, W, None, **kw)
            lin_abs = lambda x, W=W.abs(), kw=kw: F.conv2d(x, W, None, **kw)
            probe = lin(torch.zeros((1,) + shapes[ins[0]], dtype=DTYPE))
            oshape = tuple(probe.shape[1:])
            b = torch.zeros(oshape, dtype=DTYPE)
            if bias is not None:
                b = b + torch.tensor(bias, dtype=DTYPE).reshape(-1, 1, 1)
            nodes.append(Node(out, "linear", [ins[0]], oshape, lin, lin_abs, b, op))
            shapes[out] = oshape
            continue

        if op in ("Flatten", "Reshape", "Identity"):
            ishape = shapes[ins[0]]
            if op == "Flatten":
                if int(a.get("axis", 1)) != 1:
                    raise UnsupportedOp("Flatten with axis != 1")
                oshape = (int(np.prod(ishape)),)
            elif op == "Reshape":
                tgt = [int(t) for t in consts[ins[1]]]
                if tgt[0] not in (-1, 0, 1):
                    raise UnsupportedOp(f"Reshape that mixes the batch dimension: {tgt}")
                per = [t for t in tgt[1:]]
                n = int(np.prod(ishape))
                if -1 in per:
                    known = int(np.prod([t for t in per if t != -1]))
                    per[per.index(-1)] = n // known
                oshape = tuple(per)
                if int(np.prod(oshape)) != n:
                    raise UnsupportedOp(f"Reshape {ishape} -> {tgt}")
            else:
                oshape = ishape
            lin = lambda x, s=oshape: x.reshape((x.shape[0],) + s)
            nodes.append(Node(out, "linear", [ins[0]], oshape, lin, lin, torch.zeros(oshape, dtype=DTYPE), op))
            shapes[out] = oshape
            continue

        if op in ("Add", "Sub"):
            sign = 1.0 if op == "Add" else -1.0
            if var(ins[0]) and var(ins[1]):
                if shapes[ins[0]] != shapes[ins[1]]:
                    raise UnsupportedOp("broadcasting Add between variables")
                nodes.append(Node(out, "add", ins[:2], shapes[ins[0]], op_type=op, sign=sign))
                shapes[out] = shapes[ins[0]]
                continue
            v, c = (ins[0], ins[1]) if var(ins[0]) else (ins[1], ins[0])
            if op == "Sub" and not var(ins[0]):
                raise UnsupportedOp("const - variable")
            ishape = shapes[v]
            b = torch.tensor(np.broadcast_to(consts[c].reshape(consts[c].shape[-len(ishape):] if consts[c].ndim > len(ishape) else consts[c].shape), ishape).copy(), dtype=DTYPE) * sign
            ident = lambda x: x
            nodes.append(Node(out, "linear", [v], ishape, ident, ident, b, op))
            shapes[out] = ishape
            continue

        if op == "Mul":
            if var(ins[0]) and var(ins[1]):
                raise UnsupportedOp("Mul between two variable tensors")
            v, c = (ins[0], ins[1]) if var(ins[0]) else (ins[1], ins[0])
            ishape = shapes[v]
            k = torch.tensor(np.broadcast_to(consts[c], ishape).copy(), dtype=DTYPE)
            nodes.append(Node(out, "linear", [v], ishape, lambda x, k=k: x * k, lambda x, k=k.abs(): x * k,
                              torch.zeros(ishape, dtype=DTYPE), op))
            shapes[out] = ishape
            continue

        raise UnsupportedOp(f"ONNX op {op}")

    out_name = g.output[0].name
    if out_name not in shapes:
        raise UnsupportedOp("graph output is constant")
    graph = Graph(nodes, inp.name, out_name, input_shape, op_types)
    if check:
        _check_against_onnxruntime(graph, onnx_path, in_dims)
    return graph


def _check_against_onnxruntime(graph: Graph, onnx_path: str, in_dims: List[int], n: int = 8) -> None:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(n,) + graph.input_shape).astype(np.float32)
    ref = []
    for i in range(n):  # some exported models have a fixed batch of 1
        ref.append(sess.run(None, {iname: x[i:i + 1]})[0].reshape(-1))
    ref = np.stack(ref)
    ours = graph.forward(torch.tensor(x, dtype=DTYPE)).numpy()
    scale = max(1.0, float(np.abs(ref).max()))
    err = float(np.abs(ours - ref).max()) / scale
    if not np.isfinite(err) or err > 1e-4:
        raise RuntimeError(f"graph loader disagrees with onnxruntime (rel err {err:.2e}) on {onnx_path}")
