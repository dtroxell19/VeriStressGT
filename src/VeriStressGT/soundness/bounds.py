"""IBP and CROWN-style backward linear bounds on a ``graph.Graph``.

All functions are batched over B input sub-domains (boxes given by center ``c`` and radius ``r``,
each [B, *input_shape]). ``BoundOpts`` carries the planted-bug switches used by the mutant
verifiers; with default options every bound here is sound (up to float64 rounding).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .graph import DTYPE, Graph, Node

Interval = Tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class BoundOpts:
    use_crown: bool = True                  # False -> IBP only (sound, weaker)
    ibp_sign_bug: bool = False              # [BUG] interval matmul without splitting W+ / W-
    relu_no_intercept_bug: bool = False     # [BUG] ReLU upper relaxation drops its intercept
    drop_bias_ops: Tuple[str, ...] = ()     # [BUG] bias of these op types ignored in bounds
    crown_rows: int = 4096                  # max B*2*neurons for an intermediate CROWN pass (else IBP)


def _bias(n: Node, opts: BoundOpts) -> torch.Tensor:
    return torch.zeros_like(n.bias) if n.op_type in opts.drop_bias_ops else n.bias


def ibp(graph: Graph, c: torch.Tensor, r: torch.Tensor, opts: BoundOpts,
        refined: Optional[Dict[str, Interval]] = None) -> Dict[str, Interval]:
    """Interval bounds for every node. ``refined`` overrides (intersects) bounds at given nodes."""
    iv: Dict[str, Interval] = {graph.input_name: (c - r, c + r)}
    for n in graph.nodes:
        if n.kind == "input":
            continue
        if n.kind == "linear":
            lo, hi = iv[n.inputs[0]]
            b = _bias(n, opts)
            if opts.ibp_sign_bug:
                a1, a2 = n.lin(lo) + b, n.lin(hi) + b
                out = (torch.minimum(a1, a2), torch.maximum(a1, a2))
            else:
                cc, rr = (lo + hi) / 2, (hi - lo) / 2
                cn, rn = n.lin(cc) + b, n.lin_abs(rr)
                out = (cn - rn, cn + rn)
        elif n.kind == "add":
            (l1, h1), (l2, h2) = iv[n.inputs[0]], iv[n.inputs[1]]
            out = (l1 + l2, h1 + h2) if n.sign > 0 else (l1 - h2, h1 - l2)
        else:  # relu
            lo, hi = iv[n.inputs[0]]
            out = (lo.clamp(min=0), hi.clamp(min=0))
        if refined is not None and n.name in refined:
            rl, rh = refined[n.name]
            out = (torch.maximum(out[0], rl), torch.minimum(out[1], rh))
        iv[n.name] = out
    return iv


def _lin_T(n: Node, A: torch.Tensor, in_shape) -> torch.Tensor:
    """A: [M, *out_shape] -> A . L as [M, *in_shape] (vector-Jacobian product of the linear part)."""
    z = torch.zeros((A.shape[0],) + tuple(in_shape), dtype=DTYPE, requires_grad=True)
    y = n.lin(z)
    (g,) = torch.autograd.grad(y, z, grad_outputs=A.reshape(y.shape))
    return g.detach()


def crown(graph: Graph, target: str, A_target: torch.Tensor, c: torch.Tensor, r: torch.Tensor,
          pre: Dict[str, Interval], opts: BoundOpts) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lower bound of ``A_target . node(target)`` over each box.

    A_target: [B, K, *shape(target)]. ``pre`` holds interval bounds of every ReLU input that
    ``target`` depends on. Returns (lower [B, K], A_input [B, K, *input_shape]).
    """
    nodes = graph.by_name()
    B, K = A_target.shape[:2]
    A: Dict[str, torch.Tensor] = {target: A_target}
    const = torch.zeros((B, K), dtype=DTYPE)
    order = [n.name for n in graph.nodes]
    for name in reversed(order[: order.index(target) + 1]):
        if name not in A or nodes[name].kind == "input":
            continue
        n = nodes[name]
        An = A.pop(name)
        red = tuple(range(2, An.dim()))
        if n.kind == "linear":
            const = const + (An * _bias(n, opts)).sum(dim=red)
            src = nodes[n.inputs[0]]
            Ain = _lin_T(n, An.reshape((B * K,) + n.shape), src.shape).reshape((B, K) + src.shape)
            A[src.name] = A[src.name] + Ain if src.name in A else Ain
        elif n.kind == "add":
            for src_name, s in ((n.inputs[0], 1.0), (n.inputs[1], n.sign)):
                A[src_name] = A[src_name] + s * An if src_name in A else s * An
        else:  # relu
            l, u = pre[n.inputs[0]]
            l, u = l.unsqueeze(1), u.unsqueeze(1)
            active = (l >= 0).to(DTYPE)
            unstable = ((l < 0) & (u > 0)).to(DTYPE)
            denom = torch.where(unstable > 0, u - l, torch.ones_like(u))
            s_up = u / denom
            t_up = -s_up * l
            if opts.relu_no_intercept_bug:
                t_up = torch.zeros_like(t_up)
            alpha = (u > -l).to(DTYPE)
            slope_lo = active + unstable * alpha
            slope_up = active + unstable * s_up
            Apos, Aneg = An.clamp(min=0), An.clamp(max=0)
            const = const + (Aneg * unstable * t_up).sum(dim=red)
            Az = Apos * slope_lo + Aneg * slope_up
            src = n.inputs[0]
            A[src] = A[src] + Az if src in A else Az
    Ain = A.get(graph.input_name, torch.zeros((B, K) + graph.input_shape, dtype=DTYPE))
    red = tuple(range(2, Ain.dim()))
    cc, rr = c.unsqueeze(1), r.unsqueeze(1)
    lower = const + (Ain * cc).sum(dim=red) - (Ain.abs() * rr).sum(dim=red)
    return lower, Ain


def spec_lower_bounds(graph: Graph, C: torch.Tensor, c: torch.Tensor, r: torch.Tensor,
                      opts: BoundOpts) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Lower bounds of ``C . f(x)`` over each box. C: [K, n_out]. Returns ([B, K], A_input or None)."""
    B = c.shape[0]
    iv = ibp(graph, c, r, opts)
    out_lo, out_hi = iv[graph.output_name]
    out_lo, out_hi = out_lo.reshape(B, -1), out_hi.reshape(B, -1)
    Cp, Cn = C.clamp(min=0), C.clamp(max=0)
    lb_ibp = out_lo @ Cp.T + out_hi @ Cn.T
    if not opts.use_crown:
        return lb_ibp, None

    # Tighten every ReLU input with CROWN (in topological order), intersected with IBP.
    refined: Dict[str, Interval] = {}
    for n in graph.nodes:
        if n.kind != "relu":
            continue
        z = n.inputs[0]
        if z in refined:
            continue
        shape = iv[z][0].shape[1:]
        m = int(torch.tensor(shape).prod())
        if B * 2 * m > opts.crown_rows:
            refined[z] = iv[z]
            continue
        eye = torch.eye(m, dtype=DTYPE).reshape((m,) + tuple(shape))
        A0 = torch.cat([eye, -eye], dim=0).unsqueeze(0).expand((B, 2 * m) + tuple(shape)).contiguous()
        lb, _ = crown(graph, z, A0, c, r, {**iv, **refined}, opts)
        lo_c, hi_c = lb[:, :m].reshape((B,) + tuple(shape)), -lb[:, m:].reshape((B,) + tuple(shape))
        refined[z] = (torch.maximum(iv[z][0], lo_c), torch.minimum(iv[z][1], hi_c))
        iv = ibp(graph, c, r, opts, refined)  # propagate tighter bounds downstream

    out_node = graph.by_name()[graph.output_name]
    A_out = C.reshape((1, C.shape[0]) + out_node.shape).expand((B, C.shape[0]) + out_node.shape).contiguous()
    lb_crown, A_in = crown(graph, graph.output_name, A_out, c, r, {**iv, **refined}, opts)
    return torch.maximum(lb_crown, lb_ibp), A_in
