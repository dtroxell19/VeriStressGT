"""In-house reference verifier with injectable bugs (Thrust 1 synthetic verifiers).

Pipeline (bug = "none"):
  1. strict ONNX load (``graph.load_graph``); unsupported ops -> error "unsupported ..."
  2. PGD attack; SAT only if the float64 forward pass confirms a violating point in the box
  3. CROWN bounds + ReLU-split branch-and-bound; UNSAT only if every sub-domain is proved
  4. otherwise UNKNOWN (budget exhausted)

Each entry of ``bugs.MUTANT_BUGS`` switches on one planted defect. The reference ("none") and
"ibp_only" are sound controls and must never be flagged by a ground-truth benchmark.

Usage (prints an authoritative ``Result: <sat|unsat|unknown>`` line):
  python src/VeriStressGT/soundness/refverify.py --onnx M.onnx --vnnlib S.vnnlib --bug none
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":  # run by path: make this checkout's src/ win over any installed copy
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from VeriStressGT.soundness.bounds import BoundOpts, Interval, crown, ibp
from VeriStressGT.soundness.graph import DTYPE, Graph, UnsupportedOp, load_graph
from VeriStressGT.soundness.spec import Spec, parse_vnnlib

VERIFY_EPS = 1e-9        # reference: a sub-domain is proved only if its bound exceeds this


@dataclass(frozen=True)
class VerifierOpts:
    bounds: BoundOpts = BoundOpts()
    verify_threshold: float = VERIFY_EPS   # [BUG tol_unsat] -1e-3
    attack_accept: float = 0.0             # [BUG tol_sat] accept "counterexamples" with margin < this
    validate_cex: bool = True              # [BUG tol_sat] skip float64 re-check of the witness
    disjunct_last: bool = False            # [BUG] return the status of the last disjunct checked
    clip01: bool = False                   # [BUG] silently clip the input box to [0, 1]
    radius_scale: float = 1.0              # [BUG eps_half] 0.5
    bab: bool = True
    milp_max_unstable: int = 600           # exact MILP when the root has at most this many unstable ReLUs


def opts_for_bug(bug: str) -> VerifierOpts:
    base = VerifierOpts()
    table = {
        "none": base,
        "ibp_only": replace(base, bounds=BoundOpts(use_crown=False)),
        "ibp_sign": replace(base, bounds=BoundOpts(ibp_sign_bug=True)),
        "relu_no_intercept": replace(base, bounds=BoundOpts(relu_no_intercept_bug=True)),
        "conv_bias_dropped": replace(base, bounds=BoundOpts(drop_bias_ops=("Conv",))),
        "tol_unsat": replace(base, verify_threshold=-1e-3),
        "tol_sat": replace(base, attack_accept=1e-3, validate_cex=False),
        "disjunct_last": replace(base, disjunct_last=True),
        "input_clip01": replace(base, clip01=True),
        "eps_half": replace(base, radius_scale=0.5),
    }
    if bug not in table:
        raise ValueError(f"unknown bug {bug!r}; choose from {sorted(table)}")
    return table[bug]


# ── attack ────────────────────────────────────────────────────────────────────────────────

def _margins(graph: Graph, C: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (graph.forward(X) @ C.T).min(dim=1).values


def pgd_attack(graph: Graph, C: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, *,
               restarts: int = 16, steps: int = 150, seed: int = 0) -> Tuple[torch.Tensor, float]:
    """Minimise min_d C_d . f(x) over the box. Returns (best x [1, *shape], its margin)."""
    g = torch.Generator().manual_seed(seed)
    c, r = (lo + hi) / 2, (hi - lo) / 2
    shape = (restarts,) + tuple(lo.shape[1:])
    X = c + r * (2 * torch.rand(shape, generator=g, dtype=DTYPE) - 1)
    X[0] = c[0]
    X[1] = c[0] + r[0] * torch.sign(torch.randn(lo.shape[1:], generator=g, dtype=DTYPE))
    best_x, best_m = X[:1].clone(), float("inf")
    for t in range(steps):
        X = X.detach().requires_grad_(True)
        m = _margins(graph, C, X)
        md = m.detach()
        i = int(torch.argmin(md))
        if float(md[i]) < best_m:
            best_m, best_x = float(md[i]), X[i:i + 1].detach().clone()
        if best_m <= -1e-6:
            break
        (grad,) = torch.autograd.grad(m.sum(), X)
        step = 2.5 * r / steps * (1.0 + 3.0 * (1 - t / steps))
        X = torch.max(torch.min(X - step * torch.sign(grad), hi), lo)
    with torch.no_grad():
        m = _margins(graph, C, X.detach())
        i = int(torch.argmin(m))
        if float(m[i]) < best_m:
            best_m, best_x = float(m[i]), X[i:i + 1].detach().clone()
    return best_x, best_m


# ── bounds under ReLU splits ──────────────────────────────────────────────────────────────

def _bounds_with_splits(graph: Graph, C: torch.Tensor, c: torch.Tensor, r: torch.Tensor,
                        splits: Dict[str, torch.Tensor], opts: BoundOpts):
    """Lower bounds of C.f over the box restricted to the given ReLU phase splits.

    splits[z] in {-1, 0, +1} per neuron of ReLU input z (inactive / free / active). Fixing a phase
    clamps that neuron's pre-activation interval; the resulting bound is valid on the sub-domain.
    Returns (lb [B, K], infeasible [B], relu scores {z: [B, *shape]}, pre-bounds).
    """
    B = c.shape[0]
    refined: Dict[str, Interval] = {}
    infeasible = torch.zeros(B, dtype=torch.bool)

    def clamp(z: str, lo: torch.Tensor, hi: torch.Tensor) -> Interval:
        s = splits.get(z)
        if s is None:
            return lo, hi
        lo = torch.where(s > 0, lo.clamp(min=0), lo)
        hi = torch.where(s < 0, hi.clamp(max=0), hi)
        return lo, hi

    iv = ibp(graph, c, r, opts)
    for n in graph.nodes:
        if n.kind != "relu" or n.inputs[0] in refined:
            continue
        z = n.inputs[0]
        lo, hi = iv[z]
        shape = tuple(lo.shape[1:])
        m = int(np.prod(shape))
        if opts.use_crown and B * 2 * m <= opts.crown_rows:
            eye = torch.eye(m, dtype=DTYPE).reshape((m,) + shape)
            A0 = torch.cat([eye, -eye]).unsqueeze(0).expand((B, 2 * m) + shape).contiguous()
            lb, _ = crown(graph, z, A0, c, r, {**iv, **refined}, opts)
            lo = torch.maximum(lo, lb[:, :m].reshape((B,) + shape))
            hi = torch.minimum(hi, -lb[:, m:].reshape((B,) + shape))
        lo, hi = clamp(z, lo, hi)
        infeasible |= (lo > hi + 1e-12).reshape(B, -1).any(dim=1)
        refined[z] = (lo, hi)
        iv = ibp(graph, c, r, opts, refined)

    out_lo, out_hi = iv[graph.output_name]
    Cp, Cn = C.clamp(min=0), C.clamp(max=0)
    lb = out_lo.reshape(B, -1) @ Cp.T + out_hi.reshape(B, -1) @ Cn.T
    if opts.use_crown:
        out_node = graph.by_name()[graph.output_name]
        A_out = C.reshape((1, C.shape[0]) + out_node.shape).expand((B, C.shape[0]) + out_node.shape).contiguous()
        lb_c, _ = crown(graph, graph.output_name, A_out, c, r, {**iv, **refined}, opts)
        lb = torch.maximum(lb, lb_c)

    # Branching score: ambiguity of each unstable neuron (BaBSR-style width proxy).
    scores = {}
    for z, (lo, hi) in refined.items():
        unstable = (lo < 0) & (hi > 0)
        scores[z] = torch.where(unstable, torch.minimum(-lo, hi), torch.zeros_like(lo))
    return lb, infeasible, scores, iv


def bab(graph: Graph, C: torch.Tensor, c: torch.Tensor, r: torch.Tensor, opts: VerifierOpts,
        deadline: float, *, batch: int = 32, max_domains: int = 20000) -> Tuple[str, int]:
    """ReLU-split branch-and-bound. Returns ("unsat" | "unknown", domains visited)."""
    queue: List[Dict[str, torch.Tensor]] = [{}]
    visited = 0
    while queue:
        if time.time() > deadline or visited > max_domains:
            return "unknown", visited
        doms, queue = queue[:batch], queue[batch:]
        B = len(doms)
        keys = sorted({k for d in doms for k in d})
        splits = {}
        for k in keys:
            ref = next(d[k] for d in doms if k in d)
            splits[k] = torch.stack([d.get(k, torch.zeros_like(ref)) for d in doms])
        cb, rb = c.expand((B,) + c.shape[1:]), r.expand((B,) + r.shape[1:])
        lb, infeasible, scores, _ = _bounds_with_splits(graph, C, cb, rb, splits, opts.bounds)
        visited += B
        proved = infeasible | (lb.min(dim=1).values > opts.verify_threshold)
        if not opts.bab and not bool(proved.all()):
            return "unknown", visited
        for i in range(B):
            if bool(proved[i]):
                continue
            best_z, best_idx, best_s = None, None, 0.0
            for z, s in scores.items():
                si = s[i].reshape(-1)
                j = int(torch.argmax(si))
                if float(si[j]) > best_s:
                    best_z, best_idx, best_s = z, j, float(si[j])
            if best_z is None:  # no unstable neuron left: bound is exact on a linear piece
                return "unknown", visited
            for phase in (1.0, -1.0):
                child = {k: v.clone() for k, v in doms[i].items()}
                if best_z not in child:
                    child[best_z] = torch.zeros_like(scores[best_z][i])
                child[best_z].view(-1)[best_idx] = phase
                queue.append(child)
    return "unsat", visited


# ── top level ─────────────────────────────────────────────────────────────────────────────

def _accept_cex(graph: Graph, C: torch.Tensor, x: torch.Tensor, m: float, lo, hi, opts: VerifierOpts) -> bool:
    if not (m <= 0 or m < opts.attack_accept):
        return False
    if not opts.validate_cex:
        return True
    return float(_margins(graph, C, x)) <= 0 and bool(((x >= lo) & (x <= hi)).all())


def _solve(graph: Graph, C: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
           opts: VerifierOpts, deadline: float) -> Dict:
    x, m = pgd_attack(graph, C, lo, hi)
    if _accept_cex(graph, C, x, m, lo, hi, opts):
        return {"status": "sat", "stage": "attack", "witness_margin": m, "witness": x.reshape(-1).tolist()}

    c, r = (lo + hi) / 2, (hi - lo) / 2
    lb, _, scores, iv = _bounds_with_splits(graph, C, c, r, {}, opts.bounds)
    thr = opts.verify_threshold
    open_rows = [int(d) for d in torch.argsort(lb[0]) if float(lb[0, d]) <= thr]
    if not open_rows:
        return {"status": "unsat", "stage": "bounds", "attack_margin": m}

    n_unstable = sum(int((s > 0).sum()) for s in scores.values())
    if n_unstable <= opts.milp_max_unstable:
        from VeriStressGT.soundness.milp_exact import MilpModel
        model = MilpModel(graph, {k: (v[0][0], v[1][0]) for k, v in iv.items()},
                          lo[0].numpy(), hi[0].numpy(), drop_bias_ops=opts.bounds.drop_bias_ops)
        milp_thr = thr if thr < 0 else max(thr, 1e-6)   # HiGHS tolerances ~1e-7..1e-6
        undecided = False
        for k, d in enumerate(open_rows):
            left = deadline - time.time()
            if left <= 0:
                return {"status": "unknown", "stage": "milp-timeout", "unstable": n_unstable}
            st, obj, dual, x_in = model.minimize(C[d].numpy(), left / (len(open_rows) - k))
            if st == "infeasible" or (dual is not None and dual > milp_thr):
                continue
            if obj is not None and x_in is not None:
                xc = torch.tensor(x_in, dtype=DTYPE).reshape(lo.shape).clamp(lo, hi)
                mc = float(_margins(graph, C, xc))
                if _accept_cex(graph, C, xc, min(mc, obj), lo, hi, opts):
                    return {"status": "sat", "stage": "milp", "witness_margin": mc, "witness": xc.reshape(-1).tolist()}
            undecided = True
        if not undecided:
            return {"status": "unsat", "stage": "milp", "unstable": n_unstable}
        return {"status": "unknown", "stage": "milp", "unstable": n_unstable}

    status, visited = bab(graph, C, c, r, opts, deadline)
    return {"status": status, "stage": "bab", "domains": visited, "attack_margin": m}


def load_problem(onnx_path: str, vnnlib_path: str) -> Tuple[Graph, Spec]:
    graph = load_graph(onnx_path)
    n_out = graph.forward(torch.zeros((1,) + graph.input_shape, dtype=DTYPE)).shape[1]
    return graph, parse_vnnlib(vnnlib_path, n_out)


def verify(onnx_path: str, vnnlib_path: str, bug: str = "none", time_budget: float = 60.0) -> Dict:
    t0 = time.time()
    graph, spec = load_problem(onnx_path, vnnlib_path)
    return verify_box(graph, spec.C, spec.center, spec.radius, bug, time_budget - (time.time() - t0))


def verify_box(graph: Graph, C_np: np.ndarray, center_np: np.ndarray, radius_np: np.ndarray,
               bug: str = "none", time_budget: float = 60.0) -> Dict:
    t0 = time.time()
    opts = opts_for_bug(bug)
    center = torch.tensor(center_np, dtype=DTYPE).reshape((1,) + graph.input_shape)
    radius = torch.tensor(radius_np, dtype=DTYPE).reshape((1,) + graph.input_shape) * opts.radius_scale
    lo, hi = center - radius, center + radius
    if opts.clip01:
        lo, hi = lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)
        if bool((lo > hi).any()):
            return {"status": "unsat", "note": "empty box after clipping"}
    C = torch.tensor(C_np, dtype=DTYPE)
    deadline = t0 + time_budget

    if not opts.disjunct_last:
        out = _solve(graph, C, lo, hi, opts, deadline)
    else:
        # [BUG] mirrors the disjunctive-spec presolve bug in Appendix D: each disjunct is checked
        # separately and the status of the *last* one is returned instead of OR-aggregating.
        per = []
        for d in range(C.shape[0]):
            sub_deadline = time.time() + (deadline - time.time()) / (C.shape[0] - d)
            per.append(_solve(graph, C[d:d + 1], lo, hi, opts, sub_deadline)["status"])
        out = {"status": per[-1], "per_disjunct": per}
    out["wall_time_s"] = time.time() - t0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--vnnlib", required=True)
    ap.add_argument("--bug", default="none")
    ap.add_argument("--time_budget", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    try:
        out = verify(args.onnx, args.vnnlib, args.bug, args.time_budget)
    except UnsupportedOp as e:
        print(f"Error: unsupported ONNX construct: {e}", file=sys.stderr)
        return 2
    witness = out.pop("witness", None)
    print(json.dumps(out))
    if witness is not None:
        print("Counterexample: " + json.dumps(witness))
    print(f"Result: {out['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
