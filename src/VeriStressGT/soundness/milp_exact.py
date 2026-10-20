"""Exact big-M MILP for piecewise-linear DAGs, solved with scipy's HiGHS (no license needed).

For each disjunct d we solve  min_x c_d . f(x)  over the input box. Pre-activation bounds
(from ``bounds``) define the big-M constants, so wrong bounds make the MILP silently exclude real
network behaviour -- which is exactly how the planted bound bugs propagate into false UNSAT claims.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import Bounds, LinearConstraint, milp

from .bounds import Interval
from .graph import DTYPE, Graph, Node


def _dense_linear(n: Node, in_shape) -> sp.csr_matrix:
    """Matrix M (out x in) of the node's linear part, extracted by pushing an identity basis."""
    m_in = int(np.prod(in_shape))
    eye = torch.eye(m_in, dtype=DTYPE).reshape((m_in,) + tuple(in_shape))
    with torch.no_grad():
        cols = n.lin(eye).reshape(m_in, -1)          # row i = L(e_i)
    M = cols.T.numpy()
    M[np.abs(M) < 1e-300] = 0.0
    return sp.csr_matrix(M)


class MilpModel:
    """Variables: one continuous var per element of every node + one binary per unstable ReLU."""

    def __init__(self, graph: Graph, iv: Dict[str, Interval], lo_in: np.ndarray, hi_in: np.ndarray,
                 drop_bias_ops=()):
        self.graph = graph
        offs: Dict[str, int] = {}
        lb: List[np.ndarray] = []
        ub: List[np.ndarray] = []
        nvar = 0
        for n in graph.nodes:
            size = int(np.prod(n.shape))
            offs[n.name] = nvar
            nvar += size
            if n.kind == "input":
                lb.append(lo_in.reshape(-1)); ub.append(hi_in.reshape(-1))
            else:
                l, u = iv[n.name]
                lb.append(l.reshape(-1).numpy()); ub.append(u.reshape(-1).numpy())
        self.offs, self.n_cont = offs, nvar

        rows = []
        binaries = []  # (y var, z var, l, u) per ReLU element
        nodes = graph.by_name()
        for n in graph.nodes:
            o, size = offs[n.name], int(np.prod(n.shape))
            if n.kind == "linear":
                src = nodes[n.inputs[0]]
                M = _dense_linear(n, src.shape)
                b = np.zeros(size) if n.op_type in drop_bias_ops else n.bias.reshape(-1).numpy()
                rows.append(("lin", o, size, offs[src.name], M, b))           # y - M x = b
            elif n.kind == "add":
                rows.append(("add", o, size, offs[n.inputs[0]], offs[n.inputs[1]], n.sign))
            elif n.kind == "relu":
                z = nodes[n.inputs[0]]
                lz, uz = (t.reshape(-1).numpy() for t in iv[z.name])
                for i in range(size):
                    binaries.append((o + i, offs[z.name] + i, float(lz[i]), float(uz[i])))
        self.binaries = [bnd for bnd in binaries if bnd[2] < 0 < bnd[3]]
        n_bin = len(self.binaries)
        self.nvar = nvar + n_bin

        blocks = []
        for r in rows:
            if r[0] == "lin":
                _, o, size, si, M, b = r
                I = sp.identity(size, format="csr")
                blk = sp.lil_matrix((size, self.nvar))
                blk[:, o:o + size] = I
                blk[:, si:si + M.shape[1]] = -M
                blocks.append((blk.tocsr(), b, b))
            else:
                _, o, size, a1, a2, sign = r
                blk = sp.lil_matrix((size, self.nvar))
                blk[:, o:o + size] = sp.identity(size)
                blk[:, a1:a1 + size] = blk[:, a1:a1 + size] - sp.identity(size)
                blk[:, a2:a2 + size] = blk[:, a2:a2 + size] - sign * sp.identity(size)
                blocks.append((blk.tocsr(), np.zeros(size), np.zeros(size)))

        # ReLU: stable ones are equalities / zero; unstable ones use big-M with a binary.
        unstable_y = {bnd[0] for bnd in self.binaries}
        r_lo, r_hi, r_list = [], [], []
        for n in graph.nodes:
            if n.kind != "relu":
                continue
            z = nodes[n.inputs[0]]
            lz, uz = (t.reshape(-1).numpy() for t in iv[z.name])
            for i in range(int(np.prod(n.shape))):
                y, zi = offs[n.name] + i, offs[z.name] + i
                if y in unstable_y:
                    continue
                if lz[i] >= 0:      # active: y = z
                    r_list.append({y: 1.0, zi: -1.0}); r_lo.append(0.0); r_hi.append(0.0)
                else:               # inactive: y = 0
                    r_list.append({y: 1.0}); r_lo.append(0.0); r_hi.append(0.0)
        for k, (y, zi, l, u) in enumerate(self.binaries):
            a = nvar + k
            r_list.append({y: 1.0, zi: -1.0}); r_lo.append(0.0); r_hi.append(np.inf)       # y >= z
            r_list.append({y: 1.0, a: -u}); r_lo.append(-np.inf); r_hi.append(0.0)         # y <= u a
            r_list.append({y: 1.0, zi: -1.0, a: -l}); r_lo.append(-np.inf); r_hi.append(-l)  # y <= z - l(1-a)
        if r_list:
            R = sp.lil_matrix((len(r_list), self.nvar))
            for i, coefs in enumerate(r_list):
                for j, v in coefs.items():
                    R[i, j] = v
            blocks.append((R.tocsr(), np.array(r_lo), np.array(r_hi)))

        self.A = sp.vstack([b[0] for b in blocks]).tocsr()
        self.c_lo = np.concatenate([b[1] for b in blocks])
        self.c_hi = np.concatenate([b[2] for b in blocks])
        self.lb = np.concatenate(lb + [np.zeros(n_bin)])
        self.ub = np.concatenate(ub + [np.ones(n_bin)])
        self.integrality = np.concatenate([np.zeros(nvar), np.ones(n_bin)])

    def minimize(self, c_out: np.ndarray, time_limit: float) -> Tuple[str, Optional[float], Optional[float], Optional[np.ndarray]]:
        """min c_out . output. Returns (status, objective, dual_bound, x_input)."""
        cost = np.zeros(self.nvar)
        o = self.offs[self.graph.output_name]
        cost[o:o + c_out.size] = c_out
        res = milp(cost, constraints=LinearConstraint(self.A, self.c_lo, self.c_hi),
                   bounds=Bounds(self.lb, self.ub), integrality=self.integrality,
                   options={"time_limit": max(0.5, float(time_limit)), "disp": False})
        dual = getattr(res, "mip_dual_bound", None)
        x_in = None
        if res.x is not None:
            oi = self.offs[self.graph.input_name]
            x_in = res.x[oi:oi + int(np.prod(self.graph.input_shape))]
        if res.status == 0:
            return "optimal", float(res.fun), float(dual) if dual is not None else float(res.fun), x_in
        if res.status == 2:
            return "infeasible", None, None, None
        return "limit", float(res.fun) if res.x is not None else None, (float(dual) if dual is not None else None), x_in
