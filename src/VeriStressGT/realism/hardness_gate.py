"""HARDNESS_AND_TIGHTNESS_GATE.

For a candidate (model, x0, y, eps) that passed the native soundness check, compute:
  L_ana   : native rigorous lower bound on the min-margin over the box (from the constructor).
  U_wit   : best feasible witness upper bound (PGD + corners) — proves NON-robust if <= 0.
  mu_MILP : exact min-margin over the box via the exact-radius MILP (reused), when tractable.
and classify the instance (TIGHT_NATIVE / MILP_GROUND_TRUTH / EASY_REJECT / NON_ROBUST).

Conv nets are handled by converting each conv to its exact dense-equivalent LinearLayer (conv is a
linear map), then reusing solve_exact_radius(..., domain=(0,1)) — no new MILP engine.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from VeriStressGT.robust_constructions.mlp_relu.milp.exact_radius import (
    LinearLayer, solve_exact_radius, forward_layers,
)


# --------------------------------------------------------------------------- #
# torch model -> dense LinearLayer list (exact)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _linear_of_module(mod: nn.Module, in_shape: Tuple[int, ...]) -> LinearLayer:
    """Exact dense (W,b) of a linear module on flattened input (row-major), via basis probing."""
    d = int(np.prod(in_shape))
    base = mod(torch.zeros(1, *in_shape, dtype=torch.float64)).reshape(-1)
    cols = []
    for i in range(d):
        e = torch.zeros(d, dtype=torch.float64); e[i] = 1.0
        cols.append((mod(e.reshape(1, *in_shape)).reshape(-1) - base))
    W = torch.stack(cols, dim=1).cpu().numpy().astype(np.float64)   # (out, in)
    b = base.cpu().numpy().astype(np.float64)
    return LinearLayer(W=W, b=b)


@torch.no_grad()
def sequential_to_layers(modules: List[nn.Module], in_shape: Tuple[int, ...]) -> List:
    """Walk a Conv2d/ReLU/Flatten/Linear stack -> dense [Linear,'relu',...,Linear] (exact).

    Conv2d/Linear -> dense LinearLayer (bias folded); ReLU -> 'relu'; Flatten -> no-op (order kept).
    `in_shape` is (C,H,W) or (D,). Same-padding convs keep H,W. First/last entries must be Linear.
    """
    layers: List = []
    shape = tuple(in_shape)
    for m in modules:
        if isinstance(m, nn.Conv2d):
            layers.append(_linear_of_module(m, shape))
            shape = (m.out_channels, shape[1], shape[2])
        elif isinstance(m, nn.Linear):
            layers.append(_linear_of_module(m, shape))
            shape = (m.out_features,)
        elif isinstance(m, nn.ReLU):
            layers.append("relu")
        elif isinstance(m, nn.Flatten):
            shape = (int(np.prod(shape)),)
        else:
            raise ValueError(f"sequential_to_layers: unsupported {type(m).__name__}")
    return layers


@torch.no_grad()
def contractive_to_layers(model, img: int) -> List:
    """ContractiveCNN -> dense LinearLayer list."""
    mods = list(model.features) + [nn.Flatten(), model.head]
    return sequential_to_layers(mods, (1, img, img))


# --------------------------------------------------------------------------- #
# witness (feasible upper bound on min-margin)
# --------------------------------------------------------------------------- #
def witness_min_margin(model, x0: torch.Tensor, eps: float, y: int, domain=(0.0, 1.0),
                       n_steps: int = 100, n_restarts: int = 8) -> float:
    """Best (smallest) min-margin found by PGD + corners over the box (upper bound on mu*)."""
    model = model.double().eval()
    shape = x0.shape
    x0f = x0.double().reshape(-1)
    d = x0f.numel()
    lo = torch.clamp(x0f - eps, domain[0], domain[1])
    hi = torch.clamp(x0f + eps, domain[0], domain[1])

    def margin(xf):
        logits = model(xf.reshape(shape)).reshape(-1)
        other = logits.clone(); other[y] = float("inf")
        return logits[y] - other.min()  # min over k of (f_y - f_k)

    best = float("inf")
    # corners of a random subset of coords + center + random points + PGD
    cands = [x0f, lo, hi]
    for _ in range(n_restarts):
        x = (lo + torch.rand(d, dtype=torch.float64) * (hi - lo)).clone().requires_grad_(True)
        for _ in range(n_steps):
            m = margin(x)
            g, = torch.autograd.grad(m, x)
            with torch.no_grad():
                x = torch.clamp(x - (eps / 4) * g.sign(), lo, hi).requires_grad_(True)
        cands.append(x.detach())
    with torch.no_grad():
        for c in cands:
            best = min(best, float(margin(c)))
    return best


# --------------------------------------------------------------------------- #
# exact radius via reused MILP
# --------------------------------------------------------------------------- #
def exact_radius_milp(layers: List, x0: np.ndarray, Rmax: float, time_limit: float,
                      mip_gap: float, domain=(0.0, 1.0)) -> Dict:
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    return solve_exact_radius(layers, x0, Rmax=Rmax, time_limit_s=time_limit,
                              mip_gap=mip_gap, verbose=False, domain=domain)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def classify(L_ana: float, r_star: Optional[float], eps: float, U_wit: float, floor: float,
             near_frac: float = 0.88) -> Tuple[str, str]:
    """Classify an instance from the exact radius r* (MILP) and the native lower bound L_ana.

    r* is the EXACT l-inf radius to the nearest adversarial, so `eps < r*` is provably robust and
    `eps >= r*` provably non-robust (label_source = milp_exact). The witness U_wit is a diagnostic
    (a valid adversarial => U_wit <= 0). Returns (hardness_class, label_source).
    """
    if r_star is None or not np.isfinite(r_star):
        return "UNKNOWN_NUMERICAL", "none"
    if eps >= r_star:                      # exact non-robust
        return "NON_ROBUST", "milp_exact"
    near_boundary = eps >= near_frac * r_star
    if L_ana > floor and near_boundary:    # native cert also tight near the boundary (rare for Lipschitz)
        return "TIGHT_NATIVE", "native_certificate"
    if near_boundary:                      # robust + near-boundary, native cert loose -> MILP labels
        return "MILP_GROUND_TRUTH", "milp_exact"
    return "EASY_REJECT", "milp_exact"     # robust but far from boundary
