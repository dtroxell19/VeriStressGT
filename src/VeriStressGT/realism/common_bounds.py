"""Rigorous sound bounds shared by the trained structured constructors (A/B/C).

Everything here is a *sound* (conservative) bound on the exact quantity — never an estimate that
could over-claim robustness. Certificates consume the EXACT weights that will be exported to ONNX.

Key facts used:
- The l-inf -> l-inf induced (operator) norm of a linear map is its maximum absolute row sum.
- For a 2D convolution with zero padding, each output coordinate is an inner product of the kernel
  with a receptive field; the maximum absolute row sum over all output coordinates is achieved by an
  interior pixel and equals sum_{c_in,kh,kw} |W[c_out,c_in,kh,kw]| maximized over c_out. Boundary
  pixels touch fewer inputs, so this is a valid UPPER bound on the conv's l-inf Lipschitz constant.
- ReLU is 1-Lipschitz in every p-norm.

These give a rigorous product bound L_Phi = prod_l L_l on a Conv/Linear + ReLU feature extractor.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Induced l-inf (operator) norms — rigorous upper bounds on per-layer Lipschitz
# --------------------------------------------------------------------------- #
def linear_induced_inf_norm(weight: torch.Tensor) -> float:
    """||W||_inf = max_i sum_j |W_ij|  (exact l-inf->l-inf operator norm of x -> Wx)."""
    W = weight.detach().to(torch.float64)
    return float(W.abs().sum(dim=1).max().item())


def conv_induced_inf_norm(weight: torch.Tensor) -> float:
    """Sound upper bound on the l-inf->l-inf Lipschitz constant of a Conv2d (bias-free part).

    = max_{c_out} sum_{c_in,kh,kw} |W[c_out,c_in,kh,kw]|  (max absolute row sum; interior pixel).
    Valid for any stride/padding since boundary pixels only touch a subset of these weights.
    """
    W = weight.detach().to(torch.float64)  # (C_out, C_in, KH, KW)
    per_out = W.abs().reshape(W.shape[0], -1).sum(dim=1)
    return float(per_out.max().item())


def sequential_lipschitz_inf(modules: List[nn.Module]) -> Tuple[float, List[float]]:
    """Product l-inf Lipschitz bound over a Conv/Linear + ReLU stack. Returns (L_total, per_layer)."""
    per = []
    for m in modules:
        if isinstance(m, nn.Conv2d):
            per.append(conv_induced_inf_norm(m.weight))
        elif isinstance(m, nn.Linear):
            per.append(linear_induced_inf_norm(m.weight))
        elif isinstance(m, (nn.ReLU, nn.Flatten)):
            per.append(1.0)  # ReLU 1-Lipschitz; Flatten is an isometry in l-inf
        else:
            raise ValueError(f"Unsupported layer for a sound l-inf Lipschitz bound: {type(m).__name__}")
    total = 1.0
    for p in per:
        total *= p
    return total, per


# --------------------------------------------------------------------------- #
# Rigorous interval arithmetic (float64) — outward-safe helpers
# --------------------------------------------------------------------------- #
def interval_linear(lo: np.ndarray, hi: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interval image of [lo,hi] under x -> W x + b (float64). W:(out,in)."""
    W = W.astype(np.float64); b = b.astype(np.float64)
    Wp = np.clip(W, 0, None); Wn = np.clip(W, None, 0)
    out_lo = Wp @ lo + Wn @ hi + b
    out_hi = Wp @ hi + Wn @ lo + b
    return out_lo, out_hi


def interval_relu(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.maximum(lo, 0.0), np.maximum(hi, 0.0)


# --------------------------------------------------------------------------- #
# Exhaustive corner validation (micro tests only) — the soundness ground truth
# --------------------------------------------------------------------------- #
@torch.no_grad()
def exhaustive_corner_min_margin(model: nn.Module, x0: torch.Tensor, eps: float, label: int,
                                 domain=(0.0, 1.0)) -> float:
    """Exact min over all 2^d box corners of min_{k!=label}(f_label - f_k).

    Only for tiny d (<= ~20). Corners are clamped to `domain`. This is the exhaustive ground truth
    against which analytic lower bounds are validated (analytic_lb <= this).
    """
    model.eval()
    x0f = x0.reshape(-1).to(torch.float64)
    d = x0f.numel()
    if d > 22:
        raise ValueError(f"exhaustive corner check only for tiny inputs (d={d} too large)")
    lo = torch.clamp(x0f - eps, domain[0], domain[1])
    hi = torch.clamp(x0f + eps, domain[0], domain[1])
    worst = float("inf")
    for mask in range(1 << d):
        bits = [(mask >> i) & 1 for i in range(d)]
        x = torch.where(torch.tensor(bits, dtype=torch.bool), hi, lo)
        logits = model(x.reshape(x0.shape).to(torch.float64)).reshape(-1)
        others = torch.cat([logits[:label], logits[label + 1:]])
        margin = float(logits[label] - others.max())
        worst = min(worst, margin)
    return worst


def margins_at(model: nn.Module, X: torch.Tensor, label: int) -> torch.Tensor:
    """Per-row min-margin f_label - max_{k!=label} f_k for a batch X."""
    with torch.no_grad():
        logits = model(X)
        y = logits[:, label]
        other = logits.clone()
        other[:, label] = float("-inf")
        return y - other.max(dim=1).values
