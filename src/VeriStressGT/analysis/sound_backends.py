"""Arithmetic backends for the sound-arithmetic re-certification of VeriStress-GT
ground-truth margins (rebuttal: AC's "most dangerous" soundness concern).

Two rigorous backends, each of which can only *under*-claim, never over-claim, the
certified margin lower bound:

  * EXACT RATIONAL (`fractions.Fraction`): IEEE fp32/fp64 weights are exact dyadic
    rationals, so `Fraction(float(w))` is exact. Every +,-,*,integer-power and
    comparison on Fractions is exact -> a verified inequality is a *proof*, with
    zero rounding error. numpy object arrays of Fraction support exact `@`,
    `np.maximum`, `abs`, `sum`.

  * VALIDATED INTERVAL (`mpmath.iv`): outward-rounded interval arithmetic with a
    rigorous `iv.exp`. An interval enclosure of the margin over the whole box whose
    lower endpoint is > 0 is a rigorous certificate over the continuum (no sampling).

The one place a rational cannot represent the exact value is a spectral (l2) norm;
there we return a SOUND rational UPPER bound `sigma(W) <= sqrt(||W||_1 * ||W||_inf)`
via an up-rounded exact sqrt. Over-estimating a Lipschitz constant only shrinks the
certified margin -> still sound.
"""
from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import List, Sequence, Tuple

import numpy as np
import mpmath
from mpmath import iv

FR = Fraction

# Default working precision for the interval backend (bits). Raised high so the
# enclosure width, not precision, is the binding constraint even at margin_slack~1.0001.
IV_PREC_BITS = 200
iv.prec = IV_PREC_BITS


# --------------------------------------------------------------------------- #
# Exact-rational backend
# --------------------------------------------------------------------------- #
def to_frac_array(a) -> np.ndarray:
    """Exact object-array of Fraction from an fp32/fp64 array (the trust boundary).

    `Fraction(float(x))` is exact because every IEEE float is a dyadic rational.
    """
    a = np.asarray(a)
    out = np.empty(a.size, dtype=object)
    out[:] = [Fraction(float(x)) for x in a.reshape(-1)]
    return out.reshape(a.shape)


def frac_relu(x: np.ndarray) -> np.ndarray:
    """Elementwise exact ReLU on an object array of Fraction."""
    return np.maximum(x, Fraction(0))


def frac_linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact affine `W @ x + b` on object arrays of Fraction. W:(out,in), x:(in,)."""
    return W @ x + b


def frac_abs(W: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda z: abs(z))(W)


def frac_max_abs_row_sum(W: np.ndarray) -> Fraction:
    """Exact ||W||_inf = max_i sum_j |W_ij| (l_inf->l_inf induced operator norm)."""
    A = frac_abs(W)
    return max(A.sum(axis=1))


def frac_max_abs_col_sum(W: np.ndarray) -> Fraction:
    """Exact ||W||_1 = max_j sum_i |W_ij| (l_1->l_1 induced operator norm)."""
    A = frac_abs(W)
    return max(A.sum(axis=0))


def sound_sqrt_upper(P) -> Fraction:
    """Smallest rational `u` with `u**2 >= P` (a sound rational upper bound on sqrt(P)).

    For P = p/q (p,q>0): sqrt(P) = sqrt(p*q)/q; s = ceil(sqrt(p*q)) via integer isqrt;
    u = s/q satisfies u**2 = s**2/q**2 >= p*q/q**2 = P. Exact and sound.
    """
    P = Fraction(P)
    if P <= 0:
        return Fraction(0)
    p, q = P.numerator, P.denominator
    s = isqrt(p * q)
    if s * s < p * q:
        s += 1
    return Fraction(s, q)


def frac_spectral_upper(W: np.ndarray) -> Fraction:
    """Sound rational upper bound on the l2 spectral norm: sigma(W) <= sqrt(||W||_1*||W||_inf)."""
    P = frac_max_abs_col_sum(W) * frac_max_abs_row_sum(W)
    return sound_sqrt_upper(P)


def frac_conv_induced_norms(Wc: np.ndarray):
    """Exact (l1_induced, linf_induced) of a Conv2d weight (C_out,C_in,KH,KW).

    linf = max_{c_out} sum_{c_in,kh,kw} |W|   (max abs 'row' sum, interior pixel)
    l1   = max_{c_in}  sum_{c_out,kh,kw} |W|   (max abs 'col' sum)
    Both are sound upper bounds on the conv operator's induced l_inf / l_1 norms
    (boundary pixels touch fewer weights).
    """
    A = frac_abs(Wc)
    co, ci = Wc.shape[0], Wc.shape[1]
    linf = max(sum(A[o].reshape(-1)) for o in range(co))
    l1 = max(sum(A[:, i].reshape(-1)) for i in range(ci))
    return l1, linf


def frac_conv_spectral_upper(Wc: np.ndarray) -> Fraction:
    """Sound rational upper bound on a Conv2d's l2 operator norm: <= sqrt(l1*linf)."""
    l1, linf = frac_conv_induced_norms(Wc)
    return sound_sqrt_upper(l1 * linf)


def frac_margin(logits: Sequence[Fraction], label: int) -> Fraction:
    """Exact min-margin f_label - max_{k != label} f_k."""
    others = [logits[k] for k in range(len(logits)) if k != label]
    return logits[label] - max(others)


# --- exact-rational interval bound propagation (sound over-approximation) ------
def frac_interval_linear(lo, hi, W, b):
    """Exact interval image of [lo,hi] under x -> W x + b (W:(out,in) Fraction)."""
    Wp = np.where(W > Fraction(0), W, Fraction(0))
    Wn = np.where(W < Fraction(0), W, Fraction(0))
    return Wp @ lo + Wn @ hi + b, Wp @ hi + Wn @ lo + b


def frac_interval_relu(lo, hi):
    return frac_relu(lo), frac_relu(hi)


def frac_ibp(layers, lo, hi):
    """Propagate an exact-rational box through [(W,b) | 'relu', ...] layers."""
    for L in layers:
        if L == "relu":
            lo, hi = frac_interval_relu(lo, hi)
        else:
            W, b = L
            lo, hi = frac_interval_linear(lo, hi, W, b)
    return lo, hi


def frac_ibp_margin_lb(layers, lo, hi, label) -> Fraction:
    """Sound lower bound on min-margin over the box via exact-rational IBP.

    lo[label] - max_{k!=label} hi[k]  is a valid lower bound on
    min_x (f_label(x) - max_{k!=label} f_k(x)) over the box (IBP is a sound
    over-approximation, so this can only under-claim the true margin).
    """
    olo, ohi = frac_ibp(layers, lo, hi)
    others = [ohi[k] for k in range(len(ohi)) if k != label]
    return olo[label] - max(others)


# --------------------------------------------------------------------------- #
# Validated-interval backend (mpmath.iv)
# --------------------------------------------------------------------------- #
def iv_point(x) -> "mpmath.iv.mpf":
    """Point interval [x,x] from an exact float (dyadic -> representable exactly)."""
    return iv.mpf(float(x))


def iv_from_frac(fr: Fraction) -> "mpmath.iv.mpf":
    """Tight interval enclosing an exact rational (outward-rounded division)."""
    return iv.mpf(int(fr.numerator)) / iv.mpf(int(fr.denominator))


def iv_interval(lo, hi) -> "mpmath.iv.mpf":
    """Interval [lo, hi] from float/rational endpoints (enclosing, outward)."""
    a = iv_from_frac(lo) if isinstance(lo, Fraction) else iv.mpf(float(lo))
    b = iv_from_frac(hi) if isinstance(hi, Fraction) else iv.mpf(float(hi))
    return iv.mpf([a.a, b.b])


def iv_relu(z: "mpmath.iv.mpf") -> "mpmath.iv.mpf":
    """ReLU of an interval (monotone): [max(a,0), max(b,0)]."""
    a = z.a if z.a > 0 else iv.mpf(0).a
    b = z.b if z.b > 0 else iv.mpf(0).b
    return iv.mpf([a, b])


def iv_matvec(W: np.ndarray, x: List["mpmath.iv.mpf"], b=None) -> List["mpmath.iv.mpf"]:
    """Interval affine W@x(+b). W is an object array of interval mpf (or floats->points)."""
    out = []
    n_out = W.shape[0]
    for i in range(n_out):
        acc = iv.mpf(0)
        for j in range(W.shape[1]):
            wij = W[i, j] if hasattr(W[i, j], "a") else iv.mpf(float(W[i, j]))
            acc = acc + wij * x[j]
        if b is not None:
            bi = b[i] if hasattr(b[i], "a") else iv.mpf(float(b[i]))
            acc = acc + bi
        out.append(acc)
    return out


def iv_softmax(scores: List["mpmath.iv.mpf"]) -> List["mpmath.iv.mpf"]:
    """Rigorous interval softmax (shift-invariant max-subtraction for stability)."""
    hi = scores[0].b
    for s in scores[1:]:
        if s.b > hi:
            hi = s.b
    shift = iv.mpf(hi)
    exps = [iv.exp(s - shift) for s in scores]
    denom = exps[0]
    for e in exps[1:]:
        denom = denom + e
    return [e / denom for e in exps]


def iv_lower(z: "mpmath.iv.mpf") -> float:
    """Lower endpoint of an interval as a float (for reporting / positivity check)."""
    return float(z.a)


def iv_is_positive(z: "mpmath.iv.mpf") -> bool:
    """True iff the whole interval is > 0 (rigorous)."""
    return bool(z.a > 0)


def iv_to_frac_array(W: np.ndarray) -> np.ndarray:
    """Object array of point-interval mpf from a float weight array."""
    out = np.empty(W.size, dtype=object)
    out[:] = [iv.mpf(float(x)) for x in W.reshape(-1)]
    return out.reshape(W.shape)


# --------------------------------------------------------------------------- #
# Box helpers
# --------------------------------------------------------------------------- #
def outward_interval_from_box(x0, eps, domain=(0.0, 1.0)):
    """Exact rational box [max(x0-eps,lo), min(x0+eps,hi)] per coordinate."""
    x0f = to_frac_array(np.asarray(x0).reshape(-1))
    e = Fraction(float(eps))
    lo_d, hi_d = Fraction(float(domain[0])), Fraction(float(domain[1]))
    lo = np.array([max(v - e, lo_d) for v in x0f], dtype=object)
    hi = np.array([min(v + e, hi_d) for v in x0f], dtype=object)
    return lo, hi
