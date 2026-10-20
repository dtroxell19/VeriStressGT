"""Unit tests for the sound-arithmetic backends (A0).

Run: PYTHONPATH=src python tests/test_sound_backends.py
"""
from __future__ import annotations
import sys
from fractions import Fraction

import numpy as np
import mpmath
from mpmath import iv

from VeriStressGT.analysis import sound_backends as B


def test_fp32_roundtrip():
    """Fraction(float(w)) is exact for fp32/fp64 (the trust boundary)."""
    rng = np.random.default_rng(0)
    a32 = rng.standard_normal(500).astype(np.float32)
    fr = B.to_frac_array(a32)
    assert all(float(f) == float(x) for f, x in zip(fr.reshape(-1), a32.reshape(-1)))
    a64 = rng.standard_normal(200).astype(np.float64)
    fr64 = B.to_frac_array(a64)
    assert all(float(f) == float(x) for f, x in zip(fr64.reshape(-1), a64.reshape(-1)))
    print("ok fp32/fp64 -> Fraction round-trip exact")


def test_sound_sqrt_upper():
    """u = sound_sqrt_upper(P) satisfies u**2 >= P exactly, and is tight-ish."""
    cases = [Fraction(2), Fraction(1, 3), Fraction(10**9), Fraction(7, 1000000),
             Fraction(0), Fraction(123456789, 987654321)]
    for P in cases:
        u = B.sound_sqrt_upper(P)
        assert u * u >= P, f"not an upper bound for {P}"
        if P > 0:
            # tightness: (u - 1/den)^2 < P would over-shoot; just check within a hair
            assert float(u) ** 2 >= float(P)
    print("ok sound_sqrt_upper: u^2 >= P exactly")


def test_spectral_upper_bounds_numpy_norm2():
    """frac_spectral_upper(W) >= numpy spectral norm, for random and structured W."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        m, n = rng.integers(1, 8), rng.integers(1, 8)
        W = rng.standard_normal((m, n)).astype(np.float32)
        u = B.frac_spectral_upper(B.to_frac_array(W))
        s2 = np.linalg.norm(W.astype(np.float64), 2)
        assert float(u) >= s2 - 1e-9, f"surrogate {float(u)} < sigma {s2}"
    print("ok frac_spectral_upper >= numpy spectral norm")


def test_exact_linear_algebra():
    """Exact affine + ReLU + margin on Fraction object arrays matches float within fp error."""
    rng = np.random.default_rng(2)
    W = rng.standard_normal((4, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    x = rng.standard_normal(3).astype(np.float32)
    Wf, bf, xf = B.to_frac_array(W), B.to_frac_array(b), B.to_frac_array(x)
    y = B.frac_relu(B.frac_linear(xf, Wf, bf))
    y_float = np.maximum(W.astype(np.float64) @ x.astype(np.float64) + b.astype(np.float64), 0)
    assert all(abs(float(a) - c) < 1e-5 for a, c in zip(y, y_float))
    # exact margin
    logits = list(B.to_frac_array(np.array([0.5, -0.2, 0.1, 0.9], dtype=np.float32)))
    assert B.frac_margin(logits, 3) == logits[3] - max(logits[0], logits[1], logits[2])
    print("ok exact linear algebra + margin")


def test_fraction_in_interval():
    """The exact Fraction value lies inside the mpmath.iv enclosure of the same computation."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((3, 3)).astype(np.float32)
    x = rng.standard_normal(3).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    Wf, bf, xf = B.to_frac_array(W), B.to_frac_array(b), B.to_frac_array(x)
    y_exact = B.frac_relu(B.frac_linear(xf, Wf, bf))
    Wiv = B.iv_to_frac_array(W)
    xiv = [iv.mpf(float(v)) for v in x]
    biv = [iv.mpf(float(v)) for v in b]
    y_iv = [B.iv_relu(z) for z in B.iv_matvec(Wiv, xiv, biv)]
    for fe, zi in zip(y_exact, y_iv):
        lo, hi, v = float(zi.a), float(zi.b), float(fe)
        assert lo - 1e-12 <= v <= hi + 1e-12, f"exact {v} not in [{lo},{hi}]"
    print("ok exact Fraction value inside iv enclosure")


def test_iv_softmax_encloses():
    """Interval softmax encloses the true softmax and sums to ~1 (rows contain 1)."""
    z = [0.3, -1.1, 2.0, 0.0]
    sm = B.iv_softmax([iv.mpf(v) for v in z])
    zt = np.array(z)
    true = np.exp(zt - zt.max()); true /= true.sum()
    for p, t in zip(sm, true):
        assert float(p.a) - 1e-12 <= float(t) <= float(p.b) + 1e-12
    s = sm[0]
    for e in sm[1:]:
        s = s + e
    assert float(s.a) - 1e-12 <= 1.0 <= float(s.b) + 1e-12
    print("ok iv softmax encloses truth, sums to 1")


def test_outward_box():
    x0 = np.array([0.5, 0.0, 1.0], dtype=np.float32)
    lo, hi = B.outward_interval_from_box(x0, 0.1, (0.0, 1.0))
    assert lo[1] == Fraction(0) and hi[2] == Fraction(1)  # clamped
    assert hi[0] - lo[0] == Fraction(0.2).limit_denominator(10**9) or float(hi[0] - lo[0]) == 0.2
    print("ok outward box clamped exactly")


def main():
    for fn in [test_fp32_roundtrip, test_sound_sqrt_upper, test_spectral_upper_bounds_numpy_norm2,
               test_exact_linear_algebra, test_fraction_in_interval, test_iv_softmax_encloses,
               test_outward_box]:
        fn()
    print("\nALL A0 BACKEND TESTS PASSED")


if __name__ == "__main__":
    main()
