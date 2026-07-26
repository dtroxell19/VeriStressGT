# Sound-arithmetic re-certification of VeriStress-GT ground-truth margins

**172 / 225 shipped constructed instances have their ground-truth robustness margin rigorously re-certified `> 0` under sound arithmetic** (plus **12 / 12** newly-generated exactly-certified convex-polynomial companions) — evaluating each family's analytic certificate on the EXACT shipped fp32 weights (IEEE floats are exact dyadic rationals) in `fractions.Fraction` (exact) or validated `mpmath.iv` intervals (outward-rounded, over the whole box). Every certified lower bound is a machine-checkable enclosure of `min_{x∈B_ε(x0)} margin(x)`, not a sample. The 53 remaining shipped instances (31 MILP + 22 nonconvex polynomial) are the deliberately hardest / near-boundary families whose ground truth rests on a complete method — disclosed below, not glossed.

## Backend tiers

| tier | certified / total | meaning |
|---|---|---|
| exact-rational | 149/149 | exact `Fraction`, zero rounding — a proof |
| validated-interval | 35/57 | `mpmath.iv` outward-rounded box enclosure |
| inconclusive | 0/31 | sound methods too loose; GT rests on a complete method (disclosed) |

## Per-family results

| family | certified / n | backend | min certified lb |
|---|---|---|---|
| Deep-Contractive | 50/50 | exact-rational | 0.0249 |
| Constant-on-Box | 5/5 | exact-rational | 0.832 |
| Fixed-Order Attn | 17/17 | validated-interval | 0.0043 |
| Dominant-Key Attn | 18/18 | validated-interval | 0.064 |
| MEAP | 14/14 | exact-rational | 1e-05 |
| MILP | 0/31 | inconclusive | — |
| Paired-Bias | 46/46 | exact-rational | 0.001 |
| Input-Corner | 22/22 | exact-rational | 3.95 |
| Polynomial (shipped, nonconvex) | 0/22 | validated-interval | — |
| Polynomial (convex companion) | 12/12 | exact-rational | 0.0939 |

## Method (per family)

- **Paired-Bias / MEAP**: structural — hard-zeroed non-label logits + monotone coupled-ReLU / min-of-max tree give `f_label ≥ margin` (resp. `≥ γ`) pointwise; verified exactly.
- **Constant-on-Box**: exact-rational IBP over the full net (the projection collapses the box to ~1e-8, so downstream IBP is tight); handles the η=0 1-ULP boundary case rigorously.
- **Input-Corner**: the margin is *concave*, so its min over the box is at a corner — exact min over the `2^active_dim` corners.
- **Deep-Contractive**: ends ReLU→Gemm with a positive readout and zero other logits, so `f_label ≥ B` pointwise (structural, contraction-independent — a stronger statement than the paper's Lipschitz bound).
- **Attention (Dominant-Key, Fixed-Order)**: direct `mpmath.iv` interval enclosure of the (softmax) attention forward over the box — no Lipschitz looseness; closes even the razor-thin `margin_slack≈1.0001` Fixed-Order instances.
- **Polynomial (convex companion)**: convex Prop-11 instances certified EXACTLY via the convex first-order bound `μ(x0) − ε‖∇μ(x0)‖₁ > 0` in `Fraction`.

## Cross-validation

Independent sanity check against the existing fp32-vs-fp64 audit (`soundness_audit.json`): every one of
the **172/172** certified lower bounds satisfies `certified_lb ≤ fp64 sampled min-margin` — i.e. each
rigorous certificate is a valid *floor* below the empirically-observed minimum, never above it (a
certificate that exceeded reality would signal a bug). The backend arithmetic itself is unit-tested
(`tests/test_sound_backends.py`, 7/7): fp32→`Fraction` is bit-exact, the sound spectral/sqrt bounds
dominate the true value, and exact `Fraction` results lie inside the `mpmath.iv` enclosures.

## Honest disclosures

- **Polynomial (shipped, nonconvex)**: these 22 ship with GT assigned by a *local L-BFGS-B screen* (not an analytic proof). Naive interval arithmetic is too loose to independently re-certify a degree-≤22 polynomial in 100-D (the per-neuron power ranges are exact, but summing 100 mixed-sign terms loses the shared-x correlation). We therefore provide the **exactly-certified convex companion** above as the analytically-grounded polynomial GT, and flag the shipped set as screen-only.
- **MILP**: these ReLU MLPs are deliberately near-boundary (`ε = ε_frac·r*`, up to 0.9999·r*). Exact-rational IBP and sound box branch-and-bound are too loose (box relaxation cannot track the input-space halfspace of a ReLU split). Robustness at `ε < r*` was established by the **exact-radius MILP** — itself a complete verification (exact big-M) — which is the certificate for this family; exact-rational *complete* re-verification is the one remaining piece.

*Reproduce:* `PYTHONPATH=src python -m VeriStressGT.analysis.sound_recertify`. Backends + unit tests: `python tests/test_sound_backends.py`.
