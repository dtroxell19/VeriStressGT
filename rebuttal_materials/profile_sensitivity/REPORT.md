# Difficulty-Profile sensitivity study — results

Answers the AC's "sensitivity study of profile components (sample count, sampling distribution,
η/τ/grid width) with recommended defaults, plus an inter-component correlation matrix; this answers
three reviewers at once" (JJNS-Q2, gVrS-Q2, rxK4). Each knob is varied one at a time around a
production centre-point on a stratified 32-instance subset (all architectures, families, difficulty
terciles); the correlation matrix and the unified Difficulty Index use the full frozen 345-instance
table. Figures in `plots/`.

## Experiments performed
We ran six one-at-a-time sensitivity sweeps plus two whole-table dependence analyses. **(1) Sample
count:** each component was recomputed at N ∈ {50, 100, 200, 400, 800, 1600} and its normalized drift
from the N=1600 estimate tracked to locate a convergence plateau. **(2) Seed stability:** every
component was recomputed over 16 independent random seeds at the recommended N and summarized by its
coefficient of variation, isolating sampling noise from true signal. **(3) Sampling distribution:** the
box-point sampler was swept across `uniform_only`, `boundary_only`, the production `current_mixture`, and
`pgd_heavy`, measuring how much each strategy moves the profile. **(4) η (numerical-stability constant):**
the denominator floor in G_IBP and d_eff was swept over 1e-12 → 1e-3 to confirm the components are flat
across every physically-sane value. **(5) τ, two distinct knobs:** the A_τ quantization grid width was
swept over [0.05, 0.5] and its projection dimension over {5, 10, 20}, and the unstable-fraction threshold
was swept for both the legacy width test and the paper's ω_j>τ test — with a separate smooth-activation
control (`tests/test_omega_smooth.py`) to exhibit the τ-dependence that the ReLU subset masks by
construction. Each sweep varies one knob around a fixed production centre-point on a stratified
32-instance subset spanning all architectures, families, and difficulty terciles. On top of the sweeps we
computed **(6) the full 5×5 inter-component dependence matrix** (Spearman rank + distance correlation,
with per-domain variants) and **(7) two independently-derived unified Difficulty Indices** (a
predictively-weighted logistic combination and a first-principal-component combination), each validated
against real per-instance verifier timeout rates — both over the frozen 345-instance table. Every sweep
emits a CSV under `results/` and a figure under `plots/`; the sections below report the numbers.

## Headline
The profile is **not** hyperparameter-fragile. Four of the five components are essentially constant
across every knob and every random seed; the only genuinely tunable knob (A_τ's grid width) sits on a
stable plateau; and the parameters that *look* tunable (η, and the ReLU unstable-fraction threshold)
are invariant **by construction**, which the sweep confirms empirically.

## 1. Sample count N — most components converge at N=50; only A_τ is sample-hungry
Normalized drift from the N=1600 estimate (median over instances), per component:

| component | drift @ N=50 | drift @ N=200 | drift @ N=800 | plateau N (within 5%) |
|---|--:|--:|--:|--:|
| M̂_min | 1.2e-4 | 5.5e-5 | 4.2e-6 | **50** |
| G_IBP | 1.2e-7 | 5.1e-8 | 5.6e-9 | **50** |
| d_eff | 0.011 | 0.003 | 0.001 | **50** |
| A_τ | 0.442 | 0.269 | 0.082 | **~800–1600** |

M̂_min, G_IBP, d_eff are stable from as few as 50 samples. A_τ (a log-count of distinct gradient
fingerprints) needs more samples to enumerate the fingerprints — it is within 8% of the asymptote by
N=800 and converged by N=1600. See `plots/stability_vs_N.png`.

## 2. Seed / run-to-run stability — negligible
Coefficient of variation over 16 seeds at the recommended N (median over instances):

| component | median CoV |
|---|--:|
| M̂_min | 1.2e-6 |
| G_IBP | 1.6e-8 |
| **unstable fraction** | **0.0 (deterministic)** |
| A_τ | 1.3e-3 |
| d_eff | 6.0e-3 |

The IBP-based components (G_IBP, U) are essentially deterministic; the sampled ones vary by < 1% run to
run (`plots/cov_bar.png`).

## 3. Sampling distribution — small effect, boundary mixture is best
Across `uniform_only / boundary_only / current_mixture / pgd_heavy`, the normalized swing is small
(d_eff 0.039, G_IBP 0.004, M̂_min 0.002). Uniform-only under-estimates boundary hardness (higher
M̂_min); the boundary-biased mixture gives the tightest min-margin without the extra variance of
pgd_heavy — hence the recommended default.

## 4. η — flat across any sane value (a numerical constant, not a knob)
η is the numerical-stability constant in the G_IBP and d_eff denominators. It is **flat for η ≤ 1e-6**
and only a physically-absurd η=1e-3 (comparable to the quantities it stabilizes) perturbs anything:

| η | d_eff | G_IBP |
|---|--:|--:|
| 1e-12 | 35.83 | 444.15 |
| 1e-9 | 35.82 | 444.15 |
| 1e-6 | 35.64 | 443.85 |
| 1e-3 | 16.91 | 287.1 |

Recommended η = 1e-9 (deep in the flat region, still large enough to avoid division blow-up). This is a
feature, not a limitation: the component that *looks* like a tunable hyperparameter isn't one
(`plots/eta_effect.png`).

## 5. τ / grid width
- **A_τ grid width τ** (the one genuinely tunable knob, gVrS-Q2): A_τ is stable across τ ∈ [0.05, 0.2]
  (values 4.44–4.73) and across projection dim ∈ {5,10,20} (± ~0.1), drifting only at a very coarse
  τ=0.5. Recommended τ = 0.1 (mid-plateau), projection dim = 10 (`plots/atau_grid.png`).
- **Unstable-fraction threshold τ** (JJNS-Q2). τ thresholds each neuron's *slope-variation*
  ω_j = max φ′ − min φ′ over the input box (unstable ⇔ ω_j > τ). **How U changes with τ is entirely an
  activation-family question**, which is why a single reported range was incomplete:
  - **ReLU / piecewise-linear — U is exactly τ-invariant for every τ ∈ (0,1), by structure not tuning.**
    φ′ ∈ {0,1}, so ω_j is *binary*: 1 iff the pre-activation straddles 0, else 0, and `ω_j > τ` collapses
    to the exact 0-crossing test for any τ in the open unit interval. This is *why* the all-ReLU benchmark
    sweep is dead flat at U = 0.740 for every τ — τ has nothing to act on, not a lucky choice.
  - **Smooth (sigmoid/tanh) — U(τ) is a monotone non-increasing step function**: U → 1 as τ → 0 (every
    curved neuron counts) and decays to 0 once τ exceeds the largest ω_j, which is bounded by the
    activation's slope range (≤ 0.25 sigmoid, ≤ 1 tanh). On the sigmoid probe
    (`tests/test_omega_smooth.py`), U = 1.0 for τ ≤ 0.05, 0.5 at τ = 0.1, 0.0 at τ = 0.25 — a **low plateau
    up to ~20 % of the max slope-swing, then decay** (`plots/u_tau_curve.png`).

  **Recommended τ = 1e-2.** It sits on the smooth low-plateau (captures every neuron operating in its
  nonlinear region — the smooth analogue of ReLU's 0-crossing), is a *no-op* for ReLU/piecewise-linear
  nets (any τ ∈ (0,1) gives the exact test), and its small positive value discards only float-noise
  curvature in saturated neurons (ω_j ~ 1e-3). U is unchanged across τ ∈ [1e-3, 5e-2] for both families,
  so the value is not delicate; for cross-activation consistency an equivalent relative form is
  τ = 5 %·(range of φ′), which auto-scales to ≈0.0125 (sigmoid) / ≈0.05 (tanh).

## 6. Recommended defaults
From the plateaus/CoV above (`results/recommended_defaults.csv`):

| knob | recommended | why |
|---|---|---|
| sample count N | **1600** (50 suffices for all but A_τ) | A_τ plateau; the others converge by 50 |
| sampling distribution | boundary-biased mixture | tightest min-margin, low variance |
| η | 1e-9 | flat for η ≤ 1e-6; avoids blow-up |
| A_τ grid width τ | 0.1 | mid-plateau of [0.05, 0.2] |
| A_τ projection dim | 10 | flattens for ≥ 10 |
| U smooth-activation test | ω_j>τ, τ=1e-2 | on the smooth-net plateau (U flat for τ≤0.05, sigmoid); no-op for ReLU (ω binary ⇒ exact 0-crossing ∀τ∈(0,1)); U only decays for τ≳0.1 |
| seeds | ≥ 8-seed mean | removes residual < 1% sampling noise |

## 7. Inter-component dependence (full 345 table)
Spearman ρ over the five components (`results/correlation/spearman_all.csv`; distance-correlation in
`dcor_all.csv`; per-domain variants alongside; heatmaps `plots/spearman_5x5.png`, `dcor_5x5.png`):

|  | M̂_min | G_IBP | U | A_τ | d_eff |
|---|--:|--:|--:|--:|--:|
| **M̂_min** | 1.00 | −0.36 | −0.12 | −0.06 | −0.15 |
| **G_IBP** | | 1.00 | 0.16 | 0.29 | **0.58** |
| **U** | | | 1.00 | −0.18 | −0.32 |
| **A_τ** | | | | 1.00 | 0.46 |
| **d_eff** | | | | | 1.00 |

The components are moderately correlated but not redundant: the strongest coupling is G_IBP–d_eff
(ρ=0.58) and A_τ–d_eff (0.46); M̂_min is largely independent of the rest (|ρ| ≤ 0.36). This is the
"dependence structure" rxK4 asked for — related but distinct axes of hardness.

## 8. Unified Difficulty Index (rxK4's "unified measurement formula")
Two independently-derived single scalars over the standardized, direction-oriented five components,
validated against the real per-instance verifier timeout rate:

| index | construction | Spearman vs P(timeout) | Kendall | isotonic R² | decile lo→hi |
|---|---|--:|--:|--:|--:|
| **DI_pred** | mean logistic SD-weights (predictive) | 0.397 | 0.296 | 0.274 | 0.16 → 0.61 |
| **DI_PCA** | first principal component (EVR 39%) | 0.420 | 0.327 | 0.349 | 0.33 → 0.60 |

Both track real timeout rates and **agree** (that agreement is the robustness check). The predictive
weighting leans on G_IBP (2.16) and unstable-fraction (1.49); the PCA leans on d_eff/A_τ/G_IBP — two
different derivations, same monotone relationship to hardness (`plots/difficulty_index_validation.png`).

## Reproduce
```
cd rebuttal_materials/profile_sensitivity/src
python run_all.py --stage subset
python run_all.py --stage driver --axes all --n-jobs 100
python run_all.py --stage analysis && python run_all.py --stage plots
```
(oval21 + mnist_fc are excluded from the recompute — CIFAR/784-dim finite-difference cost — but remain
in the full-345 correlation matrix + Difficulty Index. `tests/test_omega_smooth.py` validates the ω_j>τ
fix.)
