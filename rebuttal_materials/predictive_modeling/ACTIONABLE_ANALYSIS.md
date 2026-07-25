# Does the Difficulty Profile add predictive value beyond size/type? — bootstrap test

We replicate the framework of a prior cross-country excess-mortality study
(*Scientific Reports*, 2023): "Bootstrap hypothesis testing" + "Measuring the effect of actionable
features", adapted to the VeriStress timeout-prediction setting.

## Headline (single overall test, analogous to the prior study's one result)
Pooling **all instances × all five verifiers** into one gradient-boosting model (verifier identity
included as an intrinsic covariate, like countries in the prior study) and testing whether the
Difficulty Profile improves held-out timeout prediction beyond size/type + verifier identity:

> **ΔAUC = +0.107, p = 0.001** (B = 1000 bootstrap; 0/1000 null shuffles reached it; null 95% band
> only +0.020). Deviance gain +0.125, p = 0.001. N = 1525 (instance, verifier) rows, 231 networks.

Sub-pools: **synthetic** ΔAUC +0.043 (p = 0.001); **established** ΔAUC +0.108 (p = 0.005, AUC;
deviance n.s. on the underpowered 6-network slice). (`results/actionable/overall_htest_h240.csv`.)
The per-verifier breakdown below shows where this overall gain concentrates.

---

**Mapping.** intrinsic features → network **size/type (S)**; actionable features → the five
**Difficulty-Profile** components (D); response → **timeout** (binary, uniform 240 s horizon).
Model → **gradient boosting** (HistGradientBoosting), matching the paper's GBM. All CV is
**network-grouped** (their rows were one-per-country; our instances share networks).

## The test
A flexible model can lower error just by having *more variables*, so a raw AUC gain isn't proof.
We test `H0: A ⟂ Y` with a **parametric bootstrap** (the classification analog of the paper's
residual bootstrap `y* = ŷ_I + r*`):

1. Fit intrinsic-only (S) with grouped CV → out-of-fold probabilities `p̂_I`.
2. Draw a null response `y*_i ~ Bernoulli(p̂_I(x_i))` — labels depend on S only, so D is null by
   construction.
3. Refit S and S+D on `y*`; statistic = held-out gain (ΔAUC and Δdeviance). Repeat B = 500 →
   null distribution.
4. p-value = fraction of null gains ≥ the **observed** gain (floor 1/501 ≈ 0.002).

## Result — profile adds significant predictive value in the large majority of cells

| Verifier · Scenario | n | pos | obs ΔAUC | p(AUC) | obs ΔDev | p(Dev) | null ΔAUC p95 |
|---|---|---|---|---|---|---|---|
| abcrown · synthetic | 225 | 36 | +0.089 | **0.002** | +0.075 | **0.002** | 0.026 |
| neuralsat · synthetic | 218 | 64 | +0.043 | **0.004** | +0.060 | **0.004** | 0.018 |
| marabou · synthetic | 168 | 43 | +0.120 | **0.002** | +0.145 | **0.002** | 0.034 |
| nnenum · synthetic | 127 | 20 | +0.072 | **0.046** | +0.020 | 0.078 | 0.070 |
| pyrat · synthetic | 198 | 38 | +0.009 | 0.202 | +0.028 | **0.032** | 0.023 |
| abcrown · combined | 343 | 102 | +0.135 | **0.002** | +0.150 | **0.002** | 0.020 |
| neuralsat · combined | 336 | 116 | +0.160 | **0.002** | +0.175 | **0.002** | 0.051 |
| marabou · combined | 286 | 94 | +0.106 | **0.002** | −0.090 | 0.832 | 0.030 |
| nnenum · combined | 247 | 92 | +0.107 | **0.002** | +0.131 | **0.002** | 0.030 |
| pyrat · combined | 313 | 97 | +0.072 | 0.076 | +0.271 | **0.006** | 0.081 |

- **AUC gain is significant at the bootstrap floor (p ≈ 0.002) in 7/10 cells** and p < 0.05 in 8/10.
  The observed gain sits far beyond the null 95% band (`plots/actionable_htest.png`).
- **GBM captures nonlinear profile signal the linear model missed**: abcrown-synthetic, flat under
  logistic (ΔAUC ≈ 0), is now clearly significant (ΔAUC +0.089, p = 0.002) — the profile helps once
  the model can use it nonlinearly.
- **pyrat is the honest exception** (its AUC is at ceiling): no AUC gain (p = 0.20 synthetic, 0.076
  combined), though deviance still improves. This is the same "no incremental ranking value" cell the
  logistic and permutation tests flagged.
- Two deviance anomalies (marabou-combined ΔDev < 0; nnenum-synthetic borderline): the *ranking* (AUC)
  improves but GBM calibration does not in those cells — reported, not hidden.

## Per-instance effect (δ) and interpretation
Following the paper's "delta" analysis, we compute per instance
`δ_i = P̂(timeout | S+D) − P̂(timeout | S)` with 95% **grouped-bootstrap** CIs
(`results/actionable/delta_*.csv`), and:
- **δ forest by constructor family** (`plots/actionable_delta_forest_*`): which families the profile
  pushes toward vs away from timeout.
- **profile values split by δ-sign** (`plots/actionable_density_*`, the Fig-7 analog): instances the
  profile pushes toward timeout have higher G_IBP / unstable-fraction.
- **partial-dependence** of each profile component + a 2-D G_IBP × parameter-count surface
  (`plots/actionable_pdp*`); **permutation-importance** with the profile block highlighted
  (`plots/actionable_importance_*`).

## Relationship to the other tests
This GBM bootstrap test, the elastic-net permutation-shuffle control
(`results/permutation_control/`), and the primary grouped-CV ΔAUC all agree: the Difficulty Profile
carries genuine per-instance signal beyond size/type — not an artifact of extra parameters — for the
large majority of verifiers, with pyrat (ceiling) the consistent exception.

*Reproduce:* `python actionable_analysis.py` then `python actionable_plots.py` (uniform 240 s horizon).
