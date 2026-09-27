# Difficulty-Profile Sensitivity Study — decisions log

Chronological record of non-obvious choices, so the study is reproducible and the
rebuttal text can cite exact rationale.

## Mapping the paper's η / τ to the code (start)
- Read the paper's Definition 1 and eqs. 10–15. **η** (G_IBP eq.11, d_eff eq.15) is
  explicitly "a small numerical-stability constant", not a difficulty knob — so its
  "sensitivity" is expected to be ~zero and we report that plainly rather than
  manufacturing a large effect.
- **τ** appears in two places: the A_τ quantization **grid width** (eq.14) and the
  smooth-activation unstable test (U, eq.12, `ω_j = sup|φ'(s)−φ'(t)| > τ`). For ReLU,
  U reduces to the exact 0-crossing test and is **τ-independent by definition**.
- Two faithfulness gaps found in the code and closed behind opt-in flags:
  - **G1**: the smooth-activation branch of `_interval_propagate` used a degenerate
    interval-width test `(hi−lo)>1e-12`, not eq.12's `ω_j>τ`. Added `_omega_range`
    (exact derivative-range for Sigmoid/Tanh/LeakyRelu) + `smooth_unstable_mode="omega"`.
  - **G2**: A_τ grid width was `round(F, decimals)` (powers of ten only). Added
    `quantize_width` for any τ.
  - **G3**: A_τ random projection was hard-seeded `default_rng(1)` (un-seedable). Now
    threads the run seed.
- All new kwargs default to prior behavior; verified `unstable_frac`, `ibp_relative_gap`,
  and `margin_sample_min` reproduce the stored `sweep_all/difficulty_profiles.json` values
  on fp7 (U=1.0) and ld7 (U=0.0); a fixed seed reproduces d_eff run-to-run.

## Recompute scope
- Reviewers asked for a "brief" analysis → recompute on a **stratified ~48-instance
  subset** (all 4 architectures, all 4 benchmarks/datasets, easy→hard difficulty terciles).
- The inter-component dependence matrix and the Difficulty Index use the full frozen
  345-instance table (`predictive_modeling/`), so they validate against real verifier
  outcomes.
- `polynomial_stress_22` has no committed ONNX (network reconstructed from generator args
  at build time). Rather than risk a silently-divergent reconstruction for 3/48 instances,
  poly is **recompute-N/A** (flagged in `subset.csv`, `na_reason=no_committed_onnx`);
  POLYNOMIAL remains fully represented in the dependence matrix + Difficulty Index.

## Sweep design
- One axis varied at a time around a production **center-point** C0 = {N=200,
  current_mixture, A_τ width=0.1, proj=10, η=1e-12, U width-mode}; each job calls only the
  sub-estimator(s) the axis touches (large runtime savings vs. full re-profiling).
- Sampling-distribution presets: `uniform_only`, `boundary_only`, `current_mixture`
  (production), `pgd_heavy`. `uniform_only`/`boundary_only` also drop the deterministic
  first-order worst-case corner so they are *pure* (no gradient-informed point). A_τ and U
  are computed on uniform samples / IBP and are **invariant to the mixture preset** — stated
  so the tornado plots are not misread.
- Jobs run in isolated processes (loky) because the estimators seed torch/numpy *global*
  RNG; each worker pins `torch.set_num_threads(1)`.

## Unified Difficulty Index
- Two variants on the frozen 345 table: **DI_PCA** (first principal component of the
  oriented-standardized 5 components) and **DI_pred** (Σ w_k·z_k with w_k = mean over the 5
  verifiers of the standardized logistic SD-coefficients from `predictive_modeling/`).
- Orientation "higher = harder": only M̂_min is inverted (higher margin = easier).
- Validated against the empirical per-instance timeout rate (fraction of the 5 verifiers
  that time out, ERROR/UNKNOWN excluded from the denominator): decile monotonicity, Spearman,
  Kendall, isotonic R². Agreement of the two variants is the robustness argument.

## Anonymization
- All paths repo-root-relative (`REPO_ROOT = parents[3]`); no home dirs, hostnames, usernames,
  or author names in any file, CSV, or figure. Git identity stays the anonymous account.
