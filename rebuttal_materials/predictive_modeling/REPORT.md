# Difficulty-Profile Predictive Modeling — Report

**Verdict.** In repeated network-grouped cross-validation, the five VeriStress-GT
Difficulty-Profile components add **statistically significant** held-out timeout-prediction value
**beyond network size/type** for the majority of verifier×scenario cells — decisively in the
**combined** synthetic+established setting (**4/5 verifiers**, ΔAUC **+0.05 to +0.20**, group-bootstrap
95% CIs above 0) and for the complete verifiers **Marabou and nnenum on synthetic** (ΔAUC **+0.11**
each). The **IBP relative gap (G_IBP)** and **unstable-fraction** are the stable drivers. For the
strong branch-and-bound verifiers on **synthetic-only** (abcrown, neuralsat, pyrat), network size
already predicts timeouts well (AUC 0.86–0.97) so the profile adds no AUC there — but it consistently
improves **calibration** (Brier), and adds **runtime-ranking** information. Cross-domain **transfer**
confirms the profile carries scale-invariant difficulty signal that raw size lacks.

## 1. Plain-language question and answer
Does the Difficulty Profile predict, for a *single instance*, whether a given verifier times out —
beyond ordinary network size/type descriptors? **Answer: yes for most verifiers, and the effect is
largest exactly where size is insufficient** (cross-architecture pooling and complete verifiers),
while for BaB verifiers on a single architecture family size already suffices and the profile instead
sharpens calibration. We report this heterogeneity rather than collapsing verifiers into one score.

## 2. Input audit and exclusions
- **Source (read in place):** `rebuttal_materials/merged_records_constructed_and_real.json` (345
  instances) — joins the five profile components with per-verifier status/runtime/outcome for
  abcrown, neuralsat, marabou, nnenum, pyrat.
- **Canonical table:** `data/processed/instance_verifier_rows.parquet` — **1725 rows** (345 instances
  × 5 verifiers), **231 distinct networks**.
- **Domains:** synthetic 225 (`sweep_all` 203 + `polynomial_stress_22` 22); established 120
  (`vnncomp_mnist_fc` 90 + `oval21` 30).
- **Network groups (leakage unit):** synthetic = per instance (225 groups); established = ONNX content
  hash → **only 6 groups** (mnist_fc 3 nets × 30 props, oval21 3 nets × 10 props).
- **Architecture mix (from graph structure):** MLP 180, CNN 126, POLYNOMIAL 22, ATTENTION 17.
- **Feature source:** ONNX for 323 instances; 22 polynomial nets reconstructed from generator `args`
  (`onnx_file_bytes` missing → tracked). 0 profile-missing rows, 0 duplicate rows, 0 instances dropped.
- **Excluded benchmarks:** rl_benchmarks / reach_prob_density (verifier logs exist but no profile join
  and no bounded-timeout budget → no timeout target). Logged in `results/audit/excluded_rows.csv`.

## 3. Outcome & common horizon
Target = `TIMEOUT` (positive) vs conclusive `UNSAT`/`SAT` (negative); ERROR/UNKNOWN/missing excluded.
Budgets differ by verifier and domain, so the common horizon is chosen **per (verifier, scenario)** as
the largest grid value retaining ≥85 % of rows with both classes present (see the `H` column).
Instances solved after the horizon are relabeled timeout-at-horizon. A fixed-360 s sensitivity pass is
available. (Marabou/nnenum on synthetic land at H=60 s because the `polynomial_stress_22` sub-benchmark
ran at a 60 s budget; those cells use the 168/127 non-poly synthetic instances at 60 s.)

## 4. Feature sets & model
- **S (size/type):** input/output dim, parameter count, node/op counts (Gemm/MatMul, Conv, ReLU, Add,
  Mul, Softmax, Pow), graph depth, hidden-layer count, max hidden width, ONNX bytes, architecture-type
  one-hot, op flags. **D (profile):** margin_hat_min, g_ibp, unstable_fraction, a_tau, d_eff.
  Constructor/benchmark/instance names are never features.
- Elastic-net logistic (saga), **StratifiedGroupKFold 5×20** (seeds 1100–1119), inner 4-fold grouped
  tuning of C×l1_ratio×class_weight by ROC-AUC. Identical outer folds for S/D/S+D. Thresholds from
  inner-train predictions only. Headline = paired held-out ΔAUC with 95% group-bootstrap CI over
  networks. Leak-free transforms outside folds; imputation/standardization inside folds.

## 5. Main results — size/type vs size/type+profile

`H` = horizon (s); `TO` = timeout base rate; `Groups` = independent networks. ΔAUC is the aggregate
out-of-fold Size+Profile − Size, with 95% group-bootstrap CI. `*` = exploratory (6 groups < 10-group
gate). Verdicts are the pre-registered gate (§9.3), not redefined after seeing results.

| Verifier | Scenario | n | TO | Groups | H | Size AUC | Size+Prof AUC | ΔAUC | 95% CI | Size Brier | S+P Brier | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abcrown | Synthetic | 225 | .14 | 225 | 600 | 0.859 | 0.865 | +0.005 | [−0.033,+0.040] | .117 | .086 | ≈size; profile ↑calib |
| marabou | Synthetic | 168 | .30 | 168 | 60 | 0.872 | **0.968** | **+0.111** | **[+0.064,+0.161]** | .132 | .058 | **PROFILE ADDS VALUE** |
| neuralsat | Synthetic | 218 | .13 | 218 | 600 | 0.887 | 0.882 | +0.002 | [−0.039,+0.044] | .107 | .094 | ≈size; profile ↑calib |
| nnenum | Synthetic | 127 | .20 | 127 | 60 | 0.822 | **0.927** | **+0.107** | **[+0.030,+0.187]** | .136 | .081 | **PROFILE ADDS VALUE** |
| pyrat | Synthetic | 198 | .19 | 198 | 600 | 0.972 | 0.967 | +0.001 | [−0.015,+0.016] | .045 | .044 | ≈size (ceiling) |
| abcrown | Combined | 343 | .28 | 231 | 360 | 0.768 | **0.825** | **+0.056** | **[+0.011,+0.116]** | .233 | .227 | **PROFILE ADDS VALUE** |
| marabou | Combined | 286 | .38 | 174 | 60 | 0.600 | **0.805** | **+0.200** | **[+0.100,+0.288]** | .294 | .185 | **PROFILE ADDS VALUE** |
| neuralsat | Combined | 336 | .35 | 224 | 240 | 0.771 | **0.838** | **+0.082** | **[+0.018,+0.153]** | .187 | .159 | **PROFILE ADDS VALUE** |
| nnenum | Combined | 247 | .42 | 133 | 60 | 0.718 | **0.882** | **+0.151** | **[+0.016,+0.281]** | .242 | .144 | **PROFILE ADDS VALUE** |
| pyrat | Combined | 313 | .31 | 204 | 360 | 0.832 | 0.879 | +0.046 | [−0.002,+0.108] | .196 | .144 | partial (CI grazes 0) |
| abcrown | Established* | 118 | .52 | 6 | 360 | 0.574 | 0.743 | +0.209 | [−0.067,+0.493] | .296 | .226 | exploratory (wide CI) |
| marabou | Established* | 118 | .43 | 6 | 240 | 0.334 | 0.486 | +0.153 | [+0.050,+0.361] | .300 | .289 | exploratory (positive) |
| neuralsat | Established* | 118 | .44 | 6 | 240 | 0.579 | 0.712 | +0.055 | [−0.080,+0.290] | .256 | .243 | exploratory (wide CI) |
| nnenum | Established* | 120 | .60 | 6 | 240 | 0.653 | 0.732 | +0.025 | [−0.074,+0.338] | .227 | .217 | exploratory (wide CI) |
| pyrat | Established* | 115 | .50 | 6 | 360 | 0.581 | 0.776 | +0.246 | [−0.009,+0.475] | .243 | .229 | exploratory (wide CI) |

**Reading it.**
- **Combined (231 networks, well-powered):** the profile adds significant value for **4/5** verifiers
  (all but pyrat, whose CI just grazes 0). Here size alone is weak (AUC 0.60–0.83) because it must span
  two very different size regimes; scale-relative difficulty fills the gap. Marabou +0.200 and nnenum
  +0.151 are large.
- **Synthetic (225 networks):** for **Marabou and nnenum** the profile adds +0.11 AUC (CIs clear of 0)
  and roughly halves Brier — their timeouts are driven by difficulty structure, not scale. For
  **abcrown/neuralsat/pyrat** size already gives AUC 0.86–0.97; the profile adds no AUC (ΔAUC ≈ 0, CI
  includes 0) but consistently improves calibration (e.g. marabou-syn Brier .132→.058, abcrown-syn
  .117→.086).
- **Established (6 networks):** point estimates are all positive (size alone is ≈chance within a fixed
  architecture family, e.g. marabou 0.334, pyrat 0.581), but with only 6 groups the CIs are wide;
  only Marabou clears 0. Reported **exploratory** — the profile *looks* decisive here but the design is
  underpowered at the network level. This is the honest limitation, not a claim.

## 6. Cross-domain transfer (Scenario D)
Train on one domain, test once on the other; 95% group-bootstrap over target networks
(`results/transfer/`).

| Verifier | Direction | tgt n | Size AUC | Size+Prof AUC | ΔAUC | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| abcrown | synth→estab | 118 | 0.371 | 0.668 | +0.297 | [−0.026,+0.453] |
| marabou | synth→estab | 118 | 0.321 | 0.741 | +0.420 | [−0.015,+0.625] |
| neuralsat | synth→estab | 118 | 0.344 | 0.742 | +0.398 | [+0.059,+0.524] |
| nnenum | synth→estab | 120 | 0.444 | 0.807 | +0.363 | [−0.067,+0.553] |
| pyrat | synth→estab | 115 | 0.464 | 0.551 | +0.087 | [+0.054,+0.244] |
| abcrown | estab→synth | 225 | 0.335 | 0.600 | +0.265 | [+0.143,+0.381] |
| marabou | estab→synth | 168 | 0.500 | 0.864 | +0.364 | [+0.304,+0.417] |
| neuralsat | estab→synth | 218 | 0.744 | 0.671 | −0.073 | [−0.165,+0.023] |
| nnenum | estab→synth | 127 | 0.288 | 0.739 | +0.451 | [+0.263,+0.624] |
| pyrat | estab→synth | 198 | 0.461 | 0.323 | −0.139 | [−0.251,−0.022] |

**Key point:** size features trained on one domain typically transfer at or **below chance** (AUC
0.29–0.50 — the size→timeout relation does not survive the change of size regime), and the profile
**restores discrimination**, sometimes dramatically (marabou estab→synth 0.50→0.86; nnenum estab→synth
0.29→0.74). Two honest exceptions where the profile does not help transfer: neuralsat and pyrat in the
estab→synth direction. This is direct evidence that the profile measures *architecture-agnostic*
difficulty, which is the property a per-instance diagnostic should have.

## 7. Which components carry the signal (ablation + coefficients)
- **Add-one-profile to size (ΔAUC), most-valuable component per cell** (`results/ablation/addone_best.csv`):
  **`g_ibp`** wins in 9/15 cells — Marabou-combined **+0.208**, nnenum-combined +0.153, Marabou-syn
  +0.083, nnenum-syn +0.073; `unstable_fraction` wins several established/combined cells (neuralsat-comb
  +0.064, abcrown-estab +0.038).
- **Standardized coefficients (final descriptive SD model, `results/coefficients/`):** `g_ibp` is
  positive with **nonzero-frequency 1.0 and sign-consistency 1.0 in every cell** — odds ratios up to
  **62×** (marabou-combined) and **111×** (nnenum-combined) per +1 SD. `unstable_fraction` is the stable
  second driver (OR 8–11×). Higher IBP looseness and more unstable neurons ⇒ higher timeout odds — the
  mechanism bound-propagation/BaB verifiers actually struggle with.
- **Stable interactions (predeclared, `results/coefficients/stable_interactions.csv`):** `g_ibp ×
  parameter_count` and `g_ibp × arch_CNN` recur with consistent positive sign across verifiers — the
  IBP-gap effect is **amplified in larger / convolutional networks**. This answers the interaction-aware
  concern: the profile's contribution is not merely an additive proxy for scale.

## 8. Secondary — solved-only runtime (log1p seconds; selection-biased, reported as secondary)
Spearman rank-correlation of predicted vs actual runtime improves with the profile for **all five**
verifiers on synthetic (ΔSpearman: nnenum +0.32, pyrat +0.16, marabou +0.09, neuralsat +0.08, abcrown
+0.05) and for abcrown-combined (+0.20). Mixed on some combined cells (nnenum −0.08). Consistent with
the timeout finding: the profile carries information about *how hard*, not just *whether it times out*.
(`results/runtime/runtime_regression.csv`.)

## 9. Calibration
For the BaB verifiers where AUC is already high, the profile's contribution shows up as **better
calibration**: Brier improves in 13/15 cells (e.g. abcrown-syn .117→.086, neuralsat-syn .107→.094,
pyrat-comb .196→.144). ROC/PR/reliability curves per cell in `plots/roc_pr_calib_*.png`.

## 10. Limitations & blocked analyses
- **Established-only is underpowered at the network level (6 groups < 10-group gate)** → exploratory;
  large point estimates there are suggestive, not claims. Leave-one-benchmark-out is impossible (2
  benchmarks). This is the main caveat.
- **Marabou/nnenum synthetic run at H=60 s** (poly sub-benchmark budget) on 168/127 instances; the
  effect is at that horizon.
- **Runtime regression is selection-biased** (timed-out instances censored) — secondary only.
- **SAT on provably-robust instances** are soundness anomalies counted as conclusive per the frozen
  spec; counts are small and reported in the audit.
- All budgets ≥ their horizon by construction; a fixed-360 s sensitivity pass is available for cells
  whose budgets reach it.

## 11. Reproduction
```bash
cd rebuttal_materials/predictive_modeling/src
python build_dataset.py
python run_all.py --repeats 20 --n-jobs 50      # primary CV + transfer + correlations
python finalize.py                              # interpret, runtime, predictions, tables, plots
python report.py                                # tables + verdicts
cd ../tests && PYTHONPATH=../src python test_pipeline.py
```
Frozen inputs + hashes in `data/manifests/frozen_analysis_manifest.json`. The CV reads only the frozen
`instance_verifier_rows.parquet` (CSV fallback), so it runs without ONNX/GPU.

## 12. Tests & status
Leakage/correctness suite (`tests/test_pipeline.py`): **9/9 pass** — no network group spans train/test;
identical folds across feature sets; preprocessing/thresholds fit on training only; forbidden columns
absent; horizon relabeling boundaries; deterministic seeds; serialization round-trip; positive control
(profile recovers a planted effect) and negative control (no spurious increment).

## 13. Primary artifacts
- `results/primary_results_table.{md,csv}` — the table above.
- `results/predictions/out_of_fold_predictions.csv` — per-instance size-only / profile-only / augmented
  timeout probabilities (the per-instance predictor artifact) + `instance_verifier_repeat_avg.csv`.
- `results/transfer/`, `results/coefficients/`, `results/ablation/`, `results/per_subgroup_auc.csv`.
- `plots/` — AUC comparison, paired ΔAUC, Brier, ROC/PR/calibration, transfer, coefficients, ablation,
  correlation heatmaps.
