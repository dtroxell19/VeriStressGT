<!-- RESULTS sections (6–15) are injected/finalized after the CV run completes. -->
# Difficulty-Profile Predictive Modeling — Report

**Verdict:** _[filled after run: one sentence on whether the Difficulty Profile predicts
per-instance verifier timeouts beyond network size/type, and for which verifiers/domains]._

## 1. Plain-language question and answer
We ask whether the five VeriStress-GT **Difficulty-Profile** components
(M̂_min, G_IBP, unstable-fraction, A_τ, d_eff) predict, for a *single instance*, whether a
given verifier will time out — **beyond** what ordinary network size/type descriptors already
predict. We answer it per verifier with repeated, network-grouped, held-out cross-validation,
comparing `Size/type` against `Size/type + Profile` on identical folds. _[answer filled after run]_

## 2. Input audit and exclusions
- **Source (read in place):** `rebuttal_materials/merged_records_constructed_and_real.json`
  (345 instances) — already joins the five profile components with per-verifier
  status/runtime/outcome for abcrown, neuralsat, marabou, nnenum, pyrat.
- **Canonical table:** `data/processed/instance_verifier_rows.parquet` — **1725 rows**
  (345 instances × 5 verifiers), **231 distinct networks**.
- **Domains:** synthetic 225 instances (`sweep_all` 203 + `polynomial_stress_22` 22),
  established 120 (`vnncomp_mnist_fc` 90 + `oval21` 30).
- **Network groups (leakage unit):** synthetic = per instance (225 groups); established = ONNX
  content hash → **only 6 groups** (mnist_fc 3 nets × 30 props, oval21 3 nets × 10 props).
- **Architecture mix (from graph structure):** MLP 180, CNN 126, POLYNOMIAL 22, ATTENTION 17.
- **Feature source:** ONNX for 323 instances; 22 polynomial nets reconstructed from generator
  `args` (`onnx_file_bytes` missing → tracked; 110 rows carry ≥1 missing size feature).
- **Coverage:** 0 profile-missing rows, 0 duplicate (instance,verifier) rows, 0 instances dropped.
- **Excluded benchmarks:** rl_benchmarks / reach_prob_density (verifier logs exist but no profile
  join and no bounded-timeout budget → no timeout target). Logged, not silently dropped.

## 3. Outcome definition and common timeout horizon
- **Target:** `TIMEOUT` (positive) vs conclusive `UNSAT`/`SAT` (negative); ERROR/UNKNOWN/missing
  excluded from the strict target. (All instances are provably robust, so any SAT is a soundness
  anomaly — counted as conclusive per spec, reported separately; counts are small.)
- **Budgets differ by verifier and domain** (inferred from TIMEOUT runtimes): synthetic ≈ 600 s;
  established 360 s (abcrown, pyrat) / 240 s (neuralsat, marabou, nnenum); `polynomial_stress_22`
  ran at 60 s for marabou/nnenum. The **common horizon is chosen per (verifier, scenario)** as the
  largest grid value retaining ≥85 % of rows with both classes present; a fixed-360 s sensitivity
  pass is provided. Instances solved after the horizon are relabeled timeout-at-horizon.

## 4. Network size/type feature definitions (baseline S)
Input/output dim, parameter count, node & op counts (Gemm/MatMul, Conv, ReLU, Add, Mul, Softmax,
Pow), graph depth (longest path), hidden-layer count, max hidden width, ONNX file bytes;
architecture-type one-hot (MLP/CNN/ATTENTION/POLYNOMIAL/HYBRID/OTHER, from graph structure) and
op-presence flags. Skewed counts get log1p; profile features get signed-log1p / log1p / fixed-eps
logit as declared. Constructor/benchmark/instance names are **never** features.

## 5. Grouped repeated cross-validation design
Elastic-net logistic regression (saga). **StratifiedGroupKFold, 5 folds × 20 repeats**
(seeds 1100–1119), inner 4-fold grouped tuning of C × l1_ratio × class_weight by ROC-AUC inside
every outer training fold. Identical outer folds for S, D, and S+D within each cell. Thresholds
chosen from inner training predictions only. Headline = paired held-out ΔAUC (S+D − S) with a 95%
group-bootstrap CI over networks; AP, Brier, calibration reported alongside.

---
<!-- ===================== RESULTS (auto/finalized post-run) ===================== -->
_Results tables, per-verifier findings, transfer, ablations, coefficients, calibration,
runtime/survival, limitations, reproduction commands, and test/commit status are appended once
`run_all.py` completes. Generated fragments live under `results/` (primary_results_table.md,
transfer/transfer_table.md, coefficients/…, ablation/…)._
