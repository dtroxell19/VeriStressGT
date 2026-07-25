# DECISIONS — Difficulty-Profile Predictive Modeling increment

Chronological record for this increment only. Root `DECISIONS.md` gets a one-line pointer.
All work is isolated under `rebuttal_materials/predictive_modeling/`; no existing artifact is modified.

## 2026-07-25 — Input audit & authoritative source
- **Authoritative joined source**: `rebuttal_materials/merged_records_constructed_and_real.json`
  (345 records). It already joins the five Difficulty-Profile components with per-verifier
  status / runtime / outcome for all five verifiers (abcrown, neuralsat, marabou, nnenum, pyrat).
  Rebuilding these joins from scratch would risk disagreeing with the numbers already used
  elsewhere in the paper, so we read this table in place and derive our copies from it.
- **Population**: 345 instances = 225 synthetic constructor instances (203 `sweep_all` +
  22 `polynomial_stress_22`) + 120 established-benchmark instances (90 `vnncomp_mnist_fc`
  + 30 `oval21`). The other real benchmarks under `real_verify/` (rl_benchmarks,
  reach_prob_density) are **excluded**: they lack the five profile components in the joined
  table and were run without a bounded timeout budget, so no timeout target can be defined.
  Recorded as an exclusion, not silently dropped.

## 2026-07-25 — Network group identity (leakage control)
- Synthetic instances are one-network-per-instance → group id = `benchmark::instance_id`
  (225 groups).
- Established instances share networks (VNN-COMP re-uses a handful of nets across many
  properties). `args` is empty for these rows, so identity is recovered by **SHA-256 of the
  local `model.onnx`**. Result: mnist_fc = 3 networks × 30 properties, oval21 = 3 × 10 →
  **only 6 established network groups**.
- **Consequence**: established-only fails the ≥10-group eligibility gate. It is therefore run
  as an **exploratory** analysis (leave-network-out), never a headline claim. Well-powered
  scenarios are synthetic-only (225 groups) and combined (231 groups).

## 2026-07-25 — Timeout target & common horizon
- Budgets differ by domain **and** verifier (inferred from the median runtime of TIMEOUT rows
  per verifier×benchmark): synthetic ≈ 600 s everywhere; established = 360 s (abcrown, pyrat)
  but 240 s (neuralsat, marabou, nnenum); the `polynomial_stress_22` sub-benchmark ran at 60 s
  for marabou/nnenum. This is exactly the "incomparable horizons" case the spec warns about.
- **Decision**: the common horizon is chosen **per (verifier, scenario)** as the largest value
  in {60,120,180,240,300,360,480,600} that retains ≥85 % of the cell's rows with both classes
  present (spec §4.2 "largest common horizon supported by every included run"). Rows from
  runs whose budget < horizon are excluded at that horizon and logged. A fixed-360 s
  sensitivity pass is provided separately (spec §10.5).
- Timeout label at horizon H: TIMEOUT (budget≥H) → 1; conclusive UNSAT/SAT solved in <H → 0;
  conclusive but runtime ≥ H → 1 (timeout-at-H); ERROR/UNKNOWN/missing → excluded. SAT/
  counterexample on these provably-robust instances is a soundness anomaly but is counted as
  "conclusive/negative" per spec §4.1 (and reported separately; counts are small).

## 2026-07-25 — Features & transforms
- Profile (D): the canonical five — `margin_hat_min, g_ibp, unstable_fraction, a_tau, d_eff`.
- Size/type (S): extracted from ONNX (input/output dim, param count, node/op counts, graph
  depth, hidden layers/width, file bytes, architecture-type one-hot from graph structure, op
  flags). The 22 polynomial nets have no committed ONNX → features reconstructed from generator
  `args` with `onnx_file_bytes` left missing (tracked). `feat_source ∈ {onnx, poly_args}`.
- Constructor/benchmark/instance names are **never** primary features (kept only for grouping
  and held-out subgroup reporting).
- Transforms are declared per feature and **stateless** (signed-log1p / log1p / fixed-eps logit
  / identity) so applying them globally leaks nothing; the only data-fit steps (median impute,
  standardize) live inside the fold pipeline.

## 2026-07-25 — Models & CV
- Headline model: elastic-net logistic regression (saga). Grid C×l1_ratio×class_weight per
  spec. Nested: inner 4-fold grouped tuning by ROC-AUC inside every outer training fold.
- Repeated **StratifiedGroupKFold** (5 folds × 20 repeats, seeds 1100–1119); falls back to
  fewer folds / GroupKFold when group/class counts require it. Identical folds across all
  feature sets within a cell (splits depend only on (y, groups)). Thresholds picked from inner
  training predictions only.
- Verdict gate is pre-registered (spec §9.3) and evaluated after the run; not redefined.
