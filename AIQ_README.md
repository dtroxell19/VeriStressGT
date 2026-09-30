> **Phase 1 submission summary (PDF):** [docs/aiq_phase1_summary.pdf](docs/aiq_phase1_summary.pdf)
> covers the approach, the comparison with existing benchmarks, expected results, BAA alignment
> and scalability, in 3 pages.

## AIQ Flow

**Phase 1 submission card: `cards/thrust1_soundness.yaml`.** The other cards are for
self-evaluation.

### Setup (one command)
Needs git and a C/C++ toolchain; conda is installed if missing. Tested on macOS (Apple Silicon);
Linux x86_64 is the expected evaluation platform.
```bash
git clone https://github.com/dtroxell19/VeriStressGT.git
cd VeriStressGT
git checkout evaluation-card
bash scripts/bootstrap.sh            # envs: VeriStressGT, alpha-beta-crown, pyrat, nnenum, neuralsat, marabou
conda activate VeriStressGT
python scripts/check_aiq_setup.py    # runs every real verifier on one UNSAT + one SAT instance
```
`bootstrap.sh` installs MAGNET pinned to `v0.1.0` (the frozen Phase 1 API) into the `VeriStressGT`
env, and each verifier into its own conda env. The runners locate the envs by name, and the
α-β-CROWN checkout by default, so no environment variables are needed.

**Marabou** is pinned to 2.0.0 (`scripts/install_marabou.sh`). PyPI has 2.0.0 wheels only for
Linux x86_64 and Intel macOS; on Apple Silicon and Linux aarch64 the script builds the `v2.0.0`
tag from source (30-60 min). The Intel wheel crashes under Rosetta on macOS 14, so it is not used.

**Gurobi.** `gurobipy` (pip, free size-limited license) is needed only to rebuild or scale the
benchmark; the committed `thrust1_bench/` runs without it.

**macOS + MAGNET v0.1.0.** Its job scripts run `chmod +x -- <file>`, which macOS `chmod` rejects,
so `magnet evaluate` fails on macOS unless a `chmod` that accepts `--` is first on `PATH`
(e.g. GNU coreutils). Linux is unaffected.

### Run
```bash
magnet evaluate cards/thrust1_soundness.yaml
```
**Expected:** `RESULT: VERIFIED`, and in the top-level `verdict.json` both metrics at 1.0
("Scoring accuracy (ground truth)", "Buggy verifiers caught (ground truth)"). Full tables are under
Thrust 1 > Sample run results.

About 30 min on a laptop CPU. The runner stops with an error if a real verifier is not working,
so a partial environment cannot silently shrink the verifier pool
(`allow_missing_verifiers: True` in the card's `algo_params` overrides this).

**Metrics** (declared with `define_metric`, reported in the top-level `verdict.json`):
`gt_scoring_accuracy` (fraction of all definitive verdicts that ground truth judges correctly;
the BAA accuracy metric) and `gt_detection_rate` (fraction of exercised planted bugs caught).

### Scaling
Set `scale: N` in the card's `algo_params`. The runner builds (once) a benchmark with `N`
re-seeded copies of every base network, each with its analytic certificate, plus `N` times the
random and attack-hard CNN SAT instances, in `thrust1_bench_xN/`, and runs the same pool on it.
Every label is re-derived from construction, so the claim should hold unchanged at any scale.

### Evaluation cards

| Card | Thrust | Metric |
|------|--------|--------|
| `cards/thrust1_soundness.yaml` | 1 (**submission**) | Scoring accuracy and buggy verifiers caught, with ground-truth labels |
| `cards/thrust2_timeout_prediction.yaml` | 2 | Per-verifier timeout-prediction AUC (target >= 0.7), held-out and live out-of-distribution |
| `cards/evaluation.yaml` | 2 (supporting) | Mini sweep: per-verifier correctness + per-component timeout AUC |

## Thrust 1: bug detection with ground-truth labels

```bash
magnet evaluate cards/thrust1_soundness.yaml
```

**Benchmark (`thrust1_bench/`, pre-built and committed).** Built from
`src/VeriStressGT/configs/thrust1.yaml` plus `python -m VeriStressGT.soundness.ground_truth`
(pass `--rebuild` to the runner to regenerate):

- 24 **UNSAT** instances, labelled by the constructors' analytic certificates (MILP exact radius,
  MEAP, corners, paired-bias CNN, contractive CNN, linear / softmax attention). This includes
  near-boundary MILP radii (eps = 0.999 r\*, 0.9999 r\*).
- 26 **SAT** instances, each shipping a concrete counterexample (`witness.json`) whose margin is
  negative in both float64 and onnxruntime float32, so no verifier has to be trusted for these
  labels. They are "twins" of robust networks with the radius scaled just past (`_sat_near`) or
  well past (`_sat_far`) the robustness threshold. MEAP and paired-bias networks are robust at
  every radius, so small random CNNs (`rcnn_*`, `tcnn_*`) supply the CNN counterexamples.
  Four of them (`tcnn_*_sat_hard`) sit so close to the threshold that PGD cannot find the
  counterexample but exact MILP can, so bugs in a verifier's proof procedure cannot hide behind a
  successful attack.

Without SAT instances, a verifier that always answers "robust" would score perfectly. They are
what exposes false UNSAT claims, the dangerous direction.

**Verifier pool.**

| Role | Verifiers |
|------|-----------|
| Sound controls | `reference` (in-house: PGD attack, then CROWN bounds, then exact MILP via scipy/HiGHS or ReLU-split BaB), `ibp_only` |
| Planted, tier (b) | 12 mutants of the reference with one injected bug each. Bound computation: `ibp_sign`, `relu_no_intercept`, `conv_bias_dropped`. Numerics: `tol_unsat`, `tol_sat`, `weights_fp16` (verifies a half-precision copy of the network). Search logic: `disjunct_last` (the Appendix D bug), `bab_any_row` (prunes a sub-domain once any disjunct is proved). Input/spec handling: `input_clip01`, `eps_half`, `hwc_layout` (reads a CHW image box as HWC), `label_off_by_one` |
| Real | `abcrown`, `pyrat`, `nnenum`, `neuralsat`, `marabou` (each skipped if not installed) |
| Planted, tier (a) | 6 output-level faults on each real verifier: `timeout_as_unsat`, `crash_as_sat` (the rebuttal-era parser bug), `conv_sat_to_unsat`, `attention_unsat_to_sat`, `cache_collision` (results cached by network file, ignoring the spec), `flaky` (nondeterministic flips on ~15% of instances) |

Bug descriptions and the failure classes they model are in `src/VeriStressGT/soundness/bugs.py`.

**Scoring.** A verifier is flagged when any SAT/UNSAT verdict contradicts the label;
timeouts and errors never flag. On benchmarks without a label for every instance (the external
comparison below), verdicts are also scored with the benchmark's labels plus majority vote on its
unlabelled instances (`labels_plus_majority`), and `coverage` counts how many verdicts each
benchmark can judge with certainty.

**Claim.** The card passes only if ground truth scores 100% of verdicts correctly, flags no sound
control, and catches at least `detection_threshold` (0.75) of the planted bugs that fired.

For planted verifiers, a flag counts only on instances where the planted bug changed the verdict
of the verifier it was derived from. Disagreements it merely inherits from that verifier are listed
separately as `gt_inherited`. Detection rates count only *exercised* planted bugs, meaning ones
that changed at least one verdict. A fault that never fires (for example a crash handler on a
verifier that never crashed) gives the benchmark nothing to catch and is listed as `not_exercised`.

The real verifiers run with counterexample search enabled (`configs/abcrown_thrust1.yaml` sets
`pgd_order: before`, and pyrat gets `--check both --attack pgd`). With the repo's default settings
both only try to prove UNSAT and never report SAT.

**Outputs** (next to `results.json`): `thrust1_verifiers.csv` (per-verifier flags),
`thrust1_evidence.json` (the instances that expose each flag), `thrust1_verdict_matrix.png`
(verifier × instance, correct / wrong / undecided).

### Sample run results

**Command:** `magnet evaluate cards/thrust1_soundness.yaml` with MAGNET v0.1.0 (card defaults:
60 s timeout, 4 parallel in-house jobs, 2 parallel real-verifier jobs). **Machine:** MacBook Pro
(Apple Silicon, 8 GB), CPU only, plugged in. **Verifiers:** abcrown, pyrat, nnenum, neuralsat,
marabou 2.0.0 (source build). Card run takes about 32 min. **Result: VERIFIED**; metrics in
`verdict.json`: scoring accuracy 1.0, buggy verifiers caught 1.0.

Benchmark: 50 instances (24 UNSAT, 26 SAT, including 4 attack-hard tiny-CNN SAT instances where
PGD fails and only a completed proof procedure decides the instance).

| Metric | Result |
|---|---|
| Scoring accuracy (BAA metric) | **2025/2025 verdicts (100%)** |
| Correct verdicts wrongly rejected / wrong verdicts wrongly accepted | 0 / 0 |
| Buggy verifiers caught | **39/39 fired bugs (100%)** (42 planted; 3 never fired) |
| Sound verifiers wrongly flagged | 0 |

Ground-truth SAT labels are re-certified from their witnesses on every run, so the 100% is
checked, not assumed.

**Compared with existing benchmarks** (same pool on VNN-COMP `mnist_fc`, 30/90 instances, and
`oval21`, each scored with the best labels it has; details in
[experiments/external_benchmarks](experiments/external_benchmarks/README.md)):

| Benchmark | Bugs that fire | Caught by own labels | Caught by own labels + majority vote | Verdicts judged with certainty |
|---|---|---|---|---|
| VeriStressGT | 39/42 | **39/42** | 39/42 | **100%** |
| `mnist_fc` (30/90) | 19/42 | 6/42 | 14/42 | 21% |
| `oval21` | 19/42 | 2/42 | 9/42 | 7% |

- **Real finding: Marabou 2.0.0 returns false SAT on provably robust instances**, flagged by
  ground truth on `meap_02`, `meap_03` and `milp_s2_a` in this run. On `meap_*` its assignment
  leaves the true class 0.001 ahead: the constraint tolerance absorbs MEAP's certified margin
  (gamma = 0.001). On `milp_s2_a` (eps = 0.99 r\*) the output values Marabou reports at its own
  assignment disagree with evaluating the network there (onnxruntime: true class ahead by 0.0036).
  Marabou 2.0.0 is also **not deterministic**: the same command on `milp_s2_a` returns UNSAT on
  some runs and SAT on others. Which of these instances Marabou gets wrong can therefore vary
  between runs; the ground-truth scoring (100%) does not. Matches the paper's Table 4, where
  Marabou also made false SAT claims.
- **Not exercised:** `pyrat+crash_as_sat` (pyrat never crashed), and
  `abcrown+/nnenum+attention_unsat_to_sat` (neither produced an UNSAT on an attention model to
  flip).
- **Hard-to-expose bugs:** `mutant:conv_bias_dropped` and `mutant:weights_fp16` only produce wrong
  answers when the attack fails and the proof procedure decides, which happens on the attack-hard
  CNN instances. `mutant:hwc_layout` goes wrong on a single instance (`rcnn_2_sat_near`), the only
  multi-channel SAT instance whose scrambled box excludes every counterexample.
- **pyrat false UNSAT (prove-only mode):** with counterexample search off, unmodified pyrat
  (`con_z`, torch float32) reported `tcnn_2_sat_near` as robust. Exact rational arithmetic on the
  model's weights confirms the stored witness lies in the box with margin -2.99e-6. With the attack
  enabled, pyrat finds the counterexample first.

**Scaling check** (`scale: 2`, sound controls only, restricted Gurobi license): 98 instances
(48 UNSAT, 50 SAT, all 50 witnesses re-certified), ground-truth scoring 173/173, no control
flagged; build + run 9 min.

![Thrust 1 verdict matrix](assets/thrust1_verdict_matrix.png)

## Thrust 2: per-verifier timeout prediction

```bash
magnet evaluate cards/thrust2_timeout_prediction.yaml
```

One predictor per verifier: elastic-net logistic regression on network size/type features plus
the five Difficulty-Profile components (`src/VeriStressGT/prediction/`).

1. **Recorded, held-out (primary).** The training data is the paper's server runs
   (`src/VeriStressGT/prediction/data/training_rows.csv`): 345 instances (225 synthetic, 120
   VNN-COMP mnist_fc / oval21) × abcrown, neuralsat, marabou, nnenum, pyrat. Every split is grouped
   by network, so no network is on both sides; this matters because VNN-COMP reuses 6 networks
   across 120 properties. Results are the mean over 10 grouped splits. Labels use a 120 s horizon
   (timed out, or solved only after 120 s), matching the live runs.
2. **Live out-of-distribution (reported).** `thrust2_ood_bench/` holds 30 fresh networks with
   seeds and parameters outside the training runs. The card computes their features on the fly,
   runs abcrown, pyrat, nnenum, neuralsat and marabou at a 120 s timeout, and scores predictors fitted on all recorded
   data.
3. **Environment diagnostic.** For each live verifier, the card also re-runs 12 recorded anchor
   instances from the hard end of its recorded runtimes. It fits local_s = overhead + slope[arch] ×
   recorded_s, which shows how far that verifier's speed has shifted from the recording environment.
   An anchored model is also reported, but is exploratory: with this few anchors it was not
   reliably better than the uncalibrated one.

**Claim.** Every verifier's mean held-out AUC is >= 0.7. Per the eval plan, this is not aligned
to the BAA's 95% goal.

Features are computed exactly as for the training table: size/type from the ONNX graph, and the
profile via `estimate_profile(..., atau_n_samples=600)`, with `a_tau` as the log-count of distinct
local fingerprints (paper Eq. 14). `tests/test_prediction.py` checks the recomputed features
against the table.

**Run it on an awake, plugged-in machine.** Runtimes are measured with a monotonic clock, so a
sleeping laptop does not inflate them, but a suspended run does not progress either.

> **Stale for Marabou:** the sample results below were run with maraboupy 1.0.0, before Marabou
> was pinned to 2.0.0. Re-run this card before quoting Marabou's live numbers.

### Sample run results

**Machine:** MacBook Pro (Apple Silicon), CPU only, plugged in and awake. **Config:** card
defaults (120 s horizon, five live verifiers, Marabou 2.0.0). Full run about 70 min. Model fits and
profile features are seeded, so re-scoring the same live runs reproduces every number exactly.

| Verifier | Held-out AUC, size + profile | Size only | Profile only | Live OOD AUC (timeouts) |
|---|---|---|---|---|
| abcrown | **0.825** ± 0.047 | 0.754 | 0.825 | **0.865** (4/30) |
| neuralsat | **0.902** ± 0.042 | 0.861 | 0.745 | **0.926** (9/27) |
| marabou | **0.768** ± 0.058 | 0.675 | 0.774 | **0.872** (12/27) |
| nnenum | **0.897** ± 0.064 | 0.755 | 0.844 | **0.963** (6/15) |
| pyrat | **0.920** ± 0.026 | 0.864 | 0.851 | **0.890** (10/30) |

Live instances a verifier cannot handle (errors, e.g. nnenum and Marabou on attention models) are
excluded from its live AUC; the denominators above show how many were scored.

- All five verifiers clear 0.7 on held-out networks, and all five predictors transfer to fresh
  networks run live on a different machine (0.87-0.96).
- Marabou's live AUC was 0.622 with maraboupy 1.0.0 (the only macOS wheel); with 2.0.0, the
  version the recorded server runs used, it is 0.872.
- The anchors show how much the environment matters. Local abcrown solves the server's attention
  anchors (142-154 s there) in 7-8 s, about 20x faster. That is too fast for the slope to be
  identifiable, and the same holds for NeuralSAT. pyrat's slope is about 0.6-1.0, nnenum's
  0.6-1.0, Marabou's 0.46. At 120 s the predictor is less exposed to this speed shift than at 60 s.
- The anchored models were not better (pyrat 0.835 vs 0.890; nnenum and Marabou unchanged), so the
  uncalibrated live AUC is the headline number.

![Thrust 2 timeout AUC](assets/thrust2_auc.png)

## Mini sweep (supporting card)

### Run
```bash
magnet evaluate cards/evaluation.yaml
```

### Expected Results

**Output files** written alongside `results.json`:

| File | Description |
|------|-------------|
| `mini_sweep_summary.csv` | Per-verifier solved/total counts and correct fraction |
| `construction_heatmap.png` | Solved/total per construction × verifier, with avg wall time |
| `timeout_auc_heatmap.png` | Timeout AUC per difficulty component × verifier |
| `timeout_auc.csv` | Numeric timeout AUC values |

**1. Correctness invariant:** Every instance is UNSAT by construction (provably robust). No verifier should ever return SAT — a SAT result indicates a soundness error in that verifier. Some verifiers will timeout or return error (i.e. unsupported) results.

**2. Timeout AUC Values Typically Above 0.5:** For each (component, verifier) pair, timeout AUC is the probability that a randomly chosen timed-out instance has a higher component value than a randomly chosen solved instance (AUROC of the component as a timeout predictor). A value > 0.5 means harder instances — as measured by that component — are more likely to cause a timeout, which is the expected direction. Values near 0.5 indicate the component doesn't predict difficulty for that verifier. We expect the majority of verifier, component combinations to have a value over 0.5.

---

### Sample Run Results

**Machine:** MacBook Pro, Apple Silicon, 8 GB RAM, macOS Sonoma 14.7.5  
**Config:** 50 instances × {pyrat, nnenum}, 60s timeout per instance

**Verifier summary:**

| Verifier | UNSAT | Timeout | Error | SAT |
|----------|------:|--------:|------:|----:|
| alpha,beta-crown    | 49   | 1      | 0     | 0   |
| pyrat    | 45    | 4       | 1     | 0   |
| nnenum   | 13     | 10       | 27    | 0   |

No verifier returned SAT. nnenum errors are structural (unsupported op types on meap, corners, and attention constructions) rather than difficulty-related. pyrat timeouts occur on the hardest `pb_cnn` (large `num_pairs`) and `milp`/`meap` instances at the top of the difficulty range.

**Timeout AUC heatmap** (pyrat + nnenum, 60s timeout):

![Timeout AUC Heatmap](assets/timeout_auc_heatmap.png)
