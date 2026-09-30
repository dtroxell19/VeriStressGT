# Thrust 1 card: bug detection with ground-truth labels

[Back to AIQ_README](../AIQ_README.md)

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
  labels. MEAP and paired-bias networks are robust at every radius, so small random CNNs (`rcnn_*`, `tcnn_*`) supply the CNN counterexamples. Four of them (`tcnn_*_sat_hard`) sit so close to the threshold that PGD cannot find the
  counterexample but exact MILP can, so bugs in a verifier's proof procedure cannot hide behind a
  successful attack.

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
of the verifier it was derived from. Disagreements it inherits from that verifier are listed
separately as `gt_inherited`. Detection rates count only bugs
that changed at least one verdict.

**Outputs** (next to `results.json`): `thrust1_verifiers.csv` (per-verifier flags),
`thrust1_evidence.json` (the instances that expose each flag), `thrust1_verdict_matrix.png`
(verifier × instance, correct / wrong / undecided).

## Sample run results

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
[experiments/external_benchmarks](../experiments/external_benchmarks/README.md)):

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

![Thrust 1 verdict matrix](../assets/thrust1_verdict_matrix.png)
