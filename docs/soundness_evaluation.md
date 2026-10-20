# Soundness evaluation with ground-truth labels

VeriStress-GT instances carry certified labels, so a verifier's SAT/UNSAT verdicts can be checked
against the truth instead of against other verifiers. `VeriStressGT-soundness` runs a pool of
verifiers, including deliberately buggy ones, on a labelled benchmark and reports which bugs the
labels expose.

```bash
VeriStressGT-soundness --out_dir runs/soundness --timeout 60
# subset / partial environments:
VeriStressGT-soundness --real_verifiers abcrown pyrat --allow_missing_verifiers
```

Outputs in `--out_dir`: `results.json` (summary, per-verifier flags, every verdict),
`verifiers.csv`, `evidence.json` (the instances behind each flag) and `verdict_matrix.png`
(verifier × instance: correct / wrong / undecided).

## Benchmark: `src/VeriStressGT/benchmarks/soundness_gt`

Built from `src/VeriStressGT/configs/soundness_gt.yaml` plus
`python -m VeriStressGT.soundness.ground_truth` (`VeriStressGT-soundness --rebuild` does both;
`--scale N` builds an N-times larger, re-seeded copy):

- 24 **UNSAT** instances, labelled by the constructors' analytic certificates (MILP exact radius,
  MEAP, corners, paired-bias CNN, contractive CNN, linear / softmax attention), including
  near-boundary MILP radii (eps = 0.999 r\*, 0.9999 r\*).
- 26 **SAT** instances, each shipping a concrete counterexample (`witness.json`) whose margin is
  negative in both float64 and onnxruntime float32, so no verifier has to be trusted for these
  labels. MEAP and paired-bias networks are robust at every radius, so small random CNNs
  (`rcnn_*`, `tcnn_*`) supply the CNN counterexamples. Four of them (`tcnn_*_sat_hard`) sit so
  close to the threshold that PGD cannot find the counterexample but exact MILP can, so bugs in a
  verifier's proof procedure cannot hide behind a successful attack.

Every SAT witness is re-certified at the start of each run.

## Verifier pool

| Role | Verifiers |
|------|-----------|
| Sound controls | `reference` (in-house `refverify`: PGD attack, then CROWN bounds, then exact MILP via scipy/HiGHS or ReLU-split BaB), `ibp_only` |
| Planted, tier (b) | 12 mutants of the reference with one injected bug each. Bound computation: `ibp_sign`, `relu_no_intercept`, `conv_bias_dropped`. Numerics: `tol_unsat`, `tol_sat`, `weights_fp16`. Search logic: `disjunct_last`, `bab_any_row`. Input/spec handling: `input_clip01`, `eps_half`, `hwc_layout`, `label_off_by_one` |
| Real | `abcrown`, `pyrat`, `nnenum`, `neuralsat`, `marabou` |
| Planted, tier (a) | 6 output-level faults on each real verifier: `timeout_as_unsat`, `crash_as_sat`, `conv_sat_to_unsat`, `attention_unsat_to_sat`, `cache_collision` (results cached by network file, ignoring the spec), `flaky` (nondeterministic flips on ~15% of instances) |

Bug descriptions and the failure classes they model are in `src/VeriStressGT/soundness/bugs.py`.
The reference verifier also runs on its own:
`python -m VeriStressGT.cli.verify_benchmark --verifier refverify --refverify_bug <bug> ...`.

## Scoring

A verifier is flagged when any SAT/UNSAT verdict contradicts the label; timeouts and errors never
flag. For a planted verifier, a flag counts only on instances where the bug changed the verdict of
the verifier it was derived from (inherited disagreements are listed as `gt_inherited`), and
detection rates count only bugs that changed at least one verdict.

On benchmarks without a label for every instance, verdicts are also scored with the benchmark's
labels plus majority vote on the unlabelled instances (`labels_plus_majority`), and `coverage`
counts how many verdicts can be judged with certainty.

## Sample results

60 s timeout, 4 parallel in-house jobs, 2 parallel real-verifier jobs; MacBook Pro (Apple Silicon,
8 GB), CPU only; abcrown, pyrat, nnenum, neuralsat, Marabou 2.0.0. About 32 min.

| Metric | Result |
|---|---|
| Verdicts scored correctly by the labels | **2025/2025 (100%)** |
| Correct verdicts wrongly rejected / wrong verdicts wrongly accepted | 0 / 0 |
| Planted bugs caught | **39/39 that fired** (42 planted; 3 never fired) |
| Sound verifiers wrongly flagged | 0 |

- **Marabou 2.0.0 returns false SAT on provably robust instances** (`meap_02`, `meap_03`,
  `milp_s2_a` in this run). On `meap_*` its assignment leaves the true class 0.001 ahead: the
  constraint tolerance absorbs MEAP's certified margin (gamma = 0.001). On `milp_s2_a`
  (eps = 0.99 r\*) the outputs Marabou reports at its own assignment disagree with evaluating the
  network there (onnxruntime: true class ahead by 0.0036). Marabou 2.0.0 is also not
  deterministic on `milp_s2_a` (UNSAT on some runs, SAT on others).
- **Not exercised:** `pyrat+crash_as_sat` (pyrat never crashed) and
  `abcrown+/nnenum+attention_unsat_to_sat` (neither produced an UNSAT on an attention model).
- **Hard-to-expose bugs:** `conv_bias_dropped` and `weights_fp16` only produce wrong answers when
  the attack fails and the proof procedure decides, i.e. on the attack-hard CNN instances.
  `hwc_layout` goes wrong on a single instance (`rcnn_2_sat_near`).
- **pyrat false UNSAT in prove-only mode:** with counterexample search off, pyrat (`con_z`, torch
  float32) reported `tcnn_2_sat_near` as robust; exact rational arithmetic confirms the stored
  witness lies in the box with margin -2.99e-6. With the attack enabled (the default here), pyrat
  finds the counterexample first.

Scaling check (`--scale 2`, sound controls only): 98 instances (48 UNSAT, 50 SAT, all witnesses
re-certified), 173/173 verdicts scored correctly, no control flagged; build + run 9 min.
