## AIQ Flow
To setup/run, complete the following:

### Benchmark Setup
```bash
git clone --recursive git@github.com:dtroxell19/VeriStressGT.git
cd VeriStressGT
git fetch origin
git checkout -b evaluation-card origin/evaluation-card
bash scripts/bootstrap.sh
conda activate VeriStressGT
```

### MAGNET Setup
```bash
pip install git+https://github.com/AIQ-Kitware/aiq-magnet.git

# Additional deps needed by Gurobi-based constructions
pip install sortedcontainers coloredlogs termcolor beartype gurobipy 
```

### α-β-CROWN Environment Variables
From root directory of repo:
```bash
export ABCROWN_VNNCOMP2024_DIR="$(pwd)/src/VeriStressGT/verifiers/alpha-beta-CROWN"
export ABCROWN_CONDA_ENV="alpha-beta-crown"
```

### Evaluation cards

| Card | Thrust | Metric |
|------|--------|--------|
| `cards/thrust1_soundness.yaml` | 1 (primary) | Fraction of planted buggy verifiers exposed by ground-truth labels (vs. majority vote) |
| `cards/evaluation.yaml` | 2 (diagnostic) | Mini sweep: per-verifier correctness + timeout AUC of Difficulty Profile components |

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
- 22 **SAT** instances, each shipping a concrete counterexample (`witness.json`) whose margin is
  negative in both float64 and onnxruntime float32, so no verifier has to be trusted for these
  labels. They are "twins" of robust networks with the radius scaled just past (`_sat_near`) or
  well past (`_sat_far`) the robustness threshold. MEAP and paired-bias networks are robust at
  every radius, so small random CNNs (`rcnn_*`, `tcnn_*`) supply the CNN counterexamples.

Without SAT instances, a verifier that always answers "robust" would score perfectly. They are
what exposes false UNSAT claims, the dangerous direction.

**Verifier pool.**

| Role | Verifiers |
|------|-----------|
| Sound controls | `reference` (in-house: PGD attack, then CROWN bounds, then exact MILP via scipy/HiGHS or ReLU-split BaB), `ibp_only` |
| Planted, tier (b) | 8 mutants of the reference with one injected bug each: `ibp_sign`, `relu_no_intercept`, `conv_bias_dropped`, `tol_unsat`, `tol_sat`, `disjunct_last` (the Appendix D bug), `input_clip01`, `eps_half` |
| Real | `abcrown`, `pyrat` (skipped if not installed) |
| Planted, tier (a) | output-level faults on each real verifier: `timeout_as_unsat`, `crash_as_sat` (the rebuttal-era parser bug), `conv_sat_to_unsat`, `attention_unsat_to_sat` |

Bug descriptions and the failure classes they model are in `src/VeriStressGT/soundness/bugs.py`.

**Scoring.** A verifier is flagged when any SAT/UNSAT verdict contradicts the label;
timeouts and errors never flag. The same verdicts are scored three ways:

- `ground_truth`: the benchmark labels.
- `majority_one_buggy`: VNN-COMP-style majority vote over one planted verifier plus every
  honest verifier. Ties are unresolved; an instance nobody decides is assumed robust.
- `majority_full`: majority vote over the whole pool.

**Claim.** Ground truth flags at least `detection_threshold` (0.75) of the planted verifiers and
flags no sound control.

For planted verifiers, a flag counts only on instances where the planted bug changed the verdict
of the verifier it was derived from. Disagreements it merely inherits from that verifier are listed
separately as `gt_inherited`.

The real verifiers run with counterexample search enabled (`configs/abcrown_thrust1.yaml` sets
`pgd_order: before`, and pyrat gets `--check both --attack pgd`). With the repo's default settings
both only try to prove UNSAT and never report SAT.

**Outputs** (next to `results.json`): `thrust1_verifiers.csv` (per-verifier flags),
`thrust1_evidence.json` (the instances that expose each flag), `thrust1_verdict_matrix.png`
(verifier × instance, correct / wrong / undecided).

### Sample run results

**Machine:** MacBook Pro (Apple Silicon), CPU only. **Config:** card defaults (60 s timeout,
4 parallel in-house jobs, 2 parallel real-verifier jobs). The full card run takes roughly 15-20 min.

| Labels used for scoring | Planted bugs detected | Sound controls flagged |
|---|---|---|
| Ground truth | **13/16 (81%)**; 13/14 of the bugs that changed any verdict | none |
| Majority vote, one buggy verifier in the pool | 11/16 (69%) | none |
| Majority vote, full pool | 12/16 (75%) | **both** (`reference`, `ibp_only`) |

- **Ground truth missed 3:** `mutant:conv_bias_dropped` never produced a wrong verdict (it only
  turned proofs into UNKNOWN). `abcrown+attention_unsat_to_sat` and `pyrat+crash_as_sat` never
  fired, because abcrown's attack crashes on the attention models and pyrat never errored. No
  output-based method can catch these three on this benchmark.
- **Majority vote, one buggy verifier:** it also misses `abcrown+crash_as_sat` and
  `pyrat+attention_unsat_to_sat`. Their false SATs land on attention instances where the only
  other decisive verifier is the one real tool that supports them, so the vote ties and nobody is
  flagged.
- **Majority vote, full pool:** the planted false-UNSAT verdicts outvote the truth on 5
  near-threshold SAT instances. Both sound controls are then flagged for correct answers, and
  `abcrown+timeout_as_unsat` goes undetected.
- **pyrat false UNSAT (prove-only mode):** with counterexample search off, unmodified pyrat
  (`con_z`, torch float32) reported `tcnn_2_sat_near` as robust. Exact rational arithmetic on the
  model's weights confirms the stored witness lies in the box with margin -2.99e-6. With the attack
  enabled, pyrat finds the counterexample first.

![Thrust 1 verdict matrix](assets/thrust1_verdict_matrix.png)

## Thrust 2: mini sweep

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
