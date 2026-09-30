# Thrust 2 card: per-verifier timeout prediction

[Back to AIQ_README](../AIQ_README.md)

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

## Sample run results

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

![Thrust 2 timeout AUC](../assets/thrust2_auc.png)
