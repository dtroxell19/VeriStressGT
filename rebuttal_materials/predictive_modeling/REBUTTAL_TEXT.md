# Rebuttal-ready text — per-instance predictive value of the Difficulty Profile

Three lengths. All numbers are held-out, from repeated network-grouped cross-validation
(StratifiedGroupKFold, 5 folds × 20 repeats), per verifier, comparing size/type features against
size/type + the five Difficulty-Profile components on identical folds. Language is deliberately
non-causal.

---

## SHORT (~110 words)
To address the concern that the Difficulty Profile is population-level and does not predict a
*particular* instance, we built an interpretable per-verifier timeout predictor and evaluated it with
repeated network-grouped cross-validation (grouping by ONNX identity to prevent leakage). Comparing
network size/type features against size/type + the five profile components on identical folds, the
profile adds significant held-out AUC for 4/5 verifiers when the benchmarks are pooled (231 networks;
ΔAUC +0.05 to +0.20, CIs above zero) and for the complete verifiers Marabou and nnenum on synthetic
(ΔAUC +0.11). The IBP relative gap and
unstable-fraction are the stable drivers. Where size already predicts timeouts, the profile improves
calibration. We report this heterogeneity rather than a single score.

## MEDIUM (~210 words)
Reviewers asked whether the Difficulty Profile predicts the outcome of an individual verification
instance, not just population trends. We trained a separate interpretable model (elastic-net logistic
regression) per verifier and evaluated timeout prediction with repeated **network-grouped**
cross-validation (5×20; groups = ONNX identity, so two properties of the same network never span
train/test). For every verifier and scenario we compare, on identical folds, **size/type features**
against **size/type + the five profile components**, and report the paired held-out ΔAUC with a 95%
group-bootstrap confidence interval.

Pooling synthetic (225 nets) and established benchmarks (231 networks total), the profile adds
significant value for **4 of 5 verifiers** — ΔAUC +0.056 (abcrown), +0.082 (neuralsat), +0.151
(nnenum), +0.200 (marabou), all CIs above zero; only pyrat is inconclusive. On synthetic alone, the
complete verifiers **Marabou (0.87→0.97)** and **nnenum (0.82→0.93)** gain +0.11 AUC and roughly half
their Brier score, while for the strong branch-and-bound verifiers (abcrown, neuralsat, pyrat) size
already predicts timeouts (AUC 0.86–0.97) and the profile instead improves calibration. Ablations and
coefficients single out the **IBP relative gap** (odds ratios up to 60–110× per SD) and
**unstable-fraction** as the stable predictors. Base rates (0.13–0.42) are reported with every cell.

## DETAILED (~430 words)
Several reviewers and the AC noted that our Difficulty-Profile analysis is population-level and does
not show that it predicts the outcome of a *particular* instance. We therefore built an interpretable
per-instance predictor and evaluated it rigorously. For each verifier we fit an elastic-net logistic
regression and predict whether that verifier times out on an instance, using **repeated
network-grouped cross-validation** (StratifiedGroupKFold, 5 folds × 20 repeats, seeds fixed). Groups
are network identity (ONNX content hash), so no two properties of the same network appear in both train
and test; all imputation, standardization, and hyperparameter tuning happen inside the training fold,
and decision thresholds are chosen from inner-training predictions only. Our pre-registered comparison
is, on **identical folds**, network **size/type** features (input/output dimension, parameter and
node/operator counts, depth, width, architecture type from the graph) versus **size/type + the five
profile components**. The headline quantity is the paired held-out ΔAUC with a 95% group-bootstrap
interval over networks.

Results are heterogeneous and we report them as such. When synthetic constructor networks (225) and
established VNN-COMP benchmarks are **pooled** (231 networks), the profile adds significant held-out
value for **four of five verifiers**: ΔAUC +0.056 (abcrown), +0.082 (neuralsat), +0.151 (nnenum),
+0.200 (marabou), each 95% CI above zero; pyrat is positive but inconclusive (+0.046, CI grazes zero).
Here size-only is comparatively weak (AUC 0.60–0.83) because it must span two very different size
regimes. On synthetic-only, the **complete verifiers Marabou (AUC 0.87→0.97) and nnenum (0.82→0.93)**
gain +0.11 AUC with roughly halved Brier, whereas the strong branch-and-bound verifiers already reach
AUC 0.86–0.97 from size alone — there the profile adds no AUC but consistently **improves calibration**
(e.g. Brier 0.12→0.09) and improves solved-instance **runtime ranking** (Spearman +0.05 to +0.32).
Cross-domain transfer is the sharpest evidence: size features trained on one domain transfer at or
below chance to the other (AUC 0.29–0.50), and adding the profile restores discrimination (e.g. Marabou
0.50→0.86, nnenum 0.29→0.74), showing the profile captures architecture-agnostic difficulty. Ablations
and standardized coefficients identify the **IBP relative gap** (odds ratios 60–110× per standard
deviation, sign-consistent in every fold) and **unstable-fraction** as the stable drivers, with a
stable **G_IBP × network-size interaction** — so the contribution is not merely a proxy for scale.

We are explicit about limits: the established benchmarks contribute only six distinct networks, so that
slice is exploratory (its large point estimates come with wide intervals); the runtime analysis is
selection-biased; and all claims are predictive, not causal. Net: the profile provides per-instance
information beyond ordinary scale descriptors for the majority of verifiers, and most where size is
insufficient.
