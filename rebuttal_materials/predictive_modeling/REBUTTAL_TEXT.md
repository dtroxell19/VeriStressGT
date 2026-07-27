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

## MEDIUM (~350 words)
Reviewers raised a fair question: the Difficulty Profile explains verifier timeouts *in aggregate*, but
does it tell you anything about a *single* instance? To find out, we built a deliberately simple,
interpretable predictor — one elastic-net logistic regression per verifier — that predicts whether that
verifier will time out on a given instance. We scored it honestly, with repeated cross-validation (5
folds × 20 repeats) that groups by network identity, so two properties of the same network never land on
opposite sides of the train/test split. On identical folds we compared ordinary network descriptors
(input/output size, depth, width, parameter count, architecture type) *alone* against those same
descriptors *plus* the five profile components, and measured the paired gain in held-out AUC with a 95%
network-bootstrap interval.

Pooling our synthetic networks with established VNN-COMP benchmarks (231 networks in all), the profile
adds genuinely significant predictive power for four of five verifiers — +0.056 AUC for α,β-CROWN, +0.082
for NeuralSAT, +0.151 for nnenum, +0.200 for Marabou, every interval clear of zero; only PyRAT, whose
accuracy is already at ceiling, is inconclusive. On the synthetic networks alone the two *complete*
verifiers benefit most (Marabou 0.87→0.97, nnenum 0.82→0.93, ~half the Brier error), while for the
branch-and-bound verifiers size already predicts timeouts and the profile instead sharpens their
calibration. Two components do most of the work: the IBP relative gap and the unstable fraction.

One objection still stands — any model handed five extra variables can look better by luck. So we ran a
real significance test: a parametric bootstrap adapted from the "actionable-features" analysis of a
published excess-mortality study. We treat the network descriptors as fixed *intrinsic* features and the
profile as the *actionable* block under test, generate null datasets whose outcomes depend on the
intrinsic features only, and ask how often the profile's measured gain could arise by chance. Pooled
across all instances and all five verifiers in a single gradient-boosted model (verifier identity
included as a covariate), the profile improves held-out prediction by **ΔAUC +0.107 — a gain not one of
1,000 null draws reached (p = 0.001; the null 95% ceiling was only +0.020)**. Run per verifier, it is
significant at the bootstrap floor in 7 of 10 verifier×dataset cells and significant in 8 of 10, with
PyRAT the one honest exception — and it even recovers nonlinear signal the linear model missed (abcrown
on synthetic, flat under logistic regression, becomes clearly significant once a nonlinear model can use
the profile). Finally, a per-instance effect δ = P(timeout | size + profile) − P(timeout | size), with
bootstrap intervals, shows *which* instances the profile flags: those it pushes toward timeout have
visibly higher IBP gap and unstable fraction. The three tests agree — the profile carries real
per-instance information beyond raw network scale, not an artifact of extra parameters.

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
