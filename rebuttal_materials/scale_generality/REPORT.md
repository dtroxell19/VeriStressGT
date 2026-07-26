# Part B: scale + arbitrary-architecture demonstration (CelebA 128x128)

**The Paired-Bias ground-truth certificate is backbone-agnostic and O(1)**, so it composes with a real, heterogeneous, downsampling CelebA CNN backbone (strided conv + BatchNorm + MaxPool) at **49,152-dim input** — 16x CIFAR, ~230x MNIST-8x8 — with no solver and cost independent of backbone size. This answers WvAC's *'scales to real-world models'* and *'not tied to specific architectures'*.

- backbone: 3x128x128 -> 2048 features, untrained (server run trains it; certificate is backbone-agnostic either way).
- ground-truth construction: **0.01s/instance** (analytic paired-bias certificate, O(1) in backbone size — MILP GT is already 231s on a 512-ReLU MNIST net and does not scale here).
- **Difficulty Profiles remain computable at scale**: autograd gradient path profiles each 49k-dim instance in **0.10s** (one backward per sample-batch; finite differences would need ~49,152 forward passes per gradient). Answers JJNS Q4.
- ONNX/ORT parity: max 2.0e-08; every instance is analytically robust (UNSAT) with all pairs unstable near x0 (a genuine stress test).

## Per-instance

| instance | input_dim | feat_dim | GT (s) | profile (s) | parity | unstable frac | margin@x0 |
|---|---|---|---|---|---|---|---|
| celeba_000 | 49,152 | 2048 | 0.019 | 0.154 | 2.0e-08 | 1.0 | 0.1 |
| celeba_001 | 49,152 | 2048 | 0.005 | 0.073 | 1.8e-08 | 1.0 | 0.1 |
| celeba_002 | 49,152 | 2048 | 0.006 | 0.083 | 1.8e-08 | 1.0 | 0.1 |

## Scale ladder

MNIST-8x8 (192-dim, `realism_smoke`) -> MNIST-28x28 (784-dim, `real_networks/paired_bias_real`) -> **CelebA-128x128 (49,152-dim, here)**. The analytic GT cost is flat across the ladder; only profiling grows (~linearly), and the autograd path keeps it feasible.

*Verify:* `python -m VeriStressGT.cli.verify_benchmark --benchmark rebuttal_materials/scale_generality/celeba_bench --verifier abcrown ...` (verifiers strain at this scale — expected; the point is that GT + profiles are constructible).
