# Part B: scale + arbitrary-architecture demonstration (real backbone, 128x128)

**The Paired-Bias ground-truth certificate is backbone-agnostic and O(1)**, so it composes with a real, heterogeneous, downsampling CNN backbone (strided conv + BatchNorm + MaxPool) at **49,152-dim input** — 16x a 32x32 image, ~230x MNIST-8x8 — with no solver and cost independent of backbone size. This answers WvAC's *'scales to real-world models'* and *'not tied to specific architectures'*.

- backbone: 3x128x128 -> 2048 features, trained on CIFAR10 (real photos, resized to 128x128; held-out test acc 0.440).
- ground-truth construction: **0.01s/instance** (analytic paired-bias certificate, O(1) in backbone size — MILP GT is already 231s on a 512-ReLU MNIST net and does not scale here).
- **Difficulty Profiles remain computable at scale**: autograd gradient path profiles each 49k-dim instance in **0.05s** (one backward per sample-batch; finite differences would need ~49,152 forward passes per gradient). Answers JJNS Q4.
- ONNX/ORT parity: max 6.4e-08; every instance is analytically robust (UNSAT) with all pairs unstable near x0 (a genuine stress test).

## Per-instance

| instance | input_dim | feat_dim | GT (s) | profile (s) | parity | unstable frac | margin@x0 |
|---|---|---|---|---|---|---|---|
| celeba_000 | 49,152 | 2048 | 0.011 | 0.057 | 6.4e-08 | 1.0 | 0.1 |
| celeba_001 | 49,152 | 2048 | 0.01 | 0.044 | 3.2e-08 | 1.0 | 0.1 |
| celeba_002 | 49,152 | 2048 | 0.013 | 0.062 | 3.5e-08 | 1.0 | 0.1 |
| celeba_003 | 49,152 | 2048 | 0.009 | 0.047 | 5.2e-08 | 1.0 | 0.1 |
| celeba_004 | 49,152 | 2048 | 0.011 | 0.046 | 4.0e-08 | 1.0 | 0.1 |
| celeba_005 | 49,152 | 2048 | 0.009 | 0.046 | 3.1e-08 | 1.0 | 0.1 |
| celeba_006 | 49,152 | 2048 | 0.01 | 0.044 | 4.2e-08 | 1.0 | 0.1 |
| celeba_007 | 49,152 | 2048 | 0.009 | 0.045 | 4.8e-08 | 1.0 | 0.1 |

## Scale ladder

MNIST-8x8 (192-dim, `realism_smoke`) -> MNIST-28x28 (784-dim, `real_networks/paired_bias_real`) -> **CIFAR-10 @ 128x128 (49,152-dim, here)**. The analytic GT cost is flat across the ladder; only profiling grows (~linearly), and the autograd path keeps it feasible. (Instance ids retain a legacy `celeba_` prefix; the backbone is a CIFAR-10-trained CNN upsampled to 128x128.)

*Verify:* `python rebuttal_materials/scale_generality/../../src/VeriStressGT/cli/verify_benchmark.py --benchmark rebuttal_materials/scale_generality/celeba_bench --verifier abcrown --out_dir rebuttal_materials/scale_generality/verify_abcrown --timeout 600 --overwrite` (verifiers strain at this scale — expected; the point is that GT + profiles are constructible).
