# Experiments

Supporting studies for VeriStress-GT's claims, beyond the main benchmark runs. Each folder has its
own README or REPORT with the full results. Run everything from the repository root with
`PYTHONPATH=src`. Most of these need `pip install -e '.[experiments]'`.

| Folder | Question | Headline result | Reproduce |
|---|---|---|---|
| [`sound_arithmetic/`](sound_arithmetic/REPORT.md) | Are the ground-truth margins sound beyond floating point? | 172/225 shipped margins re-certified > 0 in exact-rational or validated-interval arithmetic; 12/12 convex polynomial companions certified exactly | `gen_poly_convex.py`, then `sound_recertify.py` (about 3 min) |
| [`predictive_modeling/`](predictive_modeling/README.md) | Does the Difficulty Profile predict verifier timeouts beyond network size and type? | Profile adds AUC over size-only models for 4/5 verifiers in the combined scenario (+0.05 to +0.20) | `src/run_all.py` from the frozen row table |
| [`profile_sensitivity/`](profile_sensitivity/README.md) | How sensitive are the profile components to their hyperparameters (η, τ, N, sampling distribution)? | Recommended defaults; component dependence matrix; unified Difficulty Index | `run_sensitivity.sh` (recompute sweeps are server-scale) |
| [`activation_regions/`](activation_regions/REPORT.md) | Does the A_τ grid proxy track the exact ReLU activation-region count? | Spearman 0.705 against exact enumeration | `count_regions.py` |
| [`realism_stats/`](realism_stats/REPORT.md) | Do constructed instances look like real verified-robust networks? | Real networks fall inside the constructed profile range (coverage 81–100% per component) | `profile_distribution_tests.py`, `realism_pt2.py` |
| [`scale_generality/`](scale_generality/REPORT.md) | Does the paired-bias certificate scale and compose with an arbitrary backbone? | 49,152-dim input: ground truth in 0.01 s/instance, profile in 0.05 s | `run_celeba_scale.sh` (downloads CIFAR-10 / STL-10) |
| [`paper_figures/`](paper_figures/README.md) | Figures 2 and 3 of the paper, and the same figures with the convex polynomial set replacing the screened one | `paper/` reproduces the published figures exactly | `make_figures.sh` |
| [`gpu_timing/`](gpu_timing/GPU_TIMEOUT_PATTERNS.md) | Do CPU timeout patterns and soundness persist on GPU? | GPU rescales runtimes but keeps the hard core; no new soundness flips ([GPU_SOUNDNESS.md](gpu_timing/GPU_SOUNDNESS.md)) | results only (see below) |

## Shared inputs

`merged_records_constructed_and_real.json` holds the merged per-instance verifier records and
Difficulty Profiles for the constructed and VNN-COMP benchmarks (345 instances); its sha256 matches
`predictive_modeling/data/manifests/source_manifest.json`. `activation_regions/count_regions.py`,
`gpu_timing/analyze_cpu_vs_gpu.py` and `realism_stats/profile_distribution_tests.py` read it and
reproduce their committed outputs. The GPU runs `gpu_timing` compares against are in
`src/VeriStressGT/runs_standardized/*_veristress_gpu`.

## Inputs that are not in the repository

- The VNN-COMP ONNX models. `predictive_modeling/src/build_dataset.py` extracts network features
  from them, so rebuilding the frozen row table needs the VNN-COMP 2022 `mnist_fc` and `oval21`
  models; the rest of the pipeline re-runs from the committed
  `data/processed/instance_verifier_rows.parquet`.
- For `realism_stats`: the YAML configs for `sweep.py` and `sweep_analytic.py`, and the
  real-network benchmark directories `realism_pt2.py` reads (e.g. `real_cifar_biasfield`).
