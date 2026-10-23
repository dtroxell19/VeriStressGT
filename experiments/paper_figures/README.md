# Paper Figures 2 and 3

`make_figures.sh` rebuilds Figure 2 (verifier outcomes and average runtimes per constructor) and
Figure 3 (timeout AUC per Difficulty Profile component; timeout rate over U × d_eff) from the
committed benchmarks, Difficulty Profiles and `runs_standardized` results, using
`analysis/analyze_profile_updated_components.py` and
`analysis/make_targeted_timeout_heatmaps_with_auc.py`.

```bash
bash experiments/paper_figures/make_figures.sh     # writes paper/ and convex/
```

Both variants use the 225-instance VeriStress-GT suite plus MNIST_fc and oval21:

- **`paper/`**: `sweep_all` + `polynomial_stress_22`, the published figures. Its
  `merged_records.json` is identical to `experiments/merged_records_constructed_and_real.json`.
- **`convex/`**: `polynomial_convex_certified` (22 analytically certified convex polynomial
  networks) replaces `polynomial_stress_22` (22 networks labelled by a numerical screen). Only the
  polynomial rows of the inputs change.

What changes in `convex/`:

- **Figure 2, Polynomial Net row:** α,β-CROWN 22/22 (was 16/22), NeuralSAT 16/22 with 6 UNKNOWN on
  degree 18 to 22 (was 11/22), nnenum 0/22 (unsupported `Transpose` op, as before unsupported).
  Marabou and PyRAT count as 0/22 errors: their server runs on this set failed on setup, so they
  produced no verdicts. The convex α,β-CROWN and NeuralSAT runs were on the GPU server; every other
  cell is a CPU run.
- **Figure 3:** the 22 PyRAT timeouts and the α,β-CROWN / NeuralSAT timeouts on the old polynomial
  set are gone, so PyRAT's column moves most (e.g. G_IBP 0.69 → 0.80, M̂_min 0.52 → 0.64);
  α,β-CROWN and NeuralSAT move by at most 0.04, and Marabou and nnenum are unchanged.

Conventions kept from the paper: average runtimes count a timeout as 600 s for `sweep_all` and
360 s for every other benchmark (including the polynomial sets, whose runs used 600 s); outcomes
other than solved / timeout (errors, UNKNOWN) are left out of the averages and of Figure 3. The
convex Difficulty Profiles (`benchmarks/polynomial_convex_certified/difficulty_profiles.json`) use
the profiler's defaults with `--seed 0`, so they regenerate exactly. (The published profiles were
computed without a seed; re-profiling those moves A_τ by a few percent.)

## Regenerating the convex inputs

Everything the `convex/` figures read can be rebuilt from the repo; each step reproduces the
committed files (meta.json differs only in elapsed-time fields):

```bash
export PYTHONPATH=src
python -m VeriStressGT.cli.create_benchmark \
  --spec src/VeriStressGT/configs/polynomial_convex_certified.yaml --out_dir ./regen_convex
python -m VeriStressGT.difficulty_profile.cli --benchmark-dir ./regen_convex --seed 0
bash experiments/paper_figures/make_figures.sh
```

The verifier runs themselves (`runs_standardized/*_poly_convex`) record their settings in
`run_config.json`; runtimes and borderline verdicts depend on the hardware and verifier versions.
