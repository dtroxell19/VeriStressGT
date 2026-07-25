# Difficulty-Profile Sensitivity Study

Answers the reviewer/AC requests for: (1) sensitivity of the five Difficulty-Profile
components to **sample count**, **sampling distribution**, and the **η / τ / grid-width**
hyperparameters; (2) **recommended defaults**; (3) an **inter-component dependence matrix**;
and (4) a **unified measurement formula** (a single Difficulty Index).

## What varies, and where the knobs live
The paper's η is a numerical-stability constant (G_IBP eq.11, d_eff eq.15); τ is the A_τ
quantization grid width (eq.14) and the smooth-activation unstable threshold (U, eq.12; ReLU
is τ-independent by definition). These are exposed as **opt-in, backward-compatible** kwargs in
`src/VeriStressGT/difficulty_profile/{components,profile,cli}.py`; every default reproduces the
frozen 345-instance numbers exactly. The added faithful pieces (paper eq.12 `ω_j>τ` smooth test;
arbitrary A_τ grid width) are gated behind flags and disclosed in `REPORT.md`.

## Layout
```
config/experiment.json   grid + center-point + subset params
src/common.py            paths, canonical 5-component map, reuse of predictive_modeling analysis layer
src/subset.py            stratified ~48-instance subset (arch x dataset x difficulty)
src/driver.py            one-axis-at-a-time recompute sweeps -> tidy long CSVs
src/analysis.py          stability, seed-CoV, tornado, dependence (345), Difficulty Index
src/plots.py             figures
src/run_all.py           stage driver: subset|driver|analysis|plots|all (+ --smoke)
tests/test_omega_smooth.py   verifies the omega_j>tau fix on a sigmoid net
run_sensitivity.sh       server entry (full grid) + --smoke
```

## Run
```bash
# from repo root
export PYTHONPATH=$PWD/src
cd rebuttal_materials/profile_sensitivity/src
python run_all.py --stage subset
python run_all.py --stage driver --axes all --n-jobs 100   # server; recompute sweeps
python run_all.py --stage analysis
python run_all.py --stage plots
# fast local sanity check of the whole pipeline:
python run_all.py --smoke
# omega-mode faithfulness check:
cd ../tests && PYTHONPATH=../../../src python test_omega_smooth.py
```
Or on a many-core host: `bash rebuttal_materials/profile_sensitivity/run_sensitivity.sh`.

## Scope
- Recompute sweeps run on a **representative subset** (the reviewers asked for a "brief"
  analysis) spanning all 4 architectures, all 4 benchmarks/datasets, and easy→hard difficulty.
- The **inter-component dependence matrix** and the **Difficulty Index** run on the full frozen
  345-instance table (`predictive_modeling/data/processed/`), so they validate against real
  verifier timeout outcomes.
- `polynomial_stress_22` has no committed ONNX (network is reconstructed from generator args), so
  those instances are recompute-N/A; POLYNOMIAL is still covered in the dependence matrix + index.
