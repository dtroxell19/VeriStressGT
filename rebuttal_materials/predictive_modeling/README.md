# Difficulty-Profile Predictive Modeling

Does the five-component VeriStress-GT **Difficulty Profile** predict *per-instance* verifier
timeouts **beyond ordinary network size/type features**? This increment builds an interpretable,
group-aware, per-verifier predictor and answers that question with repeated held-out evaluation.

Self-contained under this folder; reads existing artifacts in place, writes only here.

## Pipeline (run order)
```bash
cd src
python build_dataset.py           # audit + canonical (instance x verifier) table + freeze
python run_all.py --repeats 20    # primary repeated grouped nested-CV (A/B/C) + transfer (D) + correlations
python interpret.py               # final coefficients, stability, interactions, shallow trees
python runtime_regression.py      # secondary solved-only runtime regression
python predictions.py             # per-instance out-of-fold predictions CSV
python report.py                  # primary results table + verdicts
python plots.py                   # all figures
cd ../tests && PYTHONPATH=../src python test_pipeline.py   # leakage/correctness tests
```
`run_all.py` checkpoints per (scenario, verifier) cell to `results/cv/` and skips completed cells,
so it resumes after interruption.

## What is compared
For every eligible verifier × scenario, on **identical grouped folds**:
`Size/type (S)`  vs  `Size/type + Difficulty-Profile (S+D)`  (plus `Profile-only (D)` as a diagnostic,
and add-one / drop-one profile ablations). Headline metric = paired held-out ΔAUC with a 95%
group-bootstrap CI; Brier, AP, calibration reported alongside.

## Scenarios
- **A Synthetic** — 225 constructor networks (well-powered, 225 groups).
- **B Established** — mnist_fc + oval21 = only **6 networks** → exploratory (fails ≥10-group gate).
- **C Combined** — 231 networks.
- **D Transfer** — train one domain, test the other (group-bootstrap over target networks).

## Key design choices (see `DECISIONS.md`)
- Network groups: synthetic per-instance; established by ONNX content hash (leakage control).
- Per-(verifier,scenario) common timeout horizon (budgets differ 60–600 s); fixed-360 sensitivity.
- Constructor/benchmark names never used as features; stateless transforms only outside folds;
  all imputation/standardization/tuning inside training folds.

## Outputs
- `REPORT.md`, `REBUTTAL_TEXT.md` — findings + rebuttal-ready text.
- `results/primary_results_table.csv`, `results/cv/*.summary.json` — per-cell metrics + verdict.
- `results/predictions/out_of_fold_predictions.csv` — the per-instance predictor artifact.
- `results/transfer/`, `results/coefficients/`, `plots/` — transfer, interpretation, figures.
- `data/manifests/frozen_analysis_manifest.json` — frozen, hashed inputs.
