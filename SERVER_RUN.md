# Rebuttal server runs

Two jobs:
1. **abcrown + NeuralSAT on all 225 constructed instances (GPU)** — the numerical-soundness / GPU
   comparison requested by reviewers.
2. **All 5 verifiers on the revised polynomial benchmark** (convex analytically-certified main set +
   the separate nonconvex screening set).

## 0. Environment (once)
```bash
cd VeriStressGT
source .env                       # sets ABCROWN_* / NEURALSAT_DIR / conda env names
export PYTHONPATH=$PWD/src
```

## 1. abcrown + NeuralSAT on the 225 constructed instances (GPU)
The benchmark `src/VeriStressGT/benchmarks/sweep_all` (225 instances) is already in the repo.
```bash
# alpha,beta-CROWN on GPU (saves counterexamples so any SAT verdict can be re-checked in float64)
python -m VeriStressGT.cli.verify_benchmark \
  --benchmark src/VeriStressGT/benchmarks/sweep_all \
  --verifier abcrown --abcrown_config src/VeriStressGT/configs/abcrown_gpu.yaml \
  --device cuda --timeout 300 --jobs 1 \
  --out_dir runs/sweep_all_abcrown_gpu --overwrite

# NeuralSAT on GPU
python -m VeriStressGT.cli.verify_benchmark \
  --benchmark src/VeriStressGT/benchmarks/sweep_all \
  --verifier neuralsat --device cuda --timeout 300 --jobs 1 \
  --out_dir runs/sweep_all_neuralsat_gpu --overwrite
```
All 225 are provably robust, so the ground truth is UNSAT. Any `SAT` verdict is a candidate soundness
failure: re-evaluate its saved counterexample in float64 to confirm (spurious CEX => unsound). Tally:
```bash
for f in runs/sweep_all_abcrown_gpu runs/sweep_all_neuralsat_gpu; do
  echo "$f:"; jq -r .status "$f/results.jsonl" | sort | uniq -c
done
```

## 2. Generate the revised polynomial benchmarks
```bash
# MAIN: convex, analytically certified (even degree, alpha>=0; NO L-BFGS-B). Aims for 22 instances.
python -m VeriStressGT.cli.create_benchmark \
  --spec src/VeriStressGT/configs/polynomial_convex_certified.yaml \
  --out_dir src/VeriStressGT/benchmarks/polynomial_convex_certified --overwrite

# SEPARATE: nonconvex screening (odd degrees; per-instance screening_status in meta gt).
python -m VeriStressGT.cli.create_benchmark \
  --spec src/VeriStressGT/configs/polynomial_nonconvex_screen.yaml \
  --out_dir src/VeriStressGT/benchmarks/polynomial_nonconvex_screen --overwrite

# convex certificate is analytic (is_robust=True for every generated instance):
jq -r '.gt.certificate_type' src/VeriStressGT/benchmarks/polynomial_convex_certified/instances/*/meta.json | sort | uniq -c
# nonconvex screening tally (numerically_certified_robust / non_robust / inconclusive):
jq -r '.gt.screening_status' src/VeriStressGT/benchmarks/polynomial_nonconvex_screen/instances/*/meta.json | sort | uniq -c
```

## 3. All 5 verifiers on the polynomial benchmark(s)
`$VERIFIERS` = the five used in the paper. abcrown needs a config; add `--device cuda` where a GPU is
available (nnenum/marabou/pyrat are CPU verifiers).
```bash
BENCH=src/VeriStressGT/benchmarks/polynomial_convex_certified     # then repeat for polynomial_nonconvex_screen
NAME=poly_convex

# abcrown (GPU)
python -m VeriStressGT.cli.verify_benchmark --benchmark $BENCH \
  --verifier abcrown --abcrown_config src/VeriStressGT/configs/abcrown_gpu.yaml \
  --device cuda --timeout 600 --jobs 1 --out_dir runs/${NAME}_abcrown --overwrite

# neuralsat (GPU)
python -m VeriStressGT.cli.verify_benchmark --benchmark $BENCH \
  --verifier neuralsat --device cuda --timeout 600 --jobs 1 --out_dir runs/${NAME}_neuralsat --overwrite

# marabou / nnenum / pyrat (CPU)
for V in marabou nnenum pyrat; do
  python -m VeriStressGT.cli.verify_benchmark --benchmark $BENCH \
    --verifier $V --timeout 600 --jobs 4 --out_dir runs/${NAME}_${V} --overwrite
done
```

## Notes
- The convex constructor is analytically certified, so **no numerical global-separation check runs**
  for the main benchmark; the odd-degree cases live only in `polynomial_nonconvex_screen`.
- `create_benchmark` now writes the certificate metadata to `gt` in each instance `meta.json`
  (`is_robust`, `certificate_type`, `certified_convex`, `norm`, `screening_status`).
- If fewer than 22 convex instances certify (boundary sampling is stochastic), add more seeds to
  `polynomial_convex_certified.yaml` and regenerate.
- l2 perturbations: the certificate math is norm-generic (`--norm l2`), but VNNLIB export currently
  supports l_inf only, so l2 runs are not wired here.
