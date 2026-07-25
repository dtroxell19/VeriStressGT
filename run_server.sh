#!/usr/bin/env bash
# Rebuttal server runs, one file. See SERVER_RUN.md for the annotated version.
#   1) abcrown + NeuralSAT on all 225 constructed instances (GPU)
#   2) generate the revised polynomial benchmarks (convex certified + nonconvex screen)
#   3) all 5 verifiers on the polynomial benchmarks
#
# Usage:
#   bash run_server.sh                 # everything
#   TIMEOUT_225=300 TIMEOUT_POLY=600 bash run_server.sh
#   SKIP_225=1 bash run_server.sh      # only the polynomial steps
#   SKIP_POLY=1 bash run_server.sh     # only the 225 GPU step
set -uo pipefail

cd "$(dirname "$0")"
source .env
export PYTHONPATH="$PWD/src"

VERIFY="python -m VeriStressGT.cli.verify_benchmark"
CREATE="python -m VeriStressGT.cli.create_benchmark"
ABCFG="src/VeriStressGT/configs/abcrown_gpu.yaml"
BM="src/VeriStressGT/benchmarks"
DEVICE="${DEVICE:-cuda}"
TIMEOUT_225="${TIMEOUT_225:-300}"
TIMEOUT_POLY="${TIMEOUT_POLY:-600}"
CPU_JOBS="${CPU_JOBS:-4}"
# NeuralSAT settings JSON (applied to every neuralsat run). Unset to disable.
export NEURALSAT_SETTING_FILE="${NEURALSAT_SETTING_FILE:-$PWD/src/VeriStressGT/configs/neuralsat_settings.json}"
mkdir -p runs

tally () {  # $1 = results dir  (jq-free)
  if [ -f "$1/results.jsonl" ]; then
    echo "  $1:"
    python3 -c "import json,collections,sys
c=collections.Counter(json.loads(l).get('status') for l in open(sys.argv[1]) if l.strip())
[print('   %5d %s'%(n,s)) for s,n in c.most_common()]" "$1/results.jsonl"
  else echo "  $1: (no results)"; fi
}

count_gt () {  # $1 = benchmark dir, $2 = gt field  (jq-free)
  python3 -c "import json,glob,collections,sys
c=collections.Counter(json.load(open(f)).get('gt',{}).get(sys.argv[2]) for f in glob.glob(sys.argv[1]+'/instances/*/meta.json'))
[print('   %5d %s'%(n,s)) for s,n in c.most_common()]" "$1" "$2"
}

# --------------------------------------------------------------------------- #
# 1) abcrown + NeuralSAT on the 225 (GPU). Ground truth = UNSAT; any SAT is a
#    soundness-failure candidate (re-check its saved counterexample in float64).
# --------------------------------------------------------------------------- #
if [ -z "${SKIP_225:-}" ]; then
  echo "=== [1] abcrown (GPU) on sweep_all (225) ==="
  $VERIFY --benchmark "$BM/sweep_all" --verifier abcrown --abcrown_config "$ABCFG" \
    --device "$DEVICE" --timeout "$TIMEOUT_225" --jobs 1 \
    --out_dir runs/sweep_all_abcrown_gpu --overwrite || echo "WARN: abcrown 225 run had errors"

  echo "=== [1] NeuralSAT (GPU) on sweep_all (225) ==="
  $VERIFY --benchmark "$BM/sweep_all" --verifier neuralsat \
    --device "$DEVICE" --timeout "$TIMEOUT_225" --jobs 1 \
    --out_dir runs/sweep_all_neuralsat_gpu --overwrite || echo "WARN: neuralsat 225 run had errors"

  echo "--- 225 tallies (expect all UNSAT; SAT = soundness-failure candidate) ---"
  tally runs/sweep_all_abcrown_gpu
  tally runs/sweep_all_neuralsat_gpu
fi

# --------------------------------------------------------------------------- #
# 2) + 3) polynomial: generate, then run all 5 verifiers.
# --------------------------------------------------------------------------- #
if [ -z "${SKIP_POLY:-}" ]; then
  echo "=== [2] generate polynomial benchmarks ==="
  $CREATE --spec src/VeriStressGT/configs/polynomial_convex_certified.yaml \
    --out_dir "$BM/polynomial_convex_certified" --overwrite || { echo "FATAL: convex generation failed"; exit 1; }
  $CREATE --spec src/VeriStressGT/configs/polynomial_nonconvex_screen.yaml \
    --out_dir "$BM/polynomial_nonconvex_screen" --overwrite || { echo "FATAL: nonconvex generation failed"; exit 1; }

  echo "--- convex certificate types (all analytic) ---"
  count_gt "$BM/polynomial_convex_certified" certificate_type
  echo "--- nonconvex screening tally ---"
  count_gt "$BM/polynomial_nonconvex_screen" screening_status

  run_all_verifiers () {  # $1 = benchmark dir, $2 = name prefix
    local bench="$1" name="$2"
    echo "=== [3] abcrown (GPU) on $name ==="
    $VERIFY --benchmark "$bench" --verifier abcrown --abcrown_config "$ABCFG" \
      --device "$DEVICE" --timeout "$TIMEOUT_POLY" --jobs 1 \
      --out_dir "runs/${name}_abcrown" --overwrite || echo "WARN: abcrown $name had errors"
    echo "=== [3] neuralsat (GPU) on $name ==="
    $VERIFY --benchmark "$bench" --verifier neuralsat \
      --device "$DEVICE" --timeout "$TIMEOUT_POLY" --jobs 1 \
      --out_dir "runs/${name}_neuralsat" --overwrite || echo "WARN: neuralsat $name had errors"
    for V in marabou nnenum pyrat; do
      echo "=== [3] $V (CPU) on $name ==="
      extra=()
      # Marabou's adapter runs a compiled binary; point it at MARABOU_BIN if set.
      [ "$V" = marabou ] && [ -n "${MARABOU_BIN:-}" ] && extra=(--marabou_bin "$MARABOU_BIN")
      $VERIFY --benchmark "$bench" --verifier "$V" \
        --timeout "$TIMEOUT_POLY" --jobs "$CPU_JOBS" "${extra[@]}" \
        --out_dir "runs/${name}_${V}" --overwrite || echo "WARN: $V $name had errors"
    done
    echo "--- $name tallies ---"
    for V in abcrown neuralsat marabou nnenum pyrat; do tally "runs/${name}_${V}"; done
  }

  run_all_verifiers "$BM/polynomial_convex_certified" poly_convex
  run_all_verifiers "$BM/polynomial_nonconvex_screen" poly_nonconvex
fi

echo "=== DONE. Results under runs/ ==="
