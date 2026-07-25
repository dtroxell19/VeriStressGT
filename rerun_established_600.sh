#!/usr/bin/env bash
# Re-run the ESTABLISHED benchmarks (mnist_fc + oval21) with a uniform 600 s timeout,
# so the predictive-modeling target can use the same 600 s horizon as the synthetic data.
# Mirrors run_server.sh conventions (GPU for abcrown/neuralsat; CPU + MARABOU_BIN for the rest).
#
# Usage:
#   MARABOU_BIN=$PWD/scripts/marabou_maraboupy.py bash rerun_established_600.sh
#   TIMEOUT=600 DEVICE=cuda CPU_JOBS=8 bash rerun_established_600.sh
#   ONLY=vnncomp_mnist_fc bash rerun_established_600.sh   # one benchmark only
set -uo pipefail
cd "$(dirname "$0")"
source .env 2>/dev/null || true
export PYTHONPATH="$PWD/src"

VERIFY="python -m VeriStressGT.cli.verify_benchmark"
ABCFG="src/VeriStressGT/configs/abcrown_gpu.yaml"
BM="src/VeriStressGT/benchmarks"
DEVICE="${DEVICE:-cuda}"
TIMEOUT="${TIMEOUT:-600}"
CPU_JOBS="${CPU_JOBS:-8}"
export NEURALSAT_SETTING_FILE="${NEURALSAT_SETTING_FILE:-$PWD/src/VeriStressGT/configs/neuralsat_settings.json}"
OUT="rebuttal_materials/predictive_modeling/server_reruns/est600"
mkdir -p "$OUT"

BENCHES="${ONLY:-vnncomp_mnist_fc oval21}"

tally () {  # $1 = results dir (jq-free)
  if [ -f "$1/results.jsonl" ]; then
    python3 -c "import json,collections,sys
c=collections.Counter(json.loads(l).get('status') for l in open(sys.argv[1]) if l.strip())
print('   '+sys.argv[2]+': '+', '.join('%d %s'%(n,s) for s,n in c.most_common()))" "$1/results.jsonl" "$(basename $1)"
  else echo "   $(basename $1): (no results)"; fi
}

for BENCH in $BENCHES; do
  echo "=========================================================="
  echo "=== $BENCH  @ ${TIMEOUT}s  (device=$DEVICE) ==="
  echo "=========================================================="
  echo "--- abcrown (GPU) ---"
  $VERIFY --benchmark "$BM/$BENCH" --verifier abcrown --abcrown_config "$ABCFG" \
    --device "$DEVICE" --timeout "$TIMEOUT" --jobs 1 \
    --out_dir "$OUT/${BENCH}_abcrown" --overwrite || echo "WARN: abcrown $BENCH had errors"

  echo "--- neuralsat (GPU) ---"
  $VERIFY --benchmark "$BM/$BENCH" --verifier neuralsat \
    --device "$DEVICE" --timeout "$TIMEOUT" --jobs 1 \
    --out_dir "$OUT/${BENCH}_neuralsat" --overwrite || echo "WARN: neuralsat $BENCH had errors"

  for V in marabou nnenum pyrat; do
    echo "--- $V (CPU, jobs=$CPU_JOBS) ---"
    extra=()
    [ "$V" = marabou ] && [ -n "${MARABOU_BIN:-}" ] && extra=(--marabou_bin "$MARABOU_BIN")
    $VERIFY --benchmark "$BM/$BENCH" --verifier "$V" \
      --timeout "$TIMEOUT" --jobs "$CPU_JOBS" "${extra[@]}" \
      --out_dir "$OUT/${BENCH}_${V}" --overwrite || echo "WARN: $V $BENCH had errors"
  done

  echo "--- $BENCH tallies (@ ${TIMEOUT}s) ---"
  for V in abcrown neuralsat marabou nnenum pyrat; do tally "$OUT/${BENCH}_${V}"; done
done

echo "=== DONE. Re-run results under $OUT/ ==="
echo "Next: rsync $OUT back, then  python src/ingest_established_600.py  to build the 600 s table."
