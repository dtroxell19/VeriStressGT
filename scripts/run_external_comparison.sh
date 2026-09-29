#!/usr/bin/env bash
# VeriStressGT vs. external benchmarks: run the Thrust 1 verifier pool on VNN-COMP mnist_fc and oval21
# (labelled only by attack-certified counterexamples) and compare which planted bugs each can expose.
# See experiments/external_benchmarks/README.md.
#
#   bash scripts/run_external_comparison.sh <thrust1 card run>/results.json
#
# Defaults use every instance (mnist_fc: 90, oval21: 30); roughly 4-5 h on a laptop CPU.
#   PER_NETWORK=10   evenly spaced subset per network (the published laptop run: mnist_fc 30/90)
#   VNNCOMP_DIR      checkout of github.com/ChristopherBrix/vnncomp2022_benchmarks (for the .onnx files)
#   OUT              output root (default: ext_runs)
set -euo pipefail
cd "$(dirname "$0")/.."

THRUST1_RESULTS="${1:?usage: $0 <thrust1 card run>/results.json}"
VNNCOMP_DIR="${VNNCOMP_DIR:-$HOME/vnncomp2022_benchmarks/benchmarks}"
PER_NETWORK="${PER_NETWORK:-0}"
OUT="${OUT:-ext_runs}"
suffix=$([ "$PER_NETWORK" = 0 ] && echo "" || echo "_n$PER_NETWORK")

compare_args=("VeriStressGT=$THRUST1_RESULTS")
for b in mnist_fc oval21; do
    src="src/VeriStressGT/benchmarks/$([ $b = mnist_fc ] && echo vnncomp_mnist_fc || echo oval21)"
    bench="ext_${b}${suffix}_bench"
    [ -f "$bench/manifest.json" ] || python aiq/build_external_bench.py --source "$src" \
        --onnx_root "$VNNCOMP_DIR/$b" --out "$bench" --per_network "$PER_NETWORK"
    python -u aiq/thrust1_runner.py --bench_dir "$bench" --run_dir "$OUT/${b}${suffix}_run" \
        --results_fpath "$OUT/${b}${suffix}/results.json"
    compare_args+=("$b=$OUT/${b}${suffix}/results.json")
done
python aiq/compare_benchmarks.py "${compare_args[@]}" --out "$OUT/benchmark_comparison${suffix}.md"
