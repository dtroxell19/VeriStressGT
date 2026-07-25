#!/usr/bin/env bash
# Difficulty-Profile sensitivity study: full recompute grid.
# Run from the repo root on a many-core host.  No hostnames / usernames / paths
# baked in — everything is $PWD-relative.
#
# Usage:
#   bash rebuttal_materials/profile_sensitivity/run_sensitivity.sh            # full grid
#   bash rebuttal_materials/profile_sensitivity/run_sensitivity.sh --smoke    # fast local check
set -euo pipefail

# activate the project env if available (no-op if already active)
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate VeriStressGT || true
fi

REPO="$PWD"
SRC="$REPO/rebuttal_materials/profile_sensitivity/src"
# module src first (common/driver/...), core src second (VeriStressGT); common.py also
# self-registers the core src, so either ordering works.
export PYTHONPATH="$SRC:$REPO/src"
# CPU-only: the profiler runs on CPU; hiding the GPU silences the harmless
# "NVIDIA driver too old" CUDA-init warning and avoids contending with GPU
# verifier jobs. Override by exporting CUDA_VISIBLE_DEVICES before calling.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
NJOBS="${NJOBS:-100}"
run() { ( cd "$SRC" && python run_all.py "$@" ); }

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[smoke] fast local sanity run"
  run --smoke --n-jobs "$(getconf _NPROCESSORS_ONLN)"
  exit 0
fi

echo "[1/4] subset selection";              run --stage subset
echo "[2/4] recompute sweeps (n-jobs=$NJOBS)"; run --stage driver --axes all --n-jobs "$NJOBS"
echo "[3/4] analysis + Difficulty Index";    run --stage analysis
echo "[4/4] plots";                          run --stage plots

echo "done. results -> rebuttal_materials/profile_sensitivity/{results,plots}/"
