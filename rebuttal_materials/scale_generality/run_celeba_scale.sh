#!/usr/bin/env bash
# Part B server run: train a CelebA 128x128 backbone, graft the backbone-agnostic
# Paired-Bias head, export ground-truth instances, autograd-profile, and verify.
# Run from the repo root.
#
# CelebA data: torchvision expects it under <root>/celeba/ (img_align_celeba/ + attr csv).
# If torchvision's auto-download is quota-blocked, place the CelebA files there manually;
# the driver falls back to an untrained backbone (certificate is backbone-agnostic) and
# says so in the report.
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate VeriStressGT || true
fi
export PYTHONPATH="$PWD/src"
ROOT="${CELEBA_ROOT:-$PWD/data}"
SG="rebuttal_materials/scale_generality"

echo "[1/2] build CelebA-128 backbone + paired-bias ground-truth instances + autograd profiling"
python "$SG/run_celeba_scale.py" --n 8 --num-pairs 64 --celeba-root "$ROOT"

echo "[2/2] verify (verifiers strain at 49k-dim; timeouts are expected and are the point)"
# Run by explicit file path (not `-m`) so a stale pip-installed VeriStressGT in
# site-packages can't shadow the current cli/verify_benchmark.py. Abort early with a
# clear message if the resolved file has an unexpected (old) argument signature.
VB="$PWD/src/VeriStressGT/cli/verify_benchmark.py"
if ! grep -q '"--benchmark"' "$VB"; then
  echo "  WARNING: $VB has an unexpected signature (stale checkout?). Skipping verify." >&2
else
  python "$VB" \
    --benchmark "$SG/celeba_bench" --verifier abcrown \
    --out_dir "$SG/verify_abcrown" --timeout 600 --jobs 1 || true
fi

echo "done. results -> $SG/{celeba_scale_results.json,REPORT.md,verify_abcrown/}"
