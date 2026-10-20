#!/usr/bin/env bash
# Run scripts/marabou_maraboupy.py with the Python of the conda env that has maraboupy installed.
# Point the marabou adapter at this file:  --marabou_bin scripts/marabou_env.sh
#
# Interpreter: $MARABOU_PYTHON if set, else <conda base>/envs/${MARABOU_CONDA_ENV:-marabou}/bin/python.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${MARABOU_PYTHON:-}"
if [ -z "$py" ]; then
  base="$(conda info --base 2>/dev/null || dirname "$(dirname "${CONDA_EXE:-$(command -v conda)}")")"
  py="$base/envs/${MARABOU_CONDA_ENV:-marabou}/bin/python"
fi
exec "$py" "$here/marabou_maraboupy.py" "$@"
