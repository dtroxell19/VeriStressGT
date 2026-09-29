#!/usr/bin/env bash
# Install Marabou 2.0.0 (maraboupy) into its own conda env, the same version on every platform.
#
#   bash scripts/install_marabou.sh            # env name: marabou (or $MARABOU_CONDA_ENV)
#
# PyPI has maraboupy 2.0.0 wheels only for Linux x86_64 and Intel macOS. Everywhere else (Apple
# Silicon, Linux aarch64) this builds the v2.0.0 tag from source (~30-60 min, needs a C++
# compiler; the build downloads Boost, protobuf, ONNX, OpenBLAS and pybind11 itself). The Intel
# wheel does not run under Rosetta on macOS 14 (illegal instruction), so Apple Silicon always
# builds natively.
set -euo pipefail

MARABOU_VERSION=2.0.0
ENV_NAME="${MARABOU_CONDA_ENV:-marabou}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${MARABOU_BUILD_DIR:-$REPO_ROOT/.deps}"
JOBS="${MARABOU_BUILD_JOBS:-4}"   # keep modest: each compile job can take ~1 GB of RAM

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -q -n "$ENV_NAME" python=3.10
fi
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
"$PY" -m pip install -q "numpy>=1.21,<2" "onnx>=1.15,<2" "onnxruntime>=1.12,<2"

OS="$(uname -s)"; ARCH="$(uname -m)"
if [[ "$OS/$ARCH" == "Linux/x86_64" || "$OS/$ARCH" == "Darwin/x86_64" ]]; then
    echo "Installing the maraboupy $MARABOU_VERSION wheel ($OS/$ARCH)"
    "$PY" -m pip install -q "maraboupy==$MARABOU_VERSION"
else
    echo "No maraboupy $MARABOU_VERSION wheel for $OS/$ARCH: building from source in $BUILD_ROOT"
    "$PY" -m pip install -q "cmake>=3.16,<4"
    src="$BUILD_ROOT/marabou-v$MARABOU_VERSION"
    if [ ! -d "$src" ]; then
        mkdir -p "$BUILD_ROOT"
        git clone -q --depth 1 --branch "v$MARABOU_VERSION" \
            https://github.com/NeuralNetworkVerification/Marabou.git "$src"
    fi
    cmake_bin="$(dirname "$PY")/cmake"
    mkdir -p "$src/build"
    (cd "$src/build" && "$cmake_bin" .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON \
        -DRUN_UNIT_TEST=OFF -DRUN_MEMORY_TEST=OFF -DPYTHON_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY" \
        && make -j"$JOBS")
    site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    "$PY" -m pip uninstall -y -q maraboupy 2>/dev/null || true
    rm -rf "$site/maraboupy"
    cp -R "$src/maraboupy" "$site/maraboupy"
    rm -rf "$site/maraboupy/test"
    echo "$MARABOU_VERSION" > "$site/maraboupy/SOURCE_VERSION"
fi

"$PY" -W ignore -c "from maraboupy import Marabou, MarabouCore; print('maraboupy OK:', Marabou.__file__)"
