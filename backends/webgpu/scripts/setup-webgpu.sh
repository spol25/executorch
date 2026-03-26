#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Setup script for the WebGPU backend.
#
# Prerequisites: an active ExecuTorch conda environment (run install_executorch.sh first).
#
# This script installs into the active conda env:
#   1. Emscripten SDK (emsdk) via conda-forge — C++ to Wasm compiler with WebGPU support
#   2. Node.js 22 via conda-forge — required for --experimental-webgpu runtime testing
#
# All artifacts stay inside $CONDA_PREFIX — no global pollution.
#
# Usage:
#   source backends/webgpu/scripts/setup-webgpu.sh
#
# After sourcing, the following will be available in your shell:
#   emcc, emcmake   — Emscripten toolchain
#   node (v22+)     — Node.js with WebGPU support
#   $EMSDK          — Emscripten root directory (inside conda env)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=== WebGPU Backend Setup ==="

# --- Verify conda environment is active ---
if [ -z "${CONDA_PREFIX}" ]; then
    echo "ERROR: No conda environment is active."
    echo "  Run: conda activate <your-executorch-env>"
    exit 1
fi
echo "Using conda env: ${CONDA_PREFIX}"

# --- Verify base ExecuTorch environment ---
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Run install_executorch.sh first."
    exit 1
fi

if ! python -c "import executorch" 2>/dev/null; then
    echo "WARNING: executorch package not importable."
    echo "  Run: pip install -e . --no-build-isolation"
fi

# --- Install emsdk and Node.js 22 into the conda env ---
# Both come from conda-forge and install entirely under $CONDA_PREFIX.
NEEDS_INSTALL=()

if ! command -v emcc &>/dev/null; then
    NEEDS_INSTALL+=("emsdk")
fi

NODE_MAJOR=$(node --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
if [ -z "${NODE_MAJOR}" ] || [ "${NODE_MAJOR}" -lt 22 ]; then
    NEEDS_INSTALL+=("nodejs>=22")
fi

if [ ${#NEEDS_INSTALL[@]} -gt 0 ]; then
    echo "Installing into conda env: ${NEEDS_INSTALL[*]}"
    conda install -y -c conda-forge "${NEEDS_INSTALL[@]}"
else
    echo "emsdk and Node.js 22+ already installed in conda env."
fi

# --- Activate emsdk environment ---
# conda-forge's emsdk package installs an activation script.
EMSDK_ENV="${CONDA_PREFIX}/bin/emsdk_env.sh"
if [ -f "${EMSDK_ENV}" ]; then
    source "${EMSDK_ENV}" 2>/dev/null || true
elif [ -n "${EMSDK}" ] && [ -f "${EMSDK}/emsdk_env.sh" ]; then
    source "${EMSDK}/emsdk_env.sh" 2>/dev/null || true
fi

# --- Verify ---
echo ""
echo "=== Verification ==="
echo "  conda env: ${CONDA_PREFIX}"
echo "  emcc:      $(emcc --version 2>/dev/null | head -1 || echo 'not found')"
echo "  node:      $(node --version 2>/dev/null || echo 'not found')"
echo "  python:    $(python --version 2>/dev/null)"
echo "  flatc:     $(flatc --version 2>/dev/null || echo 'not found')"
echo ""

NODE_MAJOR=$(node --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
if [ -n "${NODE_MAJOR}" ] && [ "${NODE_MAJOR}" -lt 22 ]; then
    echo "WARNING: Node.js v${NODE_MAJOR} detected, but 22+ is required for --experimental-webgpu"
fi

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  # Run the Python export test:"
echo "  python -m pytest backends/webgpu/test/test_webgpu_export.py -v"
echo ""
echo "  # Build the Wasm + WebGPU runtime:"
echo "  emcmake cmake -DEXECUTORCH_BUILD_WEBGPU=ON -DEXECUTORCH_BUILD_WASM=ON -B cmake-out-webgpu ."
echo "  cmake --build cmake-out-webgpu --target executorch_wasm -j\$(nproc)"
echo ""
echo "  # Or run the full end-to-end test:"
echo "  bash backends/webgpu/test/test_build_webgpu.sh"
