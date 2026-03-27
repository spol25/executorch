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
# This script:
#   1. Clones emsdk and installs Emscripten 4.0+ (includes Node.js 22)
#   2. Downloads the emdawnwebgpu port package from Dawn nightly releases
#
# Usage:
#   source backends/webgpu/scripts/setup-webgpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=== WebGPU Backend Setup ==="

if [ -z "${CONDA_PREFIX}" ]; then
    echo "ERROR: No conda environment is active."
    echo "  Run: conda activate <your-executorch-env>"
    return 1 2>/dev/null || true
fi
echo "Using conda env: ${CONDA_PREFIX}"

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Run install_executorch.sh first."
    return 1 2>/dev/null || true
fi

if ! python -c "import executorch" 2>/dev/null; then
    echo "WARNING: executorch package not importable."
    echo "  Run: pip install -e . --no-build-isolation"
fi

# --- Install emsdk (need 4.0+ for emdawnwebgpu port) ---
EMSDK_DIR="${CONDA_PREFIX}/emsdk"
EMSDK_VERSION="4.0.9"

if [ ! -f "${EMSDK_DIR}/emsdk" ]; then
    echo "Cloning emsdk into ${EMSDK_DIR}..."
    git clone https://github.com/emscripten-core/emsdk.git "${EMSDK_DIR}"
fi

if [ ! -d "${EMSDK_DIR}/upstream/emscripten" ]; then
    echo "Installing Emscripten ${EMSDK_VERSION} (this may take a few minutes)..."
    "${EMSDK_DIR}/emsdk" install "${EMSDK_VERSION}"
    "${EMSDK_DIR}/emsdk" activate "${EMSDK_VERSION}"
fi

# Set PATH directly — sourcing emsdk_env.sh from another script breaks its
# BASH_SOURCE detection. Also adds emsdk's bundled Node.js (avoids dylib
# conflicts with conda's node on macOS).
if [ -d "${EMSDK_DIR}/upstream/emscripten" ]; then
    export EMSDK="${EMSDK_DIR}"
    _NODE_BIN=$(find "${EMSDK_DIR}/node" -maxdepth 3 -name node -type f 2>/dev/null | head -1)
    export PATH="${EMSDK_DIR}:${EMSDK_DIR}/upstream/emscripten${_NODE_BIN:+:$(dirname "${_NODE_BIN}")}:${PATH}"
fi

# --- Download emdawnwebgpu port package from Dawn releases ---
_EMDAWNWEBGPU_DIR="${EXECUTORCH_ROOT}/backends/webgpu/third-party/emdawnwebgpu"
if [ ! -f "${_EMDAWNWEBGPU_DIR}/emdawnwebgpu_pkg/emdawnwebgpu.port.py" ]; then
    echo "Downloading emdawnwebgpu port package from Dawn releases..."
    _DAWN_TAG=$(curl -sL "https://api.github.com/repos/google/dawn/releases/latest" | python -c "import sys,json; print(json.load(sys.stdin)['tag_name'])" 2>/dev/null)
    if [ -n "${_DAWN_TAG}" ]; then
        _ZIP_URL="https://github.com/google/dawn/releases/download/${_DAWN_TAG}/emdawnwebgpu_pkg-${_DAWN_TAG}.zip"
        _TMP_ZIP="$(mktemp).zip"
        curl -sL -o "${_TMP_ZIP}" "${_ZIP_URL}" && \
            mkdir -p "${_EMDAWNWEBGPU_DIR}" && \
            unzip -o "${_TMP_ZIP}" -d "${_EMDAWNWEBGPU_DIR}" && \
            rm -f "${_TMP_ZIP}" && \
            echo "Downloaded emdawnwebgpu (${_DAWN_TAG})"
    else
        echo "WARNING: Could not fetch Dawn release tag."
        echo "  Download manually from https://github.com/google/dawn/releases"
    fi
fi

# --- Verify ---
echo ""
echo "=== Verification ==="
echo "  emcc:    $(emcc --version 2>/dev/null | head -1 || echo 'not found')"
echo "  emcmake: $(command -v emcmake 2>/dev/null || echo 'not found')"
echo "  node:    $(node --version 2>/dev/null || echo 'not found')"
echo "  python:  $(python --version 2>/dev/null)"

if ! command -v emcmake &>/dev/null; then
    echo "ERROR: emcmake still not found after setup."
    return 1 2>/dev/null || true
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  bash backends/webgpu/test/test_build_webgpu.sh"
