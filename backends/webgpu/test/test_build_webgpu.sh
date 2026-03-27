#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# End-to-end build and test script for the WebGPU backend.
# Usage: bash backends/webgpu/test/test_build_webgpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

# Locate emsdk and set PATH directly (sourcing emsdk_env.sh from another
# script breaks its BASH_SOURCE detection).
_EMSDK_DIR="${EMSDK:-${CONDA_PREFIX:+${CONDA_PREFIX}/emsdk}}"
_EMSDK_DIR="${_EMSDK_DIR:-$HOME/local/emsdk}"
if [[ -d "${_EMSDK_DIR}/upstream/emscripten" ]]; then
  export EMSDK="${_EMSDK_DIR}"
  _NODE_BIN=$(find "${_EMSDK_DIR}/node" -maxdepth 3 -name node -type f 2>/dev/null | head -1)
  export PATH="${_EMSDK_DIR}:${_EMSDK_DIR}/upstream/emscripten${_NODE_BIN:+:$(dirname "${_NODE_BIN}")}:${PATH}"
fi

echo "=== Step 1: Run Python export test ==="
$PYTHON_EXECUTABLE -m pytest "${SCRIPT_DIR}/test_webgpu_export.py" -v

echo "=== Step 2: Export test model ==="
$PYTHON_EXECUTABLE -c "
import dataclasses
from torch.utils._pytree import LeafSpec
if not hasattr(LeafSpec(), 'type'):
    def _leafspec_getattr(self, name):
        for f in dataclasses.fields(type(self)):
            if f.name == name:
                if f.default is not dataclasses.MISSING:
                    return f.default
                elif f.default_factory is not dataclasses.MISSING:
                    val = f.default_factory()
                    object.__setattr__(self, name, val)
                    return val
        raise AttributeError(f\"'{type(self).__name__}' object has no attribute '{name}'\")
    LeafSpec.__getattr__ = _leafspec_getattr

import torch
from executorch.backends.webgpu import WebGPUPartitioner
from executorch.exir import to_edge_transform_and_lower

class AddModule(torch.nn.Module):
    def forward(self, a, b):
        return a + b

model = AddModule()
example_inputs = (torch.randn(1024, 1024), torch.randn(1024, 1024))
ep = torch.export.export(model, example_inputs)
et_program = to_edge_transform_and_lower(
    ep, partitioner=[WebGPUPartitioner()]
).to_executorch()

with open('webgpu_add_test.pte', 'wb') as f:
    f.write(et_program.buffer)
print('Exported webgpu_add_test.pte')
"

echo "=== Step 3: Build with Emscripten ==="
if ! command -v emcmake &>/dev/null; then
    echo "SKIP: emcmake not found. Run setup-webgpu.sh first."
    exit 1
fi

BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out-webgpu"
rm -rf "${BUILD_DIR}"

EMDAWNWEBGPU_PORT_ARGS=""
LOCAL_PORT="${EXECUTORCH_ROOT}/backends/webgpu/third-party/emdawnwebgpu/emdawnwebgpu_pkg/emdawnwebgpu.port.py"
if [[ -f "$LOCAL_PORT" ]]; then
    echo "Using local emdawnwebgpu port: ${LOCAL_PORT}"
    EMDAWNWEBGPU_PORT_ARGS="-DWEBGPU_EMDAWNWEBGPU_PORT=${LOCAL_PORT}"
fi

emcmake cmake \
    -DEXECUTORCH_BUILD_WEBGPU=ON \
    -DEXECUTORCH_BUILD_WASM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DCMAKE_BUILD_TYPE=Release \
    ${EMDAWNWEBGPU_PORT_ARGS} \
    -B "${BUILD_DIR}" \
    "${EXECUTORCH_ROOT}"

cmake --build "${BUILD_DIR}" --target executorch_wasm_module -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "=== Step 4: Run runtime test ==="
_NODE_CMD=""
if [[ -n "${EMSDK}" ]]; then
    _NODE_CMD=$(find "${EMSDK}/node" -maxdepth 3 -name node -type f 2>/dev/null | head -1)
fi
[[ -z "${_NODE_CMD}" ]] && _NODE_CMD="$(command -v node 2>/dev/null || true)"

if [[ -n "${_NODE_CMD}" ]] && "${_NODE_CMD}" "${SCRIPT_DIR}/test_webgpu_runtime.js" 2>/dev/null; then
    echo "PASS: Node.js runtime test"
else
    echo "WebGPU not available in Node.js, falling back to browser test..."
    TEST_DIR="/tmp/webgpu_test"
    rm -rf "${TEST_DIR}"
    mkdir -p "${TEST_DIR}"
    cp "${BUILD_DIR}/extension/wasm/executorch_wasm.js" "${TEST_DIR}/"
    cp "${BUILD_DIR}/extension/wasm/executorch_wasm.wasm" "${TEST_DIR}/"
    cp webgpu_add_test.pte "${TEST_DIR}/"
    cp "${SCRIPT_DIR}/test_webgpu_browser.html" "${TEST_DIR}/"

    PORT=8080
    while lsof -i :"${PORT}" &>/dev/null; do
        PORT=$((PORT + 1))
    done
    echo "Open http://localhost:${PORT}/test_webgpu_browser.html in Chrome"
    echo "Press Ctrl+C to stop the server when done."
    cd "${TEST_DIR}" && $PYTHON_EXECUTABLE -m http.server "${PORT}"
fi

echo "=== Done ==="
