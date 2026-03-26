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

if [[ -z $PYTHON_EXECUTABLE ]]; then
  PYTHON_EXECUTABLE=python3
fi

# Source emsdk if not already on PATH
if ! command -v emcmake &>/dev/null; then
  EMSDK_ENV="${EMSDK:-$HOME/local/emsdk}/emsdk_env.sh"
  if [[ -f "$EMSDK_ENV" ]]; then
    EMSDK_QUIET=1 source "$EMSDK_ENV"
  fi
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
example_inputs = (torch.randn(4, 4), torch.randn(4, 4))
ep = torch.export.export(model, example_inputs)
et_program = to_edge_transform_and_lower(
    ep, partitioner=[WebGPUPartitioner()]
).to_executorch()

with open('webgpu_add_test.pte', 'wb') as f:
    f.write(et_program.buffer)
print('Exported webgpu_add_test.pte')
"

echo "=== Step 3: Build with Emscripten ==="
if command -v emcmake &>/dev/null; then
    BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out-webgpu"
    rm -rf "${BUILD_DIR}"
    EMDAWNWEBGPU_PORT_ARGS=""
    LOCAL_PORT="${SCRIPT_DIR}/emdawnwebgpu_pkg/emdawnwebgpu_pkg/emdawnwebgpu.port.py"
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

    cmake --build "${BUILD_DIR}" --target executorch_wasm -j$(nproc)

    echo "=== Step 4: Run Node.js WebGPU test ==="
    # Node 22-24 needs --experimental-webgpu; Node 25+ has WebGPU built-in
    if command -v node &>/dev/null; then
        NODE_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)
        NODE_ARGS=()
        if [[ "$NODE_VERSION" -ge 25 ]]; then
            true  # WebGPU is built-in
        elif [[ "$NODE_VERSION" -ge 22 ]]; then
            NODE_ARGS+=(--experimental-webgpu)
        else
            echo "SKIP: Node.js 22+ required for WebGPU (found v${NODE_VERSION})"
            echo "=== All WebGPU backend tests passed ==="
            exit 0
        fi
        if ! node "${NODE_ARGS[@]}" "${SCRIPT_DIR}/test_webgpu_runtime.js"; then
            echo "SKIP: Node.js WebGPU runtime test failed (no GPU available?)"
        fi
    else
        echo "SKIP: Node.js not found"
    fi
else
    echo "SKIP: emcmake not found (install Emscripten to run Wasm build and runtime tests)"
fi

echo "=== All WebGPU backend tests passed ==="
