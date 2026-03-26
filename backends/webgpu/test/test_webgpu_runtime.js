/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Runtime test for the WebGPU backend.
// Requires: Node.js 22+ with --experimental-webgpu flag.
//
// Usage:
//   node --experimental-webgpu test_webgpu_runtime.js
//
// This test:
//   1. Requests a WebGPU adapter and device
//   2. Loads the Wasm module with the WebGPU device pre-initialized
//   3. Loads a .pte file containing a WebGPU-delegated add model
//   4. Runs inference and verifies output = a + b

const fs = require("fs");
const path = require("path");

async function main() {
  // Step 1: Initialize WebGPU device
  // In Node.js 22+, the gpu object is available globally
  const gpu = navigator.gpu;
  if (!gpu) {
    console.error(
      "WebGPU not available. Run with: node --experimental-webgpu"
    );
    process.exit(1);
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    console.error("Failed to get WebGPU adapter");
    process.exit(1);
  }

  const device = await adapter.requestDevice();
  console.log("WebGPU device acquired");

  // Step 2: Load the Wasm module
  // The Wasm module expects Module.preinitializedWebGPUDevice to be set
  // before initialization so that emscripten_webgpu_get_device() works.
  const wasmPath = path.resolve(
    __dirname,
    "../../../cmake-out-webgpu/extension/wasm/executorch_wasm.js"
  );

  if (!fs.existsSync(wasmPath)) {
    console.error(`Wasm module not found at: ${wasmPath}`);
    console.error("Build first with: cmake --build cmake-out-webgpu --target executorch_wasm");
    process.exit(1);
  }

  const createModule = require(wasmPath);
  const Module = await createModule({
    preinitializedWebGPUDevice: device,
  });

  console.log("Wasm module loaded");

  // Step 3: Load the .pte model
  const ptePath = path.resolve(__dirname, "../../../webgpu_add_test.pte");
  if (!fs.existsSync(ptePath)) {
    console.error(`Model not found at: ${ptePath}`);
    console.error("Export first with the test_build_webgpu.sh script");
    process.exit(1);
  }

  const pteBuffer = fs.readFileSync(ptePath);
  const model = new Module.Module();
  model.load(new Uint8Array(pteBuffer));
  model.loadMethod("forward");
  console.log("Model loaded");

  // Step 4: Run inference
  const size = 16; // 4x4 = 16 elements
  const a_data = new Float32Array(size);
  const b_data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    a_data[i] = i * 1.0;
    b_data[i] = i * 2.0;
  }

  const a = Module.Tensor.fromArray(a_data, [4, 4]);
  const b = Module.Tensor.fromArray(b_data, [4, 4]);

  const output = model.forward([a, b]);
  const output_data = output[0].data;

  // Step 5: Verify output = a + b
  let maxError = 0;
  for (let i = 0; i < size; i++) {
    const expected = a_data[i] + b_data[i];
    const actual = output_data[i];
    const error = Math.abs(actual - expected);
    maxError = Math.max(maxError, error);
  }

  console.log(`Max error: ${maxError}`);
  if (maxError > 1e-5) {
    console.error(`FAIL: max error ${maxError} exceeds tolerance 1e-5`);
    process.exit(1);
  }

  console.log("PASS: WebGPU add test passed");
  device.destroy();
}

main().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
