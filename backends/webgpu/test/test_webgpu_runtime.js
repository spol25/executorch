/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Runtime test for the WebGPU backend.
// Requires a browser with WebGPU support (Chrome 113+) or Node.js 25+.
//
// Usage (Node.js):
//   node test_webgpu_runtime.js
//
// Usage (browser):
//   Open test_webgpu_browser.html in Chrome.

const fs = require("fs");
const path = require("path");

async function main() {
  const gpu = navigator.gpu;
  if (!gpu) {
    console.error("WebGPU not available");
    process.exit(1);
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    console.error("Failed to get WebGPU adapter");
    process.exit(1);
  }

  const device = await adapter.requestDevice();
  console.log("WebGPU device acquired");

  const wasmPath = path.resolve(
    __dirname,
    "../../../cmake-out-webgpu/extension/wasm/executorch_wasm.js"
  );
  if (!fs.existsSync(wasmPath)) {
    console.error(`Wasm module not found at: ${wasmPath}`);
    console.error(
      "Build first: cmake --build cmake-out-webgpu --target executorch_wasm_module"
    );
    process.exit(1);
  }

  const createModule = require(wasmPath);
  const Module = await createModule({
    preinitializedWebGPUDevice: device,
  });
  console.log("Wasm module loaded");

  const ptePath = path.resolve(__dirname, "../../../webgpu_add_test.pte");
  if (!fs.existsSync(ptePath)) {
    console.error(`Model not found at: ${ptePath}`);
    process.exit(1);
  }

  const pteBuffer = new Uint8Array(fs.readFileSync(ptePath));
  const model = await Module.Module.load(pteBuffer);
  await model.loadMethod("forward");
  console.log("Model loaded");

  const dim = 1024;
  const size = dim * dim;
  const aData = new Array(size);
  const bData = new Array(size);
  for (let i = 0; i < size; i++) {
    aData[i] = i * 1.0;
    bData[i] = i * 2.0;
  }

  const a = Module.Tensor.fromArray([dim, dim], aData);
  const b = Module.Tensor.fromArray([dim, dim], bData);

  const output = await model.forward([a, b]);
  const outTensor = Array.isArray(output) ? output[0] : output;
  const outputData = outTensor.data;

  let maxError = 0;
  for (let i = 0; i < Math.min(size, 1024); i++) {
    const expected = aData[i] + bData[i];
    const error = Math.abs(outputData[i] - expected);
    maxError = Math.max(maxError, error);
  }

  console.log(`Max error: ${maxError}`);
  if (maxError > 1e-3) {
    console.error(`FAIL: max error ${maxError} exceeds tolerance 1e-3`);
    process.exit(1);
  }

  console.log("PASS: WebGPU add test passed");
  device.destroy();
}

main().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
