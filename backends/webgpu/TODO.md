# WebGPU Backend — TODO

## Current State (Prototype)
- Single op: `aten.add.Tensor` (fp32, buffer storage)
- Reuses Vulkan FlatBuffer serialization format
- Built-in WGSL shader (not embedded in .pte)

## Performance: Command Encoding Overhead
WebGPU `GPUCommandBuffer` is single-use (no equivalent to Vulkan's cached command lists).
Per-dispatch Wasm→JS bridge cost: ~3-10μs. For 1000-node graphs this becomes 3-10ms per inference.

**Primary mitigation: mega-kernel fusion.** Generate fused WGSL shaders for chains of
element-wise ops (add→relu→mul→clamp) at compile time. Embed via the existing
`shaders: [VkBytes]` field in schema.fbs. This eliminates both encoding overhead and
global memory round-trips. Target: reduce 1000 dispatches to ~50-100.

## Next Steps
1. **More ops**: sub, mul, relu, linear (matmul), softmax, layer_norm
2. **WGSL codegen**: YAML+template system (like `gen_vulkan_spv.py`) for dtype variants
3. **fp16 support**: Feature-detect `shader-f16`, fallback to fp32
4. **Dynamic shapes**: Buffer reallocation on shape change
5. **Buffer pooling**: Reuse GPU buffers to avoid OOM at scale
6. **Pipeline caching**: `createComputePipelineAsync()` + IndexedDB cache
7. **Profiling**: Wire WebGPU timestamp queries into ETDump/EventTracer
8. **Embedded shaders**: Runtime parses `shaders[]` from FlatBuffer for compile-time fusion
9. **JSPI migration**: Replace ASYNCIFY with JavaScript Promise Integration for non-blocking readback
10. **LLM support**: KV cache management, Flash Attention in WGSL, quantized ops (int4/int8)
