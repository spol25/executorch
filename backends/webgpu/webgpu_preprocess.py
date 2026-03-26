# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from functools import partial
from typing import final, List

from executorch.backends.vulkan.serialization.vulkan_graph_builder import VkGraphBuilder
from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkStorageType,
)
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    serialize_vulkan_graph,
)
from executorch.backends.vulkan.vulkan_preprocess import apply_passes
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.utils import DelegateMappingBuilder
from executorch.exir.memory_planning import greedy, MemoryPlanningAlgorithmSuite
from executorch.exir.passes import MemoryPlanningPass, SpecPropPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass


@final
class WebGPUBackend(BackendDetails):
    @classmethod
    # pyre-ignore
    def preprocess(
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        # Minimal pass pipeline for WebGPU prototype:
        # 1. SpecPropPass to annotate tensor nodes with TensorSpec
        # 2. Memory planning for allocation IDs
        # No Vulkan-specific passes (TagMemoryMetaPass, insert_prepack_nodes, etc.)
        program = apply_passes(program, [SpecPropPass()])

        greedy_memory_planning = partial(
            greedy, allow_overlapping_allocations=False
        )
        mem_planning_suite = MemoryPlanningAlgorithmSuite(
            algo_list=[greedy_memory_planning]
        )
        program.graph_module.encounter_to_out_var_failure = True
        program = apply_passes(
            program,
            [
                ConstraintBasedSymShapeEvalPass(),
                MemoryPlanningPass(memory_planning_algo=mem_planning_suite),
            ],
        )

        # Build the graph using Vulkan's serialization with buffer-only storage.
        # WebGPU has no 3D storage textures, so we force BUFFER for all tensors.
        graph_builder = VkGraphBuilder(
            program,
            DelegateMappingBuilder(generated_identifiers=True),
            downcast_64_bit=True,
            force_fp16=False,
        )
        vk_graph = graph_builder.build_graph()

        # Override storage type to BUFFER for all tensor values
        for val in vk_graph.values:
            tensor = getattr(val, "value", None)
            if tensor is not None and hasattr(tensor, "storage_type"):
                tensor.storage_type = VkStorageType.BUFFER

        return PreprocessResult(
            processed_bytes=serialize_vulkan_graph(
                vk_graph, graph_builder.const_tensors, []
            ),
            debug_handle_map=graph_builder.delegate_mapping_builder.get_delegate_mapping(),
            data_store_output=graph_builder.named_data_store.get_named_data_store_output(),
        )
