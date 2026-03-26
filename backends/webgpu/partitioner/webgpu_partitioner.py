# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import final, List, Mapping, Optional, Tuple

import torch

from executorch.backends.webgpu.webgpu_preprocess import WebGPUBackend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data, tag_mutated_buffer
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

logger: logging.Logger = logging.getLogger("")
logger.setLevel(logging.INFO)

# Ops supported by the WebGPU prototype
_WEBGPU_SUPPORTED_OPS = {
    exir_ops.edge.aten.add.Tensor,
}


class WebGPUSupportedOperators(OperatorSupportBase):
    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.op != "call_function":
            return False

        if node.target not in _WEBGPU_SUPPORTED_OPS:
            return False

        # Only support fp32 tensors for now
        if "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, torch.Tensor) and val.dtype != torch.float32:
                logger.info(
                    f"[WebGPU Partitioner] Skipping {node.target}: "
                    f"unsupported dtype {val.dtype}"
                )
                return False

        return True


@final
class WebGPUPartitioner(Partitioner):
    def __init__(
        self,
        compile_options: Optional[List[CompileSpec]] = None,
    ) -> None:
        compile_specs = compile_options or []
        self.delegation_spec = DelegationSpec(
            WebGPUBackend.__name__, compile_specs
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            WebGPUSupportedOperators(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        partition_tags = {}
        for partition in partition_list:
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        num_partitions = len(partition_list)
        if num_partitions == 0:
            logger.warning("No WebGPU subgraphs can be partitioned!")
        else:
            logger.info(
                f"Found {num_partitions} WebGPU subgraph(s) to be partitioned."
            )

        tag_constant_data(exported_program)
        tag_mutated_buffer(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
