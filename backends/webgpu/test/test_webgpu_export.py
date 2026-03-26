# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
from executorch.backends.webgpu import WebGPUPartitioner
from executorch.exir import to_edge_transform_and_lower


class AddModule(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class TestWebGPUExport(unittest.TestCase):
    def test_add_module_export(self) -> None:
        model = AddModule()
        example_inputs = (torch.randn(4, 4), torch.randn(4, 4))

        ep = torch.export.export(model, example_inputs)
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[WebGPUPartitioner()]
        ).to_executorch()

        # Verify the program has a WebGPUBackend delegate
        found_webgpu = False
        for plan in et_program.executorch_program.execution_plan:
            for delegate in plan.delegates:
                if delegate.id == "WebGPUBackend":
                    found_webgpu = True
                    break
        self.assertTrue(found_webgpu, "Expected WebGPUBackend delegate in .pte")

        # Write to a temp file and verify the header
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            f.write(et_program.buffer)
            pte_path = f.name

        try:
            with open(pte_path, "rb") as f:
                pte_data = f.read()
            self.assertGreater(len(pte_data), 0, "PTE file should not be empty")

            # Verify the program can be saved and is non-trivial
            self.assertGreater(len(pte_data), 100)
        finally:
            os.unlink(pte_path)


if __name__ == "__main__":
    unittest.main()
