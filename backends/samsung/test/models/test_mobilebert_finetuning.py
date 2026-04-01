# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import os
import unittest

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig

from executorch.examples.samsung.scripts.mobilebert_finetune import MobileBertFinetune
from transformers import AutoTokenizer


def patch_mobilebert_finetuning(model_cache_dir: str):
    assert os.path.isdir(
        model_cache_dir
    ), "Can not found model cache dirrecory for mobilebert finetuning"

    def _monkeypatch_load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(model_cache_dir)
        return tokenizer

    old_func = MobileBertFinetune.load_tokenizer
    MobileBertFinetune.load_tokenizer = _monkeypatch_load_tokenizer
    return old_func


def recover_mobilebert_finetuning(old_func):
    MobileBertFinetune.load_tokenizer = old_func


class Test_Milestone_MobileBertFinetune(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert (model_cache_dir := os.getenv("MODEL_CACHE")), "MODEL_CACHE not set!"
        cls.model_cache_dir = os.path.join(model_cache_dir, "mobilebert")
        cls._old_func = patch_mobilebert_finetuning(cls.model_cache_dir)

    @classmethod
    def tearDownClass(cls):
        recover_mobilebert_finetuning(cls._old_func)

    # This model need to be fixed according new transformer version
    @unittest.skip
    def test_mobilebert_finetuning_fp16(self):
        mobilebert_finetune = MobileBertFinetune()
        model, _ = mobilebert_finetune.get_finetune_mobilebert(self.model_cache_dir)
        example_input = mobilebert_finetune.get_example_inputs()
        tester = SamsungTester(
            model, example_input, [gen_samsung_backend_compile_spec(TestConfig.chipset)]
        )

        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_input, atol=0.008)
        )
