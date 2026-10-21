# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4 SFT override claims the aligned dataloader, not the upstream text one."""

from torchtitan.config.override import OverrideConfig, apply_overrides

from torchtitan_npu.models.deepseek_v4.chat_dataset import DeepSeekV4ChatDataLoader
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_flash

_SFT = "torchtitan_npu.override.deepseek_v4.chat_dataloader.sft"


def test_sft_override_claims_the_aligned_dataloader():
    config = deepseek_v4_flash()
    # The registry installs the per-document aligned loader, so the SFT override must
    # target that concrete type; a path is required by the factory.
    config.dataloader.dataset_path = "tests/assets/c4_test/data.json"
    lines = apply_overrides(OverrideConfig(imports=[_SFT]), config)

    assert lines, "the SFT override must claim the V4 dataloader"
    assert isinstance(config.dataloader, DeepSeekV4ChatDataLoader.Config)
    assert config.dataloader.per_doc_alignment == 128
