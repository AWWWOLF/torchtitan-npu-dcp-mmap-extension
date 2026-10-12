# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 sparse-attention override: the fused SMLA core.

The LightningIndexer half lives in its own package, :mod:`...deepseek_v4_1.lightning_indexer`,
so the two can be selected independently.
"""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override

from torchtitan_npu.models.common.metadata_extension import MetadataExtension
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2

if TYPE_CHECKING:
    # Annotation-only: the runtime import stays lazy in the factory below so
    # the golden path does not require cann_ops_transformer.
    from .ascendc import AscV41MetadataExtension


@override(
    target=MetadataExtension.Config,
    description="Precompute the AscendC SMLA metadata operators onto the model-built V4.1 masks",
)
def asc_metadata(
    cfg: MetadataExtension.Config,
) -> "AscV41MetadataExtension.Config":
    # Imported lazily so the golden path does not require cann_ops_transformer.
    from .ascendc import AscV41MetadataExtension

    return derive(cfg, AscV41MetadataExtension.Config)


@override(
    target=CompressedSparseInnerAttention2.Config,
    exact=True,
    description="V4.1 SMLA forward and backward with the full-softmax LSE",
)
def asc(cfg: CompressedSparseInnerAttention2.Config):
    from .ascendc import AscV41SparseAttention

    return derive(cfg, AscV41SparseAttention.Config)
