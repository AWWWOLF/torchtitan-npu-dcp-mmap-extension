"""DeepSeek-V4 sparse-attention overrides (registry-facing module).

The AscendC fused implementation (metadata extension + kernel) lives in
``ascendc.py``, the PyPTO variant in ``pypto.py``; the registrations are defined
here so the override paths stay
``override.deepseek_v4.sparse_attn.{asc_li_metadata, asc_li, asc_metadata, asc,
pypto, pypto_li}``.  The model's ``build_attention_masks`` owns the per-batch
metadata construction (including context parallel); the ``asc_metadata``
override injects the AscendC kernel-metadata extension (the model dir stays
backend-agnostic).  The reference tier is attn-gym's own attention, i.e. the
model default, so no override selects it.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from torchtitan.config import derive, override
from torchtitan.models.common.decoder_sharding import dense_param_placement

from torchtitan_npu.models.common.metadata_extension import (
    LightningIndexerMetadata,
    SparseAttentionMetadata,
)
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.compressor import LightningIndexer

if TYPE_CHECKING:
    from .ascendc import AscCompressedSparseInnerAttention
    from .pypto import PyPTOCompressedSparseInnerAttention


@override(
    target=SparseAttentionMetadata.Config,
    exact=True,
    description=("Precompute the AscendC sparse-attention metadata kernels onto the model-built DSV4 varlen contract"),
)
def asc_metadata(
    cfg: SparseAttentionMetadata.Config,
) -> SparseAttentionMetadata.Config:
    # Imported lazily so the reference path does not require cann_ops_transformer.
    from .ascendc import AscMetadataExtension

    return derive(cfg, SparseAttentionMetadata.Config, extension=AscMetadataExtension)


@override(
    target=LightningIndexerMetadata.Config,
    exact=True,
    description="Precompute AscendC BF16 LightningIndexer metadata",
)
def asc_li_metadata(
    cfg: LightningIndexerMetadata.Config,
):
    from .ascendc import AscLightningIndexerMetadata

    return derive(cfg, AscLightningIndexerMetadata.Config)


@override(
    target=LightningIndexer.Config,
    exact=True,
    description="Use AscendC TND LightningIndexer",
)
def asc_li(cfg: LightningIndexer.Config):
    from .ascendc import AscLightningIndexer

    return derive(cfg, AscLightningIndexer.Config)


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description=("Use cann_ops_transformer TND LightningIndexer and SparseFlashMLA with fused SASG/SLIG backward"),
)
def asc(
    cfg: CompressedSparseInnerAttention.Config,
    indexer_loss_coeff: float = 1.0,
) -> "AscCompressedSparseInnerAttention.Config":
    # Imported lazily so the reference path does not require cann_ops_transformer.
    from .ascendc import AscCompressedSparseInnerAttention

    # ``sharding_config`` is copied from the source by ``derive``; the
    # sharding pass already ran before the overrides (update_from_config),
    # so the derived config keeps the sharding-pass values as-is.
    result = derive(
        cfg,
        AscCompressedSparseInnerAttention.Config,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    # The AscendC core's own ``_indexer_loss_acc`` accumulator buffer needs its
    # placement declared (a replicated scalar) for the state-distribution
    # pass.
    sharding_config = result.sharding_config
    assert sharding_config is not None, "the asc override requires the sharding config"
    sharding_config.state_shardings["_indexer_loss_acc"] = dense_param_placement(tp=spmd.R)
    return result


@override(
    target=CompressedSparseInnerAttention.Config,
    exact=True,
    description="Use PyPTO LI/LIG with AscendC TND SMLA/SMLAG",
)
def pypto(
    cfg: CompressedSparseInnerAttention.Config,
    indexer_loss_coeff: float = 1.0,
) -> "PyPTOCompressedSparseInnerAttention.Config":
    # Imported lazily so the reference path does not require cann_ops_transformer.
    from .pypto import PyPTOCompressedSparseInnerAttention

    return derive(
        cfg,
        PyPTOCompressedSparseInnerAttention.Config,
        indexer_loss_coeff=indexer_loss_coeff,
    )


@override(
    target=LightningIndexer.Config,
    exact=True,
    description="Use PyPTO LightningIndexer",
)
def pypto_li(cfg: LightningIndexer.Config):
    from .pypto import PyPTOLightningIndexer

    return derive(cfg, PyPTOLightningIndexer.Config)
