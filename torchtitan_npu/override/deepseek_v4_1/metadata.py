# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The V4.1 fused port's metadata hierarchy.

Three levels, each named for what it is:

``AicpuGeometry``
    the static numbers the four AI_CPU operators are called with (heads, head dim, window,
    the indexer's widths and selection width).  It lives beside the records because the
    context-parallel transform is handed the metadata alone and must rebuild them for this
    rank's frames; nothing in the forward or the backward reads it.

``AicpuMetadata``
    the four operator outputs of one geometry -- ``sparse_flash_mla_metadata``,
    ``sparse_flash_mla_grad_metadata``, ``lightning_indexer_metadata`` and
    ``sparse_lightning_indexer_kl_loss_grad_metadata`` -- plus the geometry they were sized
    with.  One model runs one selection width, so each is one tensor rather than a table.

``KernelFrame``
    one tensor's coordinate frame: its ragged document boundaries, the per-document prefix this
    consumer addresses, and the trailing partial group of a compressed axis.  The fused
    metadata is three of them (queries, sliding-window keys, compressed keys) plus the row.

Consumers read the fields: ``attention_metadata.cmp_k.seqused``,
``attention_metadata.aicpu_metadata.sparse_attn_grad``.  There are no lookup helpers: the entry
*is* one geometry and the model runs one selection width, so there is nothing to look up.  The
geometry is *not* cross-checked against the layer's own config either -- the override is the only
place that states it, and the kernels are where a wrong statement surfaces.
"""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.common.metadata_extension import (
    register_pytree_node_for_dataclass,
)

__all__ = ["AicpuGeometry", "AicpuMetadata", "KernelFrame"]


@dataclass(frozen=True, kw_only=True, slots=True)
class KernelFrame:
    """One tensor's coordinate frame: the boundaries needed to address it.

    ``cu_seqlens`` are the ragged document boundaries (``int32``, ``n_documents + 1``), the form
    the kernels take as ``cu_seqlens_*``.  ``seqused`` is the per-document prefix this frame's
    consumer addresses -- how an operator is told to read a prefix of its operand -- and
    ``residual`` the trailing partial group of a compressed axis (``None`` where the axis is not
    compressed, which the kernels reject a value for).
    """

    cu_seqlens: torch.Tensor
    seqused: torch.Tensor
    residual: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class AicpuGeometry:
    """The static numbers the four AI_CPU operators of one geometry are called with."""

    num_heads: int
    head_dim: int
    window_size: int
    index_topk: int
    index_num_heads: int
    index_head_dim: int


@dataclass(frozen=True, kw_only=True, slots=True)
class AicpuMetadata:
    """The four AI_CPU operator outputs of one geometry, and the numbers they were sized with.

    They depend only on the frames of one forward and that geometry -- never on the operands --
    so they are computed once per microbatch (per rank under context parallelism) at the eager
    boundary and read by every layer of the geometry, in the forward, the full-AC recompute and
    the backward alike.  The tensors are shared, which is sound only while the kernels keep
    taking ``metadata`` as a read-only input: it is the operator's own output, consulted for the
    grid, never an output buffer of an NPU kernel.

    The two indexer tensors are ``None`` on a layer with no compressed axis: such a layer runs no
    selector, so there is nothing to select and nothing to distil.
    """

    geometry: AicpuGeometry
    sparse_attn: torch.Tensor
    sparse_attn_grad: torch.Tensor
    lightning_indexer: torch.Tensor | None = None
    lightning_indexer_kl_loss_grad: torch.Tensor | None = None


register_pytree_node_for_dataclass(KernelFrame)
register_pytree_node_for_dataclass(AicpuGeometry)
register_pytree_node_for_dataclass(AicpuMetadata)
