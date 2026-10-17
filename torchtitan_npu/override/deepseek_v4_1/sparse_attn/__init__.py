# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 sparse-attention override: the fused SMLA core.

The LightningIndexer half lives in its own package, :mod:`...deepseek_v4_1.lightning_indexer`,
so the two can be selected independently.

There is one entry, not two: the precomputed SMLA records belong to the fused core's own
config (upstream #4897 ownership), so swapping the core in builds them.  A second override
on a metadata node could only be turned on against the attention it is meant to serve, and
two overrides claiming the same stack is exactly the wiring this removes.

The records are sized from the numbers this entry states, defaulting to the released V4.1-Flash
shape, because a kernel record is the one thing here the model's own config has no business
knowing: the reference core reads only the row's boundaries and its own ratio.  Nothing falls back
to a per-layer AI_CPU call -- a width or a geometry the built records do not match raises.
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2


@override(
    target=CompressedSparseInnerAttention2.Config,
    exact=True,
    description="V4.1 SMLA forward and backward with the full-softmax LSE",
)
def asc(
    cfg: CompressedSparseInnerAttention2.Config,
    *,
    num_heads: int = 64,
    head_dim: int = 512,
    index_topk: int = 512,
    index_num_heads: int = 32,
    index_head_dim: int = 128,
):
    """The fused core, with the record geometry stated on the override.

    The defaults are the released V4.1-Flash shape, so the shipped recipes need no arguments;
    a recipe that runs other widths states them here (``asc={"num_heads":8,"head_dim":64,
    "index_topk":32,"index_num_heads":8,"index_head_dim":32}``).  The reference config carries
    none of them: only a fused port's records are sized by this, and the record's own guard
    rejects a geometry that disagrees with the layer reading it.
    """
    from .ascendc import AscV41SparseAttention

    return derive(
        cfg,
        AscV41SparseAttention.Config,
        num_heads=num_heads,
        head_dim=head_dim,
        index_topk=index_topk,
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
    )
