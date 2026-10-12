# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What the fused ports read off a context-parallel forward's frames.

A sharded forward expresses each rank's per-document reach as the frame's per-sequence length, so
these tests pin the operands the operators actually receive: the query axis is the rank's chunk
grid, the KV axes are the gathered row's grids, and ``seqused_ori_kv - seqused_q`` is exactly this
rank's chunk offset inside the document -- the identity the kernels anchor their window and their
compressed causal limit at.  At one degree the same options carry the boundary differences, which
is what makes a ``cp = 1`` versus ``cp > 1`` comparison a comparison of the scheme.
"""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer import ascendc as selector_ascendc
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc as core_ascendc

# Only the frames are under test, so any window does.
_WINDOW = 128


class _Model:
    """The model's metadata builders, without the rest of the model.

    The ports are handed the metadata, not the model, so grafting the builders is enough; the
    boundaries the tests want are rebuilt as positions, exactly as the hook would see them.
    """

    compress_ratios = (0, 2, 1)
    # The hook is a method, so a stand-in presents the pieces it reaches for; they are taken from
    # the model itself, and ``_frame`` as a staticmethod -- a plain assignment would make it an
    # instance method here.
    get_attention_masks = DeepSeekV41Model.get_attention_masks
    _row_metadata = DeepSeekV41Model._row_metadata
    _sharded_attention_masks = DeepSeekV41Model._sharded_attention_masks
    _frame = staticmethod(DeepSeekV41Model._frame)


def _masks(bounds: list[int], cp_size: int = 1, cp_rank: int = 0):
    positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)
    owner = _Model()
    if cp_size == 1:
        return owner.get_attention_masks(positions)
    slab = positions.numel() // cp_size
    lo = cp_rank * slab
    return owner._sharded_attention_masks(
        owner._row_metadata(positions),
        positions[:, lo : lo + slab],
        cp_size=cp_size,
        cp_rank=cp_rank,
    )


def _lengths(bounds: list[int], dtype=torch.int32) -> torch.Tensor:
    return torch.tensor([end - begin for begin, end in pairwise(bounds)], dtype=dtype)


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("bounds", [[0, 8, 24], [0, 8, 32, 40]], ids=["two-docs", "three-docs"])
def test_one_degree_passes_the_boundary_differences(bounds, ratio):
    """At one degree every per-sequence length is the boundary difference, so they are no-ops."""
    options = core_ascendc._kernel_options(_masks(bounds), ratio, _WINDOW)
    lengths = _lengths(bounds)
    assert torch.equal(options["cu_seqlens_q"], torch.tensor(bounds, dtype=torch.int32))
    assert torch.equal(options["seqused_q"], lengths)
    assert torch.equal(options["seqused_ori_kv"], lengths)
    if ratio == 0:
        # A ratio-0 layer has no compressed stream, and the operators spell that ``None``.
        assert options["seqused_cmp_kv"] is None
        assert options["cu_seqlens_cmp_kv"] is None
        assert options["cmp_residual_kv"] is None
    else:
        assert torch.equal(options["seqused_cmp_kv"], lengths // ratio)
        assert torch.equal(options["cu_seqlens_cmp_kv"], torch.tensor(bounds, dtype=torch.int32) // ratio)


@pytest.mark.parametrize("cp_size, cp_rank", [(2, 1), (4, 3), (8, 7)], ids=["R2r1", "R4r3", "R8r7"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_the_anchor_is_this_ranks_chunk_offset(cp_size, cp_rank, ratio):
    """The core gets chunk queries over the row, and the two lengths differ by the chunk offset.

    ``seqused_ori_kv - seqused_q = r * Q_d`` is the whole kernel contract of this mode: it is how
    the window and the compressed causal limit end up anchored at this rank's chunk instead of at
    the row's end.
    """
    bounds = [0, 16, 48]
    row = torch.tensor(bounds, dtype=torch.int32)
    chunk = _lengths(bounds) // cp_size
    reach = (cp_rank + 1) * chunk
    options = core_ascendc._kernel_options(_masks(bounds, cp_size, cp_rank), ratio, _WINDOW)

    assert torch.equal(options["cu_seqlens_q"], (row // cp_size).to(torch.int32))
    assert torch.equal(options["cu_seqlens_ori_kv"], row)
    assert torch.equal(options["seqused_q"], chunk)
    assert torch.equal(options["seqused_ori_kv"], reach)
    assert torch.equal(options["seqused_ori_kv"] - options["seqused_q"], cp_rank * chunk)
    assert torch.equal(options["seqused_cmp_kv"], reach // ratio)
    assert torch.equal(options["cu_seqlens_cmp_kv"], (row // ratio).to(torch.int32))


@pytest.mark.parametrize("cp_size, cp_rank", [(2, 1), (4, 3)], ids=["R2r1", "R4r3"])
def test_the_selector_gets_the_same_chunk_and_reach(cp_size, cp_rank):
    """The selection is scored in the query frame against the row's compressed grid.

    Without these two lengths a rank would score its chunks against the whole row and spend its
    top-k on blocks it may not read.  They sit in the same option dict as the boundaries, because
    every operator this selector can pick takes the same pair -- the bf16 V2 indexer at
    ``legacy=True`` and the quantized pair otherwise -- so one builder serves all of them.
    """
    bounds = [0, 16, 48]
    row = torch.tensor(bounds, dtype=torch.int32)
    chunk = _lengths(bounds) // cp_size
    masks = _masks(bounds, cp_size, cp_rank)
    options = selector_ascendc._kernel_options(masks, 2)

    assert torch.equal(options["cu_seqlens_q"], (row // cp_size).to(torch.int32))
    assert torch.equal(options["cu_seqlens_k"], (row // 2).to(torch.int32))
    assert torch.equal(options["seqused_q"], chunk)
    assert torch.equal(options["seqused_k"], (cp_rank + 1) * chunk // 2)
