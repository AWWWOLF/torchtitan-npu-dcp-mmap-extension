# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Transformers attention patch must reproduce the model's reference attention bitwise.

The inference alignment compares the two implementations element for element, and the
model's reference core is Attention Gym's ``gather_attn``.  The patch therefore runs that
same function with the operands ``CompressedSparseInnerAttention.forward`` builds, so the
attention arithmetic is identical by construction; this test pins the property on the three
layer shapes -- window-only, HCA and CSA -- in fp32 and bf16.
"""

from types import SimpleNamespace

import pytest
import torch
from attn_gym.sparse import gather_attn

from scripts.numerical_alignment.deepseek_v4 import transformers_dsv4_patch as patch

pytestmark = pytest.mark.tooling

_HEADS, _SEQ, _DIM = 2, 8, 4
_WINDOW = 4
_COMPRESSED = 2
_SCALE = 0.5


def _operands(dtype):
    torch.manual_seed(0)
    query = torch.randn(1, _HEADS, _SEQ, _DIM, dtype=dtype)
    key = torch.randn(1, 1, _SEQ + _COMPRESSED, _DIM, dtype=dtype)
    sinks = torch.randn(_HEADS)
    return query, key, sinks


def _module(*, compressor=None, sinks):
    return SimpleNamespace(compressor=compressor, sinks=sinks, sliding_window=_WINDOW, training=False)


def _core_attention(query, local_kv, sparse_kv, kv_indices, sinks, cu_seqlens_k):
    """The call ``CompressedSparseInnerAttention.forward`` makes for one packed document."""
    return gather_attn(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink=sinks,
        sliding_window_size=_WINDOW,
        cu_seqlens=torch.tensor([0, _SEQ], dtype=torch.int32),
        cu_seqlens_k=cu_seqlens_k,
        impl="reference",
        scale=_SCALE,
    ).transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_window_only_matches_the_core(dtype):
    query, key, sinks = _operands(dtype)
    out, weights = patch._tt_transformers_attention_forward(
        _module(sinks=sinks), query, key[:, :, :_SEQ], None, None, _SCALE
    )
    offsets = torch.tensor([0, _SEQ], dtype=torch.int32)
    expected = _core_attention(
        query,
        key[:, :, :_SEQ],
        query.new_zeros(1, 1, 0, _DIM),
        query.new_empty(1, _SEQ, 0, dtype=torch.long),
        sinks,
        offsets,
    )
    assert weights is None
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hca_layer_matches_the_core(dtype):
    query, key, sinks = _operands(dtype)
    out, _ = patch._tt_transformers_attention_forward(
        _module(compressor=SimpleNamespace(compress_rate=4), sinks=sinks), query, key, None, None, _SCALE
    )
    candidates = torch.arange(_COMPRESSED).unsqueeze(0).expand(_SEQ, -1)
    limit = (torch.arange(1, _SEQ + 1) // 4).unsqueeze(1)
    indices = torch.where(candidates < limit, candidates, -1).unsqueeze(0).contiguous()
    expected = _core_attention(
        query,
        key[:, :, :_SEQ],
        key[:, :, _SEQ:],
        indices,
        sinks,
        torch.tensor([0, _COMPRESSED], dtype=torch.int32),
    )
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_csa_layer_matches_the_core(dtype):
    query, key, sinks = _operands(dtype)
    module = _module(compressor=SimpleNamespace(indexer=object()), sinks=sinks)
    picks = torch.tensor([[[0, -1], [0, 1], [1, -1], [0, 1], [1, 0], [1, 0], [1, 1], [0, 1]]])
    module._tt_compressed_indices = picks
    out, _ = patch._tt_transformers_attention_forward(module, query, key, None, None, _SCALE)
    expected = _core_attention(
        query,
        key[:, :, :_SEQ],
        key[:, :, _SEQ:],
        picks.contiguous(),
        sinks,
        torch.tensor([0, _COMPRESSED], dtype=torch.int32),
    )
    assert torch.equal(out, expected)
