# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torchao.quantization.qat import QATStep
from torchao_npu.configs.module_swap_configs.deepseek_v4_1.quant_lightning_indexer import (
    DeepSeekV41QuantLightningIndexerConfig,
    _quant_v41_lightning_indexer_transform,
)
from torchao_npu.quantized_modules.deepseek_v4_1 import lightning_indexer


def _metadata(tokens=4, compressed_tokens=2):
    compressed = SimpleNamespace(
        cu_seqlens=torch.tensor([0, compressed_tokens], dtype=torch.int32),
        residual=torch.zeros(1, dtype=torch.int32),
        seqused=torch.tensor([compressed_tokens], dtype=torch.int32),
    )
    return SimpleNamespace(
        q=SimpleNamespace(
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32),
            seqused=torch.tensor([tokens], dtype=torch.int32),
        ),
        frame_for=lambda ratio: compressed,
    )


def _selector(*, mode="full", candidate_topk_blocks=3):
    selector = lightning_indexer.QuantV41LightningIndexer()
    selector.index_topk = 4
    selector.compress_ratio = 2
    selector.candidate_topk_blocks = candidate_topk_blocks
    selector.candidate_block_size = 2 if candidate_topk_blocks > 0 else -1
    selector.mode = mode
    selector.has_candidate_pool = candidate_topk_blocks > 0
    selector.num_global_queries = torch.tensor(16.0)
    return selector


@pytest.fixture
def bypass_mxfp4_pack(monkeypatch):
    monkeypatch.setattr(lightning_indexer, "_pack_mxfp4", lambda value: (value, value.new_zeros(1)))


def _inputs():
    return (
        torch.zeros(1, 4, 2, 64, dtype=torch.bfloat16, requires_grad=True),
        torch.zeros(1, 2, 64, dtype=torch.bfloat16, requires_grad=True),
        torch.zeros(1, 4, 2, dtype=torch.float32, requires_grad=True),
    )


@pytest.mark.parametrize(
    "query_count, expected_teacher",
    [(8.0, 1.0), (32.0, 0.25), (None, None)],
    ids=["two-rows", "eight-rows", "missing-count"],
)
def test_qli_source_sorts_selection_publishes_pool_and_runs_slikg(
    monkeypatch, bypass_mxfp4_pack, query_count, expected_teacher
):
    q, k, w = _inputs()
    selector = _selector()
    selector.num_global_queries = None if query_count is None else torch.tensor(query_count)
    masks = _metadata()
    masks.q.seqused = torch.tensor([2], dtype=torch.int32)
    masks.frame_for(2).seqused = torch.tensor([1], dtype=torch.int32)
    unsorted = torch.tensor([3, -1, 1, 2], dtype=torch.int32).repeat(4, 1).unsqueeze(1)
    candidate_blocks = torch.tensor(
        [[[0, 1, -1]], [[0, -1, -1]], [[1, 0, -1]], [[1, -1, -1]]],
        dtype=torch.int32,
    )
    seen = {}

    def qli_metadata(**kwargs):
        seen["qli_metadata"] = kwargs
        return torch.tensor([7], dtype=torch.int32)

    def qli(*args, **kwargs):
        seen["qli"] = (args, kwargs)
        return unsorted, torch.empty(0), candidate_blocks, torch.empty(0)

    def slikg_metadata(**kwargs):
        seen["slikg_metadata"] = kwargs
        return torch.tensor([9], dtype=torch.int32)

    def slikg(**kwargs):
        seen["slikg"] = kwargs
        return (
            torch.full_like(kwargs["q"], 2),
            torch.full_like(kwargs["k"], 3),
            torch.full_like(kwargs["w"], 4),
            torch.empty(0),
        )

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(
            quant_lightning_indexer_metadata=qli_metadata,
            quant_lightning_indexer=qli,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad_metadata",
        slikg_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad",
        slikg,
        raising=False,
    )

    indices, scores, candidate = selector(q, k, w, masks, candidates_BL1C=None)
    # A later microbatch must not change the denominator saved with this graph.
    selector.num_global_queries = torch.tensor(64.0)
    if query_count is None:
        with pytest.raises(RuntimeError, match=r"Set Selector\.num_global_queries"):
            scores.backward(torch.full_like(scores, 8.0))
        assert "slikg" not in seen
        return
    scores.backward(torch.full_like(scores, 8.0))

    assert torch.equal(indices[0, 0], torch.tensor([3, 2, 1, -1], dtype=torch.int32))
    assert scores.shape == indices.shape and scores.requires_grad
    assert torch.equal(candidate, candidate_blocks.unsqueeze(0))
    assert seen["qli_metadata"]["candidate_topk_blocks"] == 3
    assert seen["qli"][1]["return_value"] is False
    assert torch.equal(seen["slikg"]["sparse_indices"], indices.reshape(4, 1, 4))
    assert torch.equal(seen["slikg"]["attn_softmax_l1_norm"], torch.full((4, 1, 4), expected_teacher))
    for options in (seen["qli_metadata"], seen["qli"][1], seen["slikg_metadata"], seen["slikg"]):
        torch.testing.assert_close(options["seqused_q"], torch.tensor([2], dtype=torch.int32))
        torch.testing.assert_close(options["seqused_k"], torch.tensor([1], dtype=torch.int32))
    assert torch.equal(q.grad, torch.full_like(q, 2))
    assert torch.equal(k.grad, torch.full_like(k, 3))
    assert torch.equal(w.grad, torch.full_like(w, 4))


def test_qsli_searcher_recovers_candidate_lengths_and_passes_pool_through(monkeypatch, bypass_mxfp4_pack):
    q, k, w = _inputs()
    selector = _selector(mode="reindex", candidate_topk_blocks=4)
    candidate = torch.tensor(
        [[[[0, 2, -1, -1]], [[1, -1, -1, -1]], [[0, 1, 2, -1]], [[2, -1, -1, -1]]]],
        dtype=torch.int32,
    )
    topk_indices = torch.arange(4, dtype=torch.int32).repeat(4, 1).unsqueeze(1)
    seen = {}

    def metadata(block_length, **kwargs):
        seen["block_length"] = block_length
        seen["metadata"] = kwargs
        return torch.tensor([7], dtype=torch.int32)

    def qsli(*args, **kwargs):
        seen["qsli"] = (args, kwargs)
        return topk_indices, torch.empty(0)

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(
            quant_sparse_lightning_indexer_metadata=metadata,
            quant_sparse_lightning_indexer=qsli,
        ),
        raising=False,
    )

    indices, scores, candidate_out = selector(q, k, w, _metadata(), candidates_BL1C=candidate)

    assert indices.shape == scores.shape == (1, 4, 4)
    torch.testing.assert_close(candidate_out, candidate, rtol=0, atol=0)
    assert torch.equal(seen["block_length"], torch.tensor([[2], [1], [3], [1]], dtype=torch.int32))
    assert torch.equal(seen["qsli"][0][4], candidate.squeeze(0))
    assert seen["qsli"][0][7:] == (1, 2)
    assert seen["qsli"][1]["descale_k"].numel() == 1
    for options in (seen["metadata"], seen["qsli"][1]):
        torch.testing.assert_close(options["seqused_q"], torch.tensor([4], dtype=torch.int32))
        torch.testing.assert_close(options["seqused_k"], torch.tensor([2], dtype=torch.int32))


def test_qli_without_candidate_pool_does_not_publish_kernel_candidate_outputs(monkeypatch, bypass_mxfp4_pack):
    q, k, w = _inputs()
    selector = _selector(candidate_topk_blocks=-1)
    candidate_blocks = torch.zeros((4, 1, 1), dtype=torch.int32)
    topk_indices = torch.arange(4, dtype=torch.int32).repeat(4, 1).unsqueeze(1)
    seen = {}

    def metadata(**kwargs):
        seen["metadata"] = kwargs
        return torch.tensor([7], dtype=torch.int32)

    def qli(*args, **kwargs):
        return topk_indices, torch.empty(0), candidate_blocks, torch.ones_like(candidate_blocks)

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(
            quant_lightning_indexer_metadata=metadata,
            quant_lightning_indexer=qli,
        ),
        raising=False,
    )

    indices, scores, candidate = selector(q, k, w, _metadata(), candidates_BL1C=None)

    assert indices.shape == scores.shape == (1, 4, 4)
    assert candidate is None
    assert seen["metadata"]["candidate_topk_blocks"] == -1
    assert seen["metadata"]["candidate_block_size"] == -1


def test_candidate_source_requires_an_enabled_pool():
    with pytest.raises(ValueError, match="candidate source requires"):
        lightning_indexer._validate_candidate_role(
            candidate_topk_blocks=-1,
            candidate_block_size=-1,
            is_source=True,
            candidate=None,
        )


def test_pack_mxfp4_uses_ds41_byte_layout(monkeypatch):
    q = torch.zeros(2, 3, 64, dtype=torch.bfloat16)
    calls = {}

    def dynamic_mx_quant(value, **kwargs):
        calls["value"] = value
        calls["kwargs"] = kwargs
        return torch.zeros(2, 3, 32, dtype=torch.uint8), torch.zeros(2, 3, 1, 2, dtype=torch.uint8)

    monkeypatch.setattr(lightning_indexer.torch_npu, "npu_dynamic_mx_quant", dynamic_mx_quant)

    data, scale = lightning_indexer._pack_mxfp4(q)

    assert data.shape == (2, 3, 32) and data.dtype is torch.uint8
    assert scale.shape == (2, 3, 1, 2) and scale.dtype is torch.uint8
    assert calls["value"].is_contiguous()
    assert calls["kwargs"]["block_size"] == 32


def test_module_swap_installs_and_restores_the_v41_selector_forward(monkeypatch):
    class SelectorHost(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return "original", args, kwargs

    host = SelectorHost()
    original_forward = host.forward
    captured = {}

    def quantized_forward(self, q, k, w, attention_masks, *, candidates_BL1C):
        captured.update(self=self, args=(q, k, w, attention_masks), candidate=candidates_BL1C)
        return "quantized"

    monkeypatch.setattr(lightning_indexer.QuantV41LightningIndexer, "forward", quantized_forward)
    _quant_v41_lightning_indexer_transform(host, DeepSeekV41QuantLightningIndexerConfig())
    operands = tuple(object() for _ in range(4))
    candidate = object()

    assert host(*operands, candidates_BL1C=candidate) == "quantized"
    assert captured == {"self": host, "args": operands, "candidate": candidate}

    _quant_v41_lightning_indexer_transform(host, DeepSeekV41QuantLightningIndexerConfig(step=QATStep.CONVERT))
    result = host(*operands, candidates_BL1C=candidate)

    assert result[0] == "original"
    assert host._torchao_npu_original_forward == original_forward
