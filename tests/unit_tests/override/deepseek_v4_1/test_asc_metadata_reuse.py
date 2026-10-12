# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The precomputed SMLA metadata: built once per microbatch, read per layer.

``AscV41MetadataExtension`` moves the two AI_CPU metadata operators off the
per-layer hot path (forward, full-AC recompute and backward alike) to the eager
mask-building boundary, one record per distinct ``(compress_ratio, cmp_topk)``
geometry the stack runs.  These tests pin the three contracts that make the
reuse safe: what the extension precomputes, the lookup's hit/miss/guard
semantics, and the fused core reading the record instead of launching.
"""

import importlib

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.checkpoint import DefaultDeviceType

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan.config import derive
from torchtitan_npu.models.common.metadata_extension import MetadataExtension
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import (
    AscV41BlockMetadata,
    AscV41Metadata,
    AscV41MetadataExtension,
    _asc_v41_record,
)

_TOKENS = torch.arange(128).remainder(32).unsqueeze(0)
_POSITIONS = torch.arange(128).unsqueeze(0)


def _fake_metadata_ops(monkeypatch):
    """Count the AI_CPU metadata launches and keep each call's output and geometry.

    ``raising=False`` keeps the fakes installable on runners without
    ``cann_ops_transformer`` registered at all; the operators are never really
    run either way.
    """
    state = {"fwd": [], "bwd": []}

    def _fwd(*args, **kwargs):
        out = torch.zeros(1, dtype=torch.int32)
        state["fwd"].append((kwargs, out))
        return out

    def _bwd(*args, **kwargs):
        out = torch.zeros(1, dtype=torch.int32)
        state["bwd"].append((kwargs, out))
        return out

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", _fwd, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", _bwd, raising=False)
    return state


def _hand_table(metadata, records):
    """Wrap a model-built metadata as the Asc subclass, records included.

    The records are hand-built (the operators are the extension's job), which
    is also how a caller that never ran the extension looks to the lookup.
    """
    return AscV41Metadata(
        q=metadata.q,
        swa_k=metadata.swa_k,
        cmp_k=metadata.cmp_k,
        positions_q=metadata.positions_q,
        exchange=metadata.exchange,
        asc=records,
    )


def test_the_extension_builds_one_record_per_distinct_geometry(monkeypatch, v41_metadata):
    """One record per ``(compress_ratio, cmp_topk)``, the only per-layer inputs the operators vary on."""
    metadata = v41_metadata([0, 4, 8])  # frames at ratios (1, 2)
    state = _fake_metadata_ops(monkeypatch)
    extension = AscV41MetadataExtension.Config(
        window_size=2,
        num_heads=2,
        head_dim=4,
        index_topk=3,
        compress_ratios=(0, 1, 2),
    ).build()

    extended = extension(metadata)

    assert isinstance(extended, AscV41Metadata)
    # Ratio 0 comes from the config (the frames carry only the compressed
    # ratios); every record is keyed by the selection top-k it was built for.
    assert list(extended.asc) == [(0, 0), (1, 3), (2, 3)]
    # One launch of each AI_CPU operator per record -- not per layer, which is
    # the point of the change.
    assert len(state["fwd"]) == 3
    assert len(state["bwd"]) == 3
    # The wrapped frames are the metadata's own: a record reads the same
    # boundary tensors the layer's inline call would, by construction.
    assert extended.q is metadata.q
    assert extended.swa_k is metadata.swa_k
    assert extended.cmp_k is metadata.cmp_k
    assert extended.positions_q is metadata.positions_q
    assert extended.exchange is metadata.exchange
    # Each record carries the operator outputs of its own launch, and guards
    # the geometry the config supplied.
    for record, (fwd_kwargs, fwd_out), (bwd_kwargs, bwd_out) in zip(
        extended.asc.values(), state["fwd"], state["bwd"], strict=True
    ):
        assert record.smla_metadata is fwd_out
        assert record.smla_grad_metadata is bwd_out
        assert (record.num_heads, record.head_dim, record.window_size) == (2, 4, 2)
    # The kernel geometry each record was built with: the window-only record
    # has no compressed axis at all, the others index the config's top-k.
    assert [(k["cmp_topk"], k["has_cmp_kv"], k["ori_topk"]) for k, _ in state["fwd"]] == [
        (0, False, 0),
        (3, True, 0),
        (3, True, 0),
    ]


def test_the_config_requires_its_geometry_up_front():
    """A config without the geometry fails at construction, not at the first batch.

    ``derive`` from an unfilled base copies the base's defaults (None geometry,
    window 0) straight in; both shapes of a missing fill fail here, at config
    time -- and a window of 0 would otherwise silently build records with
    ``ori_win_left = -1``, discoverable only through the first layer's guard.
    """
    with pytest.raises(TypeError, match="num_heads"):
        AscV41MetadataExtension.Config(window_size=2)
    with pytest.raises(ValueError, match="missing the model geometry"):
        derive(MetadataExtension.Config(), AscV41MetadataExtension.Config)
    with pytest.raises(ValueError, match="window_size must be positive"):
        AscV41MetadataExtension.Config(num_heads=1, head_dim=4, index_topk=3, window_size=0)
    # The guard that stays at the call site: only the model-built metadata is wrapped.
    extension = AscV41MetadataExtension.Config(num_heads=1, head_dim=4, index_topk=3, window_size=2).build()
    with pytest.raises(TypeError, match="DeepSeekV41Metadata"):
        extension(object())


def test_the_lookup_hits_misses_and_fails_loud(v41_metadata):
    """The lookup's three outcomes: miss (inline fallback), hit (reuse), guard (error)."""
    plain = v41_metadata([0, 4])
    record = AscV41BlockMetadata(
        smla_metadata=torch.zeros(1),
        smla_grad_metadata=torch.zeros(1),
        num_heads=1,
        head_dim=4,
        window_size=2,
    )
    window_only = AscV41BlockMetadata(
        smla_metadata=torch.zeros(1),
        smla_grad_metadata=torch.zeros(1),
        num_heads=1,
        head_dim=4,
        window_size=2,
    )
    table = _hand_table(plain, {(1, 3): record, (0, 0): window_only})
    q = torch.zeros(4, 1, 4)  # the packed [T, heads, dim] row the kernel takes
    k3 = torch.zeros(4, 1, 3, dtype=torch.int32)
    k5 = torch.zeros(4, 1, 5, dtype=torch.int32)

    # The plain contract (the golden path, hand-built unit-test metadata) has
    # no table: the layer computes its metadata inline.
    assert _asc_v41_record(plain, 1, k3, q, 2) is None
    # A ratio the extension did not precompute is a miss, not an error.
    assert _asc_v41_record(table, 3, k3, q, 2) is None
    # A top-k the extension did not precompute is a miss too: the key is
    # ``(ratio, topk)``, so a disagreement degrades to an inline recompute
    # rather than a wrong-geometry reuse.
    assert _asc_v41_record(table, 1, k5, q, 2) is None
    # The exact geometry hits and returns the extension's own record.
    assert _asc_v41_record(table, 1, k3, q, 2) is record
    # A window-only layer carries no selection: its key is ``(ratio, 0)``.
    assert _asc_v41_record(table, 0, None, q, 2) is window_only
    # A hit whose heads/head-dim/window disagree with the calling layer is a
    # config split between the extension and the attention configs: loud,
    # never a silent misread of a wrong-geometry record.
    with pytest.raises(ValueError, match="disagree"):
        _asc_v41_record(table, 1, k3, torch.zeros(4, 2, 4), 2)
    with pytest.raises(ValueError, match="disagree"):
        _asc_v41_record(table, 1, k3, q, 3)


def test_the_fused_core_reads_the_record_and_never_launches(monkeypatch, v41_metadata):
    """A hit serves the forward and the backward alike; a miss launches once per direction."""
    state = _fake_metadata_ops(monkeypatch)
    # The main operators: shape-preserving fakes, so the record path and the
    # inline path differ only in the metadata launch counts.
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **k: (q.clone(), torch.zeros(1, q.shape[0], q.shape[1])),
        raising=False,
    )

    def _grad(q, grad_output, output, lse, **kwargs):
        indices = kwargs["cmp_sparse_indices"]
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            torch.zeros(indices.shape, dtype=torch.float32),
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", _grad, raising=False)

    attention = build_cpu_model(
        ascendc.AscV41SparseAttention.Config(window_size=2, softmax_scale=0.5, compress_ratio=1)
    )
    plain = v41_metadata([0, 4])
    record = AscV41BlockMetadata(
        smla_metadata=torch.zeros(1, dtype=torch.int32),
        smla_grad_metadata=torch.zeros(1, dtype=torch.int32),
        num_heads=1,
        head_dim=4,
        window_size=2,
    )
    table = _hand_table(plain, {(1, 3): record})

    def consume(masks):
        q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)
        return attention(
            q,
            q.squeeze(2),
            torch.zeros(1),
            masks,
            cmp_k=torch.zeros(1, 4, 4, dtype=torch.bfloat16),
            topk_indices=torch.zeros(1, 4, 1, 3),
            topk_scores=torch.zeros(1, 4, 3, requires_grad=True),
        )

    # The plain contract: one metadata launch in the forward and one in the
    # backward -- the per-layer cost this change exists to remove.
    out = consume(plain)
    assert out.shape == (1, 4, 1, 4)
    out.sum().backward()
    assert len(state["fwd"]) == 1
    assert len(state["bwd"]) == 1

    # The table: the same forward and the backward through the saved context
    # both read the record, so neither count grows.
    out = consume(table)
    assert out.shape == (1, 4, 1, 4)
    out.sum().backward()
    assert len(state["fwd"]) == 1
    assert len(state["bwd"]) == 1


def test_the_shared_records_survive_every_consumer(monkeypatch, v41_metadata):
    """One record serves every layer, the recompute and the backward: nothing writes it.

    The kernels take ``metadata`` as a read-only input -- the AI_CPU operator's
    own output, consulted for the grid -- so sharing one record across all
    consumers is sound; this pins this side of the premise, the operator side
    being validated by the end-to-end A5 runs the PR documents.  Through
    several forward and backward cycles every kernel call reads the very same
    tensor objects, and they stay bit-for-bit what the extension produced.
    """
    _fake_metadata_ops(monkeypatch)
    received_fwd, received_bwd = [], []

    def _mla(q, **kwargs):
        received_fwd.append(kwargs["metadata"])
        return q.clone(), torch.zeros(1, q.shape[0], q.shape[1])

    def _grad(q, grad_output, output, lse, **kwargs):
        received_bwd.append(kwargs["metadata"])
        indices = kwargs["cmp_sparse_indices"]
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            torch.zeros(indices.shape, dtype=torch.float32),
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla", _mla, raising=False)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", _grad, raising=False)

    extension = AscV41MetadataExtension.Config(
        window_size=2, num_heads=1, head_dim=4, index_topk=3, compress_ratios=(1, 2)
    ).build()
    table = extension(v41_metadata([0, 4]))
    record = table.asc[(1, 3)]
    before = (record.smla_metadata.clone(), record.smla_grad_metadata.clone())

    attention = build_cpu_model(
        ascendc.AscV41SparseAttention.Config(window_size=2, softmax_scale=0.5, compress_ratio=1)
    )

    def consume():
        q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)
        out = attention(
            q,
            q.squeeze(2),
            torch.zeros(1),
            table,
            cmp_k=torch.zeros(1, 4, 4, dtype=torch.bfloat16),
            topk_indices=torch.zeros(1, 4, 1, 3),
            topk_scores=torch.zeros(1, 4, 3, requires_grad=True),
        )
        out.sum().backward()

    for _ in range(3):
        consume()

    # Every kernel call -- three layers' worth of forwards and their backwards
    # -- read the same, still-unchanged record tensors.
    assert len(received_fwd) == 3
    assert len(received_bwd) == 3
    assert all(t is record.smla_metadata for t in received_fwd)
    assert all(t is record.smla_grad_metadata for t in received_bwd)
    torch.testing.assert_close(record.smla_metadata, before[0], rtol=0, atol=0)
    torch.testing.assert_close(record.smla_grad_metadata, before[1], rtol=0, atol=0)


def test_the_records_flatten_through_the_pytree_registry(v41_metadata):
    """The Asc metadata round-trips the compiled path's pytree flattening."""
    plain = v41_metadata([0, 4])
    record = AscV41BlockMetadata(
        smla_metadata=torch.arange(3.0),
        smla_grad_metadata=torch.arange(3.0) + 10,
        num_heads=1,
        head_dim=4,
        window_size=2,
    )
    table = _hand_table(plain, {(1, 3): record})

    leaves, spec = tree_flatten(table)
    restored = tree_unflatten(leaves, spec)

    # The leaves survive the round trip as the very same objects, and the
    # rebuild keeps the key -> record mapping the lookup reads.
    assert isinstance(restored, AscV41Metadata)
    assert set(restored.asc) == {(1, 3)}
    round_trip, _ = tree_flatten(restored)
    assert len(round_trip) == len(leaves)
    for before, after in zip(leaves, round_trip, strict=True):
        assert before is after
    # The registry flattens past the dataclass and the mapping: the record's
    # tensor fields are leaves in their own right, not opaque blobs.
    assert any(leaf is record.smla_metadata for leaf in leaves)
    assert any(leaf is record.smla_grad_metadata for leaf in leaves)
    # Idempotent: the module-level registrations already ran at import.
    ascendc.register_pytree_node_for_dataclass(AscV41Metadata)
    ascendc.register_pytree_node_for_dataclass(AscV41BlockMetadata)


def _tiny_model(monkeypatch, *, asc_metadata: bool, fused_attention: bool = False):
    """The registered debug flavor at CPU-sized widths, with the metadata override and
    optionally the fused core that consumes it.

    Both swaps go through the same ``derive`` the override factories use, on
    the config nodes the model dir filled, so the test exercises the wiring
    the fused recipe actually installs.
    """
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    make_config = registry._make_v41_config

    def tiny_config(**kwargs):
        kwargs.update(
            dim=8,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            index_n_heads=2,
            index_head_dim=4,
            moe_inter_dim=16,
            vision_dim=8,
            vision_heads=2,
            vision_inter_dim=16,
        )
        return make_config(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny_config)
    config = registry.model_registry("deepseek_v4_1_debugmodel").model
    config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = 64
    if asc_metadata:
        config.metadata_extension = derive(config.metadata_extension, AscV41MetadataExtension.Config)
    if fused_attention:
        # The fused core that reads the records, on every layer with a
        # compressed stream; the ratio-0 layers keep the reference core, as
        # the lightning-indexer tests arrange the same stack.
        for layer in config.layers:
            if layer.attention.inner_attention.compress_ratio == 0:
                continue
            layer.attention.inner_attention = derive(
                layer.attention.inner_attention, ascendc.AscV41SparseAttention.Config
            )
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    return model, config


def test_the_model_swaps_its_mask_postprocess_and_the_default_is_identity(monkeypatch):
    """The extension runs at the mask-building boundary; without it, nothing changes.

    The pairing warning is part of that contract: the precompute only pays off
    when a fused core reads the records, so a swapped extension without its
    consumer is flagged at construction, and the fused pairing is not.
    """
    state = _fake_metadata_ops(monkeypatch)
    warned = []
    monkeypatch.setattr(
        "torchtitan_npu.models.deepseek_v4_1.model.logger.warning",
        lambda message, *args, **kwargs: warned.append(message),
    )

    # The default extension is the identity: the built masks pass through
    # untouched, no metadata operator ever launches and nothing is flagged.
    model, _ = _tiny_model(monkeypatch, asc_metadata=False)
    _, _, kwargs = model.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    assert not isinstance(kwargs["attention_masks"], AscV41Metadata)
    assert state["fwd"] == []
    assert state["bwd"] == []
    assert warned == []

    # With the override, the same boundary precomputes one record per distinct
    # geometry: the 40-layer stack runs ratios {0, 1, 2} at the debug top-k 32.
    # A reference attention stack reads none of them, so the build flags it.
    model, _ = _tiny_model(monkeypatch, asc_metadata=True)
    assert len(warned) == 1
    assert "asc_metadata" in warned[0]
    _, _, kwargs = model.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    extended = kwargs["attention_masks"]
    assert isinstance(extended, AscV41Metadata)
    assert set(extended.asc) == {(0, 0), (1, 32), (2, 32)}
    assert len(state["fwd"]) == 3
    assert len(state["bwd"]) == 3
    # Every record was built for the geometry the model dir filled on the
    # config -- the guard that makes a wrong-geometry reuse a loud error.
    for record in extended.asc.values():
        assert record.num_heads == 2
        assert record.head_dim == 8
        assert record.window_size == 128

    # The fused pairing is the combination the recipes ship: the same records,
    # now with a consumer, and the warning stays at exactly the one reference
    # build above.
    model, _ = _tiny_model(monkeypatch, asc_metadata=True, fused_attention=True)
    _, _, kwargs = model.build_attention_masks(_TOKENS, _TOKENS, {"positions": _POSITIONS})
    assert isinstance(kwargs["attention_masks"], AscV41Metadata)
    assert len(warned) == 1


def test_the_override_factory_carries_the_filled_geometry_along():
    """``derive`` copies the shared fields, so the class swap keeps the model dir's geometry."""
    from torchtitan_npu.override.deepseek_v4_1.sparse_attn import asc_metadata

    base = MetadataExtension.Config(
        window_size=128,
        num_heads=64,
        head_dim=512,
        index_topk=512,
        compress_ratios=(0, 1, 2),
    )
    swapped = asc_metadata(base)
    assert type(swapped) is AscV41MetadataExtension.Config
    assert (swapped.window_size, swapped.num_heads, swapped.head_dim, swapped.index_topk) == (128, 64, 512, 512)
    assert swapped.compress_ratios == (0, 1, 2)
    assert isinstance(swapped.build(), AscV41MetadataExtension)
    # The factory inherits the same construction-time guard: an unfilled base
    # cannot yield a silently broken extension.
    with pytest.raises(ValueError, match="missing the model geometry"):
        asc_metadata(MetadataExtension.Config())
