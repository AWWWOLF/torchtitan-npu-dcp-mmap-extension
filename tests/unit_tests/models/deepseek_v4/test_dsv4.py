"""DSV4 host-logic CPU tests: the AscendC metadata layer, the packed layout,
the compressor/indexer/attention numerics, and the ported DSA semantics.

The AscendC override host wiring (``test_asc_metadata_*``) and the model-dir
contract (``test_layout_*``, ``test_compressor_*``, ``test_attention_*``,
``test_indexer_*``, ``test_ported_*``) all import against the real
torchtitan checkout with the plugin's patches applied; only the
``cann_ops_transformer`` surface is faked (the conftest's recorder).

Numerics verification criterion: a match is only accepted when it is
*bitwise*, or the residual sits at the fp32 rounding floor of the outputs
(at most a few ulps of the output scale) — a floating-point artifact, not
an arithmetic difference.  The compressor and the indexer are bitwise; the
attention core is attn-gym's reference path and is pinned by shape/finiteness
and by the pool-gradient reproducibility guard below.

Every comparison runs single-threaded (``_reproducible_numerics`` below): the
pool gradient's backward accumulates duplicate contributions in a
thread-scheduling dependent order, which makes the residual a sample of a
distribution rather than a number, and that distribution crosses this file's
4-ulp budget often enough to flake.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtitan.models.common.attention import VarlenMetadata

pytestmark = pytest.mark.usefixtures("dsv4")

DIM, HD, RD = 8, 16, 4
# Two aligned documents longer than one compression block (multiples of the
# 128-token alignment the (4, 128) plan is built with).
_GRADIENT_DOCS = (256, 384)
_EPS32 = 1.1920929e-7  # fp32 machine epsilon


@pytest.fixture(autouse=True)
def _reproducible_numerics():
    """Pin the intra-op thread count so the pool gradient is reproducible.

    The core reads the compressed-KV pool through advanced indexing
    (attn-gym's ``kv[b, h, indices]``).  The pool's rows are read by many query
    rows, so that gather's backward is an accumulate scatter (ATen lowers it to
    ``index_put_`` with ``accumulate=True``), and its CPU kernel sums the
    duplicate contributions in a thread-scheduling dependent order: measured in
    isolation, ``index_put_(accumulate=True)`` and both advanced-index backwards
    give different last bits from run to run at 16 threads with a non-uniform
    upstream gradient.

    So with more than one intra-op thread the ``cmp_k`` gradient drifts against
    itself by up to ~3.5 ulps of its scale (``out``, ``q``, ``swa_k`` and
    ``sink`` stay bitwise constant).  Single-threaded every projection is exact,
    which is what ``test_attention_pool_gradient_is_reproducible`` asserts.
    ``torch.use_deterministic_algorithms(True)`` does not help (torch does not
    call these kernels nondeterministic), and neither does pinning
    ``MKL_DYNAMIC``/``OMP_NUM_THREADS`` to 16.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class IdentityRoPE(nn.Module):
    """Identity rope — the compressor-side counterpart of ``manual_compress``.

    The pooling reference applies NO rope, so the compressor's
    ``self.rope(...)`` step must be a no-op for the comparison to be exact.
    The real rope numerics are covered by the rope patch's own on-device
    verification.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        dim: int = 1
        original_max_position_embeddings: int = 1
        rope_theta: float = 1.0

    def __init__(self, config=None):
        super().__init__()

    def forward(self, x, key=None, positions=None, *, inverse=False):
        if key is None:
            return x
        return x, x


class BuildCfg:
    def __init__(self, fn):
        self._fn = fn

    def build(self):
        return self._fn()


class RMS(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return (
            x
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
                x.dtype
            )
            * self.weight
        )


def make_linear(n, m):
    lin = nn.Linear(n, m, bias=False)
    nn.init.normal_(lin.weight, std=0.02)
    return lin


# B=1, S=1024; docs A(128) B(256) C(384) D(256).  The plan is built with the
# model's ratios (1, 4, 128), so every document is a multiple of
# lcm(1, 4, 128) = 128 -- the loader's per-document alignment.
_ALIGNED_DOCS = (128, 256, 384, 256)
_SEQ_LEN = sum(_ALIGNED_DOCS)
_CU = torch.tensor(
    [0, *torch.tensor(_ALIGNED_DOCS).cumsum(0).tolist()], dtype=torch.int32
)


def _build_metadata(dsv4):
    """The 1024-token four-document aligned layout (ratios 1/4/128) as the model builds it."""
    v = VarlenMetadata(cu_seq_q=_CU, cu_seq_k=_CU.clone(), max_q=max(_ALIGNED_DOCS), max_k=max(_ALIGNED_DOCS))
    return dsv4.metadata.build_compressed_varlen_metadata(v, (1, 4, 128))


def _pack_container(dsv4, pooled, plan):
    """The plan-driven container pack (the dispatcher's ``select``)."""
    disp = dsv4.token_dispatcher.CPTokenDispatcher(
        dsv4.token_dispatcher.CPTokenDispatcher.Config()
    )
    return disp.select(pooled, plan)


def build_compressor(dsv4, ratio):
    coff = 2 if ratio == 4 else 1
    comp = dsv4.compressor.Compressor(
        dsv4.compressor.Compressor.Config(
            rope=BuildCfg(lambda: IdentityRoPE()),
            head_dim=HD,
            rope_head_dim=RD,
            compress_ratio=ratio,
            wkv=BuildCfg(lambda: make_linear(DIM, coff * HD)),
            wgate=BuildCfg(lambda: make_linear(DIM, coff * HD)),
            norm=BuildCfg(lambda: RMS(HD)),
        )
    )
    nn.init.trunc_normal_(comp.ape, std=0.02)
    return comp


def manual_compress(x, md, comp, ratio):
    """Independent per-document pooling reference (no rope).

    Reads each document's start from the plan's ``cu_seqlens_cmp_k`` and pools
    its complete blocks.  The plan is built with aligned documents only, so a
    document is a whole number of blocks and has no partial trailing tail to
    drop; the assertion keeps this oracle honest if a fixture drifts off grid.
    """
    plan = md.plans[ratio]
    flat_x = x.flatten(0, 1).float()
    out = torch.zeros(md.batch_size * (md.seq_len // ratio), HD)
    for doc_id, (qs, qe) in enumerate(
        zip(
            md.varlen.cu_seq_q[:-1].tolist(),
            md.varlen.cu_seq_q[1:].tolist(),
            strict=True,
        )
    ):
        length = qe - qs
        assert length > 0 and length % ratio == 0, (
            "the aligned contract has no empty document and no partial trailing block"
        )
        c_len = length // ratio
        tokens = flat_x[qs : qs + c_len * ratio].view(c_len, ratio, DIM)
        kv = F.linear(tokens, comp.wkv.weight.float())
        score = F.linear(tokens, comp.wgate.weight.float()) + comp.ape
        if ratio == 4:
            c_start = plan.cu_seqlens_cmp_k[doc_id].item()
            c_end = plan.cu_seqlens_cmp_k[doc_id + 1].item()
            valid = (torch.arange(c_end - c_start) != 0).view(-1, 1, 1)
            prev_idx = (torch.arange(c_end - c_start) - 1).clamp_min(0)
            kv_prev = torch.where(
                valid, kv[prev_idx, :, :HD], torch.zeros_like(kv[:, :, :HD])
            )
            score_prev = torch.where(
                valid,
                score[prev_idx, :, :HD],
                torch.full_like(score[:, :, :HD], float("-inf")),
            )
            kv = torch.cat([kv_prev, kv[:, :, HD:]], dim=1)
            score = torch.cat([score_prev, score[:, :, HD:]], dim=1)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = comp.norm(kv.to(x.dtype))
        start = plan.cu_seqlens_cmp_k[doc_id].item()
        out[start : start + c_len] = kv
    return out.view(md.batch_size, md.seq_len // ratio, HD)


def _pool_blocks_where(
    comp, block_tokens, block_positions, overlap_valid, dtype, ratio
):
    """The pre-refactor pooling math (two ``where`` masks) — the bitwise
    reference for the scatter-masking implementation (exact-zero softmax
    weights make the two forms identical)."""
    kv = comp.wkv(block_tokens)
    score = comp.wgate(block_tokens) + comp.ape
    if ratio == 4:
        head_dim = comp.head_dim
        n = kv.shape[0]
        overlap_kv = kv.new_zeros(n, 2 * ratio, head_dim)
        overlap_score = score.new_full((n, 2 * ratio, head_dim), float("-inf"))
        overlap_kv[:, ratio:] = kv[:, :, head_dim:]
        overlap_score[:, ratio:] = score[:, :, head_dim:]
        prev = (torch.arange(n) - 1).clamp_min(0)
        valid = overlap_valid.view(-1, 1, 1)
        overlap_kv[:, :ratio] = torch.where(
            valid, kv[prev, :, :head_dim], overlap_kv[:, :ratio]
        )
        overlap_score[:, :ratio] = torch.where(
            valid, score[prev, :, :head_dim], overlap_score[:, :ratio]
        )
        kv, score = overlap_kv, overlap_score
    kv = (kv * score.softmax(dim=1)).sum(dim=1)
    kv = comp.norm(kv.to(dtype))
    nope_dim = comp.head_dim - comp.rope_head_dim
    kv_nope, kv_rope = torch.split(kv, [nope_dim, comp.rope_head_dim], dim=-1)
    kv_rope = (
        comp.rope(
            kv_rope.unsqueeze(0).unsqueeze(2), positions=block_positions.unsqueeze(0)
        )
        .squeeze(0)
        .squeeze(1)
    )
    return torch.cat([kv_nope, kv_rope], dim=-1)


def _build_varlen(doc_lens):
    cu = torch.tensor(
        [0, *torch.tensor(doc_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    longest = max(doc_lens)
    return VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu.clone(), max_q=longest, max_k=longest)


def _run_extension(dsv4, doc_lens, ratios, **shape):
    """The model-built metadata + the AscendC metadata extension."""
    from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
        AscMetadataExtension,
        AscLightningIndexerMetadata,
    )

    v = _build_varlen(doc_lens)
    md = dsv4.metadata.build_compressed_varlen_metadata(v, ratios)
    md = AscLightningIndexerMetadata(
        AscLightningIndexerMetadata.Config(
            window_size=shape.get("window_size", 128),
            index_n_heads=shape.get("index_n_heads", 8),
            index_head_dim=shape.get("index_head_dim", 128),
            index_topk=shape.get("index_topk", 512),
        )
    )(md)
    ext = AscMetadataExtension(
        AscMetadataExtension.Config(
            window_size=shape.get("window_size", 128),
            num_heads=shape.get("num_heads", 16),
            head_dim=shape.get("head_dim", 512),
            index_n_heads=shape.get("index_n_heads", 8),
            index_head_dim=shape.get("index_head_dim", 128),
            index_topk=shape.get("index_topk", 512),
        )
    )
    return ext(md)


@pytest.fixture
def inputs(dsv4):
    torch.manual_seed(0)
    md = _build_metadata(dsv4)
    x = torch.randn(1, _SEQ_LEN, DIM)
    pos = torch.arange(_SEQ_LEN).unsqueeze(0)
    return dsv4, md, x, pos


# ---------------------------------------------------------------------------
# The AscendC metadata layer (handler + kernel fills)
# ---------------------------------------------------------------------------


def test_asc_metadata_unaligned_batch_rejected_before_any_asc_call(dsv4):
    """An off-grid stream is refused before any AscendC metadata is computed.

    The plain path takes aligned documents only (the loader pads every document
    to ``lcm(compress_ratios)``); a batch the plan builder rejects must never
    reach the AscendC metadata kernels.
    """
    dsv4.cann_ops.calls.clear()
    with pytest.raises(ValueError, match="aligned to the compression grid"):
        _run_extension(dsv4, (60, 60, 60, 76), (1, 4, 128))
    assert not dsv4.cann_ops.calls, (
        "AscendC metadata must not be computed for an invalid batch"
    )


def test_asc_metadata_smla_fills(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    assert md.batch_size == 1 and md.seq_len == 1024
    smla = [c for c in dsv4.cann_ops.calls if c[0] == "sparse_flash_mla_metadata"]
    assert len(smla) == 3  # ratios 1, 4, 128
    by_ratio = {c[2]["cmp_ratio"]: c for c in smla}
    assert by_ratio[4][1] == (16, 1, 512)
    assert by_ratio[4][2]["cmp_topk"] == 512 and by_ratio[4][2]["has_cmp_kv"] is True
    assert by_ratio[128][2]["cmp_topk"] == 0 and by_ratio[128][2]["has_cmp_kv"] is True
    assert by_ratio[1][2]["has_cmp_kv"] is False
    assert torch.equal(
        by_ratio[4][2]["cu_seqlens_cmp_kv"], md.plans[4].cu_seqlens_cmp_k
    )
    assert torch.equal(by_ratio[4][2]["cmp_residual_kv"], md.plans[4].block_remainder)


def test_asc_metadata_grad_fills(dsv4):
    dsv4.cann_ops.calls.clear()
    _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    grad = [c for c in dsv4.cann_ops.calls if c[0] == "sparse_flash_mla_grad_metadata"]
    assert len(grad) == 3
    assert all(
        "ori_topk_length" not in c[2] and "cmp_topk_length" not in c[2] for c in grad
    )


def test_asc_metadata_li_and_slig_fills(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    li = [c for c in dsv4.cann_ops.calls if c[0] == "lightning_indexer_metadata"]
    slig = [
        c
        for c in dsv4.cann_ops.calls
        if c[0] == "sparse_lightning_indexer_kl_loss_grad_metadata"
    ]
    assert len(li) == 1 and len(slig) == 1
    assert li[0][1][:3] == (8, 1, 128), li[0][1]
    assert li[0][1][3] == 512
    assert slig[0][1] == (8, 1, 128), slig[0][1]
    assert slig[0][2]["topk"] == 512  # keyword-only in the binding schema
    assert md.asc_plans[4].li_metadata is not None
    assert md.asc_plans[4].slig_metadata is not None
    assert (
        md.asc_plans[128].li_metadata is None
        and md.asc_plans[128].slig_metadata is None
    )


def test_asc_metadata_slim_contract(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (256, 256, 256, 256), (1, 4, 128))
    assert not hasattr(md, "reference")
    assert md.plans[4].cu_seqlens_cmp_k.shape[0] == 5
    assert md.plans[4].block_remainder.shape[0] == 4
    assert md.plans[4].gather_indices.numel() == 1024
    assert md.asc_plans[4].smla_metadata is not None
    assert md.asc_plans[128].smla_metadata is not None
    assert md.asc_plans[1].smla_metadata is not None


def test_asc_metadata_ratio128_only_model(dsv4):
    dsv4.cann_ops.calls.clear()
    md = _run_extension(dsv4, (512, 512), (128,))
    assert "lightning_indexer" not in [c[0] for c in dsv4.cann_ops.calls]
    assert md.asc_plans[128].li_metadata is None


# ---------------------------------------------------------------------------
# The packed layout (kernel contract + reference tier)
# ---------------------------------------------------------------------------


def test_layout_container_shape(dsv4):
    md = _build_metadata(dsv4)
    assert (md.batch_size, md.seq_len) == (1, _SEQ_LEN)


def test_layout_cu_seqs_and_remainders(dsv4):
    """An aligned document is a whole number of blocks at every ratio.

    The plan is the row divided by its ratio: the document boundaries are
    ``cu_seq_q // ratio`` and there is no remainder anywhere.
    """
    md = _build_metadata(dsv4)
    for ratio in (4, 128):
        plan = md.plans[ratio]
        assert not plan.is_empty_host
        assert torch.equal(plan.cu_seqlens_cmp_k, _CU // ratio)
        assert torch.equal(
            plan.block_remainder,
            torch.zeros(len(_ALIGNED_DOCS), dtype=torch.int32),
        )
        assert plan.n_cmp_blocks_host == _SEQ_LEN // ratio
        assert plan.out_width == _SEQ_LEN // ratio


def test_layout_host_cached_lengths(dsv4):
    """Metadata builders cache scalar lengths before entering compiled layers.

    The cached values must agree with the device-side cumulative-sequence
    tensors: ``seq_len`` is the total token count and each compressed plan
    records its total block count (the row divided by its ratio).  This guards
    the host cache introduced to remove repeated ``.item()`` synchronizations.
    """
    varlen = VarlenMetadata(
        cu_seq_q=_CU,
        cu_seq_k=_CU.clone(),
        max_q=max(_ALIGNED_DOCS),
        max_k=max(_ALIGNED_DOCS),
    )
    md = dsv4.metadata.build_compressed_varlen_metadata(varlen, (1, 4, 128))
    assert md.seq_len_host == _SEQ_LEN
    assert md.seq_len == _SEQ_LEN
    assert md.plans[1].n_cmp_blocks_host is None
    assert md.plans[4].n_cmp_blocks_host == _SEQ_LEN // 4
    assert md.plans[128].n_cmp_blocks_host == _SEQ_LEN // 128


def test_layout_gather_indices(dsv4):
    """The aligned pool is the row itself; block positions restart per document.

    Every token of an aligned document belongs to exactly one compression
    block, so the gather keeps every token in order and the segment-first
    blocks are the document boundaries divided by the ratio.  The block
    positions stay document-relative — the compressor's RoPE restarts at
    every document — so each document contributes its own progression.
    """
    md = _build_metadata(dsv4)
    for ratio in (4, 128):
        plan = md.plans[ratio]
        assert torch.equal(
            plan.gather_indices, torch.arange(_SEQ_LEN, dtype=torch.int64)
        )
        assert torch.equal(
            plan.block_positions,
            torch.cat(
                [
                    torch.arange(0, length, ratio, dtype=torch.int32)
                    for length in _ALIGNED_DOCS
                ]
            ),
        )
        assert torch.equal(
            plan.first_indices, (_CU[:-1] // ratio).to(torch.int64)
        )


@pytest.mark.parametrize(
    "docs",
    [
        pytest.param((200, 133), id="partial-tails"),
        pytest.param((10, 17, 20, 17), id="small-docs"),
        pytest.param((0, 128, 128), id="empty-doc"),
    ],
)
@pytest.mark.parametrize("ratios", [(4,), (1, 4, 128)], ids=["r4", "full-stack"])
def test_layout_rejects_streams_off_the_compression_grid(dsv4, docs, ratios):
    """The plain builder validates the loader's per-document alignment.

    A partial trailing block or an empty document cannot be expressed any
    more: the loader pads every document to ``lcm(compress_ratios)``, so an
    off-grid stream is an input bug and the plan build refuses it instead of
    silently dropping a tail.
    """
    varlen = _build_varlen(docs)
    with pytest.raises(ValueError, match="aligned to the compression grid"):
        dsv4.metadata.build_compressed_varlen_metadata(varlen, ratios)


def test_compressor_numerics_ratio4(inputs):
    """C4A pooling vs the independent manual reference.

    The residual must sit at the fp32 rounding floor: the two paths project
    with different matmul batching (all blocks at once vs per document), so
    the CPU kernel's shape-dependent reduction order yields 1-ulp residuals
    (the documented batch-size artifact; the same-size calls are
    deterministic).  ``_assert_rounding_floor`` accepts that — bitwise would
    only hold on shape-invariant kernels; the suite-wide 4-ulp budget keeps
    a 2.2x margin over the measured worst (at most 1.8 ulps, single-threaded)
    while still flagging any multi-ulp arithmetic regression.
    """
    dsv4, md, x, _pos = inputs
    comp4 = build_compressor(dsv4, 4)
    pooled = comp4(x, md)
    cmp4 = _pack_container(dsv4, pooled, md.plans[4])
    assert cmp4.shape == (1, _SEQ_LEN // 4, HD)
    ref = manual_compress(x, md, comp4, 4)
    _assert_rounding_floor((cmp4 - ref).abs(), ref.abs().max().item())


def test_compressor_numerics_ratio128(inputs):
    """C128A pooling vs the independent manual reference.

    The ratio-128 contract has no overlap window: an aligned document is a
    whole number of 128-token blocks and the pool is doc-major.
    """
    dsv4, md, x, _pos = inputs
    comp128 = build_compressor(dsv4, 128)
    cmp128 = _pack_container(dsv4, comp128(x, md), md.plans[128])
    assert cmp128.shape == (1, _SEQ_LEN // 128, HD)
    ref = manual_compress(x, md, comp128, 128)
    _assert_rounding_floor((cmp128 - ref).abs(), ref.abs().max().item())


# Per-ratio document lengths: every document is a whole number of blocks at
# the ratio the plan is built with (the aligned contract).  The cases keep the
# original intents -- tiny one-to-three-block documents, a single multi-block
# document, exact ratio multiples, the generic layout -- each with multi-block
# documents so the document-start borrow boundary is exercised.
_OVERLAP_CASES = [
    ("tiny", (4, 8, 12, 8, 4, 12, 8, 4), (128, 256, 384, 256, 128, 384, 256, 128)),
    ("single", (4, 4), (128, 128)),
    ("single-full", (8,), (256,)),
    ("r-multiples", (8, 12, 16, 4), (256, 384, 512, 128)),
    ("plain", (12, 16, 20, 16), (128, 256, 384, 256)),
]


@pytest.mark.parametrize(
    "name,docs,ratio",
    [
        pytest.param(name, docs, ratio, id=f"{name}-r{ratio}")
        for name, docs4, docs128 in _OVERLAP_CASES
        for ratio, docs in ((4, docs4), (128, docs128))
    ],
)
def test_compressor_overlap_scatter_bitwise(dsv4, name, docs, ratio):
    """The scatter-masked overlap (``first_indices``) is bitwise-identical
    to the old two-``where`` form: the masked borrowed rows have exactly
    zero softmax weight, so their kv content never contributes."""
    v = _build_varlen(docs)
    md = dsv4.metadata.build_compressed_varlen_metadata(v, (ratio,))
    plan = md.plans[ratio]
    comp = build_compressor(dsv4, ratio)
    torch.manual_seed(0)
    x = torch.randn(1, sum(docs), DIM)
    pooled = comp(x, md)
    assert pooled.shape[0] == sum(docs) // ratio, name
    bt = x.flatten(0, 1)[plan.gather_indices].reshape(pooled.shape[0], ratio, -1)
    bids = torch.arange(pooled.shape[0])
    seq_ids = torch.searchsorted(plan.cu_seqlens_cmp_k[1:], bids, right=True)
    overlap_valid = (bids - plan.cu_seqlens_cmp_k[seq_ids]) != 0
    old = _pool_blocks_where(
        comp, bt, plan.block_positions, overlap_valid, torch.float32, ratio
    )
    assert torch.equal(pooled, old), name


# ---------------------------------------------------------------------------
# Attention numerics
# ---------------------------------------------------------------------------


def _assert_rounding_floor(diff, scale, *, ulps: int = 4):
    """The residual must sit at the fp32 rounding floor of the outputs
    (<= ``ulps`` ulps of the output scale) — floating-point error only, not
    arithmetic (an arithmetic difference would exceed it by orders of
    magnitude).  ``ulps=4`` is the suite-wide budget; the compressor
    comparison's measured worst is 1.8 ulps, reproducible because the suite
    runs single-threaded."""
    assert scale > 0
    assert diff.max().item() <= ulps * _EPS32 * scale, (diff.max().item(), scale)


def _build_core(dsv4, ratio, topk):
    return dsv4.attention.CompressedSparseInnerAttention(
        dsv4.attention.CompressedSparseInnerAttention.Config(
            window_size=8,
            compress_ratio=ratio,
            softmax_scale=0.25,
            index_topk=topk,
        )
    )


def test_attention_numerics_ratio4_aligned_streams(dsv4):
    """The ratio-4 reference path over aligned documents — shape/finite.

    A document's leading tokens have no causally complete compression group,
    so their selection picks are padding (-1); the rest of the document's
    picks are real.
    """
    docs = (128, 256)
    seq_len = sum(docs)
    v = _build_varlen(docs)
    md_s = dsv4.metadata.build_compressed_varlen_metadata(v, (4,))
    comp_s = build_compressor(dsv4, 4)
    cmp_s = _pack_container(
        dsv4, comp_s(torch.randn(1, seq_len, DIM), md_s), md_s.plans[4]
    )
    assert cmp_s.shape == (1, seq_len // 4, 16)
    core_s = _build_core(dsv4, 4, 4)
    with torch.no_grad():
        sparse_s = dsv4.compressor.select_compressed_slots(
            torch.randn(1, seq_len, 4, 8),
            torch.randn(1, seq_len // 4, 8),
            torch.randn(1, seq_len, 4),
            md_s.varlen.cu_seq_q,
            md_s.plans[4].cu_seqlens_cmp_k,
            4,
        )
        out_s = core_s(
            torch.randn(1, seq_len, 3, HD),
            torch.randn(1, seq_len, HD),
            cmp_s,
            attn_sink=torch.randn(3),
            sparse_indices=sparse_s,
            attention_masks=md_s,
        )
    # The first document's opening tokens have no complete group, so their picks are
    # padding; every later row (the rest of the document and the second one) has real picks.
    assert (sparse_s[0, :3] == -1).all()
    assert (sparse_s[0, 3:] >= 0).any()
    assert out_s.shape == (1, seq_len, 3, HD) and torch.isfinite(out_s).all()


@pytest.mark.parametrize("ratio", (4, 128))
def test_padding_a_document_tail_is_invisible_to_its_real_tokens(dsv4, ratio):
    """A padded tail block never reaches the document's real tokens.

    The loader pads a document's segment to the compression alignment, so the
    document's last pooled block mixes real tokens with pads.  The kernel's
    visibility rule exposes a compressed slot only at that block's *last* token,
    and a padded block's last token is always a pad: no real token can see a
    pad-contaminated key.  The pads themselves can (that is why their outputs
    move when the pad value changes), and their labels are ``IGNORE_INDEX``, so
    they contribute nothing to the loss.

    Checked by changing nothing but the pad content: the real blocks of the
    compressed pool, every real query's output and an ``IGNORE_INDEX`` cross
    entropy stay bitwise identical, while every padded query's output changes.
    """
    from torchtitan.components.loss import IGNORE_INDEX

    real_len, total = 70, 128
    v = _build_varlen((total,))
    md = dsv4.metadata.build_compressed_varlen_metadata(v, (4, 128))
    comp = build_compressor(dsv4, ratio)
    x_real = torch.randn(1, real_len, DIM)

    def pool(pad_value):
        x = torch.cat([x_real, torch.full((1, total - real_len, DIM), pad_value)], dim=1)
        return _pack_container(dsv4, comp(x, md), md.plans[ratio])

    pool_a, pool_b = pool(0.0), pool(3.0)
    # Blocks that hold only real tokens are untouched by the pad content; the
    # block the pads land in (and every block after it) moves.
    n_real_blocks = real_len // ratio
    assert torch.equal(pool_a[:, :n_real_blocks], pool_b[:, :n_real_blocks])
    assert not torch.equal(pool_a[:, n_real_blocks:], pool_b[:, n_real_blocks:])

    q = torch.randn(1, total, 3, HD)
    swa = torch.randn(1, total, HD)
    sink = torch.randn(3)
    core = _build_core(dsv4, ratio, 4)
    sparse = None
    if ratio == 4:
        with torch.no_grad():
            sparse = dsv4.compressor.select_compressed_slots(
                torch.randn(1, total, 4, 8),
                torch.randn(1, total // ratio, 8),
                torch.randn(1, total, 4),
                md.varlen.cu_seq_q,
                md.plans[ratio].cu_seqlens_cmp_k,
                ratio,
            )
        # The selector only ever reaches the blocks that end before the query,
        # so no real query can pick the block the pads complete.
        picked = sparse[0, :real_len]
        assert (picked[picked >= 0] < n_real_blocks).all()

    with torch.no_grad():
        out_a = core(q, swa, pool_a, attn_sink=sink, sparse_indices=sparse, attention_masks=md)
        out_b = core(q, swa, pool_b, attn_sink=sink, sparse_indices=sparse, attention_masks=md)

    assert torch.equal(out_a[:, :real_len], out_b[:, :real_len])
    assert not torch.equal(out_a[:, real_len:], out_b[:, real_len:])

    logits_a = out_a.reshape(total, -1).float()
    logits_b = out_b.reshape(total, -1).float()
    labels = torch.randint(0, logits_a.shape[-1], (total,))
    labels[real_len:] = IGNORE_INDEX
    loss_a = F.cross_entropy(logits_a, labels, reduction="sum", ignore_index=IGNORE_INDEX)
    loss_b = F.cross_entropy(logits_b, labels, reduction="sum", ignore_index=IGNORE_INDEX)
    assert torch.equal(loss_a, loss_b)
    per_token = F.cross_entropy(logits_a, labels, reduction="none", ignore_index=IGNORE_INDEX)
    assert torch.count_nonzero(per_token[real_len:]) == 0
    assert torch.count_nonzero(per_token[:real_len]) == real_len


def _gradient_case(dsv4, ratio=4):
    """The gradient case's inputs: one pool, one selection, one draw per pass.

    One input set is built for both passes, so any difference in the gradients is arithmetic, not
    a different draw.  Raises if the pool is empty, i.e. if the case stops exercising the
    compressed path.
    """
    torch.manual_seed(0)
    docs = _GRADIENT_DOCS
    md = dsv4.metadata.build_compressed_varlen_metadata(_build_varlen(docs), (4, 128))
    seqlen = sum(docs)
    sink = torch.randn(3)
    pool = _pack_container(
        dsv4, build_compressor(dsv4, ratio)(torch.randn(1, seqlen, DIM), md), md.plans[ratio]
    )
    assert pool.shape[1] > 0, "the fixture must exercise a non-empty compressed pool"

    sparse = None
    if ratio == 4:
        n_cmp = int(md.plans[4].cu_seqlens_cmp_k[-1])
        sparse = dsv4.compressor.select_compressed_slots(
            torch.randn(1, seqlen, 4, 8),
            torch.randn(1, n_cmp, 8),
            torch.randn(1, seqlen, 4),
            md.varlen.cu_seq_q,
            md.plans[4].cu_seqlens_cmp_k,
            8,
        )

    base = (
        torch.randn(1, seqlen, 3, HD),
        torch.randn(1, seqlen, HD),
        pool.detach().clone(),
        sink,
    )
    return md, sparse, base, _build_core(dsv4, ratio, 8 if ratio == 4 else 0)


def _consumer_grads(consumer, sparse, md, base):
    """One differentiated pass; each pass gets its own differentiable copy."""
    q, swa_k, cmp_k, attn_sink = (t.detach().clone().requires_grad_() for t in base)
    out = consumer(q, swa_k, cmp_k, attn_sink=attn_sink, sparse_indices=sparse, attention_masks=md)
    upstream = torch.linspace(-1, 1, out.numel()).reshape_as(out)
    (out * upstream).sum().backward()
    return out, (q.grad, swa_k.grad, cmp_k.grad, attn_sink.grad)


def test_attention_pool_gradient_is_reproducible(dsv4):
    """The same consumer, run twice on identical inputs, must give the same pool gradient.

    This is what the single-threaded fixture buys, and the reason it exists: the
    pool gradient is the backward of the gather that reads the pool, an
    accumulate scatter whose contributions are summed in a thread-scheduling
    dependent order, so with more than one intra-op thread even one consumer
    disagrees with itself by up to ~3.5 ulps of ``cmp_k``'s scale.  The
    thread-count assertion keeps this guard deterministic; without it the two
    runs only *usually* differ.
    """
    assert torch.get_num_threads() == 1, "the module fixture must pin the intra-op threads"
    md, sparse, base, core = _gradient_case(dsv4)
    _out, (_q_grad, _swa_k_grad, first, _sink_grad) = _consumer_grads(core, sparse, md, base)
    _out, (_q_grad, _swa_k_grad, second, _sink_grad) = _consumer_grads(core, sparse, md, base)

    assert torch.equal(first, second), (
        "the pool gradient must be reproducible; a thread-dependent accumulate "
        "scatter in its backward makes any rounding-floor comparison flaky"
    )


def test_reference_core_rejects_context_parallel_frames(dsv4):
    """The default core refuses context-parallel frames.

    Under CP the query offsets describe this rank's chunks while the compressed pool is the
    gathered row, so a reference consumer would read boundaries that are not there.  The
    reference tier used to refuse the shape on the consumers' behalf; now that the default
    extension is the identity, the consumer refuses it itself, and the fused kernels keep
    taking the frames through their own metadata."""
    from torchtitan_npu.models.deepseek_v4.token_dispatcher import build_cp_plan

    cu = torch.tensor([0, 8], dtype=torch.int32)
    # the shard path's builder wants the global contract, i.e. both offsets as one object
    v = VarlenMetadata(cu_seq_q=cu, cu_seq_k=cu, max_q=8, max_k=8)
    cp_meta, plans, window = build_cp_plan(
        v, rank=1, cp_size=2, shard_len=4, window_size=4, ratios=[4]
    )
    assert not torch.equal(cp_meta.cu_seq_q, cp_meta.cu_seq_k)
    md = dsv4.metadata.CompressedVarlenMetadata(varlen=cp_meta, plans=plans, window=window)
    seqlen = int(cp_meta.cu_seq_q[-1])
    core = _build_core(dsv4, 4, 4)

    with pytest.raises(ValueError, match="context parallel"):
        core(
            torch.randn(1, seqlen, 3, HD),
            torch.randn(1, seqlen, HD),
            None,
            attn_sink=torch.randn(3),
            attention_masks=md,
        )
