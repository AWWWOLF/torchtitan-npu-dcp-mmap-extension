# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 QLI/QSLI selection with an SLIKG backward edge."""

__all__ = ["QuantV41LightningIndexer"]

import cann_ops_transformer  # noqa: F401  # register torch.ops.cann_ops_transformer
import torch
import torch_npu

# This training adapter packs both Q and K as TND with separate scale tensors.
_LAYOUT = "TND"
# Right-down causal masking matches the BF16 selector and SLIKG backward.
# Although QLI/QSLI also support mode 0, switching requires coordinated changes
# to model visibility, cmp_residual_k, metadata, and backward semantics.
_MASK_MODE = 3
# The packed-MXFP4 ABI is the only value either kernel accepts; the number labels the
# storage format (two E2M1 values per byte plus uE8M0 scales), it is not a precision knob.
_QUANT_MODE = 1
# MXFP4 uses one E8M0 scale per 32 elements. The kernels and _pack_mxfp4's
# [T, H, D/64, 2] scale layout require this group size.
_MX_BLOCK_SIZE = 32


def _kernel_options(attention_masks, ratio: int) -> dict[str, object]:
    compressed = attention_masks.frame_for(ratio)
    return {
        "cu_seqlens_q": attention_masks.q.cu_seqlens,
        "cu_seqlens_k": compressed.cu_seqlens,
        "seqused_q": attention_masks.q.seqused,
        "seqused_k": compressed.seqused,
        "cmp_residual_k": compressed.residual,
        "layout_q": _LAYOUT,
        "layout_k": _LAYOUT,
        "mask_mode": _MASK_MODE,
        "cmp_ratio": ratio,
    }


def _quantized_kernel_geometry(num_heads: int, head_dim: int, topk: int) -> dict[str, int]:
    return {
        "num_heads_q": num_heads,
        "num_heads_k": 1,
        "head_dim": head_dim,
        "topk": topk,
    }


def _slikg_kernel_geometry(num_heads: int, head_dim: int, topk: int) -> dict[str, int | None]:
    return {
        "num_heads_q": num_heads,
        "num_heads_k": 1,
        "head_dim": head_dim,
        "topk": topk,
        "max_seqlen_q": None,
        "max_seqlen_k": None,
    }


def _pack_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one ``[T, H, D]`` operand into the ds41 MXFP4 ABI."""
    if x.ndim != 3:
        raise ValueError(f"QLI operands must be rank-3 TND tensors, got shape {tuple(x.shape)}")
    rows, heads, head_dim = x.shape
    if head_dim % 64 != 0:
        raise ValueError(f"QLI head_dim must be divisible by 64, got {head_dim}")
    data, scale = torch_npu.npu_dynamic_mx_quant(
        x.contiguous(),
        axis=-1,
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=_MX_BLOCK_SIZE,
        round_mode="rint",
        scale_alg=2,
        dst_type_max=0.0,
    )
    return (
        data.view(torch.uint8).reshape(rows, heads, head_dim // 2).contiguous(),
        scale.view(torch.uint8).reshape(rows, heads, head_dim // 64, 2).contiguous(),
    )


def _candidate_length(candidate_block_indices: torch.Tensor) -> torch.Tensor:
    """Recover the valid prefix length from QLI's contiguous ``-1`` tail."""
    return (candidate_block_indices >= 0).sum(dim=-1).to(torch.int32).contiguous()


def _validate_candidate_role(
    *,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    is_source: bool,
    candidate: torch.Tensor | None,
) -> None:
    has_candidate_pool = candidate_topk_blocks > 0
    if is_source and not has_candidate_pool:
        raise ValueError("a candidate source requires candidate_topk_blocks > 0")
    if has_candidate_pool and candidate_block_size <= 0:
        raise ValueError("an enabled candidate pool requires candidate_block_size > 0")

    expects_candidate = has_candidate_pool and not is_source
    if (candidate is not None) != expects_candidate:
        role = (
            "a searcher requires the source candidate table"
            if expects_candidate
            else "this layer cannot consume a candidate table"
        )
        raise ValueError(f"the candidate table disagrees with this layer's role: {role}")
    if candidate is not None and (candidate.ndim != 4 or candidate.shape[0] != 1 or candidate.shape[2] != 1):
        raise ValueError(f"the model candidate table must have shape [1, T, 1, capacity], got {tuple(candidate.shape)}")


def _quantized_forward(
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    topk: int,
    *,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    is_source: bool,
    candidate: torch.Tensor | None,
    **kernel_options,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run QLI for a producer/pool-free layer or QSLI for a searcher."""
    _validate_candidate_role(
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        is_source=is_source,
        candidate=candidate,
    )

    geometry = _quantized_kernel_geometry(idx_q.shape[1], idx_q.shape[2], topk)
    idx_q_mxfp4, idx_q_scale = _pack_mxfp4(idx_q)
    idx_k_mxfp4, idx_k_scale = _pack_mxfp4(idx_k)

    if candidate is None:
        qli_metadata = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
            **kernel_options,
            **geometry,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )
        topk_indices, _, candidate_block_indices, _ = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer(
            idx_q_mxfp4,
            idx_k_mxfp4,
            idx_w,
            idx_q_scale,
            idx_k_scale,
            topk,
            _QUANT_MODE,
            metadata=qli_metadata,
            return_value=False,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            **kernel_options,
        )
        return topk_indices, candidate_block_indices.unsqueeze(0) if is_source else None

    block_indices = candidate.squeeze(0)
    block_length = _candidate_length(block_indices)
    qsli_metadata = torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata(
        block_length,
        **kernel_options,
        **geometry,
        quant_mode=_QUANT_MODE,
        candidate_block_size=candidate_block_size,
    )
    topk_indices, _ = torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer(
        idx_q_mxfp4,
        idx_k_mxfp4,
        idx_w,
        idx_q_scale,
        block_indices,
        block_length,
        topk,
        _QUANT_MODE,
        candidate_block_size,
        descale_k=idx_k_scale,
        metadata=qsli_metadata,
        return_value=False,
        **kernel_options,
    )
    return topk_indices, candidate


class _QuantV41LightningIndexerTND(torch.autograd.Function):
    """QLI/QSLI forward paired with the BF16 SLIKG training backward."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        idx_q,
        idx_k,
        idx_w,
        topk,
        ratio,
        attention_masks,
        candidate_topk_blocks,
        candidate_block_size,
        is_source,
        candidate,
        num_global_queries,
    ):
        topk_indices, candidate_out = _quantized_forward(
            idx_q,
            idx_k,
            idx_w,
            topk,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            is_source=is_source,
            candidate=candidate,
            **_kernel_options(attention_masks, ratio),
        )
        topk_indices = topk_indices.sort(dim=-1, descending=True).values
        topk_scores = topk_indices.new_empty(topk_indices.shape, dtype=torch.float32, requires_grad=True)

        ctx.save_for_backward(idx_q, idx_k, idx_w, topk_indices, num_global_queries)
        ctx.topk = topk
        ctx.ratio = ratio
        ctx.attention_masks = attention_masks
        return topk_indices, topk_scores, candidate_out

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx,
        grad_indices,
        attn_softmax_l1_norm,
        grad_candidate,
    ):
        del grad_indices, grad_candidate
        if attn_softmax_l1_norm is None:
            return (None,) * 11

        idx_q, idx_k, idx_w, topk_indices, num_global_queries = ctx.saved_tensors
        if num_global_queries is None:
            raise RuntimeError("Set Selector.num_global_queries before training the indexer")
        # Match the BF16 path: the trainer counts queries across the full step and DP mesh.
        attn_softmax_l1_norm = attn_softmax_l1_norm / num_global_queries
        options = _kernel_options(ctx.attention_masks, ctx.ratio)
        slikg_metadata = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
            **_slikg_kernel_geometry(idx_q.shape[1], idx_q.shape[2], ctx.topk),
            **options,
        )
        dq, dk, dw, _ = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q=idx_q,
            k=idx_k,
            w=idx_w,
            sparse_indices=topk_indices,
            attn_softmax_l1_norm=attn_softmax_l1_norm,
            metadata=slikg_metadata,
            **options,
        )
        return (
            dq,
            dk,
            dw,
            None,  # topk
            None,  # ratio
            None,  # attention_masks
            None,  # candidate_topk_blocks
            None,  # candidate_block_size
            None,  # is_source
            None,  # candidate
            None,  # num_global_queries
        )


class QuantV41LightningIndexer(torch.nn.Module):
    """Forward behavior installed on a DeepSeek-V4.1 ``Selector`` instance."""

    # The module-swap handler binds this forward to a Selector instance. Declare the
    # host-owned fields here so static analysis does not fall back to nn.Module's
    # dynamic attribute type (Module | Tensor).
    index_topk: int
    compress_ratio: int
    candidate_topk_blocks: int
    candidate_block_size: int
    has_candidate_pool: bool
    mode: str
    num_global_queries: torch.Tensor | None

    def forward(
        self,
        idx_q_BLHiDi: torch.Tensor,
        idx_k_BNDi: torch.Tensor,
        weights_BLHi: torch.Tensor,
        attention_masks,
        *,
        candidates_BL1C: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if idx_q_BLHiDi.shape[0] != 1 or idx_k_BNDi.shape[0] != 1 or weights_BLHi.shape[0] != 1:
            raise ValueError("V4.1 QLI/QSLI requires one packed TND row (model batch size 1)")

        topk = self.index_topk
        is_source = self.has_candidate_pool and self.mode == "full"
        topk_indices, topk_scores, candidate = _QuantV41LightningIndexerTND.apply(
            idx_q_BLHiDi.squeeze(0),
            idx_k_BNDi.squeeze(0).unsqueeze(1),
            weights_BLHi.squeeze(0).float(),
            topk,
            self.compress_ratio,
            attention_masks,
            self.candidate_topk_blocks,
            self.candidate_block_size,
            is_source,
            candidates_BL1C,
            self.num_global_queries,
        )
        return (
            topk_indices.reshape(1, -1, topk),
            topk_scores.reshape(1, -1, topk),
            candidate,
        )
