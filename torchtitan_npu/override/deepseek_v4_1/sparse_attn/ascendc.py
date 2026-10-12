# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""V4.1 SMLA attention and SMLAG-backed indexer distillation."""

from dataclasses import dataclass

import torch

from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.override import _IS_A5


def _kernel_options(attention_masks, ratio, window_size):
    """The kernel options, read off the metadata's frames.

    Both the metadata call and the main call take these, and they must agree
    option-for-option; reading one source is what makes that structural rather than a
    convention two call sites have to keep in step.
    """
    # Only ratio 0 has no compressed axis, and the kernels spell that ``None``.  Every
    # other ratio has one -- at ratio 1 it is the row itself and its frame carries no
    # residual, which is exactly what the kernel wants there.
    compressed = attention_masks.frame_for(ratio) if ratio > 0 else None
    return dict(
        cu_seqlens_q=attention_masks.q.cu_seqlens,
        cu_seqlens_ori_kv=attention_masks.swa_k.cu_seqlens,
        cu_seqlens_cmp_kv=None if compressed is None else compressed.cu_seqlens,
        # The per-sequence lengths actually addressed -- how an operator is told to read a prefix
        # of its operand.  Under context parallelism they carry the whole query movement:
        # ``seqused_q`` is this rank's chunk and ``seqused_ori_kv`` its reach, so the sliding
        # window and the compressed causal limit are anchored at ``seqused_ori_kv - seqused_q``,
        # this rank's chunk offset inside the document.  At ``cp_size = 1`` each of them equals the
        # corresponding boundary difference, so passing them is a no-op there rather than a second
        # code path.
        seqused_q=attention_masks.q.seqused,
        seqused_ori_kv=attention_masks.swa_k.seqused,
        seqused_cmp_kv=None if compressed is None else compressed.seqused,
        cmp_residual_kv=None if compressed is None else compressed.residual,
        cmp_ratio=max(ratio, 1),
        ori_mask_mode=4,
        cmp_mask_mode=0 if _IS_A5 and compressed is None else 3,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
    )


def _kernel_geometry(topk_indices, cmp_k):
    """The geometry both kernel calls are built with.

    K can differ between the candidate and indexer paths, so it comes off the supplied
    shape rather than the LI configuration's top-k.
    """
    return dict(
        ori_topk=0,
        cmp_topk=0 if topk_indices is None else topk_indices.shape[-1],
        has_ori_kv=True,
        has_cmp_kv=cmp_k is not None,
    )


class _SparseMLA(torch.autograd.Function):
    """SMLAG with its teacher by-product, in the TND layout of one packed row.

    The row is the batch, so every tensor arrives with a leading 1 that the kernels do not
    take: ``squeeze(0)`` on the way in and ``reshape_as(q)`` on the way out is the whole
    translation.  A ratio-1 ``cmp_k`` is the full-resolution second stream; a ratio-2 one
    is the pooled container, wider than ``swa_k`` and addressed by ``topk_indices``.

    ``topk_scores`` is carried as an input only so the incoming gradient on it is the
    teacher edge; it is never read.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        q,
        swa_k,
        cmp_k,
        topk_indices,
        sinks,
        attention_masks,
        softmax_scale,
        ratio,
        window_size,
        topk_scores,
        wants_teacher,
    ):
        options = _kernel_options(attention_masks, ratio, window_size)
        geometry = _kernel_geometry(topk_indices, cmp_k)
        smla_metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
            q.shape[1],
            1,
            q.shape[2],
            ori_topk_length=None,
            cmp_topk_length=None,
            **options,
            **geometry,
        )
        output, lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
            q,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            cmp_sparse_indices=topk_indices,
            ori_block_table=None,
            cmp_block_table=None,
            sinks=sinks,
            metadata=smla_metadata,
            softmax_scale=softmax_scale,
            return_softmax_lse=True,
            **options,
        )
        ctx.save_for_backward(
            q,
            swa_k,
            cmp_k,
            topk_indices,
            sinks,
            output,
            lse,
        )
        ctx.softmax_scale, ctx.ratio, ctx.window_size = softmax_scale, ratio, window_size
        ctx.wants_teacher = wants_teacher
        # The mask is a dataclass, so ``save_for_backward`` cannot take it; it is a
        # context attribute instead.  That is not extra retention: it holds the same
        # boundary tensors the graph already keeps for the backward, and the node --
        # and with it this reference -- is released when that backward has run.
        ctx.attention_masks = attention_masks
        # The teacher consumes the LSE as a detached constant.
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):  # pyrefly: ignore [bad-override]
        del grad_lse  # the LSE is a teacher signal, never a loss input
        # Nothing but the operand tensors is saved.  The options and the geometry are read
        # back off ``ctx.attention_masks`` and the saved operands -- the same source the
        # forward used, which is what keeps the two kernel calls from drifting apart --
        # and the gradient metadata is built from them here.
        q, swa_k, cmp_k, topk_indices, sinks, output, lse = ctx.saved_tensors
        # ``topk_scores`` is absent on purpose: this Function takes it only so that an
        # incoming gradient on it exists, and the autograd engine routes that gradient here
        # for being an input.  Nothing is ever read off it -- the teacher comes back from
        # SMLAG below, in the carrier's own layout.
        options = _kernel_options(ctx.attention_masks, ctx.ratio, ctx.window_size)
        smla_grad_metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_grad_metadata(
            q.shape[1],
            1,
            q.shape[2],
            **options,
            **_kernel_geometry(topk_indices, cmp_k),
        )
        dq, dswa_k, dcmp_k, dsinks, _, cmp_softmax_l1_norm = torch.ops.cann_ops_transformer.sparse_flash_mla_grad(
            q,
            grad_output.contiguous(),
            output,
            lse,
            ori_kv=swa_k,
            cmp_kv=cmp_k,
            ori_sparse_indices=None,
            cmp_sparse_indices=topk_indices,
            sinks=sinks,
            metadata=smla_grad_metadata,
            ori_topk_length=None,
            cmp_topk_length=None,
            softmax_scale=ctx.softmax_scale,
            **options,
        )

        # One gradient per forward input, in order: (q, swa_k, cmp_k, topk_indices,
        # sinks, attention_masks, softmax_scale, ratio, window_size, topk_scores,
        # wants_teacher).  PyTorch silently ignores extra trailing entries, so a count
        # mismatch here would not raise -- it would quietly starve a later input, which is
        # why the list is spelled out rather than padded.
        return (
            dq,
            dswa_k,
            dcmp_k if cmp_k is not None else None,
            None,
            dsinks,
            None,
            None,
            None,
            None,
            # Plain teacher, no loss scaling: SMLAG emits the raw marginal ``p`` and SLIKG
            # applies ``dI = Z * Y - p`` itself, so ``p`` is exactly the edge SLIKG must
            # receive.
            # Zeroing is an workaround for SMLAG kernel, SLIKG demands invalid positions be zeroed.
            cmp_softmax_l1_norm.masked_fill(topk_indices < 0, 0.0) if ctx.wants_teacher else None,
            None,
        )


class AscV41SparseAttention(CompressedSparseInnerAttention2):
    """The A5 TND kernel behind the reference forward contract.

    It is the port that carries the indexer's teacher: ``topk_scores`` is not a loss
    input but the message channel between the two kernels' backwards -- SMLAG's returns
    the raw marginal ``p`` on it, and the indexer's SLIKG backward consumes that edge to
    train the indexer.  That is why the tensor keeps threading through every layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CompressedSparseInnerAttention2.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        # Not optional, unlike the reference core's default: the kernel takes the sink
        # logits as part of its softmax denominator, and the model always has them
        # (``Attention.attn_sink`` is a parameter), so ``None`` is not a case to handle.
        attn_sink: torch.Tensor,
        attention_masks,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
    ):
        if (cmp_k is None) != (topk_indices is None):
            raise ValueError("cmp_k and topk_indices must be provided together")
        # The teacher is a training-only by-product carried out on ``topk_scores``'
        # gradient into the indexer's SLIKG backward.  In eval there is no indexer
        # gradient to feed and the extra ``p`` would be dead output.
        wants_teacher = self.training and cmp_k is not None
        # Shapes, dtypes and the ratio's KV pairing are the kernels' own contract, checked
        # at the operator boundary rather than restated here.
        # The selection arrives document-local from the selector, which chose it with the
        # same kernel: the two share one coordinate system, and ``-1`` is how both spell an
        # unused slot.  Order is left exactly as it came -- the selector already put it in
        # position order, and the teacher's slot positions follow that order, so permuting
        # it here would move each slot's mass onto a different key.
        #
        # Layout, on the other hand, has to be translated: the model carries the selection
        # as ``[B, L, 1, K]`` while a TND kernel wants ``[T, N2, K]``.  The carrier
        # (``topk_scores``) needs no such translation -- it is already ``[T, 1, K]``, which
        # is both what SLIKG requires next to ``sparse_indices`` and what SMLA's teacher
        # gradient arrives on.
        output, _lse = _SparseMLA.apply(
            q.squeeze(0),
            swa_k.squeeze(0).unsqueeze(1),
            None if cmp_k is None else cmp_k.squeeze(0).unsqueeze(1),
            None if topk_indices is None else topk_indices.reshape(-1, 1, topk_indices.shape[-1]),
            attn_sink.float(),
            attention_masks,
            self.softmax_scale,
            self.compress_ratio,
            self.window_size,
            # The carrier is translated exactly like the selection above: SLIKG needs it
            # as ``[T, 1, K]``, matching the ``sparse_indices`` it is paired with there.
            (None if (topk_scores is None or not wants_teacher) else topk_scores.reshape(-1, 1, topk_scores.shape[-1])),
            wants_teacher,
        )
        return output.reshape_as(q)
