#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional Transformers DSV4 patch for TorchTitan numerical alignment."""

from __future__ import annotations

import math
import types
from functools import cache

import torch
import torch.nn.functional as F

from ..utils.trace_hooks import record_trace

_ROPE_DEBUG_TRACE: dict[str, torch.Tensor] | None = None
_ROPE_DEBUG_CALL = 0


def _set_rope_debug_trace(trace: dict[str, torch.Tensor] | None) -> None:
    global _ROPE_DEBUG_TRACE, _ROPE_DEBUG_CALL
    _ROPE_DEBUG_TRACE = trace
    _ROPE_DEBUG_CALL = 0


def _tt_transformers_rms_norm_forward(module, x: torch.Tensor) -> torch.Tensor:
    """Run a Transformers DSV4 RMSNorm with the torchtitan-npu reference dtype recipe."""

    dtype = x.dtype
    x = x.float()
    variance = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + module.variance_epsilon)
    return (module.weight.float() * x).to(dtype)


def _tt_transformers_q_b_norm_forward(module, x: torch.Tensor) -> torch.Tensor:
    """Match the baseline's contiguous BSHD q_b normalization reduction."""

    if x.ndim == 4:
        bshd = x.transpose(1, 2).contiguous()
        variance = bshd.square().mean(-1, keepdim=True)
        return (bshd * torch.rsqrt(variance + module.eps)).transpose(1, 2)
    variance = x.square().mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + module.eps)


def _tt_transformers_rotary_forward(
    module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    layer_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the inference-baseline real ``cos/sin`` cache on-device."""

    # Transformers constructs ``*_inv_freq`` while the model is still on CPU
    # and later copies that buffer to NPU.  The inference baseline constructs
    # the inverse frequencies on NPU; exponentiation rounds differently across
    # the two devices, and the difference is already visible at layer-0 RoPE.
    # Rebuild the frequencies from config on the execution device, using the
    # same operation order as dsv4-infer-npu/torchtitan-npu.
    params = module.config.rope_parameters[layer_type]
    head_dim = (
        getattr(module.config, "head_dim", None) or module.config.hidden_size // module.config.num_attention_heads
    )
    dim = int(head_dim * params.get("partial_rotary_factor", 1.0))
    theta = params["rope_theta"]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    if params["rope_type"] == "yarn":
        original_seq_len = params["original_max_position_embeddings"]

        def correction_dim(num_rotations: float) -> float:
            return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))

        low = max(math.floor(correction_dim(params["beta_fast"])), 0)
        high = min(math.ceil(correction_dim(params["beta_slow"])), dim - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, dtype=torch.float32, device=x.device) - low) / (high - low)
        smooth = 1 - torch.clamp(ramp, 0, 1)
        factor = params["factor"]
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
    if _ROPE_DEBUG_TRACE is not None and layer_type == "main":
        record_trace(_ROPE_DEBUG_TRACE, "rope_debug.inv_freq", inv_freq)
        record_trace(
            _ROPE_DEBUG_TRACE,
            "rope_debug.max_seq_len_cached",
            torch.tensor([int(getattr(module, "max_seq_len_cached", -1))]),
        )
    positions = position_ids.detach().to(x.device)
    if _ROPE_DEBUG_TRACE is not None and layer_type == "main":
        record_trace(_ROPE_DEBUG_TRACE, "rope_debug.positions", positions)
    max_position = max(
        int(positions.max().item()) + 1,
        int(getattr(module, "max_seq_len_cached", positions.shape[-1])),
    )
    frequencies = torch.outer(torch.arange(max_position, dtype=torch.float32, device=x.device), inv_freq).float()
    selected = torch.stack((frequencies.cos(), frequencies.sin()), dim=-1)
    selected = selected[positions]
    scaling = getattr(module, f"{layer_type}_attention_scaling")
    if _ROPE_DEBUG_TRACE is not None and layer_type == "main":
        record_trace(
            _ROPE_DEBUG_TRACE,
            "rope_debug.scaling",
            torch.as_tensor(scaling, device=x.device).reshape(1),
        )
    return selected[..., 0] * scaling, selected[..., 1] * scaling


def _tt_transformers_apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply the inference-baseline adjacent-pair real rotation."""

    global _ROPE_DEBUG_CALL
    debug_call = _ROPE_DEBUG_CALL
    _ROPE_DEBUG_CALL += 1
    if _ROPE_DEBUG_TRACE is not None and debug_call < 2:
        record_trace(_ROPE_DEBUG_TRACE, f"rope_debug.{debug_call}.input", x)
        record_trace(_ROPE_DEBUG_TRACE, f"rope_debug.{debug_call}.cos", cos)
        record_trace(_ROPE_DEBUG_TRACE, f"rope_debug.{debug_call}.sin", sin)

    # Transformers keeps q/kv as [B, H, S, D], while the inference baseline
    # rotates an in-place slice of a contiguous [B, S, H, D] tensor.
    output = x.transpose(1, 2).contiguous()
    rope_dim = cos.shape[-1] * 2
    rope = output[..., -rope_dim:]
    pairs = rope.float().reshape(*rope.shape[:-1], -1, 2)
    cache_cos = cos.float().unsqueeze(2)
    cache_sin = sin.float().unsqueeze(2)
    x0, x1 = pairs.unbind(dim=-1)
    rotated = torch.stack(
        (x0 * cache_cos - x1 * cache_sin, x0 * cache_sin + x1 * cache_cos),
        dim=-1,
    ).flatten(-2)
    rope.copy_(rotated)
    result = output.transpose(1, 2)
    if _ROPE_DEBUG_TRACE is not None and debug_call < 2:
        record_trace(_ROPE_DEBUG_TRACE, f"rope_debug.{debug_call}.output", result)
    return result


def _tt_transformers_compressor_norm_forward(module, x: torch.Tensor) -> torch.Tensor:
    """Restore the BF16 boundary before the compressor RMSNorm."""

    return _tt_transformers_rms_norm_forward(module, x.to(torch.bfloat16))


@cache
def _tt_transformers_hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build the Hadamard matrix exactly as the refactored indexer does."""

    if dim & (dim - 1):
        raise ValueError(f"Hadamard dim must be a power of two, got {dim}")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


def _tt_transformers_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply the refactored indexer's normalized Hadamard projection."""

    dim = x.shape[-1]
    matrix = _tt_transformers_hadamard(dim, x.dtype, x.device)
    return F.linear(x, matrix) * (dim**-0.5)


def _tt_transformers_indexer_scorer_forward(
    module,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Match torchtitan-npu's BF16 Hadamard boundary before index scoring."""

    q = _tt_transformers_hadamard_rotate(q)
    compressed_kv = _tt_transformers_hadamard_rotate(compressed_kv)
    scores = torch.einsum("bshd,btd->bsht", q.float(), compressed_kv.float())
    weights = module.weights_proj(hidden_states) * (module.softmax_scale * module.weights_scaling)
    # Preserve Attention Gym's compensating FP32 scales and their rounding.
    scale = math.sqrt(q.shape[-2] * q.shape[-1])
    weights = weights.float() * scale
    return (scores.relu_() * weights.unsqueeze(-1)).sum(dim=2) * (1.0 / scale)


def _tt_transformers_fp32_linear_forward(module, x: torch.Tensor) -> torch.Tensor:
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), module.weight.float(), bias)


def _tt_transformers_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **_kwargs,
):
    """Run the model's reference gather attention on the Transformers operands.

    The reference core is Attention Gym's ``gather_attn`` (``impl="reference"``), so the
    patch calls that same implementation with the operands the core hands it -- the packed
    window pool, the compressed pool, and the document-local selection.  The attention
    arithmetic is therefore identical by construction (fp32 accumulation, the sink in the
    softmax denominator, and the reference path's bf16 probability cast included), and what
    the alignment still compares is the model around it: projections, norms, RoPE, the
    compressor, the indexer, mHC and MoE, all re-derived on the Transformers side.
    """

    if dropout or module.training or attention_mask is not None:
        raise ValueError(
            "the numerical-alignment attention expects eval mode, dropout 0 and no attention mask: "
            "it runs the packed reference attention, which ignores all three"
        )

    from attn_gym.sparse import gather_attn

    batch, _, seq_len, head_dim = query.shape
    compressor = getattr(module, "compressor", None)
    window = min(seq_len, getattr(module, "sliding_window", None) or seq_len)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
    kv = key.squeeze(1)
    compressed_len = kv.shape[-2] - seq_len
    if compressor is None or compressed_len <= 0:
        # A window-only layer: the core passes an empty pool and the row's own offsets.
        sparse_kv = query.new_zeros((batch, 1, 0, head_dim))
        kv_indices = query.new_empty((batch, seq_len, 0), dtype=torch.long)
        cu_seqlens_k = cu_seqlens
    else:
        if hasattr(compressor, "indexer"):
            compressed_indices = getattr(module, "_tt_compressed_indices", None)
            if compressed_indices is None:
                raise RuntimeError("CSA attention patch did not capture indexer output")
            compressed_indices = compressed_indices[..., :compressed_len]
        else:
            ratio = compressor.compress_rate
            candidates = torch.arange(compressed_len, device=query.device).expand(seq_len, -1)
            causal_limit = (torch.arange(1, seq_len + 1, device=query.device) // ratio).unsqueeze(1)
            compressed_indices = (
                torch.where(candidates < causal_limit, candidates, -1).unsqueeze(0).expand(batch, -1, -1)
            )
        sparse_kv = kv[:, seq_len:, :].unsqueeze(1)
        kv_indices = compressed_indices.contiguous()
        cu_seqlens_k = torch.tensor([0, compressed_len], dtype=torch.int32, device=query.device)

    output = gather_attn(
        query,
        kv[:, :seq_len, :].unsqueeze(1),
        sparse_kv,
        kv_indices,
        attention_sink=module.sinks,
        sliding_window_size=window,
        cu_seqlens=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        impl="reference",
        scale=scaling,
    )
    # The weights are only read when a caller asks for attentions; the alignment never does.
    return output.transpose(1, 2), None


def _tt_transformers_router_scores(module, hidden_states: torch.Tensor):
    flat = hidden_states.reshape(-1, module.hidden_dim)
    logits = F.linear(flat.float(), module.weight.float())
    scores = F.softplus(logits.float()).sqrt()
    return logits, scores


def _tt_transformers_topk_router_forward(module, hidden_states: torch.Tensor):
    logits, scores = _tt_transformers_router_scores(module, hidden_states)
    indices = (scores + module.e_score_correction_bias.float()).topk(module.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if getattr(module, "_tt_route_norm", True):
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * module.routed_scaling_factor, indices


def _tt_transformers_hash_router_forward(module, hidden_states: torch.Tensor, input_ids: torch.Tensor):
    logits, scores = _tt_transformers_router_scores(module, hidden_states)
    indices = module.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)
    if getattr(module, "_tt_route_norm", True):
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * module.routed_scaling_factor, indices


def _tt_transformers_experts_forward(
    module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Match TorchTitan's EP=1 GroupedExperts and local token dispatcher."""

    from torchtitan.ops.scatter_add import deterministic_scatter_add

    top_k = top_k_index.shape[-1]
    expert_ids = top_k_index.reshape(-1)
    expert_order = torch.argsort(expert_ids, stable=True)
    token_indices = expert_order // top_k
    routed_input = hidden_states[token_indices]

    tokens_per_expert = F.one_hot(top_k_index, num_classes=module.num_experts).sum(dim=(0, 1))
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    gate_weight, up_weight = module.gate_up_proj.chunk(2, dim=1)

    gate = torch._grouped_mm(
        routed_input.bfloat16(),
        gate_weight.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    up = torch._grouped_mm(
        routed_input.bfloat16(),
        up_weight.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    limit = getattr(module, "_tt_swiglu_limit", 0.0)
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    routed_hidden = F.silu(gate) * up
    sorted_weights = top_k_weights.reshape(-1)[expert_order]
    routed_hidden = (routed_hidden.float() * sorted_weights.reshape(-1, 1)).to(routed_hidden.dtype)
    routed_output = torch._grouped_mm(
        routed_hidden,
        module.down_proj.bfloat16().transpose(-2, -1),
        offs=offsets,
    ).type_as(hidden_states)

    output = torch.zeros_like(hidden_states)
    scatter_indices = token_indices.reshape(-1, 1).expand_as(routed_output)
    return deterministic_scatter_add(
        output,
        scatter_indices,
        routed_output,
    )


def _tt_transformers_hc_map_forward(
    module, hidden_streams: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match torchtitan-npu's mHC projection and explicit Sinkhorn recipe."""

    shape = hidden_streams.shape
    dtype = hidden_streams.dtype
    flat = hidden_streams.flatten(2).float()
    norm_eps = module.input_norm.eps
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, module.fn.float()) * rsqrt

    hc_mult = module.hc_mult
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    pre_scale, post_scale, comb_scale = module.scale.float().unbind(0)
    base = module.base.float()
    pre = torch.sigmoid(pre * pre_scale + base[:hc_mult]) + module.hc_eps
    post = 2 * torch.sigmoid(post * post_scale + base[hc_mult : 2 * hc_mult])
    comb = comb.unflatten(-1, (hc_mult, hc_mult))
    comb = comb * comb_scale + base[2 * hc_mult :].view(hc_mult, hc_mult)

    row_max = comb.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + module.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + module.hc_eps)
    for _ in range(module.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + module.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + module.hc_eps)

    collapsed = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
    return post, comb, collapsed.to(dtype)


def _tt_transformers_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Mix mHC residual streams in FP32 and cast only the completed result."""

    mixed = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return mixed.type_as(x)


def _tt_transformers_decoder_layer_forward(
    module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Run both Transformers decoder residual sites with torchtitan-npu mHC mixing."""

    residual = hidden_states
    post, comb, collapsed = module.attn_hc(hidden_states)
    attention, _ = module.self_attn(module.input_layernorm(collapsed), **kwargs)
    hidden_states = module._tt_hc_post(attention, residual, post, comb)

    residual = hidden_states
    post, comb, collapsed = module.ffn_hc(hidden_states)
    mlp = module.mlp(module.post_attention_layernorm(collapsed), input_ids=input_ids)
    hidden_states = module._tt_hc_post(mlp, residual, post, comb)
    return hidden_states


def _tt_transformers_hc_head_forward(module, hidden_streams: torch.Tensor) -> torch.Tensor:
    """Match torchtitan-npu's final mHC stream collapse operation order."""

    shape = hidden_streams.shape
    dtype = hidden_streams.dtype
    flat = hidden_streams.flatten(2).float()
    norm_eps = module.input_norm.eps
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(flat, module.hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * module.hc_scale.float() + module.hc_base.float()) + module.eps
    collapsed = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
    return collapsed.to(dtype)


def _bind_forward(module, function) -> None:
    module.forward = types.MethodType(function, module)


def _patch_transformers_for_tt(model) -> None:
    """Patch one Transformers DSV4 model to the TorchTitan numeric recipe."""

    _bind_forward(model.model.rotary_emb, _tt_transformers_rotary_forward)

    for layer in model.model.layers:
        _bind_forward(layer.input_layernorm, _tt_transformers_rms_norm_forward)
        _bind_forward(layer.post_attention_layernorm, _tt_transformers_rms_norm_forward)
        _bind_forward(layer.attn_hc, _tt_transformers_hc_map_forward)
        _bind_forward(layer.ffn_hc, _tt_transformers_hc_map_forward)
        layer._tt_hc_post = _tt_transformers_hc_post
        _bind_forward(layer, _tt_transformers_decoder_layer_forward)

        attention = layer.self_attn
        _bind_forward(attention.q_a_norm, _tt_transformers_rms_norm_forward)
        _bind_forward(attention.kv_norm, _tt_transformers_rms_norm_forward)
        _bind_forward(attention.q_b_norm, _tt_transformers_q_b_norm_forward)
        compressor = attention.compressor
        if compressor is not None:
            _bind_forward(compressor.rotary_emb, _tt_transformers_rotary_forward)
            if hasattr(compressor, "indexer"):
                _bind_forward(
                    compressor.indexer.rotary_emb,
                    _tt_transformers_rotary_forward,
                )
            for linear in (compressor.kv_proj, compressor.gate_proj):
                _bind_forward(linear, _tt_transformers_fp32_linear_forward)
            _bind_forward(
                compressor.kv_norm,
                _tt_transformers_compressor_norm_forward,
            )
            if hasattr(compressor, "indexer"):
                indexer = compressor.indexer
                for linear in (indexer.kv_proj, indexer.gate_proj):
                    _bind_forward(linear, _tt_transformers_fp32_linear_forward)
                _bind_forward(
                    indexer.kv_norm,
                    _tt_transformers_compressor_norm_forward,
                )
                _bind_forward(
                    indexer.scorer,
                    _tt_transformers_indexer_scorer_forward,
                )

                def attention_forward(_attention, *args, forward=attention.forward, **kwargs):
                    _attention._tt_compressed_indices = None
                    return forward(*args, **kwargs)

                def capture_compressed_indices(_indexer, *args, attention=attention, forward=indexer.forward, **kwargs):
                    output = forward(*args, **kwargs)
                    attention._tt_compressed_indices = output
                    return output

                _bind_forward(attention, attention_forward)
                _bind_forward(indexer, capture_compressed_indices)

        moe = layer.mlp
        moe.gate._tt_route_norm = bool(model.config.norm_topk_prob)
        _bind_forward(
            moe.gate,
            _tt_transformers_hash_router_forward if moe.is_hash else _tt_transformers_topk_router_forward,
        )
        moe.experts._tt_swiglu_limit = float(model.config.swiglu_limit)
        _bind_forward(moe.experts, _tt_transformers_experts_forward)

    _bind_forward(model.model.norm, _tt_transformers_rms_norm_forward)
    _bind_forward(model.model.hc_head, _tt_transformers_hc_head_forward)


patch_transformers_for_tt = _patch_transformers_for_tt
set_rope_debug_trace = _set_rope_debug_trace
tt_transformers_apply_rotary_pos_emb = _tt_transformers_apply_rotary_pos_emb
tt_transformers_attention_forward = _tt_transformers_attention_forward

__all__ = [
    "patch_transformers_for_tt",
    "set_rope_debug_trace",
    "tt_transformers_apply_rotary_pos_emb",
    "tt_transformers_attention_forward",
]
