# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep Attention Gym's reference gather off the CANN broadcast-index slow path.

``_packed_gather_attn`` selects slots as ``kv[b, h, indices]`` with the batch and
head ids as ``[1, 1, 1, 1]`` tensors.  CANN's ``Index`` op sees index tensors of
different shapes and runs its broadcast tiling -- a per-output-element SIMT
kernel -- for the whole selection (637 ms forward and 2.2 s forward+backward for
the DeepSeek-V4 debug model's ``[1, 1, 2048, 640]`` selection, against 1.1 ms and
15 ms for one flat index).  Flattening the pool and selecting with a single flat
index returns the same values in the same order and turns the gather into an
aclnn GatherV2; the attention arithmetic is untouched.  This rewrites that one
expression in the installed function.
"""

import importlib
import inspect

_ORIGINAL = """\
    slots = kv[
        torch.arange(batch, device=query.device)[:, None, None, None],
        torch.arange(kv_heads, device=query.device)[None, :, None, None],
        indices[:, None],
    ]"""

_PATCHED = """\
    kv_flat = kv.reshape(-1, kv.shape[-1])
    rows = (
        torch.arange(batch * kv_heads, device=query.device).view(batch, kv_heads, 1, 1) * kv.shape[2]
        + indices[:, None]
    )
    slots = kv_flat.index_select(0, rows.reshape(-1)).reshape(
        batch, kv_heads, tokens, indices.shape[-1], kv.shape[-1]
    )"""


def apply() -> None:
    """Replace the broadcast gather in attention-gym's reference implementation."""
    reference = importlib.import_module("attn_gym.sparse.gather_attn.impl.reference")
    source = inspect.getsource(vars(reference)["_packed_gather_attn"])
    namespace = dict(vars(reference))
    exec(compile(source.replace(_ORIGINAL, _PATCHED), "<attention-gym gather patch>", "exec"), namespace)
    vars(reference)["_packed_gather_attn"] = namespace["_packed_gather_attn"]


apply()
