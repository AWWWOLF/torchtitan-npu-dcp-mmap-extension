# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the attention-gym flat slot gather patch.

The patch must return exactly what the installed reference gather returns: it
only changes how the slots are selected, not which slots or how the attention
arithmetic around them is ordered.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu

_PATCH_PATH = Path(__file__).resolve().parents[4] / "torchtitan_npu" / "patches" / "attn_gym" / "gather_attn.py"
_REFERENCE = "attn_gym.sparse.gather_attn.impl.reference"


def _fresh_reference():
    """Reload attention-gym from its installed source, undoing any earlier patch."""
    reference = importlib.import_module(_REFERENCE)
    return importlib.reload(reference)


def _load_patch():
    spec = importlib.util.spec_from_file_location("attn_gym_patch_under_test", _PATCH_PATH)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))  # applies on import


def _case(dtype, *, documents):
    """A packed batch with ``documents`` documents and a shared KV head."""
    torch.manual_seed(0)
    batch, kv_heads, heads, per_doc, dim = 1, 1, 2, 4, 6
    window, compressed, slots = 3, per_doc // 2, 4
    tokens = documents * per_doc
    query = torch.randn(batch, heads, tokens, dim, dtype=dtype)
    local_kv = torch.randn(batch, kv_heads, tokens, dim, dtype=dtype)
    sparse_kv = torch.randn(batch, kv_heads, documents * compressed, dim, dtype=dtype)
    sink = torch.randn(heads, dtype=torch.float32)
    # Document-local picks, invalid slots padded with -1 as the API documents.
    picks = torch.tensor([[0, 1, -1, -1], [0, 1, -1, -1], [1, 0, -1, -1], [1, 0, -1, -1]])
    kv_indices = picks.repeat(documents, 1).unsqueeze(0)
    cu_seqlens = torch.arange(0, tokens + 1, per_doc, dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, documents * compressed + 1, compressed, dtype=torch.int32)
    return {
        "query": query,
        "local_kv": local_kv,
        "sparse_kv": sparse_kv,
        "kv_indices": kv_indices,
        "attention_sink": sink,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_k": cu_seqlens_k,
        "sliding_window_size": window,
        "scale": dim**-0.5,
        "slots": slots,
    }


def _gather(reference, case):
    return reference.gather_attn(
        case["query"],
        case["local_kv"],
        case["sparse_kv"],
        case["kv_indices"],
        case["attention_sink"],
        case["cu_seqlens"],
        case["cu_seqlens_k"],
        case["sliding_window_size"],
        True,
        scale=case["scale"],
    )


def _reference_call(original, case):
    return original(
        case["query"],
        case["local_kv"],
        case["sparse_kv"],
        case["kv_indices"],
        case["attention_sink"],
        case["cu_seqlens"],
        case["cu_seqlens_k"],
        case["sliding_window_size"],
        scale=case["scale"],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("documents", [1, 2], ids=["one-doc", "two-docs"])
def test_patched_gather_matches_the_installed_reference(dtype, documents):
    reference = _fresh_reference()
    original = reference._packed_gather_attn
    _load_patch()
    assert "kv_flat" in reference._packed_gather_attn.__code__.co_varnames

    case = _case(dtype, documents=documents)
    expected_output, expected_lse = _reference_call(original, case)
    output, lse = _gather(reference, case)
    torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
