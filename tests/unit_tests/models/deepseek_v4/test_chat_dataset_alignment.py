# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The SFT chat dataset pads every conversation to the compression grid.

The V4 stack pools ``lcm(compress_ratios)`` tokens into one compressed entry and the
plan builder refuses a stream whose documents are not a multiple of that grid, so the
chat path must pad like the C4 text path does: the conversation tokens stay a prefix,
the pads carry ``IGNORE_INDEX`` labels, and the positions buffer keeps running.
"""

import pytest
from datasets import Dataset
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from torchtitan_npu.models.deepseek_v4.chat_dataset import DeepSeekV4ChatDataset, _sft_messages

_TOKENS = [10, 11, 12, 13, 14]
_MASK = [False, False, True, True, True]
_SAMPLE = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}


class _StubEncoder:
    """A deterministic stand-in for the V4 chat encoder (no tokenizer asset needed)."""

    def encode_messages_with_assistant_mask(self, messages, tokenizer, eos_id):
        return None, _TOKENS, _MASK


def _dataset(*, seq_len: int, per_doc_alignment: int) -> DeepSeekV4ChatDataset:
    return DeepSeekV4ChatDataset(
        dataset=Dataset.from_list([_SAMPLE]),
        tokenizer=HuggingFaceTokenizer(tokenizer_path="tests/assets/deepseek_v3"),
        sample_processor=_sft_messages,
        seq_len=seq_len,
        dp_rank=0,
        dp_world_size=1,
        infinite=False,
        chat_encoder=_StubEncoder(),
        per_doc_alignment=per_doc_alignment,
    )


def test_chat_sample_is_padded_to_the_compression_grid():
    dataset = _dataset(seq_len=256, per_doc_alignment=128)
    input_ids, label_ids = dataset._tokenize_sample(_SAMPLE)

    # The conversation is a prefix; the pads are appended with IGNORE labels.
    assert input_ids[:4] == _TOKENS[:-1]
    assert label_ids[:4] == [IGNORE_INDEX, 12, 13, 14]
    assert len(input_ids) == len(label_ids) == 128
    assert input_ids[4:] == [0] * 124
    assert label_ids[4:] == [IGNORE_INDEX] * 124


def test_chat_dataset_rejects_a_seq_len_off_the_alignment():
    with pytest.raises(ValueError, match="must be a multiple of per_doc_alignment"):
        _dataset(seq_len=130, per_doc_alignment=128)
