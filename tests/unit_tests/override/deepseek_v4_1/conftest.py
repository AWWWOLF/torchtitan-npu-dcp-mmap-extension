# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared builders for the V4.1 sparse-attention override tests."""

from itertools import pairwise

import pytest
import torch

from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model


@pytest.fixture
def v41_metadata():
    """Build the metadata for one packed row of these document bounds, via the model.

    The test states the bounds it wants to exercise; the positions that produce
    them are rebuilt here, so the metadata comes from the model's own builder
    rather than being hand-rolled.  One ``arange`` per document, concatenated,
    is what a packed loader emits.
    """

    def _build(bounds: list[int], *, compress_ratios: tuple[int, ...] = (1, 2)):
        positions = torch.cat([torch.arange(end - begin) for begin, end in pairwise(bounds)]).unsqueeze(0)

        class _ModelOwner:
            """A stand-in for the model, as ``get_attention_masks`` sees it.

            The hook is a method, so it reaches for the metadata builder on ``self``; it is taken from
            the model class rather than restated here, so a change to the frame algebra cannot leave
            the two disagreeing.
            """

            _frame = staticmethod(DeepSeekV41Model._frame)
            _row_metadata = DeepSeekV41Model._row_metadata

        owner = _ModelOwner()
        owner.compress_ratios = compress_ratios
        return DeepSeekV41Model.get_attention_masks(owner, positions)

    return _build
