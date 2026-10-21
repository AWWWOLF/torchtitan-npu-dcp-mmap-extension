# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The document alignment a packed row's compression groups need.

V4 and V4.1 both pool every group of ``R`` consecutive tokens into one main-KV entry and
address that entry axis by the row's boundaries over ``R``.  A document whose token count
is not a multiple of ``R`` therefore gets a partial trailing group that the *next*
document's tokens complete: the pooled value mixes two documents, and the boundaries no
longer say which document the group belongs to.  The dataloaders pad each document to the
least common multiple of every ratio the stack pools with, which keeps a whole group of
each ratio inside one document.
"""

import math
from collections.abc import Sequence


def compression_alignment(compress_ratios: Sequence[int]) -> int:
    """The ``A`` a packed document length must be a multiple of, ``1`` when none.

    Taking the least common multiple of every positive ratio gives the alignment that
    keeps a whole group of each ratio inside one document; that is what the dataloaders
    pad each document to.  V4's ``(1, 4, 128)`` gives 128, V4.1's ``(0, 0, 2, ...)``
    gives 2.
    """
    alignment = 1
    for ratio in compress_ratios:
        if ratio > 0:
            alignment = math.lcm(alignment, ratio)
    return alignment
