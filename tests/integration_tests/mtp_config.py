# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config module for the MTP x CP integration case.

The runner's default config module is the model package and its default ``CONFIG`` is
``deepseek_v4_debugmodel``, whose ``num_mtp_layers`` is 0.  This module is selected by the
case's ``env_vars`` so the case named after MTP actually builds the MTP layers.
"""

from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel


def deepseek_v4_debugmodel_mtp():
    """The debug model with one MTP depth, for the MTP x CP case."""
    return deepseek_v4_debugmodel(num_mtp_layers=1)
