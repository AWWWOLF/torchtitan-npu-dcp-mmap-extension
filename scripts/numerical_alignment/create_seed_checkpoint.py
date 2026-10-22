# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Save initialized model weights using the configured distributed layout.

Launched through the baseline runner by loss_compare.py. Unlike TorchTitan's
single-rank CPU seed entry, this builds the normal parallelized model and saves
step 0 without calling train(), loading a checkpoint, or updating the optimizer.
This follows the seed branch of torchtitan v0.3.0's torchtitan/train.py main();
recheck that entry point and Trainer construction when upgrading torchtitan.
"""

from typing import TYPE_CHECKING, cast

import torch
from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger

import torchtitan_npu  # noqa: F401

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer


def main() -> None:
    init_logger()
    config = cast("Trainer.Config", ConfigManager().parse_args())
    if config.checkpoint.create_seed_checkpoint:
        raise ValueError("Distributed seed creation requires checkpoint.create_seed_checkpoint=False")
    if not config.checkpoint.enable or config.checkpoint.load_only:
        raise ValueError("Distributed seed creation requires checkpoint saving to be enabled")
    if not config.checkpoint.last_save_model_only or config.checkpoint.last_save_in_hf:
        raise ValueError("Distributed seed creation requires a model-only native checkpoint")
    if config.comm.mode == "local_tensor":
        raise ValueError("Distributed seed creation requires initialized distributed model weights")
    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )
    trainer: Trainer | None = None
    try:
        trainer = config.build()
        trainer.checkpointer.save(curr_step=0, last_step=True)
        logger.info("Created distributed seed checkpoint at step 0; no training updates performed")
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
