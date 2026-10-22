# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the seed entry lifecycle with Trainer construction isolated from NPU."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.tooling
SCRIPT = Path(__file__).resolve().parents[3] / "scripts/numerical_alignment/create_seed_checkpoint.py"


@pytest.fixture
def seed_entry(monkeypatch, tmp_path):
    trainer = Mock(spec=["checkpointer", "close"])
    trainer.checkpointer = Mock(spec=["save"])
    config = SimpleNamespace(
        checkpoint=SimpleNamespace(
            create_seed_checkpoint=False,
            enable=True,
            load_only=False,
            last_save_model_only=True,
            last_save_in_hf=False,
        ),
        comm=SimpleNamespace(mode="default"),
        debug=SimpleNamespace(enable_structured_logging=False),
        dump_folder=str(tmp_path),
        build=Mock(return_value=trainer),
    )
    config_manager = Mock()
    config_manager.return_value.parse_args.return_value = config
    # Only imports and NPU-dependent construction are replaced; main() is real.
    modules = {
        name: ModuleType(name)
        for name in (
            "torchtitan",
            "torchtitan.config",
            "torchtitan.observability",
            "torchtitan.tools",
            "torchtitan.tools.logging",
            "torchtitan_npu",
        )
    }
    modules["torchtitan.config"].ConfigManager = config_manager
    modules["torchtitan.observability"].structured_logger = Mock()
    modules["torchtitan.tools.logging"].init_logger = Mock()
    modules["torchtitan.tools.logging"].logger = Mock()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    destroy = Mock()
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    spec = importlib.util.spec_from_file_location("distributed_seed_entry", SCRIPT)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry, config, trainer, destroy


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("checkpoint", "create_seed_checkpoint", True, "create_seed_checkpoint=False"),
        ("checkpoint", "enable", False, "checkpoint saving to be enabled"),
        ("checkpoint", "load_only", True, "checkpoint saving to be enabled"),
        ("checkpoint", "last_save_model_only", False, "model-only native checkpoint"),
        ("checkpoint", "last_save_in_hf", True, "model-only native checkpoint"),
        ("comm", "mode", "local_tensor", "initialized distributed model weights"),
    ],
    ids=["cpu_seed", "saving_disabled", "load_only", "full_state", "hf_export", "local_tensor"],
)
def test_seed_entry_rejects_invalid_configuration_before_build(seed_entry, section, field, value, message):
    entry, config, trainer, _ = seed_entry
    setattr(getattr(config, section), field, value)

    with pytest.raises(ValueError, match=message):
        entry.main()

    config.build.assert_not_called()
    trainer.checkpointer.save.assert_not_called()


def test_seed_entry_saves_step_zero_and_releases_resources(seed_entry):
    entry, config, trainer, destroy = seed_entry

    entry.main()

    config.build.assert_called_once_with()
    trainer.checkpointer.save.assert_called_once_with(curr_step=0, last_step=True)
    trainer.close.assert_called_once_with()
    destroy.assert_called_once_with()


@pytest.mark.parametrize("failure", ["build", "save"])
def test_seed_entry_releases_resources_on_failure(seed_entry, failure):
    entry, config, trainer, destroy = seed_entry
    operation = config.build if failure == "build" else trainer.checkpointer.save
    operation.side_effect = RuntimeError("seed operation failed")

    with pytest.raises(RuntimeError, match="seed operation failed"):
        entry.main()

    if failure == "build":
        trainer.close.assert_not_called()
    else:
        trainer.close.assert_called_once_with()
    destroy.assert_called_once_with()
