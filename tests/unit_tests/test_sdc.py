# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torchtitan.trainer import Trainer

from torchtitan_npu.extensions.components.sdc.sdc import SDC
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx


class TorchNpuCheckerFake:
    def __init__(self, *, selected: list[bool] | None = None) -> None:
        self.hook_enabled: int | None = None
        self.with_checksum = False
        self.cooldown = 5
        self.strikes_num = 3
        self.strikes_window = 480
        self.checksum_cooldown = 180
        self.upper_thresh1 = 1_000_000
        self.upper_thresh2 = 100
        self.grad_sample_interval = 3
        self.check_stat: dict[str, dict[str, float | int]] = {}
        self.matmul_with_bf16 = False
        self.stream_initializations = 0
        self.startups = 0
        self.detected: list[tuple[torch.Tensor, str]] = []
        self._selected = iter(selected or [])

    def set_matmul_hook_enable(self, value: int) -> None:
        self.hook_enabled = value

    def set_with_checksum(self, value: bool) -> None:
        self.with_checksum = value

    def set_cooldown(self, value: int) -> None:
        self.cooldown = value

    def set_strikes_num(self, value: int) -> None:
        self.strikes_num = value

    def set_strikes_window(self, value: int) -> None:
        self.strikes_window = value

    def set_checksum_cooldown(self, value: int) -> None:
        self.checksum_cooldown = value

    def set_upper_thresh1(self, value: int) -> None:
        self.upper_thresh1 = value

    def set_upper_thresh2(self, value: int) -> None:
        self.upper_thresh2 = value

    def set_grad_sample_interval(self, value: int) -> None:
        self.grad_sample_interval = value

    def init_stream(self) -> None:
        self.stream_initializations += 1

    def parameter_filtering(self) -> bool:
        return next(self._selected)

    def _startup(self) -> None:
        self.startups += 1

    def _detect_grad(self, gradient: torch.Tensor, state_key: str) -> None:
        self.detected.append((gradient, state_key))


def _install_torch_npu(
    monkeypatch,
    *,
    selected: list[bool] | None = None,
) -> tuple[TorchNpuCheckerFake, list[tuple[str, str]]]:
    checker = TorchNpuCheckerFake(selected=selected)
    state_events: list[tuple[str, str]] = []

    torch_npu = ModuleType("torch_npu")
    vars(torch_npu)["asd"] = SimpleNamespace(asd=SimpleNamespace(matmul_check=checker))
    vars(torch_npu)["_C"] = SimpleNamespace(
        _npu_set_module_train_state=lambda state: state_events.append(("model", state)),
        _npu_set_call_state=lambda state: state_events.append(("call", state)),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    return checker, state_events


@pytest.mark.parametrize(
    "hccl_mode",
    [
        pytest.param(-1, id="below-range"),
        pytest.param(4, id="above-range"),
        pytest.param(1.5, id="non-integer"),
    ],
)
def test_sdc_config_rejects_invalid_hccl_modes(hccl_mode: Any) -> None:
    with pytest.raises(ValueError, match="must be 0, 1, 2 or 3"):
        SDC.Config(hccl_mode=hccl_mode)


def test_sdc_config_rejects_checksum_without_gradient_detection() -> None:
    with pytest.raises(ValueError, match=r"requires --sdc.gradient-enabled=true"):
        SDC.Config(with_checksum=True)


@pytest.mark.parametrize(
    ("option", "value", "expected_message"),
    [
        pytest.param("cooldown", 0, "--sdc.cooldown", id="cooldown-below-range"),
        pytest.param("strikes_num", 0, "--sdc.strikes-num", id="strikes-num-below-range"),
        pytest.param("strikes_window", 0, "--sdc.strikes-window", id="strikes-window-below-range"),
        pytest.param(
            "checksum_cooldown",
            0,
            "--sdc.checksum-cooldown",
            id="checksum-cooldown-below-range",
        ),
        pytest.param("upper_thresh1", 2, "--sdc.upper-thresh1", id="upper-thresh1-below-range"),
        pytest.param("upper_thresh2", 2, "--sdc.upper-thresh2", id="upper-thresh2-below-range"),
        pytest.param(
            "grad_sample_interval",
            0,
            "--sdc.grad-sample-interval",
            id="grad-sample-interval-below-range",
        ),
        pytest.param("cooldown", 1.5, "must be an integer", id="cooldown-non-integer"),
    ],
)
def test_sdc_config_rejects_invalid_gradient_tuning(
    option: str,
    value: Any,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        SDC.Config(gradient_enabled=True, **{option: value})


@pytest.mark.parametrize(
    ("option", "value"),
    [
        pytest.param("cooldown", 6, id="cooldown"),
        pytest.param("strikes_num", 4, id="strikes-num"),
        pytest.param("strikes_window", 481, id="strikes-window"),
        pytest.param("checksum_cooldown", 181, id="checksum-cooldown"),
        pytest.param("upper_thresh1", 1_000_001, id="upper-thresh1"),
        pytest.param("upper_thresh2", 101, id="upper-thresh2"),
        pytest.param("grad_sample_interval", 4, id="grad-sample-interval"),
    ],
)
def test_sdc_config_rejects_non_default_tuning_without_gradient_detection(option: str, value: int) -> None:
    with pytest.raises(ValueError, match=r"requires --sdc.gradient-enabled=true"):
        SDC.Config(**{option: value})


# Gradient monitoring


def _compiled_trainer_config() -> SimpleNamespace:
    return SimpleNamespace(
        compile=SimpleNamespace(enable=True, components=("model",), backend="inductor"),
        parallelism=SimpleNamespace(pipeline_parallel_degree=1),
    )


def _build_gradient_sdc(model_parts: list[Any], accumulation_steps: int = 1, **options: Any) -> SDC:
    return SDC.Config(gradient_enabled=True, **options).build(
        trainer_config=_compiled_trainer_config(),
        model_parts=model_parts,
        gradient_accumulation_steps=accumulation_steps,
    )


@pytest.fixture(autouse=True)
def _isolate_native_sdc_environment(monkeypatch):
    monkeypatch.delenv("NPU_ASD_CONFIG", raising=False)
    monkeypatch.delenv("NPU_ASD_ENABLE", raising=False)
    with torch.random.fork_rng():
        yield


@pytest.mark.parametrize(
    ("accumulation_steps", "expected_submissions"),
    [
        pytest.param(1, [1, 2, 3], id="every-step"),
        pytest.param(3, [0, 0, 1, 1], id="every-third-step"),
    ],
)
def test_sdc_submits_only_at_configured_accumulation_boundaries(
    monkeypatch,
    accumulation_steps: int,
    expected_submissions: list[int],
) -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    checker, _ = _install_torch_npu(monkeypatch, selected=[True] * 4)
    sdc = _build_gradient_sdc([model], accumulation_steps)

    for expected in expected_submissions:
        sdc.finalize_sdc_step()
        assert len(checker.detected) == expected

    gradient, name = checker.detected[0]
    assert torch.equal(gradient, model.weight.grad)
    assert name == "weight_backward"


def test_sdc_registers_only_supported_parameters_and_marks_bf16(monkeypatch) -> None:
    class MixedParameterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.supported = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
            self.vector = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
            self.fp16_matrix = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float16))
            self.frozen = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)

    checker, _ = _install_torch_npu(monkeypatch, selected=[True] * 4)

    _build_gradient_sdc([MixedParameterModel()])

    assert set(checker.check_stat) == {"supported_backward"}
    assert checker.matmul_with_bf16 is True


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdc_submits_only_targets_selected_by_native_checker(monkeypatch, dtype: torch.dtype) -> None:
    checker, _ = _install_torch_npu(monkeypatch, selected=[False])
    model = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
    model.weight.grad = torch.ones_like(model.weight)
    sdc = _build_gradient_sdc([model])

    sdc.finalize_sdc_step()

    assert checker.check_stat == {}
    assert checker.detected == []
    assert checker.matmul_with_bf16 is (dtype == torch.bfloat16)


def test_sdc_skips_targets_without_gradients(monkeypatch) -> None:
    class PartiallyUsedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Parameter(torch.ones(2, 2))
            self.unused = torch.nn.Parameter(torch.ones(2, 2))

    model = PartiallyUsedModel()
    model.used.grad = torch.ones_like(model.used)
    checker, _ = _install_torch_npu(monkeypatch, selected=[True] * 4)
    sdc = _build_gradient_sdc([model])

    sdc.finalize_sdc_step()

    assert [name for _, name in checker.detected] == ["used_backward"]


def test_sdc_checks_distinct_dtensor_gradients_sharing_parameter_storage(monkeypatch, tmp_path: Path) -> None:
    torch.distributed.init_process_group("gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        shared = torch.ones(2, 2)
        model = torch.nn.Module()
        model.first = torch.nn.Parameter(DTensor.from_local(shared, mesh, [Replicate()], run_check=False))
        model.second = torch.nn.Parameter(DTensor.from_local(shared, mesh, [Replicate()], run_check=False))
        model.empty = torch.nn.Parameter(DTensor.from_local(torch.empty(0, 2), mesh, [Replicate()], run_check=False))
        assert model.first is not model.second
        assert (
            model.first.to_local().untyped_storage().data_ptr() == model.second.to_local().untyped_storage().data_ptr()
        )
        (model.first.sum() + 2 * model.second.sum()).backward()
        checker, _ = _install_torch_npu(monkeypatch, selected=[True, True])
        sdc = _build_gradient_sdc([model, model])

        sdc.finalize_sdc_step()

        assert set(checker.check_stat) == {"model_parts.0.first_backward", "model_parts.0.second_backward"}
        assert [name for _, name in checker.detected] == [
            "model_parts.0.first_backward",
            "model_parts.0.second_backward",
        ]
        assert torch.equal(checker.detected[0][0], torch.ones(2, 2))
        assert torch.equal(checker.detected[1][0], torch.full((2, 2), 2.0))
    finally:
        torch.distributed.destroy_process_group()


def test_sdc_does_not_treat_to_local_protocol_as_dtensor(monkeypatch) -> None:
    class TensorWithToLocal(torch.Tensor):
        def to_local(self) -> torch.Tensor:
            raise AssertionError("to_local must only be called for DTensor")

    parameter = torch.ones(2, 2, requires_grad=True).as_subclass(TensorWithToLocal)

    class ModelPart:
        @staticmethod
        def named_parameters(*, remove_duplicate: bool) -> list[tuple[str, torch.Tensor]]:
            assert remove_duplicate is True
            return [("weight", parameter)]

    checker, _ = _install_torch_npu(monkeypatch, selected=[True] * 4)

    _build_gradient_sdc([ModelPart()])

    assert set(checker.check_stat) == {"weight_backward"}


# SDC facade


def test_sdc_build_finishes_runtime_initialization_in_one_call(monkeypatch) -> None:
    checker, state_events = _install_torch_npu(monkeypatch, selected=[True])
    monkeypatch.delenv("NPU_ASD_CONFIG", raising=False)
    monkeypatch.delenv("NPU_ASD_ENABLE", raising=False)
    checksum_installs: list[bool] = []
    checksum_module = ModuleType("torchtitan_npu.extensions.compile.sdc_checksum")

    def install_checksum_pass() -> None:
        checksum_installs.append(True)

    vars(checksum_module)["install_checksum_pass"] = install_checksum_pass
    monkeypatch.setitem(sys.modules, "torchtitan_npu.extensions.compile.sdc_checksum", checksum_module)
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16)))

    SDC.Config(gradient_enabled=True, with_checksum=True, hccl_mode=2, cooldown=6).build(
        trainer_config=_compiled_trainer_config(),
        model_parts=[model],
        gradient_accumulation_steps=2,
    )

    assert checker.hook_enabled == 1
    assert checker.with_checksum is True
    assert checker.cooldown == 6
    assert checker.stream_initializations == 1
    assert checker.startups == 1
    assert checker.matmul_with_bf16 is True
    assert checker.check_stat == {
        "weight_backward": {"avg": 0.0, "pre_val": 0.0, "step": 0, "none_zero_step": 0},
    }
    assert os.environ["NPU_ASD_ENABLE"] == "2"
    assert state_events == [("model", "train"), ("call", "backward")]
    assert checksum_installs == [True]


@pytest.mark.parametrize(
    ("environment_name", "environment_value"),
    [
        pytest.param("NPU_ASD_CONFIG", "enable:true", id="native-gradient"),
        pytest.param("NPU_ASD_ENABLE", "0", id="native-hccl-zero"),
    ],
)
def test_inactive_sdc_leaves_native_eager_environment_unchanged(
    monkeypatch, environment_name: str, environment_value: str
) -> None:
    original_import = builtins.__import__

    def reject_torch_npu_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch_npu":
            raise AssertionError("inactive SDC imported torch_npu")
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setattr(builtins, "__import__", reject_torch_npu_import)

    sdc = SDC.Config().build(trainer_config=object(), model_parts=[], gradient_accumulation_steps=1)
    sdc.finalize_sdc_step()

    assert os.environ[environment_name] == environment_value


@pytest.mark.parametrize(
    ("native_config", "native_hccl"),
    [(None, None), ("enable:false", "0"), ("", "0"), ("cooldown:6", None), ("enable:true,enable:false", "0")],
)
def test_active_sdc_accepts_disabled_native_environment(monkeypatch, native_config, native_hccl) -> None:
    checker, state_events = _install_torch_npu(monkeypatch)
    if native_config is not None:
        monkeypatch.setenv("NPU_ASD_CONFIG", native_config)
    if native_hccl is not None:
        monkeypatch.setenv("NPU_ASD_ENABLE", native_hccl)

    _build_gradient_sdc([])

    assert checker.hook_enabled == 1
    assert checker.startups == 1
    assert state_events == []
    assert os.environ.get("NPU_ASD_CONFIG") == native_config
    assert os.environ.get("NPU_ASD_ENABLE") == native_hccl


@pytest.mark.parametrize(
    ("environment_name", "environment_value"),
    [
        ("NPU_ASD_CONFIG", "enable:true"),
        ("NPU_ASD_CONFIG", "enable:false,enable:true"),
        ("NPU_ASD_ENABLE", "1"),
        ("NPU_ASD_ENABLE", "2"),
        ("NPU_ASD_ENABLE", "3"),
        ("NPU_ASD_CONFIG", "enable:TRUE"),
        ("NPU_ASD_CONFIG", "enable:0"),
        ("NPU_ASD_ENABLE", "false"),
        ("NPU_ASD_ENABLE", "4"),
        ("NPU_ASD_ENABLE", ""),
    ],
)
def test_active_sdc_rejects_native_eager_environment_before_runtime_changes(
    monkeypatch, environment_name: str, environment_value: str
) -> None:
    checker, state_events = _install_torch_npu(monkeypatch)
    monkeypatch.setenv("NPU_ASD_CONFIG", "enable:false")
    monkeypatch.setenv(environment_name, environment_value)

    with pytest.raises(ValueError, match=environment_name):
        _build_gradient_sdc([])

    assert checker.hook_enabled is None
    assert checker.startups == 0
    assert state_events == []
    assert os.environ[environment_name] == environment_value


def test_compiled_hccl_overrides_disabled_environment_after_import(monkeypatch) -> None:
    checker, state_events = _install_torch_npu(monkeypatch)
    monkeypatch.setenv("NPU_ASD_CONFIG", "enable:false")
    monkeypatch.setenv("NPU_ASD_ENABLE", "0")
    original_import = builtins.__import__
    import_values = []

    def record_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch_npu":
            import_values.append(os.environ["NPU_ASD_ENABLE"])
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_import)
    SDC.Config(hccl_mode=2).build(
        trainer_config=_compiled_trainer_config(), model_parts=[], gradient_accumulation_steps=1
    )

    assert import_values == ["0"]
    assert os.environ["NPU_ASD_ENABLE"] == "2"
    assert os.environ["NPU_ASD_CONFIG"] == "enable:false"
    assert state_events == [("model", "train"), ("call", "backward")]
    assert checker.startups == 0


def test_active_sdc_validates_compile_before_runtime_changes(monkeypatch) -> None:
    checker, state_events = _install_torch_npu(monkeypatch)
    trainer_config = _compiled_trainer_config()
    trainer_config.compile.enable = False

    with pytest.raises(ValueError, match=r"requires compile.enable=true"):
        SDC.Config(gradient_enabled=True).build(
            trainer_config=trainer_config, model_parts=[], gradient_accumulation_steps=1
        )

    assert checker.hook_enabled is None
    assert checker.startups == 0
    assert state_events == []


def test_sdc_applies_all_gradient_tuning_to_native_checker(monkeypatch) -> None:
    checker, _ = _install_torch_npu(monkeypatch)

    _build_gradient_sdc(
        [],
        cooldown=6,
        strikes_num=4,
        strikes_window=481,
        checksum_cooldown=181,
        upper_thresh1=1_000_001,
        upper_thresh2=101,
        grad_sample_interval=4,
    )

    assert checker.hook_enabled == 1
    assert checker.with_checksum is False
    assert checker.cooldown == 6
    assert checker.strikes_num == 4
    assert checker.strikes_window == 481
    assert checker.checksum_cooldown == 181
    assert checker.upper_thresh1 == 1_000_001
    assert checker.upper_thresh2 == 101
    assert checker.grad_sample_interval == 4


def test_sdc_registers_distinct_state_for_sampled_model_parts(monkeypatch) -> None:
    checker, _ = _install_torch_npu(monkeypatch, selected=[True, False, True])
    models = []
    for _ in range(3):
        model = torch.nn.Module()
        model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2)))
        model.weight.grad = torch.full_like(model.weight, len(models) + 1)
        models.append(model)

    sdc = _build_gradient_sdc(models)
    sdc.finalize_sdc_step()

    assert set(checker.check_stat) == {"model_parts.0.weight_backward", "model_parts.2.weight_backward"}
    assert (
        checker.check_stat["model_parts.0.weight_backward"] is not checker.check_stat["model_parts.2.weight_backward"]
    )
    assert [name for _, name in checker.detected] == ["model_parts.0.weight_backward", "model_parts.2.weight_backward"]
    assert torch.equal(checker.detected[0][0], torch.ones(2, 2))
    assert torch.equal(checker.detected[1][0], torch.full((2, 2), 3.0))
    assert checker.startups == 1
    assert checker.stream_initializations == 1


def test_hccl_only_initialization_does_not_probe_gradient_runtime(monkeypatch) -> None:
    _, state_events = _install_torch_npu(monkeypatch)
    delattr(sys.modules["torch_npu"], "asd")

    sdc = SDC.Config(hccl_mode=2).build(
        trainer_config=_compiled_trainer_config(), model_parts=[], gradient_accumulation_steps=1
    )
    sdc.finalize_sdc_step()

    assert os.environ["NPU_ASD_ENABLE"] == "2"
    assert state_events == [("model", "train"), ("call", "backward")]


def test_hccl_imports_torch_npu_before_setting_native_environment(monkeypatch) -> None:
    _install_torch_npu(monkeypatch)
    original_import = builtins.__import__
    import_environments: list[str | None] = []

    def record_torch_npu_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch_npu":
            import_environments.append(os.environ.get("NPU_ASD_ENABLE"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record_torch_npu_import)

    SDC.Config(hccl_mode=2).build(
        trainer_config=_compiled_trainer_config(), model_parts=[], gradient_accumulation_steps=1
    )

    assert import_environments == [None]
    assert os.environ["NPU_ASD_ENABLE"] == "2"


# Trainer integration


def _patch_trainer_sdc_flow(monkeypatch) -> tuple[list[object], TrainerEx.Config]:
    events: list[object] = []

    def finalize_sdc_step() -> None:
        events.append("sdc-after-step")

    def build_sdc(
        *, trainer_config: object, model_parts: list[Any], gradient_accumulation_steps: int
    ) -> SimpleNamespace:
        events.append(("sdc-create", trainer_config, tuple(model_parts), gradient_accumulation_steps))
        return SimpleNamespace(finalize_sdc_step=finalize_sdc_step)

    config = cast(
        "TrainerEx.Config",
        SimpleNamespace(
            sdc=SimpleNamespace(build=build_sdc),
            anticipatory=SimpleNamespace(enable=False),
            extension=SimpleNamespace(quantization=SimpleNamespace(enable_quantized_training=False)),
            training=SimpleNamespace(extension=SimpleNamespace(allow_hf32=True)),
        ),
    )

    def base_init(trainer: TrainerEx, received_config: object) -> None:
        events.append(("base-init", received_config))
        trainer.init_distributed()
        trainer.model_parts = ["model"]
        trainer.gradient_accumulation_steps = 2

    def init_distributed(_trainer: Trainer) -> SimpleNamespace:
        events.append("base-distributed")
        return SimpleNamespace()

    def forward_backward_step(_trainer: Trainer, **kwargs: object) -> str:
        events.append(("base-step", kwargs))
        return "base-result"

    monkeypatch.setattr(TrainerEx.__base__, "__init__", base_init)
    monkeypatch.setattr("torchtitan_npu.extensions.torchtitan.trainer.set_allow_hf32", lambda _allow_hf32: None)
    monkeypatch.setattr(TrainerEx.__base__, "init_distributed", init_distributed)
    monkeypatch.setattr(TrainerEx.__base__, "forward_backward_step", forward_backward_step)
    return events, config


def test_trainer_initializes_sdc_once_after_model_setup(monkeypatch) -> None:
    events, config = _patch_trainer_sdc_flow(monkeypatch)

    trainer = cast("TrainerEx", object.__new__(TrainerEx))
    TrainerEx.__init__(trainer, config)
    result = trainer.forward_backward_step(
        input_dict={"tokens": "batch"},  # type: ignore[dict-item]
        labels=cast("torch.Tensor", "labels"),
        global_valid_tokens=cast("torch.Tensor", "valid-tokens"),
    )

    assert result == "base-result"
    assert events == [
        ("base-init", config),
        "base-distributed",
        ("sdc-create", config, ("model",), 2),
        (
            "base-step",
            {
                "input_dict": {"tokens": "batch"},
                "labels": "labels",
                "global_valid_tokens": "valid-tokens",
            },
        ),
        "sdc-after-step",
    ]


def test_trainer_failed_backward_does_not_advance_gradient_accumulation(monkeypatch) -> None:
    checker, _ = _install_torch_npu(monkeypatch, selected=[True])
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2)))
    model.weight.grad = torch.ones_like(model.weight)
    sdc = _build_gradient_sdc([model], accumulation_steps=2)
    trainer = cast("TrainerEx", object.__new__(TrainerEx))
    vars(trainer)["_sdc"] = sdc
    vars(trainer)["anticipatory_schedule"] = None
    outcomes = iter(["ok", "failure", "ok"])

    def forward_backward_step(_trainer: Trainer, *_args: Any, **_kwargs: Any) -> str:
        outcome = next(outcomes)
        if outcome == "failure":
            raise RuntimeError("backward failed")
        return outcome

    monkeypatch.setattr(TrainerEx.__base__, "forward_backward_step", forward_backward_step)

    assert trainer.forward_backward_step() == "ok"
    assert checker.detected == []
    with pytest.raises(RuntimeError, match="backward failed"):
        trainer.forward_backward_step()
    assert checker.detected == []
    assert trainer.forward_backward_step() == "ok"

    assert len(checker.detected) == 1
    assert checker.detected[0][1] == "weight_backward"
    assert torch.equal(checker.detected[0][0], torch.ones(2, 2))
