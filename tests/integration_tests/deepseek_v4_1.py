# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from tests.integration_tests import OverrideDefinitions


def build_deepseek_v4_1_test_list() -> list[OverrideDefinitions]:
    args = (
        "--override.imports", "torchtitan_npu.override.common.rope.workaround",
        "--hf-assets-path=tests/assets/deepseek_v3", "--training.steps=2",
        "--training.seq-len=128", "--training.global-batch-size=2",
        "--parallelism.data-parallel-shard-degree=2", "--parallelism.expert-parallel-degree=2",
        "--checkpoint.enable", "--checkpoint.no-last-save-model-only",
        "--checkpoint.keep-latest-k=0", "--debug.seed=42", "--debug.deterministic",
        "--optimizer.name=AdamW",
    )
    text = OverrideDefinitions(
        test_name="dsv41_lora_engram_ep2_fsdp2",
        test_descr="V4.1 LoRA with frozen Engram, adapter updates and PEFT export",
        env_vars={"MODULE": "tests.integration_tests.v41_lora_config", "CONFIG": "deepseek_v4_1_lora_training"},
        override_args=[args], expected_steps=((1, 2),), ngpu=2,
        use_golden=True, check_loss=False,
    )
    multimodal = replace(
        text, test_name="dsv41_lora_multimodal_ep2_fsdp2",
        test_descr="V4.1 image-conditioned LoRA with frozen vision and Engram",
        env_vars={
            "MODULE": "tests.integration_tests.v41_lora_config",
            "CONFIG": "deepseek_v4_1_lora_multimodal_training",
        },
    )
    return [text, multimodal]


def build_deepseek_v4_1_checkpoint_test_list(
    *, optimizers: tuple[str, ...] = ("AdamW", "Muon"), include_adapter_only: bool = True,
) -> list[OverrideDefinitions]:
    case = build_deepseek_v4_1_test_list()[0]
    common = tuple(arg for arg in case.override_args[0] if not arg.startswith(("--training.steps=", "--optimizer.name=")))
    result = []
    for optimizer in optimizers:
        args = (*common, "--training.steps=4", "--checkpoint.interval=2", f"--optimizer.name={optimizer}")
        result.append(replace(
            case, test_name=f"dsv41_lora_{optimizer.lower()}_resume_ep2_fsdp2",
            test_descr=f"V4.1 LoRA {optimizer} exact native checkpoint resume",
            override_args=[args, (*args, "--checkpoint.load-step=2")],
            expected_steps=((1, 2, 3, 4), (3, 4)), check_resume=True,
            env_vars={
                "MODULE": "tests.integration_tests.v41_lora_config",
                "CONFIG": ("deepseek_v4_1_lora_adapter_only_training" if optimizer == "AdamW"
                           else "deepseek_v4_1_lora_training"),
            },
        ))
    if include_adapter_only and "Muon" in optimizers:
        args = (*common, "--training.steps=10", "--checkpoint.interval=5", "--optimizer.name=Muon")
        result.append(replace(
            case, test_name="dsv41_lora_muon_adapter_only_resume_ep2_fsdp2",
            test_descr="V4.1 LoRA Muon adapter-only resume with five exact post-resume steps",
            override_args=[args, (*args, "--checkpoint.load-step=5")],
            expected_steps=(tuple(range(1, 11)), tuple(range(6, 11))), check_resume=True,
            env_vars={
                "MODULE": "tests.integration_tests.v41_lora_config",
                "CONFIG": "deepseek_v4_1_lora_adapter_only_training",
            },
        ))
    return result


def build_deepseek_v4_1_quantized_test_list() -> list[OverrideDefinitions]:
    return [replace(
        case, test_name=case.test_name.replace("_resume_", "_block_fp8_resume_"),
        env_vars={"MODULE": "tests.integration_tests.v41_lora_config", "CONFIG": "deepseek_v4_1_lora_quantized_training"},
    ) for case in build_deepseek_v4_1_checkpoint_test_list(include_adapter_only=False)]
