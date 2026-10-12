# 数值对齐脚本

本目录提供训练指标对比和局部模块前向、反向对比两个入口。运行前需安装项目依赖、加载 CANN 环境，并准备空闲的 Ascend NPU。以下命令均从仓库根目录执行。

## 训练指标对比：loss_compare.py

[loss_compare.py](loss_compare.py) 比较两个 Git 分支或提交的逐 step `loss` 和 `grad_norm`。两侧分别在独立 worktree 中运行，默认加载同一份 seed checkpoint，并开启确定性训练。

例如，使用 Golden 实现比较前后两个提交：

```bash
export HF_ASSETS_PATH=/path/to/tokenizer
export CKPT_SAVE_LOAD_PATH=checkpoint

python3 scripts/numerical_alignment/loss_compare.py HEAD~1 HEAD \
  --steps 100 --seq-len 4096 --mode bitwise \
  --baseline-env USE_GOLDEN=1 --test-env USE_GOLDEN=1
```

将 `HF_ASSETS_PATH` 替换为实际 tokenizer 目录。常用参数：

- `--mode bitwise`：比特一致，默认模式；`--mode close`：按数值近似阈值判定。
- `--runner`：指定训练入口，默认使用 DeepSeek-V4 mini 单卡脚本。相对路径分别使用两侧 Git ref 中的脚本；绝对路径两侧共用。
- `--steps`、`--seq-len`、`--ngpus`：训练步数、序列长度、NPU 进程数，默认分别为 `100`、`4096`、`1`。
- `--baseline-options` / `--test-options`：分别传入训练参数；序列长度、步数、随机种子、确定性开关和 TensorBoard 固定配置由比较脚本控制，冲突时会报错。
- `--baseline-env` / `--test-env`：分别注入 `KEY=VALUE[,...]` 环境变量；seed 由 baseline 单卡生成，使用 baseline 环境，两侧模型权重需兼容。
- `--output-dir`：指定空目录；默认在 `outputs/compare_results/` 下创建独立结果目录。
- `--no-cleanup-after-test`：保留全部产物；默认在退出时清理 seed 和临时 worktree，保留训练日志、TensorBoard 数据和 `comparison.json`。

控制台输出两个指标的 `PASS` / `FAIL` 和误差摘要，详细结果写入 `comparison.json`。两侧指标必须完整覆盖 `1..steps`，缺步不能判定通过；当前不支持恢复完整训练状态的续训比较。

## 局部模块对比：compare_module_example.py

[compare_module_example.py](compare_module_example.py) 以 RMSNorm 为例，比较 Torch 实现和 AscRMSNorm 融合实现的前向输出、输入梯度和权重梯度。两侧使用相同输入、权重、dtype 和上游梯度。

使用模块入口启动：

```bash
python3 -m scripts.numerical_alignment.compare_module_example
python3 -m scripts.numerical_alignment.compare_module_example --dtype float32
```

默认 shape 为 `(2, 17, 4096)`，dtype 为 `bfloat16`，eps 为 `1e-6`，seed 为 `1234`。可通过 `--shape`、`--dtype`、`--eps`、`--seed` 和 `--device` 调整；dtype 支持 `bfloat16`、`float16`、`float32`，device 仅支持 NPU。脚本使用相对导入，不能直接通过文件路径启动。

控制台分别输出 `forward`、`input_grad`、`weight_grad` 的 `mean_ae`、`max_re`、`mean_re`、`cos` 和判定结果。判定使用 `mean_ae`、`mean_re`、`cos`，`max_re` 仅作诊断；任一部分不通过则整体 `FAIL`。

两个脚本均在精度检查通过时返回 `0`，不通过时返回 `1`。更多参数见各自的 `--help`；指标定义与验证要求见[数值对齐开发指南](../../docs/developer_guides/numerical_alignment.md)。
