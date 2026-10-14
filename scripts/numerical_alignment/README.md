# 数值对齐脚本

本目录提供局部模块前反向对比，以及两个 Git 版本的训练指标对比。以下命令从仓库根目录执行；需安装项目依赖、加载 CANN 环境，并准备空闲的 Ascend NPU。

[DeepSeek-V4 权重转换工具](../checkpoint_conversion/README.md)支持 HF 权重反量化和 DCP 转换。

## 局部模块对比

[`compare_module_example.py`](compare_module_example.py) 以 RMSNorm 为例，比较 Torch 实现与 AscRMSNorm 融合实现。
两侧共享相同输入、权重、dtype 和上游梯度，分别检查前向输出、输入梯度和权重梯度。

```bash
python -m scripts.numerical_alignment.compare_module_example
```

默认 shape 为 `(2, 17, 4096)`、dtype 为 `bfloat16`、eps 为 `1e-6`、seed 为 `1234`。`--dtype` 支持 `bfloat16`、`float16`、`float32`，`--device` 仅支持 NPU。

控制台按 `forward`、`input_grad`、`weight_grad` 打印 `mean_ae`、`max_re`、`mean_re`、`cos` 和 `PASS` / `FAIL`。
三项检查均须满足 `mean_ae <= 0.02`、`mean_re <= 0.02`、`cos >= 0.9999`；`max_re` 仅用于诊断。
全部通过返回 `0`，任一检查不通过返回 `1`。

## 训练 loss 和 grad_norm 对比

[`loss_compare.py`](loss_compare.py) 在独立 Git worktree 中运行两个版本，默认入口为
`examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh`。
脚本从 TensorBoard 事件读取逐 step 的 `loss` 和 `grad_norm`，默认比较 100 步、序列长度 4096，开启确定性算法，固定训练 seed 为 42。

执行前需确保两个 Git ref 已存在于本地，相关修改已提交；未提交改动不会出现在 worktree 中。
准备 tokenizer、checkpoint 环境变量，并为以下示例统一选择 Golden 实现：

```bash
export HF_ASSETS_PATH=/path/to/tokenizer
export CKPT_SAVE_LOAD_PATH=checkpoint
export USE_GOLDEN=1
```

### 比特一致对比

使用 Golden 实现比较前后两个提交：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD~1 HEAD
```

### 数值近似对比

同一提交下比较 Golden 和融合算子，只需覆盖 test 的环境变量：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
    --mode close --test-env USE_GOLDEN=0
```

如有不支持确定性算法的融合算子，可开启 `--no-deterministic`。

### 不同训练参数或 runner

使用 `--baseline-options` / `--test-options` 比较两组训练参数。例如，将 test 的学习率从默认 `1e-5` 改为 `2e-5`：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD HEAD \
    --mode close --test-options=--optimizer.lr=2e-5
```

步数和序列长度统一通过 `--steps`、`--seq-len` 控制。

`--runner` 可指定其他入口。相对路径使用各自 Git ref 中的脚本，可比较同一路径下不同版本的 runner；绝对路径让两侧共用同一份脚本。
例如使用 DeepSeek-V4 Flash 八卡入口：

```bash
python scripts/numerical_alignment/loss_compare.py HEAD~1 HEAD \
    --runner examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
    --ngpus 8
```

两侧卡数不同时使用 `--baseline-ngpus` / `--test-ngpus`，并通过各自 options 设置对应并行拓扑。
比较并行策略时，还需保证全局 batch、输入数据和训练配置一致。

### 输出与清理

默认在当前目录的 `outputs/compare_results/` 下创建独立结果目录；`--output-dir` 可指定空目录或尚不存在的目录。
控制台实时打印 seed 创建和两次训练日志，结束时打印指标误差摘要及 `PASS` / `FAIL`，详细结果保存为 `comparison.json`。
两侧指标须完整覆盖 `1..steps`；全部通过返回 `0`，比较不通过返回 `1`。

默认在退出时清理 seed 目录和临时 worktree，保留 baseline / test 训练日志、TensorBoard 数据及 `comparison.json`；
使用 `--no-cleanup-after-test` 保留全部产物，包括 seed checkpoint。训练失败或比较不通过时也按该开关清理。
若两侧不能共用 seed checkpoint，可用 `--no-seed-checkpoint` 改为固定种子初始化，但仍需确保两侧初始权重可比。

指标定义与验收要求见[数值对齐开发指南](../../docs/developer_guides/numerical_alignment.md)，完整参数见各入口的 `--help`。
