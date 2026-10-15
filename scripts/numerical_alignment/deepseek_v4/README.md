# DeepSeek-V4 推理精度对比

比较同一份权重在 Transformers 和 TorchTitan 上的最后一个 token 的 BF16 logits。范围为 Flash 前 1～4 层、前 4 个 routed experts、top-3，关闭 MTP；默认四层、batch 1、序列长度 4096、输入 seed 1234，无需 tokenizer。

## 准备权重与环境

从仓库根目录执行，先安装项目依赖、加载 CANN 环境，并准备空闲的 Ascend NPU。HF 权重需已反量化为 BF16；DCP 必须由同一份 HF 权重转换，并采用相同层数、四专家和 top-3 配置。已有 DCP 可直接使用；权重转换工具需另行准备。

```bash
python -m pip install -r scripts/numerical_alignment/requirements.txt
export HF_BF16=/path/to/dsv4-bf16
export TT_DCP=/path/to/dsv4-dcp-4L
```

基线固定为 `transformers==5.13.0`，入口会检查版本，并关闭 NPU matmul、conv、ACLNN 的 HF32，启用确定性算法。

## 对比 logits

```bash
python -m scripts.numerical_alignment.deepseek_v4.compare_dsv4_with_transformers \
    --hf-checkpoint-path "$HF_BF16" \
    --tt-checkpoint-path "$TT_DCP"
```

默认用 Golden override，按 `bitwise` 判定；`--mode close` 改为数值近似，要求 `mean_ae <= 0.02`、`mean_re <= 0.02`、`cos >= 0.9999`。控制台打印两侧加载过程、补丁状态、logits shape、指标及 `PASS` / `FAIL`；`max_re` 仅作诊断。通过返回 `0`，不通过返回 `1`。

Golden 模式仅支持 `--batch-size 1`，`--seq-len` 须为 128 的倍数；不支持的参数会在加载权重前报错。

### 基线来源与可选诊断

`transformers_dsv4_patch.py` 存在时默认启用数值补丁，参考为打补丁后的 Transformers；缺少该模块时使用原生 Transformers。补丁调整 RoPE、RMSNorm、mHC、indexer、attention 和 MoE 的计算顺序及 BF16 边界，不能用其通过结果证明原生 Transformers 也比特一致。

观察原生基线，只需在上述命令追加：

```bash
--no-apply-transformers-patch
```

`--apply-transformers-patch` 可显式要求补丁；`--trace-modules` 记录两侧中间输出用于定位，不参与最终判定。这两个功能需要对应可选模块；缺少模块时会在加载权重前明确报错。

`--attention test` 切换 AscendC override，需要对应 CANN 算子；结果按所选精度模式独立判定。`--n-layers` 可选择 1～4 层，须与 DCP 配置相同。完整参数见 `--help`，指标与验收要求见 [精度对齐开发指南](../../../docs/developer_guides/numerical_alignment.md)。
