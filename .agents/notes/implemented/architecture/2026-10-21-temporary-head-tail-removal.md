# Agent Note: 暂时移除 DeepSeek-V4 的 head-tail CP 支持

Status: accepted（临时决定；head-tail 恢复后删除本 Note）
Created: 2026-10-21
Last updated: 2026-10-21
Tracking: 本仓 !1063（连续 slab + 文档对齐）、!1064（纯 slab 计划推导）、!1066（metadata 接缝迁移）；上游
#4617（`prepare_cp_metadata` 的 `permutation` 入参）

## Context

DeepSeek-V4 的 CP 原先支持 head-tail 负载均衡：`cp_shard` 把全局行按 head/tail 交错切给各 rank，rank 持有的是
全局行的一个**置换**。计划推导因此必须把置换折进每一个 rank-local 推导：`rearrange` / `restore`、按文档物化的
`docs` 表、每个 rank 的 `CPVarlenMetadata.from_global`；窗口/块交换路由（按 `p // shard_len` 归属）与容器打包
（start-owner 规则）也都在置换坐标上计算。

metadata 接缝迁移（!1063 / !1064 / !1066）要求 CP 半边能**由全局行在 rank 内重建 metadata**：上游 #4617 的
`prepare_cp_metadata(metadata, *, permutation)` 只给全局 metadata 与一个置换。head-tail 下这条重建路径要把置换折进
varlen、窗口计划、块计划、交换路由与容器装配，等于在 metadata 里再实现一套置换感知的计划推导，参考 / 量化 /
cv_parallel 三条重切路径都要携带 permutation。

## Decision

V4 的 CP 只支持**连续 slab**：

- `--parallelism.context-parallel-load-balancer` 只接受 `None`；非 `None` 在 CP 入口显式报错（!1063 在模型的
  `_build_cp_metadata`，!1066 之后在融合 entry 的 `prepare_cp_metadata` 链路上）；
- `build_cp_plan` 不再接受 load balancer；!1064 起 rank 的 segment 结构由 slab 与全局文档边界的交得到，
  rank-local varlen 由段长 cumsum 推出，计划行是全局坐标上的 `range`；
- metadata 的 CP hook 直接对全局行调用 `build_cp_plan` 重建 plan + window，再按同一几何重填记录
  （!1066 起；此时才引入忽略 `permutation` 的写法）。

## Consequences

- V4 的 CP 失去 head-tail 负载均衡；CP 本身、长序列训练和 CP=1 不受影响，所有 CP recipe 显式传
  `--parallelism.context-parallel-load-balancer None`。
- token 到 rank 的映射与 head-tail 不同，因此同一份数据在 CP 下的逐位结果不应与旧 head-tail 运行对比；CP=1 与
  单卡参考路径不变（!1066 在 950 上验证与 !1063 逐位一致）。
- 计划推导与 metadata 重建只依赖全局行，不再需要置换、`docs` 表与 `_RankMesh` 垫片。!1064 合入后
  rank-local varlen 改由段长 cumsum 从 slab 推出（净减约 140 行）；!1066 合入后 CP hook 只做
  「`build_cp_plan` + 重填记录」，并忽略上游传入的 `permutation`。
- V3.2 仍拒绝 CP，V4.1 已经拒绝 head-tail；本决定只影响 V4。

## Risks

- 失去 head-tail 的负载均衡收益：连续 slab 下每个 rank 的 token 数仍然均分，但注意力计算在文档边界处可能不均；
  目前没有吞吐对比数据，恢复时需补 CP=2 的吞吐测量。
- 上游 `prepare_cp_metadata` 的签名仍携带 `permutation`；如果未来上游要求必须消费它，V4 的 hook 需要重新实现
  置换感知路径。

## Restoration

恢复分两步，顺序固定：

1. **先把 V4 的 CP 迁移到 V4.1 的模式**：V4.1 的 CP 是「query exchange + KV 保持整行」——每篇文档按 `cp_size`
   对齐（`compression_alignment`），rank 持有每篇文档的 chunk `r`，KV 在 kernel 边界做声明式 all-gather；
   metadata 只描述 `cu_seq_q` / `cu_seq_k` / `positions_q` / `exchange`（见 `deepseek_v4_1/context_parallel.py` 与
   `DeepSeekV41Metadata.prepare_cp_metadata`）。这一步与 head-tail 无关，先做可以让 V4 的 CP metadata 与 V4.1
   共用同一套契约，也让 `prepare_cp_metadata` 不再需要 slab 交集推导。
2. **在这个基础上重新添加 head-tail**：head-tail 在新模式下只是「query chunk → rank」的置换，加在 query exchange 的
   `send_index` / `positions_q` 上，不再影响 metadata 的 plan 推导。恢复时用 CP=2/4 的 head-tail recipe 在 910 上做
   逐 step loss / grad_norm 对照，并在 950 上确认 CP=1 数值不变。

两步完成前，「连续 slab」的决定保持有效；完成后删除本 Note。!1066 合入后，metadata 接缝会让 CP 半边
只依赖 `build_cp_plan` + 记录重填；迁移到 V4.1 模式时替换的是这一步，不需要回到置换感知的计划推导。

## Alternatives considered

- **在 metadata 里同时保留两套计划推导**（连续 slab / 置换）：会把 !1064 删掉的置换机制重新引入 metadata，且三条
  重切路径都要携带 permutation；暂不采用。
- **只对 V4.1 移除 head-tail**：V4 的 metadata 接缝迁移会因此保留一套置换感知的 CP hook，与本轮收敛目标冲突；
  不采用。
