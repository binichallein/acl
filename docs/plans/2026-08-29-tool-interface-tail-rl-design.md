# 行为等价工具接口下的尾部可靠性 Agentic RL：研究设计

**日期：** 2026-08-29
**目标会议：** COLM 2027（若 2026 年 12 月前主结果完整，可评估 ACL 2027 ARR）
**暂定题目：** *Same Task, Different Tools: Task-Conditioned Worst-Variant Reinforcement Learning for Reliable Tool Agents*

## 1. 核心问题

给定同一个潜在任务 `x`，不同工具接口实现 `e` 可以在工具名称、参数结构、返回格式、分页方式或调用粒度上不同，但保持任务语义和最终状态不变。现有 Agentic RL 通常优化平均回报，可能在常见接口上表现很好，却在少数行为等价接口上系统性失败。

本研究希望在保持原始接口性能和调用成本的同时，提高同一任务接口等价类中的最差变体成功率，并检验这种收益能否迁移到未见变换组合和外部工具分布。

## 2. 核心假设

- H1：相同 AppWorld 任务在行为等价接口下存在显著且可重复的成功率波动。
- H2：任务条件化的 paired worst-variant 目标优于等预算的 clean GRPO、uniform domain randomization、global CVaR 和 canonical wrapper。
- H3：收益能迁移到 held-out L2/L3 组合及 BFCL-Shift，且 clean success 基本不下降。

## 3. 技术设计

执行链如下：

```text
任务与变体工具文档
  -> Agent 生成 surface tool call
  -> SemanticAdapter 映射为 canonical semantic action
  -> 编译为一个或多个 AppWorld 原生调用
  -> AppWorld 更新数据库
  -> 返回值包装为变体 observation
  -> AppWorld evaluator 检查终态、任务成功和副作用
```

接口映射写作 `kappa_e: u -> A_e*`，允许一个语义动作对应多个表面工具调用。纯重命名和参数置换可使用动作等变约束；tool split/merge 并非双射，应使用接口等价类、action-sequence mapping 或 MDP homomorphism 描述。

## 4. 变换层级

- L1：工具重命名、参数重排、文档和序列化格式变化。
- L2：参数嵌套/展开、默认值变化、返回 schema、分页。
- L3：tool split/merge、可恢复异常、重试协议和动作粒度变化。
- L4：未见 L2/L3 组合，仅用于最终评测。

L1 只作为控制实验；论文主结论必须覆盖 canonical wrapper 难以解决的 L2/L3。

## 5. 方法目标

对于任务 `x` 的接口集合 `E_x`，优化：

```text
E_x[(1-lambda) Mean_e R(x,e) + lambda SoftMin_e R(x,e)]
  - beta * semantic_consistency
  - gamma * tool_cost
```

每个 batch 的基本采样单位是“任务等价类”：相同任务、初始数据库和环境种子下，同时 rollout clean 和多个接口变体。任务内估计接口风险，再跨任务平均，以避免把任务本身的难度误当作接口脆弱性。

## 6. 主指标与统计单位

主指标是 task-conditioned worst-variant success：

```text
TC-Worst = mean_x min_e p_hat(x,e)
```

同时报告 clean success、mean shifted success、paired success flip、transformation-induced regret、pass^1/pass^2/pass^4、invalid call、error recovery、collateral damage、token/turn/tool-call/wall-clock cost。

统计抽样单位是 base task，而不是单条 rollout。正式实验至少 3 个训练 seed；使用按 task 配对的 hierarchical bootstrap，并为 clean performance 做预注册的非劣检验。

## 7. 强基线

- Base/SFT
- Clean-GRPO
- Uniform DR-GRPO
- Canonical wrapper + GRPO
- Global-CVaR-GRPO
- Transform-group DRO
- Paired mean-GRPO
- Paired task-conditioned worst-variant RL（本文）

所有方法匹配 rollout、token、tool-call、训练步数和模型规模预算。

## 8. Go/No-Go 条件

- 若两个模型的接口 worst-variant drop 均低于约 5 个百分点，H1 不成立。
- 若 canonical wrapper 恢复超过 90% 的损失，停止简单 L1/L2 路线并升级到 L3。
- 若等预算 DR-GRPO 或 global CVaR 与本文方法持平，算法贡献不成立。
- 若收益只出现在 seen/L1 变换，未见组合泛化主张不成立。
- 若 clean success 下降超过预注册非劣界，或成本/副作用显著恶化，只能报告 Pareto 而不能声称全面改进。
- 若任一变体无法通过状态等价 contract tests，不得用于训练或评测。

## 9. 范围边界

本研究不声称解决任意新工具语义、真实 API 的所有版本漂移或在线 continual adaptation。它只研究经过可执行 contract 验证的行为等价接口变化。AppWorld 用于主训练和机制验证；官方 BFCL v4 与独立 BFCL-Shift 用于外部泛化评测，二者分表报告。
