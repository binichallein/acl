# Tool Interface Tail-Reliability RL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一套在行为等价工具接口下训练和评测任务条件最坏变体可靠性的 Agentic RL 系统，并在 AppWorld 与 BFCL v4 上形成可投稿实验闭环。

**Architecture:** 在 AppWorld 外部增加 schema transformation 与 semantic adapter 层，不修改受保护 bundle。Agent 只看到变体接口，adapter 将表面调用映射到原生 AppWorld 调用并把 observation 反向包装；训练器对相同任务、快照和种子的多个接口变体做 paired rollout，在任务内部计算 soft-min/worst-variant 权重。

**Tech Stack:** Python 3.10/3.11、pytest、Hypothesis、AppWorld、BFCL v4、verl 或 OpenRLHF、vLLM/SGLang、PyTorch、Hydra/OmegaConf、Polars/Pandas、SciPy/statsmodels。

---

## Task 0：建立项目与版本清单（W1）

**Files:**
- Create: `toolshift-rl/pyproject.toml`
- Create: `toolshift-rl/configs/system/ml2.yaml`
- Create: `toolshift-rl/data/manifests/dependencies.yaml`
- Create: `toolshift-rl/tests/smoke/test_environment.py`

**Steps:**

1. 在 ml2 的共享盘建立项目、模型和 trajectory 目录，避免把大文件写入根分区。
2. 固定 Python、CUDA、PyTorch、AppWorld bundle、BFCL evaluator、模型和训练框架 commit。
3. 写 smoke test，检查 AppWorld 世界可以创建、复位、执行一个原生 API 并关闭。
4. 运行 `pytest tests/smoke/test_environment.py -v`；预期全部通过。
5. 记录 `nvidia-smi`、CPU、内存、共享盘路径和容器摘要。

**Gate 0a：** 同一 oracle trajectory 重复执行时，最终状态 hash 和 evaluator 分数一致率至少 99%。

## Task 1：定义 canonical 语义层（W1）

**Files:**
- Create: `toolshift-rl/src/toolshift/types.py`
- Create: `toolshift-rl/src/toolshift/adapters/semantic.py`
- Create: `toolshift-rl/tests/unit/test_semantic_types.py`

**Steps:**

1. 先为 `SemanticAction`、`SurfaceToolSpec`、`SchemaVariant`、`ExecutionTrace` 写失败测试。
2. 运行 `pytest tests/unit/test_semantic_types.py -v`；预期因对象不存在而失败。
3. 实现最小数据结构以及 `surface_to_semantic`、`semantic_to_base_calls`、`base_observation_to_surface`、`canonicalize_trace` 接口。
4. 再次运行测试；预期通过。
5. 增加 manifest 序列化与 SHA256 稳定性测试。

## Task 2：建立 contract test 框架（W1-W2）

**Files:**
- Create: `toolshift-rl/src/toolshift/contracts/schema.py`
- Create: `toolshift-rl/src/toolshift/contracts/denotation.py`
- Create: `toolshift-rl/src/toolshift/contracts/state.py`
- Create: `toolshift-rl/src/toolshift/contracts/trace.py`
- Create: `toolshift-rl/tests/contracts/test_equivalence.py`

**Steps:**

1. 写 intentionally bad adapter 测试，确认错误映射会被拒绝。
2. 写 reset 后数据库 hash 恢复、oracle trace 同分、无额外副作用测试。
3. 实现 schema、单步 denotation、轨迹和任务终态四层检查器。
4. 运行 `pytest tests/contracts -v`；预期坏 adapter 失败、正确 identity adapter 通过。
5. 将“所有 contract 通过”设为变体进入数据集的硬条件。

## Task 3：逐级实现接口变换（W2）

**Files:**
- Create: `toolshift-rl/src/toolshift/transforms/base.py`
- Create: `toolshift-rl/src/toolshift/transforms/rename.py`
- Create: `toolshift-rl/src/toolshift/transforms/restructure.py`
- Create: `toolshift-rl/src/toolshift/transforms/return_schema.py`
- Create: `toolshift-rl/src/toolshift/transforms/split_merge.py`
- Create: `toolshift-rl/src/toolshift/transforms/errors.py`
- Create: `toolshift-rl/src/toolshift/transforms/compose.py`
- Create: `toolshift-rl/tests/property/test_transforms.py`

**Steps:**

1. 依次为 L1、L2、L3 写 schema 与状态等价失败测试。
2. 每次只实现一种最小变换，运行对应 test，再运行全部 contracts。
3. 使用 property-based tests 覆盖参数边界、列表、嵌套和默认值。
4. split/merge adapter 必须在 episode reset 时清除中间状态。
5. 每个变体生成含 operator、seed、映射、组合顺序、版本 hash 的 manifest。

**Gate 0b：** 未通过 oracle trace、终态等价或副作用测试的变体一律丢弃并记录原因。

## Task 4：完成现象 Pilot，暂不训练新方法（W3）

**Files:**
- Create: `toolshift-rl/scripts/run_pilot.py`
- Create: `toolshift-rl/configs/eval/pilot.yaml`
- Create: `toolshift-rl/src/toolshift/evaluation/metrics.py`
- Create: `toolshift-rl/src/toolshift/evaluation/statistics.py`
- Create: `toolshift-rl/tests/unit/test_metrics.py`

**Steps:**

1. 从 AppWorld train/dev 分层抽取 100-150 个任务，不查看 test_normal/test_challenge 单题结果。
2. 每题运行 clean 加 5 个变体，每变体 4 次 paired rollout；先用 4B，再用 7B/8B 复核。
3. 比较 Base、已有 clean checkpoint、uniform randomization、只看 surface schema/docs 的 generic normalizer，以及拥有 clean schema/exact inverse 的 privileged oracle canonicalizer。
4. 输出 clean-to-worst drop、TC-Worst、paired flip、regret、长度/工具数分层曲线。
5. 使用按 task 配对 bootstrap 给出 95% CI，并基于 pilot 方差做正式评测 power analysis。

**Gate 1：** 至少两个 L2/L3 家族出现可重复的 10-15pp clean-to-worst gap，且等信息 generic normalizer 后仍保留有意义差距；privileged oracle 用作双射 L1/L2 的可消除性上界。若 generic normalizer 恢复超过 90%，将该 family 降为 H1/机制控制并升级 L3。

## Task 5：打通强基线（W4）

**Files:**
- Create: `toolshift-rl/src/toolshift/envs/appworld_env.py`
- Create: `toolshift-rl/src/toolshift/rewards/components.py`
- Create: `toolshift-rl/src/toolshift/algorithms/domain_randomized_grpo.py`
- Create: `toolshift-rl/src/toolshift/algorithms/global_cvar_grpo.py`
- Create: `toolshift-rl/configs/train/baselines/*.yaml`
- Create: `toolshift-rl/tests/integration/test_baseline_training.py`

**Steps:**

1. 建立成功、可验证 milestone、副作用、surface/physical call cost 奖励。
2. 用小数据写 one-step training smoke test，确认 loss、KL、梯度和 checkpoint 正常。
3. 实现 Clean-GRPO、DR-GRPO、Global-CVaR-GRPO、generic normalizer+GRPO；另实现 privileged oracle canonicalizer 作为双射 L1/L2 ceiling，二者分开报告。
4. 用相同 rollout、token、tool-call 和训练步数预算运行短实验。
5. 审计 reward hacking；如 proxy reward 与 AppWorld gold evaluator 分离，先修评分器。

## Task 6：实现 paired sampler（W5）

**Files:**
- Create: `toolshift-rl/src/toolshift/sampling/snapshot.py`
- Create: `toolshift-rl/src/toolshift/sampling/paired.py`
- Create: `toolshift-rl/tests/unit/test_paired_sampler.py`

**Steps:**

1. 测试 pair 成员共享 task、初始状态和环境 seed，但 mutable state 相互独立。
2. 测试变体顺序不改变 batch，断点恢复能复现下一 batch。
3. 实现以任务等价类为采样单位的 `clean + K-1 variants` sampler。
4. 每条轨迹记录 task、variant manifest、初末状态 hash、surface/semantic/base calls、所有 seed 和成本。
5. 运行 unit 与小型 e2e tests；预期完全可复现。

## Task 7：实现任务条件 soft-min/worst-variant 目标（W5-W6）

**Files:**
- Create: `toolshift-rl/src/toolshift/algorithms/paired_worst_variant.py`
- Create: `toolshift-rl/tests/unit/test_paired_worst_variant.py`
- Create: `toolshift-rl/configs/train/ours.yaml`

**Steps:**

1. 写手工小例子，验证低回报变体获得更大权重。
2. 验证变体顺序置换不改变 loss；`lambda=0` 时严格退化为 mean-GRPO。
3. 验证所有变体回报相同时退化为普通 GRPO，constant reward 不出现 NaN。
4. 实现任务内 soft-min 权重，再跨任务聚合 advantage。
5. 第一版不加入复杂表示一致性；先验证 paired task-conditioned objective。
6. 若主目标有效，再加入 exact 变换的动作等变约束和 L2/L3 的 canonical trace consistency。

**Gate 2：** 相同预算下，相对最强 DR/CVaR 基线，held-out TC-Worst 达到预注册最小效应，clean 下降不超过 2pp；否则不得扩模。

## Task 8：冻结方法并完成 AppWorld 主实验（W7-W8）

**Files:**
- Create: `toolshift-rl/configs/eval/appworld_main.yaml`
- Create: `toolshift-rl/scripts/train.py`
- Create: `toolshift-rl/scripts/evaluate.py`
- Create: `toolshift-rl/scripts/analyze.py`

**Steps:**

1. 冻结算法、变换、训练划分、主指标和超参数。
2. 先完成 4B 全基线与 8B 单 seed，再运行关键方法 3 seeds。
3. 正式评测 clean、seen operator、held-out operator、held-out composition、higher severity。
4. 每个 task/variant 正式评测建议 8 次 rollout；测试集不用于调参。
5. 输出每个 seed、绝对提升、95% CI、clean 非劣检验和 Holm 校正结果。

**Gate 3：** 主指标的 task-level bootstrap CI 排除 0，clean、成本和副作用满足预注册条件。

## Task 9：接入 BFCL v4 与 BFCL-Shift（W9-W10）

**Files:**
- Create: `toolshift-rl/src/toolshift/envs/bfcl_eval.py`
- Create: `toolshift-rl/configs/eval/bfcl_official.yaml`
- Create: `toolshift-rl/configs/eval/bfcl_shift.yaml`
- Create: `toolshift-rl/tests/integration/test_bfcl_scorer.py`

**Steps:**

1. 固定 BFCL evaluator commit，复现一个官方基线分数。
2. 官方 BFCL 不参与训练和超参数选择。
3. 为允许修改的公开 schema 构建独立 BFCL-Shift，并运行相应 contract tests。
4. 分表报告官方 Multi-Turn/Hallucination/Format 等结果与 BFCL-Shift 结果。
5. 检查 AppWorld 增益是否迁移，且官方 BFCL 没有显著退化。

**Gate 4：** BFCL-Shift 对强基线有正增益；否则删除跨环境泛化主张，降级为 AppWorld 内部可靠性论文。

## Task 10：完成消融、统计和失败分析（W10-W11）

**Files:**
- Create: `toolshift-rl/configs/eval/ablations.yaml`
- Create: `toolshift-rl/reports/claim_evidence_matrix.md`
- Create: `toolshift-rl/reports/failure_taxonomy.md`

**Steps:**

1. 消融 pairing、soft-min、task conditioning、clean anchor、semantic consistency、K、温度和 L1/L2/L3 curriculum。
2. 分析不同 horizon、工具数、变换强度和组合深度。
3. 对正式数据运行 hierarchical bootstrap、mixed-effects logistic regression 和 clean 非劣检验。
4. 将每个论文 claim 映射到主表、消融或失败案例；没有证据的 claim 删除。
5. 固定数字，只允许修 bug 后整组重跑。

## Task 11：论文与复现包（W12）

**Files:**
- Create: `toolshift-rl/paper/main.tex`
- Create: `toolshift-rl/paper/sections/*.tex`
- Create: `toolshift-rl/artifact/README.md`
- Create: `toolshift-rl/artifact/model_card.md`
- Create: `toolshift-rl/artifact/data_card.md`

**Steps:**

1. 先完成主表、图、方法伪代码和失败案例，再写 Results、Methods、Introduction 和 Abstract。
2. 从干净环境复现一个代表性 baseline 与一个本文模型。
3. 发布配置、manifest、SHA256、seed、训练预算、失败任务和排除变体原因。
4. 明确区分官方 BFCL 与 BFCL-Shift，遵守 AppWorld bundle 派生物许可。
5. 不在最后一周新增方法；只修复复现和内部审稿发现的问题。

## 12 周里程碑

| 周次 | 产出 |
|---|---|
| W1 | AppWorld baseline、语义层、环境复现 |
| W2 | L1-L3 adapters 与 contract tests |
| W3 | 现象 pilot、wrapper 强基线、Gate 1 |
| W4 | Clean/DR/CVaR 基线与奖励审计 |
| W5 | paired sampler、方法 v1 |
| W6 | 方法 v2、初步消融、Gate 2 |
| W7 | 冻结方案、4B/8B 主训练 |
| W8 | 3 seeds、AppWorld 主结果、Gate 3 |
| W9 | BFCL 官方管线与 BFCL-Shift contracts |
| W10 | 跨环境评测与关键消融、Gate 4 |
| W11 | 统计、成本、副作用与失败分析 |
| W12 | 论文初稿和可复现 artifact |

## 执行原则

- 先验证 H1，再实现新算法；Gate 1 不过就停止。
- 每个变换、sampler 和 loss 都先写失败测试，再写最小实现。
- 4×A100 优先保证 4B/8B、3 seeds 和强基线，不增加 32B、GUI 或全参数训练。
- 为失败重跑预留约 40% 算力；CPU 环境并发和共享盘吞吐需先压测。
- 2026-12-15 前若主结果和关键消融完整，可评估 ACL 2027；否则按 COLM 2027 节奏完成。
