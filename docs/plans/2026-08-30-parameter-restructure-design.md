# L2 Selective Parameter Grouping Design

**日期：** 2026-08-30
**状态：** Task 3 设计增补
**范围：** 只实现 canonical flat arguments 到 surface nested arguments 的一层、确定性、双射变换

## 1. 决策摘要

下一项 L2 operator 采用“选择性参数分组”：对一个或多个 closed-object 工具，把选定的顶层参数移入一个新鲜、始终必填的 object container，工具名、canonical semantic action、base call 和 observation 保持不变。

```text
canonical: {"query": "alpha", "limit": 10, "sort": "new"}
surface:   {"request": {"query": "alpha", "limit": 10}, "sort": "new"}
```

论文主变体使用非空真子集；同一实现允许移动全部参数，作为 whole-wrap control。首版不实现 canonical nested 到 surface flat、不实现任意 JSON path、不处理 defaults、返回 schema、分页、split/merge 或 composition。

## 2. 为什么选择这个范围

比较过三种方案：

1. 整体 arguments envelope：最容易证明，但主要测试固定 wrapper 遵循，结构变化较弱。
2. 选择性一层分组：同时保留顶层和嵌套字段，能产生 misplaced、hybrid、missing-container 等可解释错误，又仍可严格证明双射。采用此方案。
3. 通用 path relocation/flatten：需要处理 optional empty object、路径前缀、数组、引用、collision 和 required 传播，实际上会提前引入 compose 的大部分复杂度。

任何 manifest-aware、能访问 clean schema 和 exact inverse 的 privileged canonicalizer 都可以完全消除确定性双射 L1/L2。因此本 operator 用于 H1、机制分析和 curriculum；privileged canonicalizer 是 ceiling，不是期望被算法击败的 baseline。H2 的公平比较对象是等信息 generic normalizer，主要 wrapper residual 仍需后续 stateful/non-bijective L3。

## 3. 形式定义

对一个 canonical 工具的 closed-object arguments schema，令顶层属性集合为 `K`、required 集合为 `R`。一条规则选择非空集合 `S ⊆ K` 和不属于 `K` 的 fresh container `g`。

对任意 schema-valid canonical arguments `a`：

```text
F(a) = a 中 K-S 的已出现成员
       ∪ {g: a 中 S 的已出现成员}
```

对任意 schema-valid surface arguments `b`：

```text
G(b) = (b 去掉 g) ∪ b[g]
```

`g` 在 surface schema 中始终 required。即使 `S` 全为 optional，canonical 的“全部缺失”也唯一映射为 `{g: {}}`，因此 missing 与 explicit empty 不会形成两个 surface 表示。转换按 key presence 而非 truthiness 判断，必须保持 `null`、`false`、`0`、空字符串、空数组和空对象。

生成 schema：

```text
outer.properties = properties[K-S] + {g: inner}
outer.required = (R-S) + {g}
inner.properties = properties[S]
inner.required = R∩S
outer.additionalProperties = false
inner.additionalProperties = false
```

当 `R∩S` 为空时省略 inner `required`；outer `required` 始终包含 `g`。

## 4. 支持的 JSON Schema 子集

首版只 admission 以下 root schema：

- root 必须是 mapping，且 `type` 精确为 `"object"`；
- `properties` 必须是非空 mapping；property 名必须是非空有效 UTF-8；
- `required` 可缺省；存在时必须是无重复字符串序列，且是 `properties` 的子集；
- `additionalProperties` 必须精确为 `false`；
- root 除上述关键字外，只允许 `$schema`、`title`、`description` 和 `$comment`；
- 递归拒绝 `$ref`、`$dynamicRef`、`$recursiveRef`、`$id`、`$anchor`、`$dynamicAnchor`、`default` 和 `examples` 等位置或实例形状敏感关键字；
- property subschema 作为完整子树移动，不拆 leaf；布尔 schema 允许。

因此首版明确拒绝 root combinator/conditional、`patternProperties`、`propertyNames`、`dependent*`、`unevaluated*` 和 `minProperties/maxProperties`。这一保守 grammar 与 property tests 共同证明全定义域双射；现有 schema contract 仍负责 variant 完整性、probe 覆盖和 adapter mapping，不能替代此 admission grammar。

该限制与 JSON Schema Draft 2020-12 的 object 语义一致：`properties` 只约束匹配成员，`required` 决定成员存在性，而 `additionalProperties` 处理未声明成员。参考：

- https://json-schema.org/draft/2020-12/json-schema-core
- https://json-schema.org/draft/2020-12/json-schema-validation

## 5. 公共 API 与 manifest

新增公开 API：

```python
PARAMETER_RESTRUCTURE_VERSION_HASH
ParameterGroupRule
ParameterRestructureTransform
ParameterRestructureAdapter
build_parameter_restructure_transform(...)
apply_parameter_restructure(...)
```

`ParameterGroupRule` 是 frozen、slotted、可重建校验的值对象：

```python
ParameterGroupRule(
    tool_name="search",
    container_name="request",
    moved_parameters=("query", "limit"),
)
```

builder 接收 exact tuple rules；每个工具最多一条规则。规则按 `tool_name`、移动参数按名称规范化排序，使调用者输入顺序不改变 variant ID。

operator 固定为：

```text
operator = "parameter_restructure"
level = "L2"
operator_id = "restructure-000"
```

manifest parameters 的稳定形状为：

```json
{
  "mode": "nest_top_level",
  "rules": [
    {
      "tool_name": "search",
      "container_name": "request",
      "moves": [
        {"source_path": ["limit"], "surface_path": ["request", "limit"]},
        {"source_path": ["query"], "surface_path": ["request", "query"]}
      ]
    }
  ]
}
```

路径使用 JSON string segment 序列，不使用点串或 JSON Pointer，避免转义歧义。base schema fingerprint 已绑定完整 source tools；operator seed 继续由 master seed、base fingerprint、operator id/name 和 position 0 派生。对象 property presentation order 不纳入本 operator 的研究主张。

## 6. Call、trace 与 adapter 语义

### 6.1 Surface call 到 canonical

- tool name 必须属于 source inventory；identity tool 的 call 只做冻结快照并原样委托；
- changed tool 的 `arguments` 必须是 mapping；
- container 必须存在且为 mapping；
- container child 必须全属于 `S`，root 不得同时出现 `S` 成员；
- root 只允许未移动的 canonical properties 加 container；
- required root/inner 成员必须存在；值类型由 source adapter 和 contract evidence继续判定；
- 移除 container、展开其成员，保留 call-level 其他字段，返回深冻结新快照。

### 6.2 Canonical call 到 surface

- changed tool 的 canonical arguments 不得含 fresh container；
- 只接受 declared canonical properties并检查 required；
- 始终生成 container，包括 `{}`；
- 移动实际存在的 `S` 成员，未移动成员留在 root；
- `G(F(a))` 和 `F(G(b))` 必须 canonical-JSON 相等。

### 6.3 Trace

工具名不变，不能复用 rename 的名称模式检测。对整条 trace：

- changed tool 有 container 且无 moved-at-root：surface mode；
- changed tool 无 container：canonical mode；
- 同一 call 同时有 container 和 moved-at-root：hybrid，拒绝；
- identity-only trace 为 neutral；
- 同一 trace 同时出现 canonical 与 surface changed calls：mixed，拒绝；
- canonical trace 原样委托；surface trace 只重写 surface-call channel，semantic actions 和 base calls 保持原对象/内容。

### 6.4 Adapter

- `surface_to_semantic`：执行 `G` 后委托 source 一次，返回 source tuple/action 原对象；
- `semantic_to_base_calls`：action 原对象直接委托；
- `base_observation_to_surface`：执行 `G` 后委托；actions、observation groups 和返回 observation 保持对象身份；
- `canonicalize_trace`：按上述整条 trace mode 处理后委托。

source `ValueError` 转成静态、payload-free `TransformValidationError`；其他 source exception 在 binding 完整时保持现有传播策略。所有 source callback 前后都复验 binding，`finally` 检测到的 rebind 必须覆盖 callback 返回或异常。

## 7. Runtime、安全与 composition 边界

第二个 transform 出现后，先新增私有 `transforms/_runtime.py`，从 rename 机械抽取：

- raw adapter variant 读取；
- mapping/schema/operator root seals 及 make/match；
- 通用 before/finally binding delegate helper。

rename 改为导入这些低层原语，行为和公开 API 不变。不抽象通用 Transform 基类，不把 operator-specific schema/call/trace 逻辑放入 runtime。

构造期深验证 source schema、rule、operator 和 expected variant；在线只做 O(1) schema/operator/root binding seal，加 `O(call payload)` 翻译，不允许每次调用遍历全部 schema。

在 `compose.py` 落地前，helper、direct Transform ctor、direct Adapter ctor 三层都拒绝 source manifest 已为 `toolshift_interface_variant`，因此 rename→restructure、restructure→rename 和同类堆叠都失败并返回固定 composition 错误。

## 8. 验证策略

### 8.1 Schema-domain property tests

开发依赖加入 `jsonschema>=4.26,<5`，使用 `Draft202012Validator` 对受限 schema 做实际验证。Hypothesis 生成 bounded I-JSON canonical arguments，要求：

```text
base_validator.is_valid(a)
surface_validator.is_valid(F(a))
G(F(a)) == a
F(G(F(a))) == F(a)
```

并覆盖 optional-only group、whole-wrap、选择性分组、missing/null/false/zero/empty containers、nested list/object、Unicode，以及所有拒绝边界。

### 8.2 Synthetic Gate 0b

使用两个工具、两步 episode：

- `search` 把 required `query` 和 optional `limit` 移入 `request`，保留 `locale` 顶层；
- `summarize` 保持 identity。

两个工具各有 `SchemaProbe` 和 `DenotationCase`；candidate trace 复用同一 denotation 对象。reference flat call 与 candidate nested call 必须产生完全相同的 canonical actions、base calls、observations、scores、physical effects、final/reset state 和 collateral digest，并在 suite 后立即以同一对象调用 `require_dataset_admission`。合法构造但错误 source adapter 必须产生具体 schema/denotation diagnostics 并拒绝 admission。

### 8.3 回归与质量门

- runtime 抽取前后 rename focused/full 结果不变；
- deterministic manifest/seed/version hash 固定向量；
- callback snapshot、TOCTOU、tamper、payload-free errors 和 binding recheck；
- large-schema 在线路径 sentinel；
- Python 3.10/3.11、Ruff、format、C901、full contracts/security；
- 不接触 AppWorld protected bundle、held-out data 或真实 benchmark trajectory。

## 9. 风险与后续

- `additionalProperties:false` 会降低真实 schema eligibility；先在公开 train/dev schema 上统计 admission/exclusion 原因，再决定是否扩展，不为覆盖率放松证明条件。
- deterministic bijective L2 可被 privileged oracle 消除；不能据此声称 canonicalization 无效。
- 同一 interface mapping 不同 seed 会有不同 variant ID；接口级 dedup 留给 dataset/composition 层。
- 下一独立 ABI 才考虑 canonical nested → surface flat；return schema、pagination、split/merge、errors 和 compose 继续按总计划分片实现。
