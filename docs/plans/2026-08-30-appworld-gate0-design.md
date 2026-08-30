# AppWorld External Adapter and Real Gate 0 Smoke Design

**Date:** 2026-08-30
**Milestone:** M3A
**Status:** Approved for implementation

## 1. Goal

Build the first real AppWorld vertical slice for ToolShift without importing protected
benchmark content into Git:

```text
task-scoped AppWorld function catalog
  -> pure AppWorldSemanticAdapter
  -> deterministic ToolShift transform
  -> separate in-process AppWorld executor
  -> fresh-world clean/candidate replay
  -> four-layer contract suite
  -> immediate hard admission
  -> aggregate-only smoke summary
```

M3A proves that the existing L1 rename and L2 parameter-restructure implementations can
drive the pinned AppWorld runtime while preserving native calls, observations, final state,
evaluator outcome, and reset behavior. It is an integration smoke, not a model result and
not a formal Gate 0b claim.

## 2. Fixed upstream and environment

- AppWorld repository: `https://github.com/StonyBrookNLP/appworld`
- AppWorld commit: `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`
- AppWorld package: `0.2.0.dev0`
- AppWorld data and base DB: `0.2.0`
- Python: `3.11.15`
- Git LFS: `3.8.0`
- ToolShift core remains importable and testable on Python 3.10.

The AppWorld dependency stays external and is imported lazily. GitHub/package downloads on
`ml2` may use the operator-provided `ml2-proxy` only in the live shell. Verification and all
localhost traffic run with every proxy variable unset. ModelScope downloads do not use that
proxy.

## 3. Data and license boundary

The following values are protected runtime material and must stay in memory or in a private
external directory:

- task IDs, instructions, supervisors, app descriptions, and task-specific API catalogs;
- raw or normalized AppWorld schemas, tool names, descriptions, and transform manifests;
- ground-truth programs, evaluator requirements, database snapshots, and state projections;
- API arguments, responses, raw traces, per-task diagnostics, and screenshots;
- fingerprints derived from protected schema, task, call, response, or state payloads.

The repository may contain only ToolShift-owned code, synthetic fixtures, official upstream
identifiers, environment/dependency pins, protocol counts, and aggregate status that cannot
be reversed into task or API content. M3A does not serialize a `ContractSuiteResult`; its
attestation is intentionally process-local. If public AppWorld experiment artifacts are ever
needed, use AppWorld's encrypted packing mechanism rather than a custom plaintext export.

Only `train` is used by the M3A smoke. `dev` is reserved for later tuning. The smoke never
opens or enumerates `test_normal` or `test_challenge` content.

## 4. Architecture choice

### Rejected: execution inside `SemanticAdapter`

`SemanticAdapter` is a deterministic parse/compile/wrap/canonicalize interface. Contract
checkers call it repeatedly, so putting a live world or side effects inside it would make the
existing contracts unsound.

### Deferred: MCP or remote API servers

MCP is useful for distributed training, but it adds server lifecycle, optional dependencies,
ports, and cross-process state isolation. It is unnecessary for the first correctness slice.

### Selected: pure adapter plus separate executor

`AppWorldSemanticAdapter` owns only immutable task-scoped tool bindings and JSON Schema
validators. `AppWorldEpisodeExecutor` owns a live world for exactly one episode and executes
compiled native calls through the world's requester. This keeps semantic mapping pure while
making lifecycle and side effects explicit.

## 5. Task-scoped catalog ingestion

The runtime obtains tools from `world.task.api_docs.function_calling()`. Each entry must have
this exact outer shape:

```json
{
  "type": "function",
  "function": {
    "name": "app__api",
    "description": "...",
    "parameters": {"type": "object", "properties": {}, "required": []}
  }
}
```

Construction snapshots all callback-bearing mappings before validation. Tool names split at
the first `__` into non-empty ASCII identifier segments; this follows AppWorld's `app__api`
codec without assuming that later underscores are separators. Names are unique and sorted
before the `SchemaVariant` is built.

The schema policy is explicit:

1. preserve every AppWorld property schema, `required` member, annotation, and `default`;
2. require a root object with a properties mapping and a unique list of declared required
   names, preserving its order exactly;
3. add only `additionalProperties: false` when absent, matching AppWorld construction with
   `raise_on_extra_parameters=True`;
4. reject a conflicting non-false `additionalProperties` value;
5. never apply JSON Schema defaults during parsing;
6. validate calls using Draft 2020-12 semantics and I-JSON-safe snapshots.

Adding the closed-root constraint is recorded in the adapter manifest as a schema policy, not
silently presented as raw upstream schema. `jsonschema` therefore becomes a ToolShift runtime
dependency rather than a test-only dependency.

L2 eligibility is stricter than adapter ingestion. A tool may be restructured only when the
existing transform's conservative grammar accepts the preserved schema. In particular,
`default`, references, ambiguous composition keywords, or unsupported root grammar make that
tool ineligible. M3A never deletes such fields to manufacture eligibility.

## 6. Pure AppWorld semantic adapter

For one native tool call:

```json
{"name": "app__api", "arguments": {"key": "value"}}
```

the adapter performs:

- `surface_to_semantic`: validate name and arguments and return exactly one
  `SemanticAction(name, arguments)`;
- `semantic_to_base_calls`: validate the action and return exactly one call with the same
  native name and arguments;
- `base_observation_to_surface`: require one action, one base call observation, and return a
  deeply frozen copy of that observation;
- `canonicalize_trace`: re-parse every surface call and require its semantic actions and base
  calls to match exactly.

Call-level metadata is preserved by transforms but is not part of AppWorld semantic arguments.
All public errors are static and payload-free. Construction and every public method recheck
variant, validator, and tool-binding root integrity before and after callback-bearing input is
consumed.

## 7. AppWorld episode executor

The executor accepts a live world via duck typing and never imports AppWorld at module import
time. It splits a validated native tool name and calls:

```python
world.requester.request(_app_name=app, _api_name=api, **arguments)
```

It then calls `world.save()` before evaluation or state inspection. It does not generate or
execute Python code, expose `load_state`, or allow an agent to roll the world back. Each reset
is a close followed by construction of a fresh `AppWorld` with the same task and seed.

The executor returns an in-memory immutable step record containing the surface call, semantic
action, native base call, and observation. Its representation and exceptions must not include
task IDs, tool names, arguments, responses, URLs, or paths.

## 8. Oracle capture and paired replay

The reference oracle is executed only to obtain a correct native call plan for contract
verification. During `world.execute`, a temporary wrapper around the exact requester instance
captures successful high-level `app__api` calls, arguments, and returned observations in
memory. The original method is restored in `finally`, including error paths. The compiled
solution and captured payloads are never returned by a public API or serialized.

The paired protocol is:

1. create a fresh reference world and capture a non-empty successful oracle trajectory;
2. build the complete task-scoped base adapter from the in-memory catalog;
3. deterministically select an oracle-used tool for L1 rename;
4. search only train tasks, in official order, for an oracle-used tool satisfying unchanged
   L2 eligibility; build one selective restructure when possible;
5. replay clean and transformed surface calls in separate fresh worlds;
6. collect per-step observations, indexed native calls, before/after whole-state hashes,
   final whole-state hash, evaluator stats, and collateral digest;
7. construct additional fresh worlds to prove reset-to-initial behavior;
8. materialize `SchemaProbe`, `DenotationCase`, `StateEvidence`, and `TraceEvidence` objects;
9. run `evaluate_contract_suite` and immediately pass the same object to
   `require_dataset_admission` in the same process.

The full task-scoped catalog remains the declared interface. Schema probes cover every
declared tool using deterministic, non-executed minimal JSON instances generated from the
preserved schemas. The runner must not shrink the interface to oracle-used tools.

L1 and current L2 are expected to compile to the identical ordered native call sequence. M3A
does not cover split/merge or other L3 transforms whose physical sequence may differ.

## 9. Aggregate-only smoke summary

The public-safe summary uses an exact allowlist:

- schema version and `mode: smoke`;
- AppWorld commit/package/data/Python pins;
- ToolShift adapter and runner ABI hashes;
- split name, seed, workers, screened task count, and admitted task count;
- requested, eligible, attempted, admitted, and excluded counts by transform family;
- aggregate diagnostic-code counts and execution/cleanup exception counts;
- `gate_evaluable: false` and `formal_gate_passed: false`;
- `smoke_passed`, true only when clean, L1, and L2 each produce a non-vacuous admitted suite.

It excludes task-set, schema, trace, state, call, and suite fingerprints even when one-way.
Failure logs remain private. A successful summary is written atomically with mode `0600`; a
failed run is never promoted as evidence.

## 10. Testing and acceptance

Local tests use synthetic catalogs and fake worlds only. They cover:

- lazy import and Python 3.10 compatibility;
- catalog shape, closed-root policy, preserved defaults, duplicate names, and tamper cases;
- valid and invalid parse/compile/wrap/trace behavior;
- deterministic schema-probe generation without execution;
- exact requester call and save ordering;
- requester restoration after oracle success and failure;
- fresh-world clean/candidate/reset lifecycle and cleanup;
- all four contract layers, immediate admission, and negative controls;
- aggregate summary allowlist, atomic write, exit status, and repository privacy scans.

The real ml2 smoke must use the fixed environment above, one worker, no proxy variables,
fresh worlds, at least one native oracle call, one admitted L1 rename suite, and one admitted L2
restructure suite. Any inability to find an unchanged-schema L2-eligible train case is a smoke
failure, not a silent skip. The output must remain marked non-formal.

## 11. Deferred formal Gate 0b

After M3A passes, M3B will preregister task scope, variant coverage, exclusions, seeds, failure
policy, and the public-safe evidence schema before a full train/dev run. Unlike Gate 0a's
environment consistency threshold, formal Gate 0b is a hard per-task/per-variant admission
gate: any contract failure excludes that pair and is counted. No formal claim is made by this
design or its smoke output.

## 12. Primary upstream references

The interface and lifecycle claims above are tied to the pinned source, not a moving branch:

- [pinned AppWorld commit](https://github.com/StonyBrookNLP/appworld/commit/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a)
- [package and Python metadata](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/pyproject.toml)
- [function-calling catalog conversion](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/api_docs.py)
- [task-scoped API-doc collection](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/collections/api_docs.py)
- [environment lifecycle and defaults](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/environment.py)
- [requester behavior](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/requester.py)
- [evaluator behavior](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/evaluator.py)
- [installation, split restrictions, and protected-data license](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/README.md)
