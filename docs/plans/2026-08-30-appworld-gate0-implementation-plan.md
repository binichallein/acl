# AppWorld External Adapter and Real Gate 0 Smoke Implementation Plan

> **Required execution skill:** use `superpowers:executing-plans` task by task, and use
> `superpowers:test-driven-development` for every production behavior.

**Goal:** Build a pure task-scoped AppWorld semantic adapter and a separate fresh-world
runtime that admits clean, L1 rename, and L2 parameter-restructure variants through all four
ToolShift contract layers on one real train task, while persisting only aggregate non-formal
smoke evidence.

**Architecture:** Protected AppWorld catalogs and oracle calls exist only inside one process.
The adapter snapshots and validates a task catalog, but never owns a live world. A separate
executor compiles surface calls through any `SemanticAdapter`, invokes the AppWorld requester,
and captures opaque state/effect/evaluator evidence. The runner builds all contract inputs,
evaluates them once, immediately admits the same suite object, and reduces the result to a
strict aggregate allowlist.

**Tech stack:** Python 3.10/3.11, pytest, jsonschema Draft 2020-12, pinned external AppWorld
`a072b7a86e7c1d5b1d7175659d750ebb9b79f10a` / data `0.2.0`, existing ToolShift immutable
types, L1/L2 transforms, and four-layer contracts.

---

## Fixed scope and invariants

- AppWorld remains an opt-in external runtime; no import of `appworld` at package import time.
- Only the `train` split may be opened in M3A. No held-out split is loaded or enumerated.
- No task ID, instruction, schema, tool name, arguments, observation, trace, state, evaluator
  requirement, protected-derived fingerprint, proxy setting, or machine path may be committed.
- Base calls use the exact shape `{"name": name, "arguments": arguments}`.
- The source adapter is one-call/one-action/one-native-call. L1 and L2 candidates must compile
  to the same ordered native calls as the clean reference.
- Every declared candidate tool receives a non-executed schema probe. Every executed candidate
  step contributes the same immutable `DenotationCase` object to denotation and trace evidence.
- State reset means constructing a fresh world, never `load_state` or checkpoint rollback.
- Admission is process-local: call `require_dataset_admission(variant, suite)` immediately with
  the exact object returned by `evaluate_contract_suite`.
- A real smoke passes only if clean, L1, and L2 each produce a non-vacuous admitted suite.
- M3A output always says `mode=smoke`, `gate_evaluable=false`, and
  `formal_gate_passed=false`.

## Task 1: Freeze the public adapter boundary and dependency

**Files:**

- Modify: `pyproject.toml`
- Create: `src/toolshift/adapters/appworld.py`
- Modify: `src/toolshift/adapters/__init__.py`
- Create: `tests/unit/test_appworld_adapter.py`

### Step 1: Write RED catalog-construction tests

Use synthetic function-calling catalogs only. Assert the desired public entry point:

```python
adapter = build_appworld_adapter(catalog)
assert isinstance(adapter, AppWorldSemanticAdapter)
assert adapter.variant.manifest == {
    "kind": "appworld_source_interface",
    "schema_policy": "closed-root-v1",
}
```

Cover:

- exact outer `type/function/name/description/parameters` shape;
- split at the first `__` into two non-empty ASCII identifier segments;
- unique names and deterministic lexical tool order;
- root `type: object`, mapping `properties`, exact unique declared `required` members;
- preserve property schemas, annotations, and `default` exactly;
- add only missing root `additionalProperties: false`;
- reject any conflicting `additionalProperties` value;
- callback-bearing catalog mappings are exhausted into snapshots before validation;
- caller objects are unchanged and resulting variant content is deeply immutable;
- all errors are static and do not contain synthetic payload values.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_adapter.py -q
```

Expected: collection fails because `toolshift.adapters.appworld` does not exist.

### Step 3: Promote jsonschema and implement construction only

Move `jsonschema>=4.26,<5` from the dev-only list into core dependencies. Implement:

```python
class AppWorldSemanticAdapter(SemanticAdapter): ...
def build_appworld_adapter(
    function_catalog: Sequence[Mapping[str, object]],
) -> AppWorldSemanticAdapter: ...
```

Construction creates one `Draft202012Validator` per snapshotted schema, calls
`check_schema`, binds the exact `SchemaVariant`, and installs identity/root seals for the
variant, bindings, validators, and canonical snapshots. Do not implement live execution.

Use generic manifest values only; do not include catalog size, tool names, or fingerprints in
the manifest.

### Step 4: Run GREEN and dependency checks

```bash
python -m pip install -e ".[dev]"
pytest tests/unit/test_appworld_adapter.py -q
ruff check src/toolshift/adapters/appworld.py tests/unit/test_appworld_adapter.py
python -m pip check
```

### Step 5: Commit

```bash
git add pyproject.toml src/toolshift/adapters/appworld.py \
  src/toolshift/adapters/__init__.py tests/unit/test_appworld_adapter.py
git commit -m "feat: add AppWorld catalog adapter"
```

## Task 2: Implement the pure semantic codec and full-catalog probes

**Files:**

- Modify: `src/toolshift/adapters/appworld.py`
- Modify: `tests/unit/test_appworld_adapter.py`

### Step 1: Write RED semantic-method tests

Cover the four `SemanticAdapter` methods:

- accept only an exact mapping with `name` and `arguments`;
- validate arguments without applying JSON Schema defaults;
- reject missing required, extra, wrong-type, unknown-tool, bool-as-int, and non-I-JSON calls;
- parse to exactly one `SemanticAction` and compile to exactly one identical native call;
- wrap exactly one action and one singleton observation group;
- preserve nested JSON and key absence, and deeply freeze outputs;
- canonicalize only internally consistent `ExecutionTrace` values;
- reject reordered, extra, missing, hybrid, or tampered trace channels;
- recheck integrity before and after callback-bearing input is consumed;
- return payload-free `ValueError` messages on all rejected inputs.

### Step 2: Write RED schema-probe tests

Add:

```python
def build_schema_probes(adapter: SemanticAdapter) -> tuple[SchemaProbe, ...]: ...
```

The helper must work for the base adapter, `RenameAdapter`, and
`ParameterRestructureAdapter`. It creates exactly one generic-indexed case per declared tool,
never executes a callback or requester, and produces an instance accepted by the declared
Draft 2020-12 schema.

The deterministic minimal-instance generator must cover the schema forms needed by public
AppWorld function catalogs: `const`, `enum`, `anyOf`/`oneOf`, nullable types, object required
properties, arrays with `minItems`, strings with `minLength`, integer/number lower bounds, and
booleans. It must fail closed on references, recursion, unsatisfiable branches, unknown type
sets, or unsupported combinators. Case IDs are `schema-0000`, not tool-derived.

### Step 3: Run RED

```bash
pytest tests/unit/test_appworld_adapter.py -q
```

### Step 4: Implement the minimal pure methods and probes

All public methods use `try/finally` integrity checks. Snapshot arguments before validator
callbacks. Convert `jsonschema` exceptions to static `ValueError` messages. The probe builder
derives expected actions by calling the candidate adapter once, so transformed names never
need special-case logic.

### Step 5: Run GREEN and mutation regressions

```bash
pytest tests/unit/test_appworld_adapter.py -q
pytest tests/contracts/test_rename_gate0b.py tests/contracts/test_restructure_gate0b.py -q
ruff check .
```

### Step 6: Commit

```bash
git add src/toolshift/adapters/appworld.py tests/unit/test_appworld_adapter.py
git commit -m "feat: add AppWorld semantic codec"
```

## Task 3: Add the isolated AppWorld episode executor

**Files:**

- Create: `src/toolshift/benchmarks/appworld_runtime.py`
- Create: `tests/unit/test_appworld_runtime.py`

### Step 1: Write RED fake-world tests

Use a fake requester, fake models record-hash interface, fake tracker, and fake world. Assert:

- importing the module never imports `appworld`;
- `AppWorldEpisodeExecutor(world, adapter)` holds one live world but exposes no reset/load API;
- each surface call is parsed, compiled, invoked, grouped, and wrapped in exact order;
- requester receives `_app_name`, `_api_name`, and copied arguments, with no code generation;
- state is hashed before and after every physical call;
- `world.save()` occurs before `world.evaluate()`;
- the final record contains initial/final state, compact evaluator score, evaluator collateral
  digest, full `ExecutionTrace`, immutable step cases, and indexed `PhysicalCallEffect` values;
- empty episodes, execution failures, malformed requester results, save failures, and evaluator
  failures fail closed;
- public exceptions and `repr` never reveal world, name, argument, observation, URL, or path;
- executor and requester bindings are restored/guarded after callback mutation or exceptions.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_runtime.py -q
```

### Step 3: Implement the executor

Reuse `canonical_state_sha256` and `evaluator_sha256` from `appworld_replay`. Keep protected
records private (`repr=False`) and frozen. A compact score contains only scalar evaluator
counts and success; it stays in memory. Effect IDs are generic indexed values whose digest
binds the before/after whole-state hashes.

### Step 4: Run GREEN

```bash
pytest tests/unit/test_appworld_runtime.py tests/unit/test_appworld_replay.py -q
ruff check src/toolshift/benchmarks/appworld_runtime.py tests/unit/test_appworld_runtime.py
```

### Step 5: Commit

```bash
git add src/toolshift/benchmarks/appworld_runtime.py tests/unit/test_appworld_runtime.py
git commit -m "feat: add AppWorld episode executor"
```

## Task 4: Capture an oracle plan in memory with exact restoration

**Files:**

- Modify: `src/toolshift/benchmarks/appworld_runtime.py`
- Modify: `tests/unit/test_appworld_runtime.py`

### Step 1: Write RED capture tests

Add a private capture function used only by the runner. With fake `world.execute`, assert:

- it temporarily replaces the exact requester instance's `request` method;
- positional and keyword `_app_name`/`_api_name` calls are captured after successful return;
- captured calls are native `name/arguments` values in original order;
- failed requester calls are not appended;
- compiled oracle text, returned observations, and call payloads are never serialized or
  included in exceptions/repr;
- original requester method identity is restored on success, execution failure, requester
  failure, evaluation failure, and `BaseException`;
- an empty trace, AppWorld execution-failure prefix, or unsuccessful oracle is rejected;
- `world.save()` precedes evaluation.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_runtime.py -q
```

### Step 3: Implement minimal in-memory capture

Read `world.task.ground_truth.compiled_solution_code` only inside the function, execute the
official solution, and keep the resulting `_CapturedOraclePlan` private and non-serializable.
The wrapper delegates to the original bound method exactly once and restores it in `finally`.

### Step 4: Run GREEN and commit

```bash
pytest tests/unit/test_appworld_runtime.py -q
ruff check .
git add src/toolshift/benchmarks/appworld_runtime.py tests/unit/test_appworld_runtime.py
git commit -m "feat: capture AppWorld oracle calls in memory"
```

## Task 5: Materialize paired evidence and immediate hard admission

**Files:**

- Create: `src/toolshift/benchmarks/appworld_gate0.py`
- Create: `tests/unit/test_appworld_gate0.py`

### Step 1: Write RED paired-bundle tests

Build synthetic clean, rename, and restructure adapters over a fake two-tool catalog. Inject a
fresh-world factory and assert one variant bundle:

- uses separate reference, candidate, reference-reset, and candidate-reset worlds;
- translates oracle native calls to candidate surface calls through the transform;
- produces exactly one schema probe for every declared candidate tool;
- creates one `DenotationCase` per executed candidate step;
- reuses those exact case objects in `TraceEvidence.candidate_steps`;
- aligns the same generic episode ID in `StateCase` and `TraceCase`;
- uses cached providers that return evidence for only that exact case;
- runs all four layers and immediately admits the exact returned suite object;
- never exposes or persists the suite, schema fingerprint, calls, or task identity.

Add negative controls for wrong mapping, observation, physical call, effect, final state, reset
state, collateral/evaluator, score, and cleanup. Each must yield a payload-free diagnostic or
failure and must not be admitted.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_gate0.py -q
```

### Step 3: Implement paired execution and evidence providers

Implement a private `_admit_variant_pair(...)` around:

```python
suite = evaluate_contract_suite(...)
admitted = require_dataset_admission(adapter.variant, suite)
assert admitted is suite
```

The function returns only a small process-local success record with counts and diagnostic
codes; it never returns the suite or protected records. Cleanup is owned by the injected
fresh-world context factory.

### Step 4: Run GREEN and all contract tests

```bash
pytest tests/unit/test_appworld_gate0.py tests/contracts -q
ruff check .
```

### Step 5: Commit

```bash
git add src/toolshift/benchmarks/appworld_gate0.py tests/unit/test_appworld_gate0.py
git commit -m "feat: admit paired AppWorld variants"
```

## Task 6: Select deterministic L1/L2 variants and run the smoke protocol

**Files:**

- Modify: `src/toolshift/benchmarks/appworld_gate0.py`
- Modify: `tests/unit/test_appworld_gate0.py`

### Step 1: Write RED selection and orchestration tests

Test an injected ordered train-task loader and world factory. Assert:

- tasks are screened in official loader order without sorting or emitting IDs;
- the catalog comes from `world.task.api_docs.function_calling()`;
- L1 selects an oracle-used tool and a collision-free generic alias;
- L2 tries oracle-used tools and sorted parameter choices through the existing conservative
  restructure constructor, never deleting `default` or changing the source schema;
- the preferred L2 case moves one proper-subset parameter; whole-wrap is not accepted for the
  real smoke;
- ineligible tasks increment only aggregate screened/excluded counters;
- one eligible task is used for clean, L1, and L2; each variant uses fresh worlds;
- no eligible L2 case, empty oracle, any suite failure, cleanup failure, or wrong pin makes the
  smoke fail rather than skip;
- only `train` is passed to the task loader.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_gate0.py -q
```

### Step 3: Implement deterministic selection and run function

Add:

```python
def run_appworld_gate0_smoke(
    *,
    seed: int = 100,
    task_loader: Callable[[str], Sequence[str]],
    world_context_factory: WorldContextFactory,
) -> AppWorldGate0Summary: ...
```

The production defaults lazily import `load_task_ids` and reuse the guarded
`appworld_world_context`. Before loading tasks, require the pinned checkout revision, package
version, Python 3.11.15-compatible runtime, and public data/base-DB version pins without
printing paths.

### Step 4: Run GREEN

```bash
pytest tests/unit/test_appworld_gate0.py -q
pytest tests/unit/test_appworld_adapter.py tests/unit/test_appworld_runtime.py -q
ruff check .
```

### Step 5: Commit

```bash
git add src/toolshift/benchmarks/appworld_gate0.py tests/unit/test_appworld_gate0.py
git commit -m "feat: orchestrate AppWorld Gate 0 smoke"
```

## Task 7: Add the aggregate-only summary, private writer, and CLI

**Files:**

- Create: `src/toolshift/benchmarks/_evidence.py`
- Modify: `src/toolshift/benchmarks/appworld_replay.py`
- Modify: `src/toolshift/benchmarks/appworld_gate0.py`
- Create: `scripts/verify_appworld_gate0.py`
- Modify: `tests/unit/test_appworld_replay.py`
- Modify: `tests/unit/test_appworld_gate0.py`
- Modify: `tests/test_repository_security.py`

### Step 1: Write RED evidence-boundary tests

Define an exact `AppWorldGate0Summary.to_dict()` key allowlist. Test type/range/cross-field
integrity for all pins and counters, and assert:

```python
summary.mode == "smoke"
summary.gate_evaluable is False
summary.formal_gate_passed is False
summary.smoke_passed is (clean == l1 == l2 == 1 and failures == 0)
```

Serialize a canary fake run and assert the JSON excludes every task/tool/schema/call/response/
state/trace/suite canary and excludes keys containing `task_id`, `schema_sha`, `trace_sha`,
`state_sha`, `suite`, or `fingerprint`.

Extract the existing mode-0600 atomic JSON primitive into `_evidence.py` without changing the
Gate 0a output. Test symlink rejection, fsync/replace behavior, temp cleanup, existing-file mode
repair, JSON `allow_nan=False`, and nonzero CLI status for failed smoke.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_gate0.py tests/unit/test_appworld_replay.py \
  tests/test_repository_security.py -q
```

### Step 3: Implement the summary and thin CLI

The script only parses `--output`, calls the production smoke runner, writes a successful or
failed aggregate summary, and returns `0` exactly for `smoke_passed`. It must not accept task
IDs, dev/test split names, raw log output paths, or proxy arguments.

### Step 4: Run GREEN and repository scans

```bash
pytest tests/unit/test_appworld_gate0.py tests/unit/test_appworld_replay.py \
  tests/test_repository_security.py -q
ruff check .
```

### Step 5: Commit

```bash
git add src/toolshift/benchmarks/_evidence.py \
  src/toolshift/benchmarks/appworld_replay.py \
  src/toolshift/benchmarks/appworld_gate0.py \
  scripts/verify_appworld_gate0.py tests/unit/test_appworld_replay.py \
  tests/unit/test_appworld_gate0.py tests/test_repository_security.py
git commit -m "feat: report aggregate AppWorld smoke evidence"
```

## Task 8: Add an opt-in integration smoke and correct the ml2 runbook

**Files:**

- Create: `tests/smoke/test_appworld_gate0.py`
- Modify: `docs/environment/ml2-appworld.md`
- Modify: `data/manifests/appworld-ml2.yaml`
- Modify: `README.md`
- Modify: `tests/test_benchmark_manifests.py`

### Step 1: Write the opt-in smoke wrapper

Mark it `pytest.mark.appworld`. Skip only when the external runtime is absent; once AppWorld
is present, any missing pin, missing data, no L2-eligible train task, contract failure, or
cleanup error is a test failure. Assert aggregate fields only.

### Step 2: Update documentation and manifest tests

Document the verified install order for the pinned source:

```text
git clone/LFS -> Python 3.11 environment -> editable install -> appworld install --repo
-> appworld download data --version 0.2.0 --mode minimal -> verify tests/tasks
```

Document that uv cache/install staging must remain on node-local storage when the shared NFS
mount does not support uv's atomic persistence, then copy/link the finished standalone Python
or use `--link-mode copy`. Do not include the real shared root, proxy port, token, username,
or host-specific secrets.

Add official fixed-commit source links and state that M3A is non-formal integration evidence.
Do not add a real smoke evidence JSON until the real run succeeds.

### Step 3: Run local GREEN

```bash
pytest tests/smoke/test_appworld_gate0.py -q
pytest tests/test_benchmark_manifests.py tests/test_repository_security.py -q
ruff check .
```

Expected locally: one explicit opt-in skip if AppWorld is absent; all static tests pass.

### Step 4: Commit

```bash
git add tests/smoke/test_appworld_gate0.py docs/environment/ml2-appworld.md \
  data/manifests/appworld-ml2.yaml README.md tests/test_benchmark_manifests.py
git commit -m "docs: add AppWorld Gate 0 smoke protocol"
```

## Task 9: Verify the pinned external environment and run the real M3A smoke

**External-only actions; no protected payload output.**

### Step 1: Finish official AppWorld verification

Run in the detached pinned checkout with all proxy variables unset:

```bash
appworld verify tests
appworld verify tasks --num-processes 4
```

Redirect detailed logs to a mode-0600 private external directory. Retain only exit status and
aggregate pass/skip counts in the operator record; delete successful detailed logs.

### Step 2: Freeze dependencies outside Git

Capture an external `uv pip freeze`/lock for the exact ml2 environment. Check for resolver
drift and incompatibilities with `python -m pip check`. The repository may record public
package/version pins, but never external paths.

### Step 3: Run the real smoke

From the pinned AppWorld checkout, with ToolShift source installed and proxies unset:

```bash
python /path/to/toolshift/scripts/verify_appworld_gate0.py --output /private/path/smoke.json
```

Inspect only the aggregate allowlisted JSON. Required result: `smoke_passed=true`, exactly one
clean/L1/L2 admitted suite, zero diagnostics/exceptions/cleanup failures, and
`formal_gate_passed=false`.

### Step 4: Add only safe aggregate evidence

If and only if the real smoke passes, copy a manually audited aggregate summary into
`data/evidence/appworld/gate0b-smoke/`. Re-run repository security tests before staging it.
Never commit a failed report or private log.

### Step 5: Commit

```bash
git add data/evidence/appworld/gate0b-smoke tests/test_benchmark_manifests.py
git commit -m "data: record AppWorld Gate 0 smoke"
```

## Task 10: Full verification, review, PR, and merge

### Step 1: Run local gates on Python 3.10 and 3.11

```bash
ruff check .
pytest -q
python -m pip check
```

Expected: all tests pass, with only the documented opt-in AppWorld skip on machines without
the external runtime.

### Step 2: Audit the diff and repository boundary

```bash
git diff --check main...HEAD
git status --short
git log --oneline main..HEAD
```

Search the entire staged diff for proxy variables, machine roots, task IDs, tool/schema/call
canaries, raw trajectories, credentials, and protected-derived fingerprints.

### Step 3: Request two independent reviews

- correctness review: adapter semantics, requester lifecycle, evidence construction, hard
  admission, negative controls;
- privacy/reproducibility review: protected boundary, aggregate allowlist, fixed upstream pins,
  install/runbook accuracy.

Resolve findings with new RED regressions before production changes.

### Step 4: Push a topic branch and open one PR

Require duplicated GitHub push/PR checks on Python 3.10 and 3.11. Do not push directly to
`main`. Use a conventional PR title and include the exact local/real verification summaries,
with no protected data.

### Step 5: Merge and post-merge verify

Squash merge after every required check is green. Pull `main`, create a fresh clean venv from
the merge commit, and rerun Ruff, the full suite, and `pip check`. Remove the completed topic
worktree/branch, notify the user, and send the repository-required Feishu completion message.
