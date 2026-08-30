# AppWorld External Adapter and Real Gate 0 Smoke Implementation Plan

> **Required execution skill:** use `superpowers:executing-plans` task by task, and use
> `superpowers:test-driven-development` for every production behavior.

**Goal:** Build a pure per-world snapshot adapter for AppWorld's pinned full non-admin catalog
and a separate fresh-world runtime that admits clean, L1 rename, and L2
parameter-restructure variants through all four ToolShift contract layers on one real train
task, while persisting non-formal results only in private external storage.

**Architecture:** Protected AppWorld catalogs and oracle calls exist only inside one process.
The adapter snapshots and validates the full catalog, but never owns a live world. A separate
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
  requirement, protected-derived fingerprint, aggregate runtime result, proxy setting, or
  machine path may be committed. Actual M3A records remain private/encrypted pending written
  maintainer approval.
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

Also reject schema properties that collide with pinned Requester control names:
`_app_name`, `_api_name`, `client`, `raise_on_failure`, `show`, `track`, and
`_system_datetime`. Reject root `patternProperties` so a regex cannot re-admit those names.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_adapter.py -q
```

Expected: collection fails because `toolshift.adapters.appworld` does not exist.

### Step 3: Promote jsonschema and implement construction only

Move `jsonschema>=4.26,<5` from the dev-only list into core dependencies. Implement:

```python
class AppWorldSemanticAdapter(SemanticAdapter):
    def __init__(
        self,
        function_catalog: Sequence[Mapping[str, object]],
    ) -> None: ...

def build_appworld_adapter(
    function_catalog: Sequence[Mapping[str, object]],
) -> AppWorldSemanticAdapter: ...
```

Construction creates one `Draft202012Validator` with `FormatChecker` per snapshotted schema,
calls
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

## Task 2: Implement the pure semantic codec and full-catalog source witnesses

**Files:**

- Modify: `src/toolshift/adapters/appworld.py`
- Modify: `src/toolshift/adapters/__init__.py`
- Modify: `tests/unit/test_appworld_adapter.py`

### Step 1: Write RED semantic-method tests

Cover the four `SemanticAdapter` methods:

- accept any `Mapping` implementation but require its I-JSON snapshot to have exactly the
  `name` and `arguments` keys; generic transforms may preserve call-level metadata, but the
  AppWorld source boundary rejects those extra fields;
- validate arguments without applying JSON Schema defaults;
- reject missing required, extra, wrong-type, unknown-tool, bool-as-int, and non-I-JSON calls;
- parse to exactly one `SemanticAction` and compile to exactly one identical native call;
- wrap exactly one action and one singleton observation group;
- preserve nested JSON and key absence, and deeply freeze outputs;
- canonicalize only internally consistent `ExecutionTrace` values;
- accept a structurally consistent empty `ExecutionTrace`; rejecting an empty captured oracle
  trajectory belongs to Task 4, not to the pure semantic codec;
- reject reordered, extra, missing, hybrid, or tampered trace channels;
- recheck integrity before and after callback-bearing input is consumed;
- return payload-free `ValueError` messages on all rejected inputs.

### Step 2: Write RED source-witness tests

Add only the source witness generator in the adapter module:

```python
def build_minimal_source_calls(
    source_adapter: AppWorldSemanticAdapter,
) -> tuple[Mapping[str, JSONValue], ...]: ...
```

The helper creates exactly one frozen native call per declared source tool, never executes a
requester, and independently confirms each witness with a fresh Draft 2020-12 validator, the
sealed source binding validator, and source parsing. Cross-adapter `SchemaProbe` materialization,
generic case IDs, and trusted transform translation belong to Task 5.

The deterministic minimal-instance generator must cover the schema forms needed by pinned
AppWorld function catalogs: `const`, `enum`, `anyOf`/`oneOf`, nullable types, object required
properties, arrays with bounded `minItems`/`maxItems`, strings with bounded lengths and admitted
formats, integer/number bounds, positive integer `multipleOf` for integer schemas, positive
decimal `multipleOf` for number schemas, and booleans. It is a finite grammar and bounded
neighbor generator, not a general JSON Schema SAT solver: references, recursion, unsupported
keywords/combinators, over-budget explicit values, or any schema for which its finite candidate
pool finds no witness fail closed. Any full catalog that cannot produce every witness fails
closed with the private exclusion code `catalog_unprobeable`.

### Step 3: Run RED

```bash
pytest tests/unit/test_appworld_adapter.py -q
```

### Step 4: Implement the minimal pure methods and source witnesses

All public methods use `try/finally` integrity checks. Snapshot arguments before validator
callbacks. Convert `jsonschema` exceptions to static `ValueError` messages. The source witness
builder is implemented here; the cross-adapter `SchemaProbe` construction is added in Task 5
so the adapter layer does not depend on contract modules.

### Step 5: Run GREEN and mutation regressions

```bash
pytest tests/unit/test_appworld_adapter.py -q
pytest tests/contracts/test_rename_gate0b.py tests/contracts/test_restructure_gate0b.py -q
ruff check .
```

### Step 6: Commit

```bash
git add docs/plans/2026-08-30-appworld-gate0-implementation-plan.md \
  src/toolshift/adapters/appworld.py src/toolshift/adapters/__init__.py \
  tests/unit/test_appworld_adapter.py
git commit -m "feat: generate AppWorld schema witnesses"
```

## Task 3: Add the isolated AppWorld episode executor

**Files:**

- Create: `src/toolshift/benchmarks/appworld_runtime.py`
- Create: `tests/unit/test_appworld_runtime.py`

### Step 1: Write RED fake-world tests

Use a fake requester, fake models record-hash interface, fake tracker, and fake world. Assert:

- importing the module never imports `appworld`;
- `AppWorldEpisodeExecutor(world, adapter)` holds one live world but exposes no reset/load API;
- the executor is one-shot: success, `Exception`, and `BaseException` paths all consume it,
  and re-entry is rejected before the new plan is inspected;
- each surface call is parsed, compiled, invoked, grouped, and wrapped in exact order;
- requester receives `_app_name`, `_api_name`, and copied arguments, with no code generation;
- the exact physical order is `pre-hash -> request -> world.save() -> post-hash`, repeated for
  every call, followed by evaluation;
- the final record contains initial/final state, compact evaluator score, evaluator collateral
  digest, full `ExecutionTrace`, immutable step cases, and indexed `PhysicalCallEffect` values;
- empty episodes, execution failures, malformed requester results, save failures, and evaluator
  failures fail closed;
- public exceptions and `repr` never reveal world, name, argument, observation, URL, or path;
- executor and requester bindings are guarded after callback mutation or exceptions.

Task 3 installs and restores no monkeypatches; exact requester and low-level-method restoration
belongs to Task 4. World construction, close, reset, and the one-live-world-per-process invariant
belong to the injected context/orchestrator. The complete surface plan is snapshotted before its
first callback. Each step has exactly one action and one base call, independently recomputes its
pre- and post-state hashes, and requires every later pre-hash to equal the prior post-hash. A
requester return is saved and post-hashed before validation, its root must be an exact built-in
`dict` or `list`, and the duck-typed `save()` return value is ignored. Evaluation uses
`suppress_errors=False` exactly once after the final post-hash. A well-formed compact evaluator
record may contain `success: false`; Task 4 rejects an unsuccessful reference oracle.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_runtime.py -q
```

### Step 3: Implement the executor

Use these concrete private shapes:

```python
class AppWorldEpisodeExecutor:
    def __init__(
        self,
        world: object,
        adapter: SemanticAdapter,
        *,
        state_hasher: Callable[[object], str] = canonical_state_sha256,
        evaluator_hasher: Callable[[object], str] = evaluator_sha256,
    ) -> None: ...

    def execute_plan(
        self,
        surface_calls: Sequence[Mapping[str, JSONValue]],
    ) -> _EpisodeRecord: ...

@dataclass(frozen=True, slots=True, repr=False)
class _ExecutedStep:
    surface_call: Mapping[str, JSONValue]
    action: SemanticAction
    base_call: Mapping[str, JSONValue]
    base_observation: JSONValue
    surface_observation: JSONValue
    before_state_sha256: str
    after_state_sha256: str

@dataclass(frozen=True, slots=True, repr=False)
class _EpisodeRecord:
    steps: tuple[_ExecutedStep, ...]
    trace: ExecutionTrace
    initial_state_sha256: str
    final_state_sha256: str
    evaluator_score: JSONValue
    evaluator_digest: str
    effects: tuple[PhysicalCallEffect, ...]
    transition_digest: str
```

Reuse `canonical_state_sha256` and `evaluator_sha256` from `appworld_replay`. Define
`effect_digest = SHA256(domain || before_state_sha256 || after_state_sha256)` and bind the
ordered effects, final state, and evaluator digest into `transition_digest`. Keep protected
records private (`repr=False`) and frozen. A compact score contains exactly `pass_count`,
`fail_count`, `total_count`, `num_tests`, and `success`; it stays in memory. The pinned tracker
invariants are `total_count == pass_count + fail_count`, `total_count <= num_tests`, and
`success == (pass_count == num_tests)`. The required physical-call order for every step is
`parse -> compile -> pre-call hash -> request -> world.save() -> post-call hash ->
snapshot/wrap`; evaluation follows the final post-call hash. Effect IDs are generic indexed
values whose digest binds the before/after whole-state hashes. Unit tests and the opt-in real
smoke must prove repeated `save()` is supported.

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
- the replacement has the exact pinned
  `request(_app_name, _api_name, client, raise_on_failure, show, track, **data)` signature;
- positional and keyword control arguments are forwarded but never recorded as API data;
- returned/completed calls are captured after return, including JSON error responses when
  `raise_on_failure=False`;
- captured calls are native `name/arguments` values in original order;
- failed requester calls are not appended;
- ToolShift capture objects never serialize compiled text, observations, or call payloads and
  never include them in exceptions/repr; AppWorld's own logs remain in private `APPWORLD_ROOT`;
- original requester method identity is restored on success, execution failure, requester
  failure, evaluation failure, and `BaseException`;
- an empty trace, AppWorld execution-failure prefix, or unsuccessful oracle is rejected;
- the Requester tracker delta must cover every captured high-level call exactly, so a low-level
  bypass is rejected;
- public `get/post/put/patch/delete` guards reject direct entry, including `track=False`, while
  permitting calls reached from the wrapped high-level `request`; all guards restore in
  `finally`;
- no `save()` occurs inside the requester callback and no redundant explicit save follows;
  pinned `world.execute` performs its own terminal save before evaluation.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_runtime.py -q
```

### Step 3: Implement minimal in-memory capture

Read `world.task.ground_truth.compiled_solution_code` only inside the function, execute the
official solution, and keep the resulting `_CapturedOraclePlan` private and non-serializable.
The wrapper delegates to the original bound method exactly once and restores it in `finally`.
Use the complete reference init profile (`ground_truth_mode=full`, `raise_on_failure=False`,
fixed seed, unique private experiment name). The private AppWorld root is outside ToolShift
Git, mode 0700 under umask 077; AppWorld's unavoidable execute/save/evaluate logs follow the
documented retention policy.

The pinned-runtime probe established that nested `world.save()` from inside the execute
callback is unsupported. Per-call state/effect evidence therefore comes only from later direct
clean/candidate replay, where the executor enforces `request -> save -> post-hash`.

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

Add the cross-adapter probe builder in the benchmark layer:

```python
def build_schema_probes(
    source_adapter: AppWorldSemanticAdapter,
    candidate_adapter: SemanticAdapter,
    canonical_call_to_surface: Callable[
        [Mapping[str, JSONValue]], Mapping[str, JSONValue]
    ],
) -> tuple[SchemaProbe, ...]: ...
```

It consumes `build_minimal_source_calls`, works for the base adapter, `RenameAdapter`, and
`ParameterRestructureAdapter`, and creates exactly one generic-indexed case per declared
source/candidate tool. `expected_actions` come only from parsing the canonical witness with
the source adapter. The candidate surface call comes only from the trusted transform
translator (or a frozen identity translator for clean); the candidate adapter never
manufactures its own expected actions. Case IDs are `schema-0000`, never tool-derived.

Build synthetic clean, rename, and restructure adapters over a fake two-tool catalog. Inject a
fresh-world factory and assert one variant bundle:

- uses separate reference, candidate, reference-reset, and candidate-reset worlds;
- translates oracle native calls to candidate surface calls through the transform;
- produces exactly one schema probe for every declared candidate tool;
- creates one candidate-surface `DenotationCase` per executed step whose expected action,
  base-call group, observation group, and expected surface observation come independently
  from the clean source replay;
- compares each actual candidate observation with that clean expectation before constructing
  contract evidence;
- reuses those exact case objects in `TraceEvidence.candidate_steps`;
- aligns the same generic episode ID in `StateCase` and `TraceCase`;
- uses cached providers that return evidence for only that exact case;
- runs all four layers and immediately admits the exact returned suite object;
- proves the candidate adapter canonicalizes both the canonical reference trace and the
  transformed candidate trace;
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

Use exact-match cached providers rather than closures that accept arbitrary cases:

```python
class _StateEvidenceLookup:
    def __init__(
        self,
        entries: tuple[tuple[StateCase, StateEvidence], ...],
    ) -> None: ...
    def __call__(self, case: StateCase) -> StateEvidence: ...

class _TraceEvidenceLookup:
    def __init__(
        self,
        entries: tuple[tuple[TraceCase, TraceEvidence], ...],
    ) -> None: ...
    def __call__(self, case: TraceCase) -> TraceEvidence: ...
```

`_TraceEvidenceLookup` is constructed with `candidate_steps=denotation_cases`, preserving the
same tuple and exact `DenotationCase` object identities used by the denotation layer. Both
lookups accept only the exact case object identity registered at construction; a merely equal
replacement raises a static `ValueError`. Providers perform no world execution.

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
- the preferred L2 case moves one proper-subset parameter. This intentionally validates the
  paper's selective-grouping operator rather than its whole-wrap control; inability to find
  one is a reported private smoke failure, not an eligibility relaxation;
- ineligible tasks increment only aggregate screened/excluded counters;
- one eligible task is used for clean, L1, and L2; each variant uses fresh worlds;
- no eligible L2 case, empty oracle, any suite failure, cleanup failure, or wrong pin makes the
  smoke fail rather than skip;
- only `train` is passed to the task loader.
- every reference/clean/candidate/reset factory call shares the frozen execution flags:
  `raise_on_failure=False`, `raise_on_extra_parameters=True`, `remote_apis_url=None`,
  `remote_environment_url=None`, `remote_mcp_url=None`, `remote_docker=False`,
  `parse_datetimes=False`,
  `wrap_response=False`, `unwrap_response=False`, and `munchify_response=False`; reference
  alone adds `ground_truth_mode=full`.

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
    workers: int = 1,
    task_loader: Callable[[str], Sequence[str]] = _load_appworld_task_ids,
    world_context_factory: WorldContextFactory = appworld_world_context,
    pin_checker: Callable[[], None] = require_pinned_appworld_runtime,
) -> AppWorldGate0Summary: ...
```

The production defaults lazily import `load_task_ids` and reuse the guarded
`appworld_world_context`. Before loading tasks, require workers exactly one, all proxy variables
unset, a dedicated absolute non-Git private root with mode 0700, the pinned editable checkout
revision, installed package version, Python runtime, data version, and base-DB version without
printing paths. Do not reuse the existing sorted `TaskSet` loader.

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

## Task 7: Add the private aggregate record, private writer, and CLI

**Files:**

- Create: `src/toolshift/benchmarks/_evidence.py`
- Modify: `src/toolshift/benchmarks/appworld_replay.py`
- Modify: `src/toolshift/benchmarks/appworld_gate0.py`
- Create: `scripts/verify_appworld_gate0.py`
- Modify: `tests/unit/test_appworld_replay.py`
- Modify: `tests/unit/test_appworld_gate0.py`
- Modify: `tests/test_repository_security.py`

### Step 1: Write RED private-record boundary tests

Define an exact operator-private `AppWorldGate0Summary.to_dict()` key allowlist. Test
type/range/cross-field integrity for all pins and counters, and assert:

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
repair, JSON `allow_nan=False`, nonzero CLI status for failed smoke, and a repository scan that
rejects any actual M3A runtime record under Git.

### Step 2: Run RED

```bash
pytest tests/unit/test_appworld_gate0.py tests/unit/test_appworld_replay.py \
  tests/test_repository_security.py -q
```

### Step 3: Implement the summary and thin CLI

The script only parses an absolute `--output` under the validated private AppWorld root, calls
the production smoke runner, writes a successful or failed private aggregate record, and
returns `0` exactly for `smoke_passed`. It must not accept task
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
git commit -m "feat: write private AppWorld smoke records"
```

## Task 8: Add an opt-in integration smoke and correct the ml2 runbook

**Files:**

- Create: `tests/smoke/test_appworld_gate0.py`
- Modify: `docs/environment/ml2-appworld.md`
- Modify: `data/manifests/appworld-ml2.yaml`
- Modify: `README.md`
- Modify: `tests/test_benchmark_manifests.py`

### Step 1: Write the opt-in smoke wrapper

Mark it `pytest.mark.appworld`. Skip unless
`TOOLSHIFT_RUN_APPWORLD_GATE0_SMOKE=1` is explicitly set.
Once enabled, a missing runtime, wrong pin, missing data, no L2-eligible train task, contract
failure, repeated-save failure, or cleanup error is a test failure and must never become a
skip. Assert aggregate fields only.

### Step 2: Update documentation and manifest tests

Document the verified install order for the pinned source:

```text
git lfs install -> clone -> checkout --detach PIN -> git lfs pull
-> Python 3.11 environment -> editable install -> appworld install --repo
-> download minimal data 0.2.0 into dedicated private APPWORLD_ROOT
-> verify tests against that root -> run the train-only lifecycle probe
```

Document that uv cache/install staging must remain on node-local storage when the shared NFS
mount does not support uv's atomic persistence, then copy/link the finished standalone Python
or use `--link-mode copy`. Do not include the real shared root, proxy port, token, username,
or host-specific secrets.

Add official fixed-commit source links and state that M3A is non-formal private integration
evidence. State explicitly that actual M3A derived records must not be added to Git without a
maintainer-approved encrypted workflow.

### Step 3: Run local GREEN

```bash
pytest tests/smoke/test_appworld_gate0.py -q
pytest tests/test_benchmark_manifests.py tests/test_repository_security.py -q
ruff check .
```

Expected locally: one explicit opt-in skip whenever the opt-in environment flag is unset; once
set, no runtime, pin, data, contract, or cleanup failure may become a skip.

### Step 4: Commit

```bash
git add tests/smoke/test_appworld_gate0.py docs/environment/ml2-appworld.md \
  data/manifests/appworld-ml2.yaml README.md tests/test_benchmark_manifests.py
git commit -m "docs: add AppWorld Gate 0 smoke protocol"
```

## Task 9: Verify the pinned external environment and run the real M3A smoke privately

**External-only actions; no protected payload output.**

### Step 1: Finish no-task official/static verification

Run in the detached pinned checkout with all proxy variables unset:

```bash
appworld verify tests --root "$APPWORLD_ROOT"
```

Redirect detailed logs to a mode-0600 private external directory. Retain only exit status and
aggregate pass/skip counts in the operator record; delete successful detailed logs.

Do not use `appworld verify tasks` for M3A: at the pinned revision it always opens train and
dev, and its multiprocessing path is not the one-worker protocol. Separately verify pins,
proxy absence, dedicated-root permissions, package version, checkout, data, and base DB before
opening any task.

Then, before selection/full smoke, run a private train-only live compatibility probe for:

- full non-admin catalog outer shape and reserved-control collisions;
- exact Requester/low-level guard replacement/restoration and tracker coverage, including a
  synthetic `track=False` bypass negative control;
- I-JSON requester responses;
- oracle capture relying on `world.execute`'s terminal save, followed by direct replay using
  `request -> save -> post-hash` with repeated saves;
- unique experiment lifecycle and cleanup in the dedicated 0700 root.

Keep its result private and print no protected value.

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

### Step 4: Keep the result outside Git

Manually audit the private allowlisted record and retain or delete it according to the lab's
private policy. Do not stage it, quote it in a public PR, or call it formal Gate 0b evidence.
Publication requires written AppWorld maintainer approval and an approved encrypted workflow;
the official leaderboard packer is not assumed to support this train/custom artifact.

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
`main`. Use a conventional PR title and include exact synthetic/local verification summaries
only. Do not quote private real-smoke results or any protected-derived aggregate.

### Step 5: Merge and post-merge verify

Squash merge after every required check is green. Pull `main`, create a fresh clean venv from
the merge commit, and rerun Ruff, the full suite, and `pip check`. Remove the completed topic
worktree/branch, notify the user, and send the repository-required Feishu completion message.
