# AppWorld External Adapter and Real Gate 0 Smoke Design

**Date:** 2026-08-30
**Milestone:** M3A
**Status:** Approved for implementation

## 1. Goal

Build the first real AppWorld vertical slice for ToolShift without importing protected
benchmark content into Git:

```text
per-world snapshot of AppWorld's pinned full non-admin catalog
  -> pure AppWorldSemanticAdapter
  -> deterministic ToolShift transform
  -> separate in-process AppWorld executor
  -> fresh-world clean/candidate replay
  -> four-layer contract suite
  -> immediate hard admission
  -> private aggregate-only smoke record
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

The reproducible installation order is fixed: install Git LFS; clone AppWorld; checkout the
commit above in detached mode; pull LFS objects; create the Python 3.11 environment; install
the checkout editable; run `appworld install --repo`; download minimal data version `0.2.0`
into a dedicated private `APPWORLD_ROOT`; then run the official test verifier against that
root and the M3A train-only lifecycle probe. The official task verifier is not used because it
opens both train and dev at this revision. The checkout occurs before editable installation and
data setup. Runner startup independently checks the installed distribution version, editable
checkout revision, data version, and base DB version.

## 3. Data and license boundary

The following values are protected runtime material and must stay in memory or in a private
external directory:

- task IDs, instructions, supervisors, app descriptions, and the full non-admin API catalog;
- raw or normalized AppWorld schemas, tool names, descriptions, and transform manifests;
- ground-truth programs, evaluator requirements, database snapshots, and state projections;
- API arguments, responses, raw traces, per-task diagnostics, and screenshots;
- fingerprints derived from protected schema, task, call, response, or state payloads.

The repository may contain only ToolShift-owned code, synthetic fixtures, official upstream
identifiers, environment/dependency pins, and the protocol definition. AppWorld's fixed
license provides no exception for irreversible or aggregate derivatives: actual screened,
eligible, admitted, diagnostic, state, score, or smoke results remain private or encrypted
unless the maintainers provide written permission for a publication workflow. File mode
`0600` protects a private file locally but does not make it safe to push.

M3A never serializes a `ContractSuiteResult`; its attestation is intentionally process-local.
The official `appworld pack` leaderboard path is limited to its supported test splits and
allowlist and is not assumed to cover a custom train-split contract artifact. Any custom
public artifact requires maintainer approval and an approved encryption/distribution path.
Existing historical Gate 0a evidence in this repository needs a separate license audit and is
not treated as precedent for M3A.

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

`AppWorldSemanticAdapter` owns only immutable per-world snapshots of the pinned full
non-admin tool bindings and JSON Schema
validators. `AppWorldEpisodeExecutor` owns a live world for exactly one episode and executes
compiled native calls through the world's requester. This keeps semantic mapping pure while
making lifecycle and side effects explicit.

## 5. Full non-admin catalog ingestion

The runtime obtains tools from `world.task.api_docs.function_calling()`. At the pinned commit,
`Task` attaches the same full non-admin app catalog to every world; it is not reduced to the
current task's oracle-used tools. Each entry must have this exact outer shape:

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
6. validate calls using Draft 2020-12 plus `FormatChecker` and I-JSON-safe snapshots;
7. reject any property colliding with Requester control parameters such as `_app_name`,
   `_api_name`, `client`, `raise_on_failure`, `show`, `track`, or `_system_datetime`;
8. reject root `patternProperties`, which could otherwise re-admit a Requester control name
   despite the explicit-property check.

Adding the closed-root constraint is recorded in the adapter manifest as a schema policy, not
silently presented as raw upstream schema. `jsonschema` therefore becomes a ToolShift runtime
dependency rather than a test-only dependency. JSON Schema validation remains a conservative
surface check; execution through AppWorld/Pydantic is the final runtime admission boundary.
The preserved `required` order is an integrity/canonicalization rule, not JSON Schema meaning.

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

Generic transforms preserve call-level metadata mechanically, but the AppWorld source-call
boundary accepts only the exact native outer shape with `name` and `arguments`; any extra
outer field is rejected and never enters AppWorld semantic arguments or native base calls.
All public errors are static and payload-free. Construction and every public method recheck
variant, validator, and tool-binding root integrity before and after callback-bearing input is
consumed.

## 7. AppWorld episode executor

The executor accepts a live world via duck typing and never imports AppWorld at module import
time. It splits a validated native tool name and calls:

```python
world.requester.request(_app_name=app, _api_name=api, **arguments)
```

The initial fresh state may be inspected before any call. After every direct requester call,
the exact order is `request -> world.save() -> post-call state inspection`; evaluation occurs
only after the last save. The pinned-runtime smoke must therefore exercise repeated `save()`
calls. This ordering applies to direct clean/candidate replay, not inside the callback of
`world.execute`: a pinned-runtime compatibility probe showed nested `save()` there is not a
supported lifecycle. Oracle capture relies on the terminal save performed by `world.execute`
and supplies no state/effect evidence; it does not issue a redundant explicit save. The
executor does not generate or execute Python code, expose
`load_state`, or allow an agent to roll the world back. Each reset is a close followed by
construction of a fresh `AppWorld` with the same complete initialization profile.

All roles share `raise_on_failure=False`, `raise_on_extra_parameters=True`, the fixed seed,
`remote_apis_url=None`, `remote_environment_url=None`, `remote_mcp_url=None`,
`remote_docker=False`,
`parse_datetimes=False`, `wrap_response=False`, `unwrap_response=False`, and
`munchify_response=False`; reference capture additionally uses `ground_truth_mode=full`.
Every phase has a unique private experiment name, only one world is live in a process at a
time, evaluation occurs while it is open, and close/close-all failures fail the smoke.

The executor returns an in-memory immutable step record containing the surface call, semantic
action, native base call, and observation. Its representation and exceptions must not include
task IDs, tool names, arguments, responses, URLs, or paths.

Whole-state evidence uses the existing pinned-internal projection
`canonical_state_sha256(world.models)`: clear AppWorld's record-hash cache, then hash sorted
app/model/id/record-hash rows, including empty tables. Failure of that internal interface
rejects the task. AppWorld's output `model_hashes.json` is not substituted for this projection,
and the resulting protected-derived digest never leaves process/private storage.

## 8. Oracle capture and paired replay

The reference oracle is executed only to obtain a correct native call plan for contract
verification. During `world.execute`, a temporary wrapper around the exact requester instance
uses the pinned `Requester.request(_app_name, _api_name, client, raise_on_failure, show, track,
**data)` signature. It separates all control parameters and captures only `**data` as native
arguments after a request returns. Because `raise_on_failure=False` can return an error JSON,
capture means completed/returned, not necessarily HTTP-successful. The original method is
restored in `finally`, including `BaseException` paths.

The runner snapshots `request_tracker` before and after oracle execution and requires complete
coverage between captured high-level calls and newly tracked low-level requests. Because a
direct low-level call with `track=False` is invisible to that tracker, capture also installs
guards on the public `get`, `post`, `put`, `patch`, and `delete` methods. A guard permits entry
only while the wrapped high-level `request` is delegating; any other entry is a bypass and
rejects the task. Every method is restored in `finally`. Capture requires `track=True`, a
non-empty trace, an execute result without the `Execution failed` prefix, and
`world.evaluate().success`.

ToolShift does not independently serialize its capture objects. AppWorld itself necessarily
writes code/output, API-call logs, state, model hashes, and evaluation reports during
`execute`, `save`, and `evaluate`. Therefore `APPWORLD_ROOT` is a dedicated absolute directory
outside the ToolShift Git worktree, created under umask `077` with directory mode `0700` and a
documented private retention/deletion policy. Every world closes through `finally`.

The paired protocol is:

1. create a fresh reference world and capture a non-empty successful oracle trajectory;
2. build the complete full-catalog base adapter from the in-memory per-world snapshot;
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

The full non-admin catalog remains the declared interface. Schema probes cover every
declared tool using deterministic, non-executed minimal JSON instances generated from the
preserved source schemas. Expected actions come only from the source AppWorld adapter over a
canonical witness. The transform translates that witness into a candidate surface call; the
candidate adapter is evaluated against the source-derived expectation and never supplies its
own oracle. The runner must not shrink the interface to oracle-used tools.

Expected actions, base-call groups, and observation groups in each `DenotationCase` come from
the clean source replay. Candidate execution is independent; its actual observation must equal
the clean expected observation before evidence construction. The same candidate adapter must
canonicalize both the canonical reference trace and transformed candidate trace.

If any declared schema has no deterministic valid witness in the supported generator grammar,
the task receives the private exclusion code `catalog_unprobeable` and is rejected. M3A first
runs a private grammar census; it does not weaken full-catalog coverage or silently skip tools.

Opaque evidence algorithms are fixed rather than filled with constants:

- `base_call_fingerprint` uses the existing contract helper;
- `effect_digest = SHA256(domain || before_state_sha256 || after_state_sha256)`;
- collateral digest binds the complete ordered transition sequence, final-state digest, and
  evaluator digest;
- these protected-derived digests remain only in the live evidence bundle/private root.

L1 and current L2 are expected to compile to the identical ordered native call sequence. M3A
does not cover split/merge or other L3 transforms whose physical sequence may differ.

## 9. Private aggregate-only smoke record

The operator-only private record uses an exact allowlist:

- schema version and `mode: smoke`;
- AppWorld commit/package/data/Python pins;
- ToolShift adapter and runner ABI hashes;
- split name, seed, workers, screened task count, and admitted task count;
- requested, eligible, attempted, admitted, and excluded counts by transform family;
- aggregate diagnostic-code counts and execution/cleanup exception counts;
- `gate_evaluable: false` and `formal_gate_passed: false`;
- `smoke_passed`, true only when clean, L1, and L2 each produce a non-vacuous admitted suite.

It excludes task-set, schema, trace, state, call, and suite fingerprints even when one-way.
Failure logs remain private. A record is written atomically with mode `0600` in the external
private root and is never committed in plaintext. A failed run is never promoted as evidence.

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
- private aggregate-record allowlist, atomic write, exit status, and repository privacy scans.

The real ml2 smoke must use the fixed environment above, one worker, no proxy variables,
fresh worlds, at least one native oracle call, one admitted L1 rename suite, and one admitted L2
restructure suite. Any inability to find an unchanged-schema L2-eligible train case is a smoke
failure, not a silent skip. The private output must remain marked non-formal. Before this full
smoke, a no-task static probe verifies pins/root/proxy configuration; then a train-only live
probe verifies the full catalog shape, Requester and low-level guard restoration, exact I-JSON
responses, direct-replay repeated saves, and private-root lifecycle without printing protected
values. The official task verifier is not part of M3A because it always opens both train and
dev at this pinned revision.

## 11. Deferred formal Gate 0b

After M3A passes, M3B will preregister task scope, variant coverage, exclusions, seeds, failure
policy, and a maintainer-approved encrypted evidence workflow before a full train/dev run.
Unlike Gate 0a's
environment consistency threshold, formal Gate 0b is a hard per-task/per-variant admission
gate: any contract failure excludes that pair and is counted. No formal claim is made by this
design or its smoke output. No plaintext publication claim is made from private M3A results.

## 12. Primary upstream references

The interface and lifecycle claims above are tied to the pinned source, not a moving branch:

- [pinned AppWorld commit](https://github.com/StonyBrookNLP/appworld/commit/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a)
- [package and Python metadata](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/pyproject.toml)
- [runtime/data version constants](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/common/constants.py)
- [Task holder for the full non-admin catalog](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/task.py)
- [application inventory](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/apps/__init__.py)
- [function-calling catalog conversion](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/api_docs.py)
- [per-Task holder of the full non-admin API-doc collection](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/collections/api_docs.py)
- [API wrapper and Requester control parameters](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/collections/apis.py)
- [environment lifecycle and defaults](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/environment.py)
- [requester behavior](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/requester.py)
- [evaluator behavior](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/evaluator.py)
- [official verification](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/verify.py)
- [download behavior](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/download.py)
- [CLI and packing boundary](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/cli.py)
- [leaderboard allowlist](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/leaderboard.py)
- [record-hash state interface](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/src/appworld/collections/models.py)
- [installation, split restrictions, and protected-data license](https://github.com/StonyBrookNLP/appworld/blob/a072b7a86e7c1d5b1d7175659d750ebb9b79f10a/README.md)
