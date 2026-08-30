# L2 Selective Parameter Grouping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a deterministic, behavior-preserving L2 transform that moves selected flat tool parameters into one required surface container, with schema-domain property tests and synthetic Gate 0b admission evidence.

**Architecture:** First extract only the low-level runtime seals and guarded delegation shared by all transforms. Then add a conservative closed-object schema rewriter, immutable rule/transform manifest, bidirectional call translator, and semantic adapter. Construction performs full schema/rule attestation; online rollout performs only O(1) schema-root checks plus O(call payload) translation.

**Tech Stack:** Python 3.10/3.11, pytest, Hypothesis, jsonschema Draft 2020-12, existing ToolShift contracts and immutable I-JSON types.

---

## Fixed scope and invariants

- Operator: `parameter_restructure`, level `L2`, ABI constant `PARAMETER_RESTRUCTURE_VERSION_HASH`.
- Direction: canonical flat arguments to surface nested arguments, one required container per changed tool.
- Public rule: exact immutable `ParameterGroupRule(tool_name, container_name, moved_parameters)`.
- Multiple changed tools are allowed, but at most one rule per tool.
- Main variants move a proper subset; moving all parameters is supported only as a whole-wrap control.
- Changed schemas must be closed root objects in the grammar defined by `2026-08-30-parameter-restructure-design.md`.
- No flatten, defaults, return wrapping, pagination, split/merge, errors, or composition in this slice.
- No real AppWorld/BFCL tasks, protected data, held-out data, proxies, secrets, or raw trajectories.

## Task 1: Extract shared transform runtime primitives

**Files:**

- Create: `src/toolshift/transforms/_runtime.py`
- Modify: `src/toolshift/transforms/rename.py`
- Create: `tests/property/test_transform_runtime.py`
- Test: `tests/property/test_transforms.py`

### Step 1: Write the failing package-internal runtime contract tests

Add focused tests for a new `_delegate_with_binding_guard` and shared seal helpers. The delegate test must use real callbacks and assert this exact sequence:

```python
events: list[str] = []
result = _delegate_with_binding_guard(
    lambda: events.append("check"),
    lambda: events.append("operation") or sentinel,
    "source rejected operation",
)
assert result is sentinel
assert events == ["check", "operation", "check"]
```

Also test:

- source `ValueError("PRIVATE")` becomes payload-free `TransformValidationError`;
- an unexpected `RuntimeError` propagates if the final binding check succeeds;
- a final binding failure overrides a callback return or callback exception;
- schema/mapping/operator seals detect root replacement without scanning nested schema nodes.

### Step 2: Run RED

Run:

```bash
pytest tests/property/test_transform_runtime.py -q
```

Expected: collection fails because `toolshift.transforms._runtime` does not exist.

### Step 3: Implement the minimal shared runtime module

Mechanically move these primitives from `rename.py` without changing semantics:

```python
_raw_adapter_variant
_MappingRootSeal
_make_mapping_root_seal
_mapping_root_seal_matches
_SchemaRuntimeSeal
_make_schema_runtime_seal
_schema_runtime_seal_matches
_OperatorRuntimeSeal
_make_operator_runtime_seal
_operator_runtime_seal_matches
_delegate_with_binding_guard
```

The delegate must be equivalent to:

```python
def _delegate_with_binding_guard(check, operation, expected_error):
    check()
    try:
        try:
            return operation()
        except ValueError:
            raise TransformValidationError(expected_error) from None
    finally:
        check()
```

Keep schema construction/deep validation out of this module. Runtime seal matching may read only exact object fields and frozen root storage identities.

### Step 4: Switch rename to the shared primitives

Remove the mechanically duplicated definitions from `rename.py`, import the shared functions/classes, and make `RenameAdapter._delegate` call `_delegate_with_binding_guard`. Do not change rename public API, error strings, manifest, trace rules, or online complexity.

### Step 5: Run GREEN and regression gates

Run:

```bash
pytest tests/property/test_transform_runtime.py tests/property/test_transforms.py tests/contracts/test_rename_gate0b.py -q
pytest -q
ruff check .
ruff check --select C901 src/toolshift/transforms/_runtime.py src/toolshift/transforms/rename.py
```

Expected: all pass; the full-suite count increases only by the new runtime tests.

### Step 6: Commit

```bash
git add src/toolshift/transforms/_runtime.py src/toolshift/transforms/rename.py tests/property/test_transform_runtime.py
git commit -m "refactor: share transform runtime guards"
```

## Task 2: Add immutable rules, schema admission, manifest, and variant construction

**Files:**

- Modify: `pyproject.toml`
- Create: `src/toolshift/transforms/restructure.py`
- Create: `tests/property/test_restructure.py`

### Step 1: Add the test dependency and failing schema/manifest tests

Add `jsonschema>=4.26,<5` to the dev extra. Add tests that import the desired public objects:

```python
from toolshift.transforms.restructure import (
    PARAMETER_RESTRUCTURE_VERSION_HASH,
    ParameterGroupRule,
    ParameterRestructureTransform,
    build_parameter_restructure_transform,
)
```

The first focused tests must assert:

- fixed ABI hash vector;
- `ParameterGroupRule` rejects blank/invalid UTF-8 names, empty/duplicate/non-string moved parameters, and post-construction tamper;
- equivalent rule/field input order yields the same normalized rules, manifest and variant ID;
- seed changes change operator seed/manifest/variant ID;
- exact manifest shape contains mode and path-segment moves;
- selective grouping and whole-wrap generate the exact required/optional closed schemas;
- identity tools preserve their exact `SurfaceToolSpec` object;
- base schema and caller rules remain unmodified.

### Step 2: Install dependencies and run RED

Run:

```bash
python -m pip install -e ".[dev]"
pytest tests/property/test_restructure.py -q
```

Expected: collection fails because `toolshift.transforms.restructure` does not exist.

### Step 3: Implement rule snapshot and conservative schema admission

Implement a frozen, slotted, unhashable rule with a canonical snapshot. Normalize rules by tool name and moved parameters lexicographically only after exhausting exact tuple inputs and rebuilding every rule.

Admission pseudocode:

```python
require schema["type"] == "object"
require nonempty Mapping schema["properties"]
require schema["additionalProperties"] is False
require required is absent or an exact list/tuple of unique declared names
require set(schema) <= ALLOWED_ROOT_KEYS
reject forbidden position-sensitive keys anywhere in the schema graph
require container not in properties
require moved_parameters is a nonempty subset of properties
```

All callback-bearing external values must be snapshotted before fingerprinting the source variant. All errors are static and payload-free.

### Step 4: Implement deterministic schema rewrite and manifest construction

For each rule, build:

```python
inner = {
    "type": "object",
    "properties": moved_property_schemas,
    "additionalProperties": False,
}
if moved_required:
    inner["required"] = moved_required

surface_required = unselected_required + (container_name,)
```

Preserve allowed root annotations, tool name and description. Reuse identity `SurfaceToolSpec` objects. Build one `OperatorManifestEntry` with derived position-0 seed and one `SchemaVariant` via `build_transformed_variant`.

Direct `ParameterRestructureTransform` construction must rebuild and compare the expected operator parameters and variant. Reject a source manifest already marked `toolshift_interface_variant` with one fixed composition message.

### Step 5: Run GREEN

Run:

```bash
pytest tests/property/test_restructure.py -q
pytest tests/property/test_transforms.py tests/property/test_transform_runtime.py -q
ruff check .
```

Expected: schema/manifest tests pass and rename remains green.

### Step 6: Commit

```bash
git add pyproject.toml src/toolshift/transforms/restructure.py tests/property/test_restructure.py
git commit -m "feat: add parameter restructure schemas"
```

## Task 3: Add bidirectional call translation, adapter, trace modes, and Gate 0b

**Files:**

- Modify: `src/toolshift/transforms/restructure.py`
- Modify: `tests/property/test_restructure.py`
- Create: `tests/contracts/test_restructure_gate0b.py`

### Step 1: Write failing translation and adapter tests

Cover these real behaviors before adding production methods:

- canonical→surface always creates the required container, including `{}` for optional-only moved fields;
- surface→canonical expands only actually present keys;
- round trip preserves key absence separately from `None`, `False`, `0`, `""`, `[]`, and `{}`;
- nested object/list values and call-level metadata survive unchanged;
- changed calls reject missing/non-mapping container, moved-at-root, hybrid shape, unknown root/inner keys, unknown tool, malformed arguments and missing required members;
- identity-tool calls round-trip as deep-frozen equal values;
- input objects are unchanged and outputs are deeply immutable;
- parse/compile/wrap/canonicalize delegate exactly once and preserve source tuple/action/groups/observation identity;
- canonical trace, surface trace and identity-only neutral trace are accepted; mixed or hybrid traces are rejected;
- helper, direct Transform and direct Adapter paths reject implicit composition.

### Step 2: Write failing synthetic Gate 0b

Create a two-tool, two-step suite:

```text
search(query required, limit optional, locale optional)
  -> request{query, limit} + locale
summarize(value required)
  -> identity
```

The candidate nested call must parse to the same canonical action/base call as the flat reference. Reuse the two exact `DenotationCase` instances in `TraceEvidence.candidate_steps`; require equal scores, every effect index/call fingerprint/effect digest, state hashes and collateral digests; call `require_dataset_admission` immediately on the same suite object. A valid transform bound to an intentionally wrong source adapter must produce schema and denotation mismatch diagnostics and fail admission.

### Step 3: Run RED

Run:

```bash
pytest tests/property/test_restructure.py tests/contracts/test_restructure_gate0b.py -q
```

Expected: failures for missing translation methods/adapter/build helper and Gate behavior.

### Step 4: Implement the minimal translator and adapter

Add:

```python
ParameterRestructureTransform.surface_call_to_canonical
ParameterRestructureTransform.canonical_call_to_surface
ParameterRestructureTransform._trace_for_source
ParameterRestructureAdapter
apply_parameter_restructure
```

Precompute exact tool/rule/property/required sets during construction. Online translation may walk only the call payload and precomputed small rule data, never the source schema graph.

The adapter must bind the exact raw/public source variant and exact final variant, check bindings before and after every callback, sanitize source `ValueError`, propagate unexpected source exceptions only when bindings remain intact, and let final binding failure override callback outcome.

### Step 5: Run GREEN and contracts

Run:

```bash
pytest tests/property/test_restructure.py tests/contracts/test_restructure_gate0b.py -q
pytest tests/contracts -q
```

Expected: focused tests and all four-layer contracts pass; bad adapter is rejected.

### Step 6: Commit

```bash
git add src/toolshift/transforms/restructure.py tests/property/test_restructure.py tests/contracts/test_restructure_gate0b.py
git commit -m "feat: add parameter restructure adapter"
```

## Task 4: Prove schema-domain bijection and harden boundaries

**Files:**

- Modify: `src/toolshift/transforms/restructure.py`
- Modify: `tests/property/test_restructure.py`

### Step 1: Write failing property, mutation and hot-path regressions

Use `Draft202012Validator` plus Hypothesis to generate valid canonical instances for required/optional, selective/whole-wrap and nested-value cases. Assert both validators accept the corresponding instances and both round trips are canonical-JSON equal.

Add RED regressions for:

- direct constructor/operator/rule/source schema tamper;
- external callback mutation before schema fingerprint;
- payload-bearing mapping/property callbacks at every constructor boundary;
- public/raw source/final variant lies and replacements;
- source rebind during parse, compile, wrap and canonicalize, including transient/finally cases;
- large schema sentinel that forbids `schema_fingerprint`, schema rebuild, canonical serialization, seal builders and any `SurfaceToolSpec` node access after construction;
- online cost independent of total schema size, while remaining O(call payload);
- Python type hints keep rule names/path segments precise.

Verify each new regression fails for the intended missing guard before production changes.

### Step 2: Implement the minimal hardening

Install construction snapshots, canonical fingerprints and shared root seals analogous to rename. Keep full graph validation at construction/Gate; keep online checks root-only. Split helpers before they exceed configured complexity limits.

Do not broaden the supported schema grammar to make a property test pass. Invalid or ambiguous cases must remain rejected.

### Step 3: Run GREEN and all quality gates

Run:

```bash
pytest tests/property/test_restructure.py tests/contracts/test_restructure_gate0b.py -q
pytest tests/property/test_transform_runtime.py tests/property/test_transforms.py -q
pytest tests/contracts -q
pytest tests/test_repository_security.py -q
pytest -q
ruff check .
ruff check --select C901,PLR0911,PLR0912,PLR0915 src/toolshift/transforms
ruff format --check src/toolshift/transforms tests/property tests/contracts/test_restructure_gate0b.py
python - <<'PY'
import ast
from pathlib import Path
files = list(Path("src").rglob("*.py")) + list(Path("tests").rglob("*.py"))
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
print(f"Python 3.10 grammar: {len(files)} files passed")
PY
git diff --check origin/main...HEAD
git status --short --branch
```

Expected: all commands exit 0; only the opt-in AppWorld smoke may be skipped.

### Step 4: Commit

```bash
git add src/toolshift/transforms/restructure.py tests/property/test_restructure.py
git commit -m "test: harden parameter restructure boundaries"
```

## Task 5: Export, audit, review and integrate

**Files:**

- Modify: `src/toolshift/transforms/__init__.py`
- Modify: `README.md` only if status text needs a factual update

### Step 1: Write a failing public export test

Add a test that imports all new public names from `toolshift.transforms` and checks `__all__` has the exact expected additions without exporting `_runtime` internals.

### Step 2: Run RED, export, and run GREEN

Run the focused test, observe the missing exports, then update `transforms/__init__.py` and rerun it.

### Step 3: Perform implementer self-review

Review `origin/main...HEAD` for:

- extra L2/L3 behavior outside scope;
- manifest/hash cycles or order dependence;
- schema-domain gaps not covered by Draft 2020-12 validation;
- per-call schema traversal;
- error payload leaks;
- identity/binding regressions;
- benchmark/proxy/secret/artifact leakage.

Fix every finding with a failing regression first.

### Step 4: Commit exports and any review fixes

```bash
git add src/toolshift/transforms/__init__.py tests/property/test_restructure.py README.md
git commit -m "feat: export parameter restructure transform"
```

Omit `README.md` from the command if unchanged.

### Step 5: Independent review sequence

1. Spec reviewer checks the complete design and this plan against `origin/main...HEAD`.
2. Implementer fixes every spec issue; spec reviewer re-reviews until APPROVED.
3. Code-quality reviewer starts only after spec approval.
4. Implementer fixes every Critical/Important issue and justified Minor issue; quality reviewer re-reviews until APPROVED.
5. Root independently reruns the full fresh gates.

### Step 6: PR and main integration

Push `research/w2-parameter-restructure`, create a focused PR using the repository template, wait for Python 3.10/3.11 CI, squash merge only when all checks are green, fast-forward local `main`, rerun the full suite on the merge commit, and remove only this clean worktree/branch. Keep official benchmark and derived synthetic Gate evidence clearly separated in the PR body.
