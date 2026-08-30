from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from toolshift.benchmarks.appworld_replay import ReplayMode, ReplaySummary

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APPWORLD_RUNBOOK = REPOSITORY_ROOT / "docs/environment/ml2-appworld.md"
README = REPOSITORY_ROOT / "README.md"
GATE_0A_EVIDENCE_RELATIVE_PATH = Path("data/evidence/appworld/gate0a/formal-2026-08-30.json")
GATE_0A_EVIDENCE = REPOSITORY_ROOT / GATE_0A_EVIDENCE_RELATIVE_PATH
FULL_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")

APPWORLD_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
BFCL_COMPARABLE_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
BFCL_OBSERVED_HEAD = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
GATE_0A_EVIDENCE_BYTES = 741
GATE_0A_EVIDENCE_SHA256 = "57508fde225276408238e745eb6cc932fabf6b06b3445f69961cebd865aa39a4"
GATE_0A_TASK_SET_SHA256 = "11a5fc9bb4ae1329a45fd0f7012b2b2432de539d01930f329740f012e4c30602"
GATE_0A_PRIVACY_ALLOWLIST = frozenset(
    {
        "schema_version",
        "mode",
        "pinned_appworld_commit",
        "splits",
        "split_counts",
        "task_set_sha256",
        "task_count",
        "repetitions",
        "episode_count",
        "seed",
        "workers",
        "initial_state_match_rate",
        "final_state_match_rate",
        "evaluator_match_rate",
        "trace_match_rate",
        "task_consistency_rate",
        "oracle_success_rate",
        "execution_failure_count",
        "exception_count",
        "failure_counts",
        "gate_evaluable",
        "gate_passed",
        "smoke_passed",
    }
)
GATE_0A_PAYLOAD: dict[str, Any] = {
    "schema_version": 1,
    "mode": "gate",
    "pinned_appworld_commit": APPWORLD_COMMIT,
    "splits": ["dev", "train"],
    "split_counts": {"dev": 57, "train": 90},
    "task_set_sha256": GATE_0A_TASK_SET_SHA256,
    "task_count": 147,
    "repetitions": 3,
    "episode_count": 441,
    "seed": 100,
    "workers": 1,
    "initial_state_match_rate": 1.0,
    "final_state_match_rate": 1.0,
    "evaluator_match_rate": 1.0,
    "trace_match_rate": 1.0,
    "task_consistency_rate": 1.0,
    "oracle_success_rate": 1.0,
    "execution_failure_count": 0,
    "exception_count": 0,
    "failure_counts": {},
    "gate_evaluable": True,
    "gate_passed": True,
    "smoke_passed": False,
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_unique_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    assert isinstance(payload, dict)
    return payload


def _reconstruct_replay_summary(payload: dict[str, Any]) -> ReplaySummary:
    assert payload["schema_version"] == 1
    return ReplaySummary(
        mode=ReplayMode(payload["mode"]),
        pinned_appworld_commit=payload["pinned_appworld_commit"],
        splits=tuple(payload["splits"]),
        split_counts=tuple(payload["split_counts"].items()),
        task_set_sha256=payload["task_set_sha256"],
        task_count=payload["task_count"],
        repetitions=payload["repetitions"],
        episode_count=payload["episode_count"],
        seed=payload["seed"],
        workers=payload["workers"],
        initial_state_match_rate=payload["initial_state_match_rate"],
        final_state_match_rate=payload["final_state_match_rate"],
        evaluator_match_rate=payload["evaluator_match_rate"],
        trace_match_rate=payload["trace_match_rate"],
        task_consistency_rate=payload["task_consistency_rate"],
        oracle_success_rate=payload["oracle_success_rate"],
        execution_failure_count=payload["execution_failure_count"],
        exception_count=payload["exception_count"],
        failure_counts=tuple(payload["failure_counts"].items()),
        gate_evaluable=payload["gate_evaluable"],
        gate_passed=payload["gate_passed"],
        smoke_passed=payload["smoke_passed"],
    )


def _load_manifest(filename: str) -> dict[str, Any]:
    path = REPOSITORY_ROOT / "data/manifests" / filename
    assert path.is_file(), f"{path.relative_to(REPOSITORY_ROOT)} must exist"

    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), f"{filename} must contain a YAML mapping"
    return manifest


def _unresolved_paths(node: Any, prefix: tuple[str, ...] = ()) -> set[str]:
    if isinstance(node, dict):
        return {
            path
            for key, value in node.items()
            for path in _unresolved_paths(value, (*prefix, str(key)))
        }
    if isinstance(node, list):
        return {
            path
            for index, value in enumerate(node)
            for path in _unresolved_paths(value, (*prefix, str(index)))
        }
    if node == "UNRESOLVED":
        return {".".join(prefix)}
    return set()


def _read_appworld_runbook() -> str:
    assert APPWORLD_RUNBOOK.is_file(), "AppWorld ml2 runbook must exist"
    return APPWORLD_RUNBOOK.read_text(encoding="utf-8")


def _bash_blocks(markdown: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)\n```", markdown, flags=re.DOTALL)


def _normalize_shell(block: str) -> str:
    return " ".join(block.replace("\\\n", " ").split())


def _bash_function(markdown: str, function_name: str) -> str:
    marker = f"{function_name}() {{\n"
    start = markdown.index(marker)
    end = markdown.index("\n}\n", start) + len("\n}\n")
    return markdown[start:end]


def test_appworld_manifest_pins_checkout_package_and_runtime() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")

    assert manifest["schema_version"] == 1
    benchmark = manifest["benchmark"]
    assert benchmark == {
        "name": "appworld",
        "repository": {
            "url": "https://github.com/StonyBrookNLP/appworld.git",
            "commit": APPWORLD_COMMIT,
            "checkout": "detached",
        },
        "package": {"version": "0.2.0.dev0", "python_requires": ">=3.11"},
        "runtime": {"python_version": "3.11.15"},
    }
    assert FULL_GIT_OBJECT.fullmatch(benchmark["repository"]["commit"])


def test_appworld_manifest_records_verified_uv_and_editable_resolution_scope() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")

    assert manifest["installation"] == {
        "uv_version": "0.11.8",
        "dependency_resolution": "upstream_editable",
        "dependency_overrides": {"python-dotenv": "1.2.2"},
        "project_training_dependency_lock": {
            "status": "not_completed",
            "checksum": None,
        },
    }


def test_appworld_runbook_preflights_shared_root_before_creating_directories() -> None:
    blocks = _bash_blocks(_read_appworld_runbook())
    assert blocks, "runbook must contain executable bash blocks"
    first_block = blocks[0]
    normalized = _normalize_shell(first_block)

    assert first_block.startswith("set -euo pipefail\n")
    assert ': "${TOOLSHIFT_SHARED_ROOT:?TOOLSHIFT_SHARED_ROOT must be set and non-empty}"' in (
        first_block
    )
    assert 'case "${TOOLSHIFT_SHARED_ROOT}" in' in first_block
    assert 'echo "TOOLSHIFT_SHARED_ROOT must be an absolute path" >&2' in first_block
    assert 'TOOLSHIFT_SHARED_ROOT="$(realpath -m -- "${TOOLSHIFT_SHARED_ROOT}")"' in (normalized)
    assert 'if [ "${TOOLSHIFT_SHARED_ROOT}" = "/" ]; then' in first_block
    assert 'echo "TOOLSHIFT_SHARED_ROOT must not resolve to filesystem root" >&2' in (first_block)
    assert "readonly TOOLSHIFT_SHARED_ROOT" in first_block

    normalization = normalized.index("realpath -m --")
    root_rejection = normalized.index('if [ "${TOOLSHIFT_SHARED_ROOT}" = "/" ]')
    mkdir = normalized.index("mkdir -p")
    assert normalization < root_rejection < mkdir
    assert re.search(r"(?:echo|printf)[^\n]*\$\{TOOLSHIFT_SHARED_ROOT\}", first_block) is None


def test_appworld_runbook_configures_isolated_lfs_before_detached_checkout() -> None:
    first_block = _normalize_shell(_bash_blocks(_read_appworld_runbook())[0])
    commands = [
        "GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout",
        'git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" config --local lfs.storage',
        'git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" lfs install --local',
        (
            'GIT_LFS_SKIP_SMUDGE=1 git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" '
            "checkout --detach"
        ),
        'git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" lfs pull',
    ]

    positions = [first_block.index(command) for command in commands]
    assert positions == sorted(positions)


def test_appworld_runbook_uses_only_repository_local_lfs_configuration() -> None:
    runbook = _read_appworld_runbook()
    checkout_block = _normalize_shell(_bash_blocks(runbook)[0])

    assert "Git LFS executable" in runbook
    assert "available on `PATH`" in runbook
    assert re.search(r"(?m)^\s*git lfs install(?:\s|$)", runbook) is None
    assert 'git -C "${TOOLSHIFT_SHARED_ROOT}/repos/appworld" lfs install --local' in checkout_block


def test_appworld_runbook_versions_are_derived_from_manifest() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")
    runbook = _read_appworld_runbook()
    blocks = [_normalize_shell(block) for block in _bash_blocks(runbook)]

    checkout_block = blocks[0]
    install_block = blocks[1]
    commit = manifest["benchmark"]["repository"]["commit"]
    python_version = manifest["benchmark"]["runtime"]["python_version"]
    data_version = manifest["data"]["version"]
    data_mode = manifest["data"]["mode"]
    lfs_version = manifest["git_lfs"]["version"]
    uv_version = manifest["installation"]["uv_version"]

    assert f"checkout --detach {commit}" in checkout_block
    assert f"uv venv --python {python_version}" in install_block
    assert f"download data --version {data_version} --mode {data_mode}" in install_block
    assert f"Git LFS is version `{lfs_version}`" in runbook
    assert 'uv_runtime_version="$(uv --version)"' in install_block
    assert f'"uv {uv_version}" | "uv {uv_version} "*)' in install_block
    assert "Unexpected uv version" in install_block
    assert "upstream editable dependency resolution" in runbook
    assert "project training dependency lock is not complete" in runbook
    assert "no lock checksum is recorded" in runbook


def test_appworld_manifest_distinguishes_inventory_digest_from_content_integrity() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")
    data = manifest["data"]

    assert data["version"] == "0.2.0"
    assert data["mode"] == "minimal"
    assert data["version_files"] == [
        {"path": "data/version.txt", "version": "0.2.0"},
        {"path": "data/base_dbs/version.txt", "version": "0.2.0"},
    ]

    integrity = data["integrity"]
    assert integrity["official_manifest_available"] is False
    assert integrity["official_checksum_available"] is False
    assert integrity["official_content_checksum"] == "UNRESOLVED"

    inventory = integrity["path_size_inventory"]
    assert inventory["record_count"] == 14885
    assert inventory["sha256"] == (
        "9a7f2826556f8aa18ecc34ebd96ab88ff9d40257ad220ee463e7904c1b8031e5"
    )
    assert SHA256.fullmatch(inventory["sha256"])
    assert inventory["algorithm"] == {
        "input_line": "<relative-path><TAB><decimal-byte-size><LF>",
        "relative_to": "data/",
        "sort_locale": "C",
        "sort_order": "bytewise",
        "exclude": [
            "*/__pycache__/*",
            "*.pyc",
            "*.pyo",
            "*-journal",
            "*-wal",
            "*-shm",
        ],
    }
    assert inventory["content_integrity"] is False


def test_appworld_manifest_records_verified_encrypted_lfs_bundles() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")
    lfs = manifest["git_lfs"]

    assert lfs["version"] == "3.8.0"
    assert lfs["pointer_worktree_object_hashes_matched"] is True
    assert lfs["encrypted_bundles"] == [
        {
            "path": "generate/.source/data.bundle",
            "bytes": 1508806,
            "sha256": "42c2a3c929c60cc891c94e8924d8ce39c2ffd779bd51ca819b26944483de6106",
        },
        {
            "path": "generate/.source/tasks.bundle",
            "bytes": 163959,
            "sha256": "471f225cf4d85db1ba61d24e2fef881afb68f4f9ca0b8bf76cde712745d60433",
        },
        {
            "path": "src/appworld/.source/apps.bundle",
            "bytes": 193950,
            "sha256": "88d21fc526c1655bb3eee4adfca78ccac793921e4506f28f734ecdb19af77a62",
        },
        {
            "path": "src/appworld/.source/tests.bundle",
            "bytes": 204426,
            "sha256": "04aa898cb015c53468c355d5ded662757c5234835a4de5cf4f7d7947bef159ec",
        },
    ]
    assert all(bundle["bytes"] > 0 for bundle in lfs["encrypted_bundles"])
    assert all(SHA256.fullmatch(bundle["sha256"]) for bundle in lfs["encrypted_bundles"])


def test_appworld_manifest_records_official_verification_results() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")
    verification = manifest["official_verification"]

    assert verification["observed_at"] == "2026-08-29"
    assert verification["tests"] == {
        "command": "appworld verify tests",
        "exit_code": 0,
        "elapsed_seconds": 612,
        "suite_summaries": [
            {"passed": 1652, "skipped": 0},
            {"passed": 76, "skipped": 0},
            {"passed": 110, "skipped": 2},
        ],
    }
    assert verification["tasks"] == {
        "command": "appworld verify tasks --num-processes 4",
        "exit_code": 0,
        "elapsed_seconds": 246,
        "passed": 147,
        "total": 147,
    }


def test_appworld_gate_0a_evidence_is_canonical_replay_summary() -> None:
    encoded = GATE_0A_EVIDENCE.read_bytes()

    assert len(encoded) == GATE_0A_EVIDENCE_BYTES
    assert hashlib.sha256(encoded).hexdigest() == GATE_0A_EVIDENCE_SHA256

    payload = _load_unique_json(GATE_0A_EVIDENCE)
    assert len(payload) == 23
    assert payload == GATE_0A_PAYLOAD

    summary = _reconstruct_replay_summary(payload)
    assert summary.to_dict() == payload
    canonical = (
        json.dumps(
            summary.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert canonical == encoded


def test_appworld_gate_0a_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"mode": "gate", "mode": "smoke"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key: mode"):
        _load_unique_json(duplicate)


def test_appworld_gate_0a_evidence_uses_only_aggregate_privacy_fields() -> None:
    payload = _load_unique_json(GATE_0A_EVIDENCE)

    assert frozenset(payload) == GATE_0A_PRIVACY_ALLOWLIST
    assert payload["splits"] == ["dev", "train"]
    assert payload["failure_counts"] == {}

    keys: set[str] = set()
    string_values: list[str] = []
    pending: list[Any] = [payload]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            keys.update(node)
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, str):
            string_values.append(node)

    forbidden_keys = {
        "task_id",
        "task_ids",
        "checkout_path",
        "shared_path",
        "compiled_solution",
        "compiled_solutions",
        "trajectory",
        "trajectories",
        "request",
        "requests",
        "request_trace",
        "request_traces",
        "api_arguments",
        "execution_output",
        "evaluator_requirements",
    }
    assert keys.isdisjoint(forbidden_keys)
    forbidden_text = re.compile(
        r"https?://|(?:^|\s)/(?:home|mnt|tmp|var)/|"
        r"proxy|credential|token|secret|trajectory|request[ _-]?trace|"
        r"api[ _-]?arguments|execution[ _-]?output|evaluator[ _-]?requirements|"
        r"task[ _-]?ids?",
        flags=re.IGNORECASE,
    )
    assert all(forbidden_text.search(value) is None for value in string_values)


def test_appworld_manifest_cross_binds_gate_0a_evidence_and_provenance() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")
    gate = manifest["gate_0a"]

    assert gate == {
        "observed_at": "2026-08-30",
        "run_label": "formal-os-exit-c84049ad-20260829T212809Z",
        "evidence": {
            "path": GATE_0A_EVIDENCE_RELATIVE_PATH.as_posix(),
            "bytes": GATE_0A_EVIDENCE_BYTES,
            "sha256": GATE_0A_EVIDENCE_SHA256,
        },
        "verifier_commit": "c84049adaa4875c2ae4d06652e451db38d2cf588",
        "appworld_commit": APPWORLD_COMMIT,
        "protocol": {
            "splits": ["dev", "train"],
            "split_counts": {"dev": 57, "train": 90},
            "task_count": 147,
            "repetitions": 3,
            "episode_count": 441,
            "seed": 100,
            "workers": 1,
        },
        "result": {
            "initial_state_match_rate": 1.0,
            "final_state_match_rate": 1.0,
            "evaluator_match_rate": 1.0,
            "trace_match_rate": 1.0,
            "task_consistency_rate": 1.0,
            "oracle_success_rate": 1.0,
            "execution_failure_count": 0,
            "exception_count": 0,
            "failure_counts": {},
            "gate_evaluable": True,
            "gate_passed": True,
            "smoke_passed": False,
        },
        "os_exit": {
            "code": 0,
            "artifact_sha256": ("9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"),
        },
        "runtime_log": {
            "schema_version": 1,
            "content_reviewed": False,
            "log_nonempty": True,
            "process_log_bytes": 135254,
            "process_log_deleted": True,
            "source_metadata_sha256": (
                "7ecd7ee397db15811f067dc63965ea004e10b31a76a07e2c97ac1b27d4138b00"
            ),
        },
        "protected_content_included": False,
        "raw_artifacts_committed": False,
        "provenance_limitations": {
            "summary_schema_version": 1,
            "summary_binds": [
                "appworld_commit",
                "task_set_fingerprint",
                "fixed_protocol",
                "aggregate_results",
            ],
            "summary_does_not_bind": [
                "toolshift_commit",
                "wrapper_version",
                "runtime_version",
                "command_digest",
                "timestamps",
            ],
            "verifier_commit_source": "audited_run_label_and_operational_chain",
            "future_schema_requirement": "atomic_completion_record",
        },
    }

    payload = _load_unique_json(REPOSITORY_ROOT / gate["evidence"]["path"])
    assert gate["evidence"]["bytes"] == len(GATE_0A_EVIDENCE.read_bytes())
    assert gate["evidence"]["sha256"] == hashlib.sha256(GATE_0A_EVIDENCE.read_bytes()).hexdigest()
    assert gate["appworld_commit"] == payload["pinned_appworld_commit"]
    for field_name, value in gate["protocol"].items():
        assert payload[field_name] == value
    for field_name, value in gate["result"].items():
        assert payload[field_name] == value


def test_appworld_runbook_records_gate_0a_without_machine_paths_or_overclaiming() -> None:
    runbook = _read_appworld_runbook()
    normalized = " ".join(runbook.split())

    assert "/home/" not in runbook
    assert "/mnt/" not in runbook
    assert "wait for the verifier to finish" in normalized
    assert "atomically write a mode-0600 OS-exit record" in normalized
    assert "validate exit code zero and the exact summary integrity and hash" in normalized
    assert "capture only safe process-log byte metadata" in normalized
    assert "delete the exact resolved private process-log path" in normalized
    assert (
        "atomically write aggregate runtime metadata with `process_log_deleted=true`" in normalized
    )
    assert "promote the evidence and checksums" in normalized
    wrapper_steps = [
        "wait for the verifier to finish",
        "atomically write a mode-0600 OS-exit record",
        "validate exit code zero and the exact summary integrity and hash",
        "capture only safe process-log byte metadata",
        "delete the exact resolved private process-log path",
        "atomically write aggregate runtime metadata with `process_log_deleted=true`",
        "promote the evidence and checksums",
    ]
    assert [normalized.index(step) for step in wrapper_steps] == sorted(
        normalized.index(step) for step in wrapper_steps
    )

    assert "Schema v1 binds" in normalized
    assert "does not bind the ToolShift commit" in normalized
    assert "wrapper version" in normalized
    assert "runtime version" in normalized
    assert "command digest" in normalized
    assert "timestamps" in normalized
    assert "atomic completion record" in normalized
    assert "temporary or nonzero run retains its private process log" in normalized
    assert "must not claim `process_log_deleted=true`" in normalized
    assert "never a glob or symbolic link" in normalized


def test_readme_scopes_gate_0a_as_train_dev_environment_evidence() -> None:
    readme = README.read_text(encoding="utf-8")

    assert (
        "[Gate 0a aggregate environment evidence]"
        f"({GATE_0A_EVIDENCE_RELATIVE_PATH.as_posix()})" in readme
    )
    assert "all 90 train and 57 dev tasks" in readme
    assert "147 tasks \N{MULTIPLICATION SIGN} 3 repetitions = 441 episodes" in readme
    assert "all six aggregate rates were `1.0`" in readme
    assert "zero execution failures and zero exceptions" in readme
    assert "environment evidence only" in readme
    assert "not a model result or a held-out evaluation" in readme


def test_appworld_runbook_freezes_replay_gate_runtime_safety() -> None:
    runbook = _read_appworld_runbook()
    normalized_blocks = [_normalize_shell(block) for block in _bash_blocks(runbook)]

    assert "must run from the pinned AppWorld checkout" in runbook
    assert "fresh AppWorld instance" in runbook
    assert "load_state()" in runbook
    assert "double time-freezer" in runbook
    assert "TOOLSHIFT_RUN_APPWORLD_SMOKE=1" in runbook
    assert "scripts/verify_appworld_replay.py" in runbook
    assert "--output" in runbook
    assert "--max-tasks" not in runbook
    assert any("clear_proxy_environment" in block for block in normalized_blocks)


def test_appworld_manifest_limits_warnings_and_unresolved_claims() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")

    assert manifest["warnings"] == [
        "starlette_httpx_compatibility",
        "cryptography_cfb_deprecation",
    ]
    assert manifest["license"] == {
        "public_code": "Apache-2.0",
        "protected_code_and_data": {
            "redistribution_constraint": (
                "Public redistribution of protected content or its derivatives must remain "
                "encrypted."
            ),
            "protected_content_included": False,
        },
    }
    assert _unresolved_paths(manifest) == {"data.integrity.official_content_checksum"}


def test_appworld_manifest_defines_private_nonformal_m3a_protocol_only() -> None:
    manifest = _load_manifest("appworld-ml2.yaml")

    assert manifest["gate_0_m3a"] == {
        "kind": "private_nonformal_integration_smoke_protocol",
        "upstream_sources": {
            "repository": (f"https://github.com/StonyBrookNLP/appworld/tree/{APPWORLD_COMMIT}"),
            "cli": (
                "https://github.com/StonyBrookNLP/appworld/blob/"
                f"{APPWORLD_COMMIT}/src/appworld/cli.py"
            ),
            "path_store": (
                "https://github.com/StonyBrookNLP/appworld/blob/"
                f"{APPWORLD_COMMIT}/src/appworld/common/path_store.py"
            ),
            "verifier": (
                "https://github.com/StonyBrookNLP/appworld/blob/"
                f"{APPWORLD_COMMIT}/src/appworld/verify.py"
            ),
        },
        "protocol": {
            "split": "train",
            "seed": 100,
            "workers": 1,
            "transform_families": ["clean", "l1", "l2"],
            "fresh_world_per_role": True,
            "full_non_admin_catalog": True,
            "formal_gate_evaluable": False,
            "formal_gate_claim": False,
        },
        "evidence_policy": {
            "plaintext_location": "external_private_appworld_root",
            "runtime_record_committed": False,
            "public_result_claim": False,
            "publication_requires": ("appworld_maintainer_approved_encrypted_workflow"),
        },
    }


def test_appworld_runbook_freezes_private_m3a_install_and_execution_boundary() -> None:
    runbook = _read_appworld_runbook()
    normalized = " ".join(runbook.split())
    normalized_blocks = [_normalize_shell(block) for block in _bash_blocks(runbook)]

    assert "dedicated private `APPWORLD_ROOT`" in normalized
    assert "umask `077`" in normalized
    assert "mode `0700`" in normalized
    assert "outside every Git worktree" in normalized
    assert "node-local storage" in normalized
    assert "--link-mode copy" in normalized
    assert "PYTHON_DOTENV_DISABLED=1" in normalized
    assert "PYTHONDONTWRITEBYTECODE=1" in normalized
    assert 'APPWORLD_CACHE="${APPWORLD_ROOT}/.cache"' in normalized
    assert "ModelScope" in normalized
    assert "without the external proxy" in normalized
    assert "private, non-formal M3A integration evidence" in normalized
    assert "must never be added to Git" in normalized
    assert "maintainer-approved encrypted workflow" in normalized
    assert "historical Gate 0a" in normalized
    assert "does not set precedent for M3A" in normalized

    install_sequence = [
        "git clone",
        "lfs install --local",
        "checkout --detach",
        "lfs pull",
        "uv venv",
        "uv pip install",
        'bin/appworld" install --repo',
        "cp -a",
        "download data --version",
        "verify tests --root",
        "scripts/verify_appworld_gate0.py",
    ]
    m3a_runbook = normalized[normalized.index("Ensure the Git LFS executable") :]
    cursor = 0
    for step in install_sequence:
        cursor = m3a_runbook.index(step, cursor) + len(step)
    assert all(
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY" not in block
        for block in normalized_blocks
    )
    assert "TOOLSHIFT_RUN_APPWORLD_GATE0_SMOKE=1" in runbook
    assert 'test ! -e "${APPWORLD_ROOT}/tests"' in runbook
    assert (
        'cp -a "${TOOLSHIFT_SHARED_ROOT}/repos/appworld/tests" "${APPWORLD_ROOT}/tests"'
    ) in normalized
    assert "verify tasks" not in runbook.split("## Historical Gate 0a", 1)[0]
    assert "test_normal" in runbook
    assert "test_challenge" in runbook
    assert "never enumerates or opens" in runbook


def test_appworld_runbook_builds_and_runs_entire_environment_on_node_local_storage() -> None:
    runbook = _read_appworld_runbook()
    normalized = _normalize_shell(runbook)
    install_block = _normalize_shell(_bash_blocks(runbook)[1])

    assert 'export APPWORLD_ENV="${TOOLSHIFT_NODE_LOCAL}/venvs/appworld-0.2.0"' in install_block
    assert "readonly APPWORLD_ENV" in install_block
    assert 'uv venv --python 3.11.15 "${APPWORLD_ENV}"' in install_block
    assert '"python-dotenv==1.2.2"' in install_block
    assert '--python "${APPWORLD_ENV}/bin/python"' in normalized
    assert '"${APPWORLD_ENV}/bin/appworld"' in normalized
    assert '"${APPWORLD_ENV}/bin/python"' in normalized
    assert '"${APPWORLD_ENV}/bin/pytest"' in normalized
    assert "${TOOLSHIFT_SHARED_ROOT}/venvs" not in runbook

    for block in _bash_blocks(runbook):
        if "uv pip install" in block:
            assert '--python "${APPWORLD_ENV}/bin/python"' in _normalize_shell(block)


def test_appworld_runbook_normalizes_pinned_runtime_path_permissions() -> None:
    runbook = _read_appworld_runbook()
    install_block = _normalize_shell(_bash_blocks(runbook)[1])

    download = install_block.index("download data --version 0.2.0")
    required_after_download = [
        'test -d "${APPWORLD_ROOT}/data"',
        'test ! -L "${APPWORLD_ROOT}/data"',
        'test -O "${APPWORLD_ROOT}/data"',
        'chmod 0700 "${APPWORLD_ROOT}/data"',
        'test -d "${APPWORLD_ROOT}/data/base_dbs"',
        'test ! -L "${APPWORLD_ROOT}/data/base_dbs"',
        'test -O "${APPWORLD_ROOT}/data/base_dbs"',
        'chmod 0700 "${APPWORLD_ROOT}/data/base_dbs"',
        'test -f "${APPWORLD_ROOT}/data/version.txt"',
        'test ! -L "${APPWORLD_ROOT}/data/version.txt"',
        'test -O "${APPWORLD_ROOT}/data/version.txt"',
        'chmod 0600 "${APPWORLD_ROOT}/data/version.txt"',
        'test -f "${APPWORLD_ROOT}/data/base_dbs/version.txt"',
        'test ! -L "${APPWORLD_ROOT}/data/base_dbs/version.txt"',
        'test -O "${APPWORLD_ROOT}/data/base_dbs/version.txt"',
        'chmod 0600 "${APPWORLD_ROOT}/data/base_dbs/version.txt"',
    ]
    cursor = download
    for step in required_after_download:
        cursor = install_block.index(step, cursor) + len(step)


def test_appworld_runbook_proxy_cleanup_is_generic_fail_closed_and_silent() -> None:
    runbook = _read_appworld_runbook()
    helper = _bash_function(runbook, "clear_proxy_environment")
    synthetic_name = "MiXeD_PRIVATE_PROXY"
    synthetic_value = "synthetic-sensitive-value"
    script = (
        "set -euo pipefail\n"
        f"{helper}\n"
        "clear_proxy_environment\n"
        f'test -z "${{{synthetic_name}+present}}"\n'
        'test "${SAFE_MARKER}" = "preserved"\n'
    )
    environment = {
        "PATH": os.environ["PATH"],
        synthetic_name: synthetic_value,
        "SAFE_MARKER": "preserved",
    }

    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert synthetic_name not in result.stdout + result.stderr
    assert synthetic_value not in result.stdout + result.stderr
    assert "*_proxy" in helper.lower()
    assert "compgen -e" in helper
    assert "proxy environment cleanup failed" in helper
    assert "before invoking any approved domestic mirror" in _normalize_shell(runbook)
    assert runbook.count("clear_proxy_environment\n") >= 3

    official_section = runbook.split("## Official verification", 1)[1].split(
        "## Private, non-formal M3A", 1
    )[0]
    m3a_section = runbook.split("## Private, non-formal M3A", 1)[1].split(
        "## Historical Gate 0a", 1
    )[0]
    assert official_section.index("clear_proxy_environment") < official_section.index(
        "verify tests"
    )
    assert m3a_section.index("clear_proxy_environment") < m3a_section.index(
        "scripts/verify_appworld_gate0.py"
    )
    assert m3a_section.index("uv pip install") < m3a_section.index("clear_proxy_environment")

    gate0a_section = runbook.split("## Historical Gate 0a", 1)[1].split("## Recorded Gate 0a", 1)[0]
    assert gate0a_section.index("uv pip install") < gate0a_section.index("clear_proxy_environment")
    assert gate0a_section.index("clear_proxy_environment") < gate0a_section.index(
        "TOOLSHIFT_RUN_APPWORLD_SMOKE"
    )


def test_appworld_runbook_privately_audits_and_safely_deletes_official_log() -> None:
    runbook = _read_appworld_runbook()
    official_section = runbook.split("## Official verification", 1)[1].split(
        "## Private, non-formal M3A", 1
    )[0]
    normalized = _normalize_shell(official_section)

    required_steps = [
        'VERIFY_LOG="${APPWORLD_ROOT}/operator-records/official-tests.log"',
        ': > "${VERIFY_LOG}"',
        'chmod 0600 "${VERIFY_LOG}"',
        '>"${VERIFY_LOG}" 2>&1',
        "verify_status=$?",
        'if [ "${verify_status}" -ne 0 ]; then',
        '"${APPWORLD_ENV}/bin/python" - "${APPWORLD_ROOT}" "${VERIFY_LOG}"',
        "os.O_NOFOLLOW",
        "os.unlink(log_name, dir_fd=records_fd)",
        "os.fsync(records_fd)",
    ]
    cursor = 0
    for step in required_steps[:-2]:
        cursor = normalized.index(step, cursor) + len(step)
    unlink_position = normalized.index(required_steps[-2], cursor)
    assert unlink_position < normalized.rindex(required_steps[-1])
    assert "Official AppWorld test verification failed" in official_section
    assert "Official AppWorld test verification audit failed" in official_section
    assert "1652" in official_section
    assert "76" in official_section
    assert "110" in official_section
    assert "2" in official_section
    assert "cat " not in official_section
    assert "tail " not in official_section


def test_appworld_runbook_explicitly_binds_tracked_clean_checkout_check() -> None:
    runbook = _read_appworld_runbook()
    before_verify = runbook.split("## Official verification", 1)[0]
    normalized = _normalize_shell(before_verify)

    required_steps = [
        '"${APPWORLD_ENV}/bin/appworld" install --repo',
        "clear_git_environment",
        '"${APPWORLD_ENV}/bin/python" - "${APPWORLD_CHECKOUT}"',
        'f"--git-dir={git_directory}"',
        'f"--work-tree={checkout}"',
        '"rev-parse", "--show-toplevel", "--absolute-git-dir"',
        '"ls-files", "-v", "-z"',
        'tag == b"S" or tag.islower()',
        '"ls-files", "--others", "-z", "--"',
        "untracked_paths.add(raw_path)",
        'git_directory / "info" / "attributes"',
        '"diff-index"',
        '"--cached"',
        '"--no-ext-diff"',
        '"--ignore-submodules=none"',
        '"ls-tree", "-r", "-z", "--full-tree", "HEAD"',
        '"cat-file", "--batch"',
        "git_blob_sha1(payload)",
        "head_modes = {relative: mode",
        "lfs_pointer(expected)",
        "hashlib.sha256()",
        "ast.parse(",
        "PBKDF2HMAC(",
        "zipfile.ZipFile(",
        "if target in seen_bundle_targets:",
        "member_payload = archive.read(info)",
        "if target in head_payloads:",
        "tracked_pointer = lfs_pointer(tracked_payload)",
        "untracked_paths != set(expected_untracked)",
        "path.read_bytes() != expected",
    ]
    cursor = normalized.index(required_steps[0]) + len(required_steps[0])
    for step in required_steps[1:]:
        cursor = normalized.index(step, cursor) + len(step)
    assert "core.fsmonitor=false" in before_verify
    assert "core.hooksPath=/dev/null" in before_verify
    assert "--no-replace-objects" in before_verify
    assert "checkout_metadata.st_uid != os.geteuid()" in before_verify
    assert "stat.S_IMODE(checkout_metadata.st_mode) != 0o700" in before_verify
    assert "Pinned AppWorld checkout binding validation failed" in before_verify
    assert "Pinned AppWorld tracked checkout is dirty" in before_verify


def test_appworld_gate0_smoke_only_skips_before_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke_path = REPOSITORY_ROOT / "tests/smoke/test_appworld_gate0.py"
    spec = importlib.util.spec_from_file_location("appworld_gate0_smoke_wrapper", smoke_path)
    assert spec is not None and spec.loader is not None
    smoke_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_module)

    monkeypatch.delenv("TOOLSHIFT_RUN_APPWORLD_GATE0_SMOKE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        smoke_module.test_private_appworld_gate0_smoke()

    monkeypatch.setenv("TOOLSHIFT_RUN_APPWORLD_GATE0_SMOKE", "1")

    def fail_runtime() -> None:
        raise RuntimeError("synthetic pinned-runtime failure")

    monkeypatch.setattr(smoke_module, "run_appworld_gate0_smoke", fail_runtime)
    with pytest.raises(RuntimeError, match="synthetic pinned-runtime failure"):
        smoke_module.test_private_appworld_gate0_smoke()


def test_bfcl_manifest_pins_comparable_source_and_keeps_observed_head_as_audit() -> None:
    manifest = _load_manifest("bfcl-v4.yaml")

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "external_official_evaluation"
    assert manifest["benchmark_version"] == "BFCL_v4"

    repository = manifest["repository"]
    assert repository["name"] == "Gorilla"
    assert repository["subdirectory"] == "berkeley-function-call-leaderboard"
    assert repository["default_branch"] == "main"
    assert repository["pinned_commit"] == BFCL_COMPARABLE_COMMIT
    assert repository["version_tag"] is None
    assert repository["observed_upstream"] == {
        "head": BFCL_OBSERVED_HEAD,
        "observed_at": "2026-08-29",
        "purpose": "audit_only",
    }
    assert FULL_GIT_OBJECT.fullmatch(repository["pinned_commit"])
    assert FULL_GIT_OBJECT.fullmatch(repository["observed_upstream"]["head"])
    assert repository["observed_upstream"]["head"] != repository["pinned_commit"]


def test_bfcl_manifest_records_evaluator_and_repository_bundled_dataset() -> None:
    manifest = _load_manifest("bfcl-v4.yaml")

    assert manifest["evaluator"] == {
        "distribution": "bfcl-eval",
        "version": "2025.12.17",
        "python_requires": ">=3.10",
        "recommended_python": "3.10",
        "wheel_sha256": "UNRESOLVED",
    }
    assert manifest["dataset"] == {
        "source": "repository_bundled",
        "path": "berkeley-function-call-leaderboard/bfcl_eval/data",
        "revision": BFCL_COMPARABLE_COMMIT,
        "git_tree": "5e49f820dd465850cbaebc241806d2ef7b893471",
        "hugging_face": {"dataset": None, "revision": None},
        "loading_note": ("BFCL v4 is not loaded from the current BFCL v3 Hugging Face dataset."),
    }
    assert FULL_GIT_OBJECT.fullmatch(manifest["dataset"]["git_tree"])


def test_bfcl_manifest_separates_cli_categories_and_reporting_labels() -> None:
    manifest = _load_manifest("bfcl-v4.yaml")
    categories = manifest["categories"]

    assert categories["multi_turn"] == {
        "scoring": True,
        "cli_categories": [
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ],
    }
    assert categories["web_search"] == {
        "scoring": True,
        "cli_categories": ["web_search_base", "web_search_no_snippet"],
    }
    assert categories["memory"] == {
        "scoring": True,
        "cli_categories": ["memory_kv", "memory_vector", "memory_rec_sum"],
    }
    assert categories["hallucination"] == {
        "scoring": True,
        "reporting_family": "Hallucination Measurement",
        "single_cli_category": False,
        "underlying_categories": [
            "irrelevance",
            "live_irrelevance",
            "live_relevance",
        ],
    }
    assert categories["format_sensitivity"] == {
        "scoring": False,
        "use": "prompts_only",
    }
    assert manifest["reporting"] == {
        "official_label": "BFCL_v4",
        "derived_label": "BFCL-Shift",
        "combine_scores": False,
    }


def test_bfcl_manifest_limits_unresolved_fields_to_known_reproducibility_gaps() -> None:
    manifest = _load_manifest("bfcl-v4.yaml")

    assert manifest["runtime"] == {
        "memory_vector_encoder_revision": "UNRESOLVED",
        "sglang_version": "UNRESOLVED",
        "web_search_external_state": "UNRESOLVED",
        "raw_trajectory_redistribution_terms": "UNRESOLVED",
    }
    assert manifest["license"] == {
        "code": "Apache-2.0",
        "third_party_materials": ["web_page_content", "model_outputs"],
        "third_party_release_terms": "UNRESOLVED",
    }
    assert _unresolved_paths(manifest) == {
        "evaluator.wheel_sha256",
        "license.third_party_release_terms",
        "runtime.memory_vector_encoder_revision",
        "runtime.raw_trajectory_redistribution_terms",
        "runtime.sglang_version",
        "runtime.web_search_external_state",
    }
