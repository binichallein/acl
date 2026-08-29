from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APPWORLD_RUNBOOK = REPOSITORY_ROOT / "docs/environment/ml2-appworld.md"
FULL_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")

APPWORLD_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
BFCL_COMPARABLE_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
BFCL_OBSERVED_HEAD = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"


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
    assert 'TOOLSHIFT_SHARED_ROOT="$(realpath -m -- "${TOOLSHIFT_SHARED_ROOT}")"' in (
        normalized
    )
    assert 'if [ "${TOOLSHIFT_SHARED_ROOT}" = "/" ]; then' in first_block
    assert 'echo "TOOLSHIFT_SHARED_ROOT must not resolve to filesystem root" >&2' in (
        first_block
    )
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
    assert f'test "$(uv --version)" = "uv {uv_version}"' in install_block
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
    assert _unresolved_paths(manifest) == {
        "data.integrity.official_content_checksum"
    }


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
        "loading_note": (
            "BFCL v4 is not loaded from the current BFCL v3 Hugging Face dataset."
        ),
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
