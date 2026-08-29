from __future__ import annotations

import re
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

VERIFIED_DEPENDENCIES = {"appworld", "bfcl"}
UNRESOLVED_DEPENDENCIES = {"base_model", "training_framework"}


def _load_yaml(relative_path: str) -> dict[object, object]:
    path = REPOSITORY_ROOT / relative_path
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), f"{relative_path} must contain a YAML mapping"
    return manifest


def test_dependency_manifest_distinguishes_verified_pins_from_candidates() -> None:
    path = REPOSITORY_ROOT / "data/manifests/dependencies.yaml"
    assert path.is_file(), "data/manifests/dependencies.yaml must exist"

    manifest = _load_yaml("data/manifests/dependencies.yaml")
    dependencies = manifest["dependencies"]

    assert set(dependencies) == {
        "appworld",
        "base_model",
        "bfcl",
        "training_framework",
    }
    for dependency_name, dependency in dependencies.items():
        assert set(dependency) == {"revision", "source", "status"}
        assert dependency["source"]

        if dependency_name in VERIFIED_DEPENDENCIES:
            assert FULL_COMMIT_SHA.fullmatch(dependency["revision"])
            assert dependency["status"] == "verified"
        else:
            assert dependency_name in UNRESOLVED_DEPENDENCIES
            assert dependency["revision"] == "UNRESOLVED"
            assert dependency["status"] == "candidate"


def test_verified_dependency_revisions_match_specialized_benchmark_manifests() -> None:
    dependencies = _load_yaml("data/manifests/dependencies.yaml")["dependencies"]
    appworld = _load_yaml("data/manifests/appworld-ml2.yaml")
    bfcl = _load_yaml("data/manifests/bfcl-v4.yaml")

    assert dependencies["appworld"]["revision"] == appworld["benchmark"]["repository"][
        "commit"
    ]
    assert dependencies["bfcl"]["revision"] == bfcl["repository"]["pinned_commit"]
