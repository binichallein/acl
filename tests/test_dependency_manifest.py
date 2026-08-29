from __future__ import annotations

from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_dependency_manifest_records_sources_without_fabricated_revisions() -> None:
    path = REPOSITORY_ROOT / "data/manifests/dependencies.yaml"
    assert path.is_file(), "data/manifests/dependencies.yaml must exist"

    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    dependencies = manifest["dependencies"]

    assert set(dependencies) == {
        "appworld",
        "base_model",
        "bfcl",
        "training_framework",
    }
    for dependency in dependencies.values():
        assert set(dependency) == {"revision", "source", "status"}
        assert dependency["source"]
        assert dependency["revision"] == "UNRESOLVED"
        assert dependency["status"]
