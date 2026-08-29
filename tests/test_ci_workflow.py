from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")


def _extract_action_references(node: object) -> list[str]:
    references: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                references.append(value)
            references.extend(_extract_action_references(value))
    elif isinstance(node, list):
        for value in node:
            references.extend(_extract_action_references(value))
    return references


def _find_unpinned_action_references(references: list[str]) -> list[str]:
    return [
        reference
        for reference in references
        if not reference.startswith(("./", "docker://"))
        and FULL_COMMIT_SHA.fullmatch(reference.rpartition("@")[2]) is None
    ]


def _load_workflow(path: Path) -> dict[object, object]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AssertionError(f"could not read workflow {path}: {error}") from error

    try:
        workflow = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise AssertionError(f"could not parse workflow YAML {path}: {error}") from error

    if not isinstance(workflow, dict):
        raise AssertionError(f"workflow {path} must contain a top-level YAML mapping")
    return workflow


def _find_unpinned_workflow_actions(workflow_directory: Path) -> list[str]:
    paths = sorted((*workflow_directory.glob("*.yml"), *workflow_directory.glob("*.yaml")))
    violations: list[str] = []
    for path in paths:
        references = _extract_action_references(_load_workflow(path))
        for reference in _find_unpinned_action_references(references):
            violations.append(f"{path.name}: {reference}")
    return violations


def test_third_party_github_actions_are_pinned_to_full_commit_sha() -> None:
    workflow_directory = REPOSITORY_ROOT / ".github/workflows"
    paths = sorted((*workflow_directory.glob("*.yml"), *workflow_directory.glob("*.yaml")))
    assert paths, "repository must contain at least one GitHub Actions workflow"

    references = [
        reference
        for path in paths
        for reference in _extract_action_references(_load_workflow(path))
    ]
    assert references, "GitHub Actions workflows must contain at least one action reference"
    assert _find_unpinned_workflow_actions(workflow_directory) == []


def test_list_shorthand_action_is_extracted_and_reported_unpinned() -> None:
    workflow = yaml.safe_load(
        "jobs:\n  test:\n    steps:\n      - uses: actions/checkout@v7\n"
    )

    references = _extract_action_references(workflow)

    assert references == ["actions/checkout@v7"]
    assert _find_unpinned_action_references(references) == ["actions/checkout@v7"]


def test_workflow_scan_reads_yml_and_yaml_files(tmp_path: Path) -> None:
    pinned_sha = "a" * 40
    (tmp_path / "first.yml").write_text(
        f"jobs:\n  one:\n    steps:\n      - uses: owner/one@{pinned_sha}\n",
        encoding="utf-8",
    )
    (tmp_path / "second.yaml").write_text(
        "jobs:\n  two:\n    steps:\n      - uses: owner/two@v1\n",
        encoding="utf-8",
    )
    violations = _find_unpinned_workflow_actions(tmp_path)

    assert len(violations) == 1
    assert "second.yaml: owner/two@v1" in violations[0]


def test_workflow_loader_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text("- uses: owner/action@v1\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="must contain a top-level YAML mapping"):
        _load_workflow(path)


def test_workflow_loader_reports_file_read_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.yml"

    with pytest.raises(AssertionError, match="could not read workflow"):
        _load_workflow(path)


def test_workflow_loader_reports_yaml_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yml"
    path.write_text("jobs: [\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="could not parse workflow YAML"):
        _load_workflow(path)


def test_dependabot_maintains_github_action_pins_weekly() -> None:
    path = REPOSITORY_ROOT / ".github/dependabot.yml"
    assert path.is_file(), ".github/dependabot.yml must exist"

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["version"] == 2
    assert config["updates"] == [
        {
            "package-ecosystem": "github-actions",
            "directory": "/",
            "schedule": {"interval": "weekly"},
        }
    ]
