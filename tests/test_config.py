from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml


def _load_system_config_function() -> Callable[..., dict[str, Any]]:
    try:
        module = importlib.import_module("toolshift.config")
    except ModuleNotFoundError:
        pytest.fail("toolshift.config.load_system_config is not implemented")

    function = getattr(module, "load_system_config", None)
    if function is None:
        pytest.fail("toolshift.config.load_system_config is not implemented")
    return function


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "machine": "ml2",
        "resources": {
            "gpu_model": "NVIDIA A100-SXM4-80GB",
            "gpus": 4,
            "gpu_memory_gib": 80,
            "cpus": 112,
            "memory_gib": 935,
        },
        "storage": {"shared_root_env": "TOOLSHIFT_SHARED_ROOT"},
    }


def _write_yaml(tmp_path: Path, payload: Mapping[str, Any]) -> Path:
    path = tmp_path / "system.yaml"
    path.write_text(yaml.safe_dump(dict(payload)), encoding="utf-8")
    return path


def _write_text(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "system.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _without(payload: dict[str, Any], dotted_path: str) -> dict[str, Any]:
    modified = copy.deepcopy(payload)
    segments = dotted_path.split(".")
    target = modified
    for segment in segments[:-1]:
        target = target[segment]
    del target[segments[-1]]
    return modified


def test_load_system_config_resolves_shared_root_from_given_environment(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    load_system_config = _load_system_config_function()
    config = load_system_config(
        _write_yaml(tmp_path, valid_payload),
        environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/ml2-shared/toolshift"},
    )

    assert config["schema_version"] == 1
    assert config["machine"] == "ml2"
    assert config["resources"] == valid_payload["resources"]
    assert config["storage"] == {
        "shared_root_env": "TOOLSHIFT_SHARED_ROOT",
        "shared_root": "/mnt/ml2-shared/toolshift",
    }


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "machine",
        "resources",
        "resources.gpu_model",
        "resources.gpus",
        "resources.gpu_memory_gib",
        "resources.cpus",
        "resources.memory_gib",
        "storage",
        "storage.shared_root_env",
    ],
)
def test_load_system_config_rejects_missing_required_field(
    tmp_path: Path, valid_payload: dict[str, Any], field: str
) -> None:
    load_system_config = _load_system_config_function()
    path = _write_yaml(tmp_path, _without(valid_payload, field))

    with pytest.raises(ValueError, match=f"missing required field: {field}"):
        load_system_config(path, environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"})


@pytest.mark.parametrize("field", ["gpus", "cpus"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, math.inf])
def test_load_system_config_rejects_non_positive_or_non_integer_resources(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    field: str,
    value: int | float,
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["resources"][field] = value

    with pytest.raises(
        ValueError,
        match=rf"resources\.{field} must be a positive integer",
    ):
        load_system_config(
            _write_yaml(tmp_path, valid_payload),
            environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"},
        )


@pytest.mark.parametrize("field", ["gpu_memory_gib", "memory_gib"])
@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, math.nan, math.inf, -math.inf, "80"],
)
def test_load_system_config_rejects_non_finite_or_non_positive_memory_resources(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["resources"][field] = value

    with pytest.raises(
        ValueError,
        match=rf"resources\.{field} must be a finite positive number",
    ):
        load_system_config(
            _write_yaml(tmp_path, valid_payload),
            environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("gpu_memory_gib", 80.5), ("memory_gib", 935.5)],
)
def test_load_system_config_accepts_finite_positive_memory_resources(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    field: str,
    value: float,
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["resources"][field] = value

    config = load_system_config(
        _write_yaml(tmp_path, valid_payload),
        environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"},
    )

    assert config["resources"][field] == value


@pytest.mark.parametrize("value", ["", "   ", None, 42])
def test_load_system_config_rejects_empty_or_non_string_machine(
    tmp_path: Path, valid_payload: dict[str, Any], value: Any
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["machine"] = value

    with pytest.raises(ValueError, match="machine must be a non-empty string"):
        load_system_config(
            _write_yaml(tmp_path, valid_payload),
            environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"},
        )


@pytest.mark.parametrize("value", ["", "   ", None, 42])
def test_load_system_config_rejects_empty_or_non_string_gpu_model(
    tmp_path: Path, valid_payload: dict[str, Any], value: Any
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["resources"]["gpu_model"] = value

    with pytest.raises(ValueError, match=r"resources\.gpu_model must be a non-empty string"):
        load_system_config(
            _write_yaml(tmp_path, valid_payload),
            environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"},
        )


@pytest.mark.parametrize(
    "environment",
    [{}, {"TOOLSHIFT_SHARED_ROOT": ""}, {"TOOLSHIFT_SHARED_ROOT": "   "}],
    ids=["missing", "empty", "whitespace"],
)
def test_load_system_config_rejects_unset_or_blank_shared_root_environment_variable(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    environment: dict[str, str],
) -> None:
    load_system_config = _load_system_config_function()

    with pytest.raises(
        ValueError,
        match="environment variable TOOLSHIFT_SHARED_ROOT is not set",
    ):
        load_system_config(_write_yaml(tmp_path, valid_payload), environ=environment)


def test_load_system_config_reads_process_environment_when_environ_is_none(
    tmp_path: Path,
    valid_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_system_config = _load_system_config_function()
    monkeypatch.setenv("TOOLSHIFT_SHARED_ROOT", "/temporary/test-only/shared")

    config = load_system_config(_write_yaml(tmp_path, valid_payload))

    assert config["storage"]["shared_root"] == "/temporary/test-only/shared"


def test_load_system_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    load_system_config = _load_system_config_function()

    with pytest.raises(ValueError, match="invalid YAML system config"):
        load_system_config(_write_text(tmp_path, "resources: [\n"), environ={})


def test_load_system_config_rejects_non_mapping_yaml_document(tmp_path: Path) -> None:
    load_system_config = _load_system_config_function()

    with pytest.raises(ValueError, match="system config must be a YAML mapping"):
        load_system_config(_write_text(tmp_path, "- ml2\n"), environ={})


@pytest.mark.parametrize("value", ["", "   ", None, 42])
def test_load_system_config_rejects_invalid_shared_root_environment_variable_name(
    tmp_path: Path, valid_payload: dict[str, Any], value: Any
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["storage"]["shared_root_env"] = value

    with pytest.raises(
        ValueError,
        match=r"storage\.shared_root_env must be a non-empty string",
    ):
        load_system_config(_write_yaml(tmp_path, valid_payload), environ={})


@pytest.mark.parametrize("value", [True, False, "1", 1.0])
def test_load_system_config_rejects_non_integer_schema_version(
    tmp_path: Path, valid_payload: dict[str, Any], value: Any
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["schema_version"] = value

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_system_config(
            _write_yaml(tmp_path, valid_payload),
            environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"},
        )


def test_load_system_config_rejects_minimal_unsupported_schema_before_v1_fields(
    tmp_path: Path,
) -> None:
    load_system_config = _load_system_config_function()

    with pytest.raises(ValueError, match="unsupported schema_version: 2"):
        load_system_config(_write_yaml(tmp_path, {"schema_version": 2}), environ={})


@pytest.mark.parametrize("value", [True, "1", 1.0])
def test_load_system_config_rejects_schema_type_before_v1_fields(
    tmp_path: Path, value: Any
) -> None:
    load_system_config = _load_system_config_function()

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_system_config(_write_yaml(tmp_path, {"schema_version": value}), environ={})


def test_load_system_config_rejects_minimal_document_without_schema_version(
    tmp_path: Path,
) -> None:
    load_system_config = _load_system_config_function()

    with pytest.raises(ValueError, match="missing required field: schema_version"):
        load_system_config(_write_yaml(tmp_path, {}), environ={})


def test_load_system_config_rejects_unsupported_schema_version(
    tmp_path: Path, valid_payload: dict[str, Any]
) -> None:
    load_system_config = _load_system_config_function()
    valid_payload["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported schema_version: 2"):
        load_system_config(
            _write_yaml(tmp_path, valid_payload),
            environ={"TOOLSHIFT_SHARED_ROOT": "/mnt/shared"},
        )


def test_ml2_system_config_records_expected_non_sensitive_metadata() -> None:
    load_system_config = _load_system_config_function()
    repository_root = Path(__file__).resolve().parents[1]
    path = repository_root / "configs/system/ml2.yaml"
    assert path.is_file(), "configs/system/ml2.yaml must exist"

    config = load_system_config(
        path,
        environ={"TOOLSHIFT_SHARED_ROOT": "/runtime-only/shared/root"},
    )

    assert config["machine"] == "ml2"
    assert config["resources"] == {
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "gpus": 4,
        "gpu_memory_gib": 80,
        "cpus": 112,
        "memory_gib": 935,
    }
    assert config["storage"]["shared_root"] == "/runtime-only/shared/root"
