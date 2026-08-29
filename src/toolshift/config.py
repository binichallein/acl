"""Load and validate reproducible system metadata."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_SUPPORTED_SCHEMA_VERSION = 1
_V1_REQUIRED_FIELDS = (
    "machine",
    "resources",
    "resources.gpu_model",
    "resources.gpus",
    "resources.gpu_memory_gib",
    "resources.cpus",
    "resources.memory_gib",
    "storage",
    "storage.shared_root_env",
)
_INTEGER_RESOURCES = ("gpus", "cpus")
_NUMERIC_RESOURCES = ("gpu_memory_gib", "memory_gib")


def _required_value(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for segment in dotted_path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise ValueError(f"missing required field: {dotted_path}")
        value = value[segment]
    return value


def _is_positive_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _is_finite_positive_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    return isinstance(value, float) and math.isfinite(value) and value > 0


def load_system_config(
    path: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load validated system metadata and resolve its shared storage root.

    The shared filesystem location is intentionally supplied at runtime through
    the environment variable named by ``storage.shared_root_env``. This keeps
    machine-specific paths and credentials out of tracked configuration files.
    """

    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML system config: {error}") from error

    if not isinstance(loaded, Mapping):
        raise ValueError("system config must be a YAML mapping")

    schema_version = _required_value(loaded, "schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("schema_version must be an integer")
    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    for field in _V1_REQUIRED_FIELDS:
        _required_value(loaded, field)

    machine = loaded["machine"]
    if not isinstance(machine, str) or not machine.strip():
        raise ValueError("machine must be a non-empty string")

    resources = loaded["resources"]
    gpu_model = resources["gpu_model"]
    if not isinstance(gpu_model, str) or not gpu_model.strip():
        raise ValueError("resources.gpu_model must be a non-empty string")

    for field in _INTEGER_RESOURCES:
        value = resources[field]
        if not _is_positive_integer(value):
            raise ValueError(f"resources.{field} must be a positive integer")
    for field in _NUMERIC_RESOURCES:
        if not _is_finite_positive_number(resources[field]):
            raise ValueError(f"resources.{field} must be a finite positive number")

    shared_root_env = loaded["storage"]["shared_root_env"]
    if not isinstance(shared_root_env, str) or not shared_root_env.strip():
        raise ValueError("storage.shared_root_env must be a non-empty string")

    environment = os.environ if environ is None else environ
    shared_root = environment.get(shared_root_env)
    if not isinstance(shared_root, str) or not shared_root.strip():
        raise ValueError(f"environment variable {shared_root_env} is not set")

    resolved = copy.deepcopy(dict(loaded))
    resolved["storage"]["shared_root"] = shared_root
    return resolved
