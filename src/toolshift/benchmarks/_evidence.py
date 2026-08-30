"""Shared atomic primitive for aggregate-only private evidence files."""

from __future__ import annotations

import json
import os
import secrets
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path


def write_private_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically replace one JSON file with mode 0600 and stable bytes."""

    if not isinstance(payload, Mapping):
        raise ValueError("private evidence payload must be a mapping")
    output_path = Path(path)
    if not output_path.name:
        raise ValueError("output path must name a JSON file")
    if output_path.is_symlink():
        raise ValueError("output path must not be a symbolic link")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            os.fchmod(temporary.fileno(), 0o600)
            json.dump(
                payload,
                temporary,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output_path)
        temporary_name = None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(output_path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _open_private_directory(
    name: str,
    *,
    parent_fd: int,
    normalize_mode: bool = False,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        if normalize_mode:
            os.fchmod(descriptor, 0o700)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ValueError("private evidence directory is invalid")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_binding_matches(
    root_fd: int,
    parent_parts: tuple[str, ...],
    expected_parent: os.stat_result,
) -> bool:
    current = os.dup(root_fd)
    try:
        for part in parent_parts:
            next_descriptor = _open_private_directory(part, parent_fd=current)
            os.close(current)
            current = next_descriptor
        actual = os.fstat(current)
        return (actual.st_dev, actual.st_ino) == (
            expected_parent.st_dev,
            expected_parent.st_ino,
        )
    except (OSError, ValueError):
        return False
    finally:
        os.close(current)


def _private_binding_matches(
    root_path: Path,
    root_fd: int,
    opened_root: os.stat_result,
    parent_parts: tuple[str, ...],
    parent_metadata: os.stat_result,
) -> bool:
    try:
        current_root = root_path.lstat()
        return (
            stat.S_ISDIR(current_root.st_mode)
            and not stat.S_ISLNK(current_root.st_mode)
            and current_root.st_uid == os.geteuid()
            and stat.S_IMODE(current_root.st_mode) == 0o700
            and (current_root.st_dev, current_root.st_ino)
            == (opened_root.st_dev, opened_root.st_ino)
            and _directory_binding_matches(root_fd, parent_parts, parent_metadata)
        )
    except (OSError, ValueError):
        return False


def _private_leaf_matches(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> bool:
    try:
        actual = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return (
            stat.S_ISREG(actual.st_mode)
            and actual.st_uid == os.geteuid()
            and stat.S_IMODE(actual.st_mode) == 0o600
            and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
        )
    except OSError:
        return False


def _discard_expected_leaf(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
) -> None:
    try:
        actual = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        return
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def write_private_json_beneath(
    root: str | Path,
    path: str | Path,
    payload: Mapping[str, object],
) -> None:
    """Atomically write JSON through no-follow descriptors beneath a private root."""

    if not isinstance(payload, Mapping):
        raise ValueError("private evidence payload must be a mapping")
    root_path = Path(root)
    output_path = Path(path)
    root_fd: int | None = None
    opened_directories: list[int] = []
    temporary_fd: int | None = None
    temporary_name: str | None = None
    temporary_metadata: os.stat_result | None = None
    try:
        if (
            not root_path.is_absolute()
            or not output_path.is_absolute()
            or output_path == root_path
            or getattr(os, "O_NOFOLLOW", 0) == 0
            or getattr(os, "O_DIRECTORY", 0) == 0
        ):
            raise ValueError
        relative = output_path.relative_to(root_path)
        if (
            len(relative.parts) < 1
            or any(part in ("", ".", "..") for part in relative.parts)
            or not relative.name
        ):
            raise ValueError
        root_metadata = root_path.lstat()
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
            or root_path.resolve(strict=True) != root_path
        ):
            raise ValueError
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        root_fd = os.open(root_path, flags)
        opened_directories.append(root_fd)
        opened_root = os.fstat(root_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (
            root_metadata.st_dev,
            root_metadata.st_ino,
        ):
            raise ValueError

        parent_fd = root_fd
        parent_parts = tuple(relative.parts[:-1])
        for part in parent_parts:
            try:
                child_fd = _open_private_directory(part, parent_fd=parent_fd)
            except FileNotFoundError:
                created = False
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.chmod(part, 0o700, dir_fd=parent_fd, follow_symlinks=False)
                child_fd = _open_private_directory(
                    part,
                    parent_fd=parent_fd,
                    normalize_mode=created,
                )
                os.fsync(parent_fd)
            opened_directories.append(child_fd)
            parent_fd = child_fd
        parent_metadata = os.fstat(parent_fd)
        output_name = relative.name
        try:
            existing = os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_uid != os.geteuid()
        ):
            raise ValueError

        temporary_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(128):
            candidate = f".{output_name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_fd = os.open(
                    candidate,
                    temporary_flags,
                    0o600,
                    dir_fd=parent_fd,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        if temporary_fd is None or temporary_name is None:
            raise ValueError
        os.fchmod(temporary_fd, 0o600)
        temporary_file = os.fdopen(temporary_fd, mode="w", encoding="utf-8")
        temporary_fd = None
        with temporary_file as temporary:
            json.dump(
                payload,
                temporary,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_metadata = os.fstat(temporary.fileno())
        if temporary_metadata is None:
            raise ValueError

        if not _private_binding_matches(
            root_path,
            root_fd,
            opened_root,
            parent_parts,
            parent_metadata,
        ):
            raise ValueError
        os.replace(
            temporary_name,
            output_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        if not (
            _private_binding_matches(
                root_path,
                root_fd,
                opened_root,
                parent_parts,
                parent_metadata,
            )
            and _private_leaf_matches(parent_fd, output_name, temporary_metadata)
        ):
            _discard_expected_leaf(parent_fd, output_name, temporary_metadata)
            raise ValueError
        os.fsync(parent_fd)
        if not (
            _private_binding_matches(
                root_path,
                root_fd,
                opened_root,
                parent_parts,
                parent_metadata,
            )
            and _private_leaf_matches(parent_fd, output_name, temporary_metadata)
        ):
            _discard_expected_leaf(parent_fd, output_name, temporary_metadata)
            raise ValueError
    except BaseException:
        if temporary_name is not None and opened_directories:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=opened_directories[-1])
        raise ValueError("private evidence path is invalid") from None
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


__all__ = ["write_private_json", "write_private_json_beneath"]
