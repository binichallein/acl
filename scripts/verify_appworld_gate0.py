#!/usr/bin/env python3
"""Run and privately persist the aggregate-only AppWorld Gate 0 smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from toolshift.benchmarks.appworld_gate0 import (
    _appworld_gate0_preflight_failure_summary,
    _require_private_appworld_root,
    run_appworld_gate0_smoke,
    write_appworld_gate0_summary,
)


class _PrivateArgumentParser(argparse.ArgumentParser):
    """Fail argument parsing without echoing caller-controlled values."""

    def error(self, message: str) -> None:
        del message
        self.exit(2)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        del message
        raise SystemExit(status)


def _argument_parser() -> argparse.ArgumentParser:
    parser = _PrivateArgumentParser(
        description="Run the private aggregate-only AppWorld Gate 0 smoke."
    )
    parser.add_argument("--output", required=True)
    return parser


def _validated_output(value: str, private_root: Path) -> Path:
    error = "output must be an absolute path under the private AppWorld root"
    try:
        output = Path(value)
        resolved_root = private_root.resolve(strict=True)
        resolved_output = output.resolve(strict=False)
        if (
            not output.is_absolute()
            or output == private_root
            or output.is_symlink()
            or not resolved_output.is_relative_to(resolved_root)
        ):
            raise ValueError
        return output
    except BaseException:
        raise ValueError(error) from None


def main(argv: list[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    try:
        private_root = _require_private_appworld_root()
        output = _validated_output(arguments.output, private_root)
    except (RuntimeError, ValueError):
        return 2
    try:
        summary = run_appworld_gate0_smoke()
    except (RuntimeError, ValueError):
        summary = _appworld_gate0_preflight_failure_summary()
    try:
        write_appworld_gate0_summary(
            output,
            summary,
        )
    except (RuntimeError, ValueError):
        return 2
    return 0 if summary.smoke_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
