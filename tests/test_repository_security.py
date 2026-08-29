from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATTERNS = {
    "private key": re.compile(
        r"-----BEGIN (?:[A-Z0-9]+(?: [A-Z0-9]+)* )?PRIVATE KEY-----"
    ),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "GitHub fine-grained PAT": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenAI token": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "credential assignment": re.compile(
        r"(?:^|[{,])\s*(?:(?:export|ENV|ARG)\s+)?[\"']?"
        r"(?:[A-Z][A-Z0-9]*_)*(?:API_KEY|TOKEN|SECRET_ACCESS_KEY|SECRET_KEY|PASSWORD)"
        r"[\"']?\s*[:=]\s*[\"']?[^\s#\"']+",
        re.IGNORECASE | re.MULTILINE,
    ),
    "credential in URL": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "proxy assignment": re.compile(
        r"^\s*(?:(?:export|ENV|ARG)\s+)?"
        r"(?:http_proxy|https_proxy|all_proxy|no_proxy)\s*[:=]",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def _find_forbidden_content(content: str) -> list[str]:
    return [label for label, pattern in FORBIDDEN_PATTERNS.items() if pattern.search(content)]


@pytest.mark.parametrize(
    ("content", "expected_label"),
    [
        pytest.param(
            "OPENAI_API" + "_KEY=" + "sk-" + "proj_" + "v1-" + "a" * 24,
            "OpenAI token",
            id="modern-openai-key",
        ),
        pytest.param(
            "AWS_SECRET" + "_ACCESS_KEY=" + "b" * 40,
            "credential assignment",
            id="aws-secret-key",
        ),
        pytest.param(
            "github_" + "pat_" + "c" * 24 + "_" + "d" * 32,
            "GitHub fine-grained PAT",
            id="fine-grained-github-pat",
        ),
        pytest.param(
            "{" + '"' + "AWS_SECRET" + '_ACCESS_KEY": "' + "e" * 40 + '"}',
            "credential assignment",
            id="json-object-credential-assignment",
        ),
        pytest.param(
            '{"safe": "value", "' + "VENDOR_API" + '_TOKEN": "' + "f" * 40 + '"}',
            "credential assignment",
            id="json-later-credential-assignment",
        ),
        pytest.param(
            "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----",
            "private key",
            id="encrypted-private-key",
        ),
        pytest.param(
            "-----BEGIN " + "PRIVATE KEY-----",
            "private key",
            id="generic-private-key",
        ),
        pytest.param(
            "-----BEGIN " + "RSA PRIVATE KEY-----",
            "private key",
            id="traditional-private-key",
        ),
        pytest.param(
            "ENV HTTP" + "S_PROXY=http://127.0.0.1:7890",
            "proxy assignment",
            id="docker-env-proxy",
        ),
        pytest.param(
            "ARG VENDOR_API" + "_KEY=temporary-value",
            "credential assignment",
            id="docker-arg-vendor-key",
        ),
        pytest.param(
            "export ACME_" + "TOKEN=temporary-value",
            "credential assignment",
            id="export-vendor-token",
        ),
        pytest.param(
            "CUSTOM_SECRET_ACCESS" + "_KEY: temporary-value",
            "credential assignment",
            id="yaml-vendor-secret-key",
        ),
    ],
)
def test_forbidden_content_examples_are_detected(content: str, expected_label: str) -> None:
    assert expected_label in _find_forbidden_content(content)


def _candidate_text_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [REPOSITORY_ROOT / relative_path for relative_path in output.splitlines()]


def test_repository_text_files_do_not_contain_secrets_or_proxy_assignments() -> None:
    violations: list[str] = []
    for path in _candidate_text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label in _find_forbidden_content(content):
            violations.append(f"{path.relative_to(REPOSITORY_ROOT)}: {label}")

    assert violations == []


def test_env_example_contains_only_shared_root_placeholder() -> None:
    env_example = REPOSITORY_ROOT / ".env.example"
    assert env_example.is_file(), ".env.example must exist"

    assert env_example.read_text(encoding="utf-8").splitlines() == [
        "TOOLSHIFT_SHARED_ROOT=/path/on/ml2/shared/filesystem"
    ]
