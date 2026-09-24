"""Sensitive-data redaction and safe repository diagnostics."""
from __future__ import annotations

import re
from pathlib import Path

from ..file_filter import is_ignored
from ..git_utils import (
    get_staged_changed_paths,
    get_staged_diff,
    get_status,
    get_unstaged_changed_paths,
    get_unstaged_diff,
)
from ..workspace import get_test_results_file
from .files import _is_context_internal_artifact


MAX_TEST_RESULT_CHARS = 60_000


PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?"
    r"-----END [^-]*PRIVATE KEY-----",
    re.DOTALL,
)


KNOWN_SECRET_PATTERNS = [
    re.compile(
        r"\bsk-[A-Za-z0-9_-]{20,}\b"
    ),
    re.compile(
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"
    ),
    re.compile(
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"
    ),
    re.compile(
        r"\bAKIA[0-9A-Z]{16}\b"
    ),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}"
        r"\.[A-Za-z0-9_-]{10,}"
        r"\.[A-Za-z0-9_-]{10,}\b"
    ),
]


QUOTED_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (
        \b
        (?:
            password
            | passwd
            | api[_-]?key
            | secret
            | token
            | access[_-]?token
            | refresh[_-]?token
            | client[_-]?secret
        )
        \b
        \s*
        [:=]
        \s*
        ["']
    )
    [^"'\r\n]{6,}
    (["'])
    """
)


UNQUOTED_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (
        \b
        (?:
            password
            | passwd
            | api[_-]?key
            | secret
            | token
            | access[_-]?token
            | refresh[_-]?token
            | client[_-]?secret
        )
        \b
        \s*
        =
        \s*
    )
    [^\s,;\#\r\n]{8,}
    """
)


BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)(Authorization\s*:\s*Bearer\s+)"
    r"[A-Za-z0-9._~+/=-]{8,}"
)


def build_safe_status(
    repo: Path,
) -> str:
    status = get_status(repo)

    if not status:
        return ""

    safe_lines: list[str] = []
    omitted_count = 0

    for line in status.splitlines():
        if len(line) < 4:
            continue

        raw_path = line[3:].strip()

        paths = [raw_path]

        if " -> " in raw_path:
            paths = raw_path.split(" -> ", 1)

        should_omit = False

        for status_path in paths:
            path = repo / status_path.strip()

            if (
                is_ignored(path, repo)
                or _is_context_internal_artifact(path, repo)
            ):
                should_omit = True
                break

        if should_omit:
            omitted_count += 1
            continue

        safe_lines.append(line)

    if omitted_count:
        safe_lines.append(
            "[ChatCode omitted "
            f"{omitted_count} sensitive or ignored "
            "file(s) from Git status.]"
        )

    return "\n".join(safe_lines)


def redact_sensitive_text(text: str) -> str:
    text = PRIVATE_KEY_PATTERN.sub(
        "[REDACTED PRIVATE KEY]",
        text,
    )

    for pattern in KNOWN_SECRET_PATTERNS:
        text = pattern.sub(
            "[REDACTED SECRET]",
            text,
        )

    text = QUOTED_SECRET_ASSIGNMENT.sub(
        r"\1[REDACTED]\2",
        text,
    )

    text = UNQUOTED_SECRET_ASSIGNMENT.sub(
        r"\1[REDACTED]",
        text,
    )

    text = BEARER_TOKEN_PATTERN.sub(
        r"\1[REDACTED]",
        text,
    )

    return text


def build_safe_diff(
    repo: Path,
    staged: bool,
) -> str:
    if staged:
        changed_paths = (
            get_staged_changed_paths(repo)
        )
    else:
        changed_paths = (
            get_unstaged_changed_paths(repo)
        )

    safe_paths: list[str] = []
    omitted_count = 0

    for raw_path in changed_paths:
        path = repo / raw_path

        if (
            is_ignored(path, repo)
            or _is_context_internal_artifact(path, repo)
        ):
            omitted_count += 1
            continue

        safe_paths.append(raw_path)

    if staged:
        diff = get_staged_diff(
            repo,
            safe_paths,
        )
    else:
        diff = get_unstaged_diff(
            repo,
            safe_paths,
        )

    if diff:
        diff = redact_sensitive_text(
            diff
        )

    if omitted_count:
        note = (
            "[ChatCode omitted "
            f"{omitted_count} sensitive or "
            "ignored file(s) from this diff.]"
        )

        if diff:
            diff = (
                f"{diff}\n\n{note}"
            )
        else:
            diff = note

    return diff


def build_failed_test_context(
    repo: Path,
) -> str:
    test_results_file = (
        get_test_results_file(repo)
    )

    if not test_results_file.exists():
        return ""

    try:
        content = test_results_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""

    if "Status: FAILED" not in content:
        return ""

    content = redact_sensitive_text(
        content
    )

    if len(content) > MAX_TEST_RESULT_CHARS:
        content = (
            "[Earlier test output truncated "
            "by ChatCode]\n\n"
            + content[-MAX_TEST_RESULT_CHARS:]
        )

    return content
