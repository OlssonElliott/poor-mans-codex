"""Refresh a saved test-failure repair context from current repository state."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from ..models import ApplyResult
from ...history import (
    HistoryError,
    get_history_patch_file,
    get_latest_applied_entry,
)
from ...test_runner import TestResult, _failed_test_ids
from ...workspace import get_repair_context_file, get_test_results_dir


def test_result_from_saved_report(path: Path):
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = dict(re.findall(
        r"^(Status|Exit code|Command|Duration):\s*(.+)$",
        content,
        flags=re.MULTILINE,
    ))
    try:
        returncode = int(fields["Exit code"])
        duration = float(fields.get("Duration", "0").split()[0])
        command = fields["Command"]
    except (KeyError, ValueError):
        return None

    return TestResult(
        command=command,
        returncode=returncode,
        duration_seconds=duration,
        output_file=path,
        failed_tests=_failed_test_ids(content, ""),
    )


def refresh_test_failure_repair_context(
    repo: Path,
    *,
    classify_test_validation_fn: Callable,
    extract_patch_paths_fn: Callable[[str], set[str]],
    build_test_failure_repair_context_fn: Callable,
) -> Path | None:
    """Rebuild a saved test-failure context from current files on demand."""
    repair_context = get_repair_context_file(repo)
    try:
        if not repair_context.read_text(
            encoding="utf-8",
            errors="replace",
        ).startswith("# ChatCode Test Failure Repair Context"):
            return None
        history_entry = get_latest_applied_entry(repo)
        patch_text = get_history_patch_file(history_entry).read_text(
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, HistoryError):
        return None

    reports_dir = get_test_results_dir(repo)
    baseline = test_result_from_saved_report(reports_dir / "baseline.md")
    targeted = test_result_from_saved_report(reports_dir / "targeted.md")
    full = test_result_from_saved_report(reports_dir / "full-suite.md")
    if baseline is None or full is None:
        return None

    validation = classify_test_validation_fn(baseline, targeted, full)
    result = ApplyResult(
        paths=extract_patch_paths_fn(patch_text),
        history_entry=history_entry,
    )
    return build_test_failure_repair_context_fn(repo, result, validation)
