"""Bridge failed background test runs into repair contexts."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..errors import PatchError
from ..models import ApplyResult, TestValidation
from ...history import HistoryError, get_history_patch_file, get_latest_applied_entry
from ...test_runner import TestResult
from ...workspace import get_applied_history_dir, get_repair_context_file, get_test_results_dir


def ensure_background_failure_repair_context(
    repo: Path,
    background: dict,
    *,
    get_stale_reason_fn: Callable[[Path], str | None],
    snapshot_from_payload_fn: Callable[[dict], object],
    capture_snapshot_fn: Callable[[Path], object],
    extract_patch_paths_fn: Callable[[str], set[str]],
    build_test_failure_context_fn: Callable[[Path, ApplyResult, TestValidation], Path],
    write_background_status_fn: Callable[..., None],
) -> tuple[Path, bool]:
    """Create one safe repair artifact for the current failed background run."""
    try:
        returncode = int(background.get("returncode", 0))
    except (TypeError, ValueError) as exc:
        raise PatchError("The background suite result is invalid.") from exc
    if background.get("state") != "completed" or returncode == 0:
        raise PatchError("The background suite did not fail.")

    failures = frozenset(
        str(item) for item in background.get("failed_tests", []) if item
    )
    if not failures:
        raise PatchError(
            "The background suite failed without stable test identifiers."
        )

    expected_context = get_repair_context_file(repo).resolve()
    recorded_context = background.get("repair_context")
    if recorded_context:
        candidate = Path(str(recorded_context)).resolve()
        if (
            candidate == expected_context
            and candidate.is_file()
            and get_stale_reason_fn(repo) is None
        ):
            return candidate, False

    try:
        tested_snapshot = snapshot_from_payload_fn(background["snapshot"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PatchError(
            "The background result has no valid repository snapshot."
        ) from exc
    if capture_snapshot_fn(repo) != tested_snapshot:
        raise PatchError(
            "The working tree changed after the failed background suite; "
            "a repair context was not generated from stale test output."
        )

    application = background.get("application")
    entry: Path
    recorded_paths: set[str] | None = None
    if isinstance(application, dict):
        entry_name = str(application.get("history_entry", ""))
        if not entry_name or Path(entry_name).name != entry_name:
            raise PatchError("The background patch history reference is invalid.")
        entry = get_applied_history_dir(repo) / entry_name
        raw_paths = application.get("paths", [])
        if not isinstance(raw_paths, list):
            raise PatchError("The background patch path list is invalid.")
        recorded_paths = {str(path) for path in raw_paths}
    else:
        try:
            entry = get_latest_applied_entry(repo)
        except HistoryError as exc:
            raise PatchError(
                "No applied ChatCode patch could be associated with this run."
            ) from exc

    applied_dir = get_applied_history_dir(repo).resolve()
    try:
        resolved_entry = entry.resolve()
    except OSError as exc:
        raise PatchError("The applied patch history could not be resolved.") from exc
    if resolved_entry.parent != applied_dir or not resolved_entry.exists():
        raise PatchError("The background patch history entry is unavailable.")

    try:
        paths = extract_patch_paths_fn(
            get_history_patch_file(resolved_entry).read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except OSError as exc:
        raise PatchError("The applied patch history could not be read.") from exc
    if not paths or (recorded_paths is not None and recorded_paths != paths):
        raise PatchError(
            "The background run does not match its applied patch history."
        )

    report = Path(str(background.get("report", "")))
    try:
        resolved_report = report.resolve()
    except OSError as exc:
        raise PatchError("The background test report could not be resolved.") from exc
    if (
        resolved_report.parent != get_test_results_dir(repo).resolve()
        or not resolved_report.is_file()
    ):
        raise PatchError("The background test report is unavailable.")

    full_result = TestResult(
        command=str(background.get("command", "unknown test command")),
        returncode=returncode,
        duration_seconds=float(background.get("duration_seconds", 0.0)),
        output_file=resolved_report,
        failed_tests=failures,
    )
    validation = TestValidation(
        baseline=None,
        targeted=None,
        full=full_result,
        status="unclassified_failure",
    )
    context = build_test_failure_context_fn(
        repo,
        ApplyResult(paths=paths, history_entry=resolved_entry),
        validation,
    )
    updated = dict(background)
    updated["repair_context"] = str(context)
    write_background_status_fn(repo, updated, run_file=True)
    return context, True
