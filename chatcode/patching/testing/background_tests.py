"""Detached full-suite execution and status persistence."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from .baseline import RepositorySnapshot
from ...workspace import atomic_write_text, get_repo_workspace, get_test_results_dir


BACKGROUND_FULL_SUITE_STATUS = "background-full-suite.json"
BACKGROUND_RUN_ARTIFACT = re.compile(
    r"background-full-suite-[0-9a-f]{32}\.(?:json|md)\Z"
)


def status_file(repo: Path) -> Path:
    return get_repo_workspace(repo) / BACKGROUND_FULL_SUITE_STATUS


def run_file(repo: Path, run_id: str) -> Path:
    return get_repo_workspace(repo) / f"background-full-suite-{run_id}.json"


def report_file(repo: Path, run_id: str) -> Path:
    return get_test_results_dir(repo) / f"background-full-suite-{run_id}.md"


def cleanup_artifacts(repo: Path, *, keep_run_id: str) -> None:
    """Remove obsolete run-owned artifacts without touching other files."""
    locations = (get_repo_workspace(repo), get_test_results_dir(repo))
    keep_names = {
        f"background-full-suite-{keep_run_id}.json",
        f"background-full-suite-{keep_run_id}.md",
    }
    for location in locations:
        try:
            resolved_location = location.resolve()
            candidates = list(location.iterdir())
        except OSError:
            continue
        for candidate in candidates:
            if (
                candidate.name in keep_names
                or not BACKGROUND_RUN_ARTIFACT.fullmatch(candidate.name)
            ):
                continue
            try:
                resolved = candidate.resolve()
                if resolved.parent != resolved_location or not candidate.is_file():
                    continue
                candidate.unlink()
            except OSError:
                continue

    try:
        (get_test_results_dir(repo) / "background-full-suite.md").unlink(
            missing_ok=True
        )
    except OSError:
        pass


def write_status(repo: Path, payload: dict, *, run_file_status: bool = False) -> None:
    destination = (
        run_file(repo, str(payload["run_id"]))
        if run_file_status
        else status_file(repo)
    )
    atomic_write_text(destination, json.dumps(payload, indent=2) + "\n")


def get_status(repo: Path) -> dict | None:
    """Return the current run status without accepting an older worker."""
    try:
        pointer = json.loads(status_file(repo).read_text(encoding="utf-8"))
        run_id = str(pointer["run_id"])
        run_path = run_file(repo, run_id)
        if run_path.is_file():
            status = json.loads(run_path.read_text(encoding="utf-8"))
            if status.get("run_id") == run_id:
                return status
        return pointer
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def snapshot_from_payload(payload: dict) -> RepositorySnapshot:
    return RepositorySnapshot(
        branch=str(payload["branch"]),
        file_hashes=tuple(
            (str(path), str(digest))
            for path, digest in payload["file_hashes"]
        ),
        status=str(payload["status"]),
        staged_diff_sha256=str(payload["staged_diff_sha256"]),
    )


def complete_full_suite(
    repo_name: str,
    snapshot_payload: dict,
    run_id: str,
    application_payload: dict | None = None,
    *,
    capture_snapshot_fn: Callable[[Path], RepositorySnapshot],
    save_baseline_fn: Callable,
    get_status_fn: Callable[[Path], dict | None] = get_status,
    cleanup_fn: Callable[..., None] = cleanup_artifacts,
) -> None:
    """Worker entry point for the detached post-apply full test suite."""
    repo = Path(repo_name)
    snapshot = snapshot_from_payload(snapshot_payload)
    report = report_file(repo, run_id)
    is_current = False
    try:
        from ...test_runner import run_project_tests

        result = run_project_tests(repo, output_file=report)
        current_snapshot = capture_snapshot_fn(repo)
        current_status = get_status_fn(repo)
        is_current = (
            current_status is not None
            and current_status.get("run_id") == run_id
        )
        if is_current and current_snapshot == snapshot:
            save_baseline_fn(repo, snapshot, result)
        status = {
            "run_id": run_id,
            "state": "completed",
            "snapshot": snapshot_payload,
            "returncode": result.returncode,
            "report": str(result.output_file),
            "command": result.command,
            "duration_seconds": result.duration_seconds,
            "failed_tests": sorted(result.failed_tests),
            "completed_at": time.time(),
        }
    except Exception as exc:
        status = {
            "run_id": run_id,
            "state": "error",
            "snapshot": snapshot_payload,
            "error": str(exc),
            "completed_at": time.time(),
        }
    if application_payload is not None:
        status["application"] = application_payload
    write_status(repo, status, run_file_status=True)
    if is_current:
        cleanup_fn(repo, keep_run_id=run_id)


def complete_full_suite_from_status(
    repo_name: str,
    run_id: str,
    *,
    complete_fn: Callable,
) -> None:
    """Load the large snapshot from disk so Windows argv stays small."""
    repo = Path(repo_name)
    try:
        payload = json.loads(run_file(repo, run_id).read_text(encoding="utf-8"))
        if payload.get("run_id") != run_id or payload.get("state") != "running":
            return
        snapshot_payload = payload["snapshot"]
        if not isinstance(snapshot_payload, dict):
            raise TypeError("background snapshot is not an object")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        write_status(
            repo,
            {
                "run_id": run_id,
                "state": "error",
                "error": f"Could not load background test state: {exc}",
                "completed_at": time.time(),
            },
            run_file_status=True,
        )
        return

    application_payload = payload.get("application")
    if isinstance(application_payload, dict):
        complete_fn(repo_name, snapshot_payload, run_id, application_payload)
    else:
        complete_fn(repo_name, snapshot_payload, run_id)


def start_full_suite(
    repo: Path,
    snapshot: RepositorySnapshot,
    application=None,
    *,
    snapshot_payload_fn: Callable[[RepositorySnapshot], dict],
    cleanup_fn: Callable[..., None] = cleanup_artifacts,
    popen: Callable = subprocess.Popen,
) -> None:
    """Launch a full suite that can outlive the current chatcode apply process."""
    run_id = uuid.uuid4().hex
    snapshot_payload = snapshot_payload_fn(snapshot)
    cleanup_fn(repo, keep_run_id=run_id)
    initial_status = {
        "run_id": run_id,
        "state": "running",
        "snapshot": snapshot_payload,
        "started_at": time.time(),
    }
    if application is not None:
        initial_status["application"] = {
            "history_entry": application.history_entry.name,
            "paths": sorted(application.paths),
        }
    write_status(repo, initial_status, run_file_status=True)
    write_status(repo, initial_status)
    worker = (
        "import sys; from chatcode.patching.testing.background_service import "
        "detached_full_suite_from_status; "
        "detached_full_suite_from_status(sys.argv[1], sys.argv[2])"
    )
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    try:
        popen(
            [sys.executable, "-c", worker, str(repo), run_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )
    except OSError:
        status_file(repo).unlink(missing_ok=True)
        run_file(repo, run_id).unlink(missing_ok=True)
        raise
