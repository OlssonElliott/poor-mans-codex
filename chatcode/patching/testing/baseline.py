"""Repository snapshots and verified baseline cache for patch validation."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ...context_state import _collect_file_hashes
from ...git_utils import get_branch, get_status, run_git
from ...test_runner import TestResult
from ...workspace import (
    atomic_write_text,
    get_repo_workspace,
    get_verified_baseline_cache_file,
)


VERIFIED_BASELINE_CACHE_VERSION = 1
VERIFIED_BASELINE_MAX_AGE_SECONDS = 30 * 60


@dataclass(frozen=True)
class RepositorySnapshot:
    """The full working state that a speculative baseline represents."""
    branch: str
    file_hashes: tuple[tuple[str, str], ...]
    status: str
    staged_diff_sha256: str


def capture_repository_snapshot(repo: Path) -> RepositorySnapshot:
    staged = run_git("diff", "--cached", "--binary", "--no-ext-diff", cwd=repo)
    return RepositorySnapshot(
        branch=get_branch(repo),
        file_hashes=tuple(sorted(_collect_file_hashes(repo).items())),
        status=get_status(repo),
        staged_diff_sha256=hashlib.sha256(staged.encode("utf-8")).hexdigest(),
    )


def test_config_identity(repo: Path) -> str | None:
    try:
        from ...test_runner import detect_test_command
        return detect_test_command(repo).display
    except Exception:
        return None


def snapshot_payload(snapshot: RepositorySnapshot) -> dict:
    return {
        "branch": snapshot.branch,
        "file_hashes": [list(item) for item in snapshot.file_hashes],
        "status": snapshot.status,
        "staged_diff_sha256": snapshot.staged_diff_sha256,
    }


def load_verified_baseline(
    repo: Path,
    snapshot: RepositorySnapshot,
    *,
    test_config_identity_fn: Callable[[Path], str | None] = test_config_identity,
):
    try:
        payload = json.loads(
            get_verified_baseline_cache_file(repo).read_text(encoding="utf-8")
        )
        if payload.get("version") != VERIFIED_BASELINE_CACHE_VERSION:
            return None
        if time.time() - float(payload["created_at"]) > VERIFIED_BASELINE_MAX_AGE_SECONDS:
            return None
        if payload.get("repository") != str(repo.resolve()):
            return None
        if payload.get("snapshot") != snapshot_payload(snapshot):
            return None
        if payload.get("test_config") != test_config_identity_fn(repo):
            return None
        report = Path(payload["report"])
        if not report.is_file():
            return None
        result = payload["result"]
        return TestResult(
            str(result["command"]),
            int(result["returncode"]),
            float(result["duration_seconds"]),
            report,
            frozenset(result["failed_tests"]),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def save_verified_baseline(
    repo: Path,
    snapshot: RepositorySnapshot,
    result,
    *,
    test_config_identity_fn: Callable[[Path], str | None] = test_config_identity,
) -> None:
    """Persist only a completed full-suite result for this exact state."""
    if test_config_identity_fn(repo) is None:
        return
    report = get_repo_workspace(repo) / "verified-baseline.md"
    try:
        shutil.copyfile(result.output_file, report)
    except OSError:
        return
    payload = {
        "version": VERIFIED_BASELINE_CACHE_VERSION,
        "created_at": time.time(),
        "repository": str(repo.resolve()),
        "snapshot": snapshot_payload(snapshot),
        "test_config": test_config_identity_fn(repo),
        "report": str(report),
        "result": {
            "command": result.command,
            "returncode": result.returncode,
            "duration_seconds": result.duration_seconds,
            "failed_tests": sorted(result.failed_tests),
        },
    }
    atomic_write_text(
        get_verified_baseline_cache_file(repo),
        json.dumps(payload, indent=2) + "\n",
    )
