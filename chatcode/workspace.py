from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path


TEMPORARY_REPOSITORY_NAME = re.compile(r"tmp[a-z0-9_]{8}\Z")
# Workspaces for repositories created by tempfile are test/tooling artifacts.
# Keep them long enough for an interrupted command to be inspected, but do not
# let ordinary test runs accumulate a week's worth of abandoned workspaces.
TEMPORARY_WORKSPACE_MAX_AGE_SECONDS = 24 * 60 * 60


def get_workspace_root() -> Path:
    configured_root = os.environ.get(
        "CHATCODE_WORKSPACE_ROOT"
    )

    if configured_root:
        workspace_root = (
            Path(configured_root)
            .expanduser()
            .resolve()
        )
    else:
        chatcode_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        workspace_root = (
            chatcode_root / "workspace"
        )

    workspace_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return workspace_root


def cleanup_stale_temporary_workspaces(
    workspace_root: Path,
    *,
    active_workspace: Path | None = None,
    now: float | None = None,
) -> None:
    """Remove old workspaces created for Python-style temporary repositories.

    This deliberately ignores all normal repository names.  A generous age
    limit also avoids disturbing a long-running command, while the active
    workspace is explicitly excluded regardless of its age.
    """
    workspace_root = workspace_root.resolve()
    active_workspace = (
        active_workspace.resolve() if active_workspace is not None else None
    )
    cutoff = (
        time.time() if now is None else now
    ) - TEMPORARY_WORKSPACE_MAX_AGE_SECONDS

    try:
        candidates = list(workspace_root.iterdir())
    except OSError:
        return

    for candidate in candidates:
        if (
            not candidate.is_dir()
            or candidate.is_symlink()
            or not TEMPORARY_REPOSITORY_NAME.fullmatch(candidate.name)
        ):
            continue

        try:
            resolved_candidate = candidate.resolve()
            if resolved_candidate.parent != workspace_root:
                continue
            if (
                active_workspace is not None
                and active_workspace.is_relative_to(resolved_candidate)
            ):
                continue
            if candidate.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(candidate)
        except OSError:
            # Cleanup is opportunistic: a locked or concurrently used folder
            # is left for a future invocation.
            continue


def atomic_write_text(
    destination: Path,
    content: str,
    *,
    newline: str | None = None,
) -> None:
    """Publish generated context only after its complete content is on disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline=newline) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def get_repo_workspace(
    repo: Path,
) -> Path:
    repo = repo.resolve()

    repo_name = repo.name

    repo_hash = hashlib.sha256(
        str(repo)
        .lower()
        .encode("utf-8")
    ).hexdigest()[:8]

    workspace_root = get_workspace_root()
    workspace = (
        workspace_root
        / repo_name
        / f"_{repo_hash}"
    )

    workspace.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleanup_stale_temporary_workspaces(
        workspace_root,
        active_workspace=workspace,
    )

    return workspace


def get_repo_patches_dir(
    repo: Path,
) -> Path:
    patches_dir = (
        get_repo_workspace(repo)
        / "patches"
    )

    patches_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return patches_dir


def get_default_patch_file(
    repo: Path,
) -> Path:
    return (
        get_repo_patches_dir(repo)
        / "incoming.diff"
    )


def get_repair_context_file(
    repo: Path,
) -> Path:
    return (
        get_repo_workspace(repo)
        / "PATCH_REPAIR_CONTEXT.md"
    )


def get_existing_failure_context_file(
    repo: Path,
) -> Path:
    """A new-task context for debt discovered during patch validation."""
    return get_repo_workspace(repo) / "EXISTING_FAILURE_CONTEXT.md"


def get_followup_context_file(repo: Path) -> Path:
    """A fresh task when validation passed but the user reports failure."""
    return get_repo_workspace(repo) / "FOLLOWUP_CONTEXT.md"


def get_followup_state_file(repo: Path) -> Path:
    """Structured lifecycle state for the repository's unresolved follow-up."""
    return get_repo_workspace(repo) / "followup-state.json"


def get_check_repair_context_file(
    repo: Path,
) -> Path:
    """A standalone health-check repair task, never tied to a patch."""
    return get_repo_workspace(repo) / "CHECK_REPAIR_CONTEXT.md"


def get_verified_baseline_cache_file(repo: Path) -> Path:
    return get_repo_workspace(repo) / "verified-baseline.json"


def get_test_results_dir(
    repo: Path,
) -> Path:
    results_dir = (
        get_repo_workspace(repo)
        / "test-results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return results_dir


def get_test_results_file(
    repo: Path,
) -> Path:
    return (
        get_test_results_dir(repo)
        / "latest.md"
    )


def get_history_dir(
    repo: Path,
) -> Path:
    history_dir = (
        get_repo_workspace(repo)
        / "history"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return history_dir


def get_applied_history_dir(
    repo: Path,
) -> Path:
    applied_dir = (
        get_history_dir(repo)
        / "applied"
    )

    applied_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return applied_dir


def get_undone_history_dir(
    repo: Path,
) -> Path:
    undone_dir = (
        get_history_dir(repo)
        / "undone"
    )

    undone_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return undone_dir
