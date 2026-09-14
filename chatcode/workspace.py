from __future__ import annotations

import hashlib
from pathlib import Path


def get_workspace_root() -> Path:
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

    workspace = (
        get_workspace_root()
        / repo_name
        / f"_{repo_hash}"
    )

    workspace.mkdir(
        parents=True,
        exist_ok=True,
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
