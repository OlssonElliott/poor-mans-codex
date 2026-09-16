from __future__ import annotations

from pathlib import Path

from ..file_filter import is_ignored, iter_repository_files
from ..workspace import get_workspace_root

INDEXABLE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".php"
}


def is_indexable(path: Path, repo: Path) -> bool:
    return (
        not is_ignored(path, repo)
        and not _is_repo_local_workspace_path(path, repo)
        and path.suffix.lower() in INDEXABLE_SUFFIXES
    )


def _is_repo_local_workspace_path(path: Path, repo: Path) -> bool:
    try:
        workspace = get_workspace_root().resolve()
        workspace.relative_to(repo.resolve())
        path.resolve().relative_to(workspace)
        return True
    except (OSError, ValueError):
        return False


def scan_project(repo: Path) -> list[Path]:
    return [path for path in iter_repository_files(repo) if is_indexable(path, repo)]
