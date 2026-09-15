from __future__ import annotations

from pathlib import Path

from ..file_filter import is_ignored, iter_repository_files

INDEXABLE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".php"
}


def is_indexable(path: Path, repo: Path) -> bool:
    return not is_ignored(path, repo) and path.suffix.lower() in INDEXABLE_SUFFIXES


def scan_project(repo: Path) -> list[Path]:
    return [path for path in iter_repository_files(repo) if is_indexable(path, repo)]
