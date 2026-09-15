from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path


# Shared by context retrieval and project indexing. These are directory names,
# not path fragments, so the rule also prunes nested runtimes and dependencies.
IGNORED_DIRS = {
    ".git",
    ".chatcode",
    ".runtime",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".hypothesis",
    ".eggs",
    "site-packages",
    "coverage",
    ".coverage",
    "htmlcov",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    "vendor",
    ".vite",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".parcel-cache",
    ".turbo",
    ".cache",
    ".gradle",
    ".m2",
    ".npm",
    ".yarn",
    ".pnpm-store",
    "bower_components",
    "tmp",
    "temp",
    ".idea",
    ".vscode",
    ".aws",
    ".ssh",
}

SECRET_FILENAMES = {
    "credentials.json",
    "secrets.json",
    "service-account.json",
    "serviceaccount.json",
    ".npmrc",
    ".pypirc",
}

SECRET_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
}


def is_excluded_directory_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in IGNORED_DIRS or lowered.endswith(".egg-info")


def is_secret(path: Path) -> bool:
    name = path.name.lower()
    return (
        name == ".env"
        or name.startswith(".env.")
        or name in SECRET_FILENAMES
        or path.suffix.lower() in SECRET_SUFFIXES
    )


def is_ignored(path: Path, repo: Path) -> bool:
    try:
        relative = path.relative_to(repo)
    except ValueError:
        return True
    # A file symlink is still yielded by os.walk even with followlinks=False.
    # Never let a repository-relative link read content outside the repository.
    try:
        path.resolve().relative_to(repo.resolve())
    except (OSError, ValueError):
        return True
    return any(is_excluded_directory_name(part) for part in relative.parts) or is_secret(path)


def iter_repository_files(repo: Path) -> Iterator[Path]:
    """Yield eligible repository files while pruning ignored trees early."""
    for root, directories, filenames in os.walk(repo, topdown=True, followlinks=False):
        root_path = Path(root)
        directories[:] = sorted(
            name
            for name in directories
            if not is_excluded_directory_name(name)
            and not is_ignored(root_path / name, repo)
        )
        for filename in sorted(filenames):
            path = root_path / filename
            if not is_ignored(path, repo):
                yield path
