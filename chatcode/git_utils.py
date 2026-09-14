from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _run_git(
    *args: str,
    cwd: Path | None = None,
    strip: bool = True,
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        message = result.stderr.strip() or "Git command failed."
        raise GitError(message)

    if strip:
        return result.stdout.strip()

    return result.stdout.rstrip("\r\n")


def run_git(*args: str, cwd: Path | None = None) -> str:
    return _run_git(*args, cwd=cwd)


def get_repo_root() -> Path:
    try:
        root = run_git(
            "rev-parse",
            "--show-toplevel",
        )
    except GitError as exc:
        raise GitError(
            "Du verkar inte stå i ett Git-repository."
        ) from exc

    return Path(root).resolve()


def get_branch(repo: Path) -> str:
    branch = run_git(
        "branch",
        "--show-current",
        cwd=repo,
    )

    return branch or "(detached HEAD)"


def get_status(repo: Path) -> str:
    return _run_git(
        "status",
        "--short",
        cwd=repo,
        strip=False,
    )


def _get_changed_paths(
    repo: Path,
    staged: bool,
) -> list[str]:
    args = ["diff"]

    if staged:
        args.append("--cached")

    args.extend([
        "--name-only",
        "-z",
        "--no-renames",
        "--",
    ])

    output = run_git(
        *args,
        cwd=repo,
    )

    if not output:
        return []

    return [
        path
        for path in output.split("\0")
        if path
    ]


def get_unstaged_changed_paths(
    repo: Path,
) -> list[str]:
    return _get_changed_paths(
        repo,
        staged=False,
    )


def get_staged_changed_paths(
    repo: Path,
) -> list[str]:
    return _get_changed_paths(
        repo,
        staged=True,
    )


def get_untracked_paths(
    repo: Path,
) -> list[str]:
    output = run_git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        cwd=repo,
    )

    if not output:
        return []

    return [
        path
        for path in output.split("\0")
        if path
    ]


def _get_diff(
    repo: Path,
    paths: list[str],
    staged: bool,
) -> str:
    if not paths:
        return ""

    args = ["diff"]

    if staged:
        args.append("--cached")

    args.append("--no-renames")
    args.append("--")
    args.extend(paths)

    return run_git(
        *args,
        cwd=repo,
    )


def get_unstaged_diff(
    repo: Path,
    paths: list[str],
) -> str:
    return _get_diff(
        repo,
        paths,
        staged=False,
    )


def get_staged_diff(
    repo: Path,
    paths: list[str],
) -> str:
    return _get_diff(
        repo,
        paths,
        staged=True,
    )
