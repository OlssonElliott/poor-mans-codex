"""Repository file discovery and deterministic task-file resolution."""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from .. import workspace
from ..file_filter import is_ignored, iter_repository_files
from ..git_utils import (
    get_staged_changed_paths,
    get_unstaged_changed_paths,
    get_untracked_paths,
)
from ..indexing.project_graph import load_map, save_map


SOURCE_SUFFIXES = {
    ".py",
    ".java",
    ".kt",
    ".php",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".properties",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".md",
}


IMPORTANT_FILES = {
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "package.json",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
    "pyproject.toml",
}


def is_source_file(path: Path) -> bool:
    return (
        path.name.lower() in IMPORTANT_FILES
        or path.suffix.lower() in SOURCE_SUFFIXES
    )


def _internal_workspace_root(repo: Path) -> Path | None:
    """Return ChatCode state root when that state lives inside the target repo."""
    repo_root = repo.resolve()
    try:
        # Resolve through the module so the workspace provider remains the
        # single authority (and can be replaced by an embedding application).
        repo_workspace = workspace.get_repo_workspace(repo).resolve()
    except OSError:
        return None

    for candidate in (repo_workspace, *repo_workspace.parents):
        if candidate.parent == repo_root:
            return candidate
        if candidate == repo_root:
            break

    return None


def _is_internal_workspace_path(
    path: Path,
    repo: Path,
    workspace_root: Path | None = None,
) -> bool:
    workspace_root = workspace_root or _internal_workspace_root(repo)
    if workspace_root is None:
        return False

    try:
        path.resolve().relative_to(workspace_root)
        return True
    except (OSError, ValueError):
        return False


def _is_context_internal_artifact(
    path: Path,
    repo: Path,
) -> bool:
    """Identify ChatCode-owned state without hiding ordinary user files."""
    if _is_internal_workspace_path(path, repo):
        return True

    try:
        relative = path.resolve().relative_to(repo.resolve())
    except (OSError, ValueError):
        return False

    parts = tuple(part.lower() for part in relative.parts)
    if not parts:
        return False

    if parts[0] == ".chatcode":
        return True

    is_chatcode_source_repo = (
        (repo / "chatcode" / "context_builder.py").is_file()
        and (repo / "chatcode" / "workspace.py").is_file()
    )
    if not is_chatcode_source_repo:
        return False

    if len(parts) == 1 and parts[0] in {
        "upload_to_chatgpt.md",
        "patch_repair_context.md",
        "context.md",
        "context-state.json",
        "project-map.json",
    }:
        return True

    return parts[0] in {
        "history",
        "patches",
        "test-results",
    }


def _iter_project_files(repo: Path):
    workspace_root = _internal_workspace_root(repo)
    for path in iter_repository_files(repo):
        if _is_internal_workspace_path(path, repo, workspace_root):
            continue
        yield path


def _iter_context_files(repo: Path):
    for path in _iter_project_files(repo):
        if _is_context_internal_artifact(path, repo):
            continue
        yield path


def _task_file_tokens(task: str) -> list[str]:
    tokens = re.findall(
        r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+",
        task,
    )
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip(".,:;!?()[]{}'\"`")
        if token:
            cleaned.append(token)
    return cleaned


def _is_selectable_task_file(
    path: Path,
    repo: Path,
) -> bool:
    try:
        path.resolve().relative_to(repo.resolve())
    except (OSError, ValueError):
        return False

    return (
        path.is_file()
        and is_source_file(path)
        and not is_ignored(path, repo)
        and not _is_context_internal_artifact(path, repo)
        and not _is_internal_workspace_path(path, repo)
    )


def _resolve_task_file_references(
    repo: Path,
    task: str,
) -> list[Path]:
    """Resolve deterministic file references from task text against the repo."""
    repo_root = repo.resolve()
    resolved: list[Path] = []
    context_files: list[Path] | None = None

    def all_context_files() -> list[Path]:
        nonlocal context_files
        if context_files is None:
            context_files = [
                path.resolve()
                for path in _iter_context_files(repo)
                if _is_selectable_task_file(path, repo)
            ]
        return context_files

    for raw_token in _task_file_tokens(task):
        normalized = raw_token.replace("\\", "/")
        posix = PurePosixPath(normalized)
        parts = posix.parts

        if (
            not parts
            or posix.is_absolute()
            or ".." in parts
            or re.match(r"^[A-Za-z]:", normalized)
        ):
            continue

        candidate = repo_root.joinpath(*parts)
        if _is_selectable_task_file(candidate, repo):
            candidate = candidate.resolve()
            if candidate not in resolved:
                resolved.append(candidate)
            continue

        lowered = normalized.lower()
        matches: list[Path] = []

        if "/" in normalized:
            matches = [
                path
                for path in all_context_files()
                if path.relative_to(repo_root).as_posix().lower() == lowered
            ]
        elif "." in normalized:
            matches = [
                path
                for path in all_context_files()
                if path.name.lower() == lowered
            ]
        else:
            try:
                matches = [
                    path.resolve()
                    for path in repo_root.iterdir()
                    if _is_selectable_task_file(path, repo)
                    and (
                        path.name.lower() == lowered
                        or path.stem.lower() == lowered
                    )
                ]
            except OSError:
                matches = []

        if len(matches) == 1 and matches[0] not in resolved:
            resolved.append(matches[0])

    return resolved


def _ambiguous_task_file_paths(repo: Path, task: str) -> set[Path]:
    """Find bare filename references that cannot be resolved safely."""
    bare_filenames = [
        token.replace("\\", "/")
        for token in _task_file_tokens(task)
        if "/" not in token.replace("\\", "/") and "." in token
    ]
    if not bare_filenames:
        return set()
    candidates = [path.resolve() for path in _iter_context_files(repo)]
    ambiguous: set[Path] = set()
    for normalized in bare_filenames:
        matches = [
            path for path in candidates
            if path.name.casefold() == normalized.casefold()
        ]
        if len(matches) > 1:
            ambiguous.update(matches)
    return ambiguous


def _ensure_explicit_task_files(
    repo: Path,
    task: str,
    files: list[Path],
) -> list[Path]:
    selected = list(files)
    for path in _resolve_task_file_references(repo, task):
        if path not in selected:
            selected.append(path)
    return selected


def _purge_internal_workspace_from_index(repo: Path) -> None:
    workspace_root = _internal_workspace_root(repo)
    if workspace_root is None:
        return

    try:
        graph = load_map(repo)
    except Exception:
        return

    files = graph.get("files")
    if not isinstance(files, dict):
        return

    removed = [
        raw_path
        for raw_path in files
        if _is_internal_workspace_path(
            repo.joinpath(*PurePosixPath(raw_path).parts),
            repo,
            workspace_root,
        )
    ]
    if not removed:
        return

    for raw_path in removed:
        files.pop(raw_path, None)

    try:
        save_map(repo, graph)
    except Exception:
        pass


def _index_scope_paths(repo: Path) -> list[str] | None:
    """Prevent a repo-local ChatCode workspace from entering the project index."""
    workspace_root = _internal_workspace_root(repo)
    if workspace_root is None:
        return None

    paths = {
        path.relative_to(repo).as_posix()
        for path in _iter_project_files(repo)
        if is_source_file(path)
    }

    changed_paths = set(get_unstaged_changed_paths(repo))
    changed_paths.update(get_staged_changed_paths(repo))
    changed_paths.update(get_untracked_paths(repo))

    for raw_path in changed_paths:
        normalized = raw_path.replace("\\", "/")
        candidate = (repo / normalized).resolve()
        if _is_internal_workspace_path(candidate, repo, workspace_root):
            continue
        if is_source_file(Path(normalized)):
            paths.add(normalized)

    return sorted(paths)


def build_tree(
    repo: Path,
    max_files: int = 500,
) -> str:
    lines: list[str] = []
    file_count = 0

    for path in _iter_context_files(repo):

        file_count += 1

        if file_count > max_files:
            lines.append(
                f"... tree truncated after "
                f"{max_files} files"
            )
            break

        relative = path.relative_to(repo)

        lines.append(
            str(relative)
        )

    return "\n".join(lines)


def get_changed_files(
    repo: Path,
) -> set[Path]:
    changed: set[Path] = set()

    paths = set(
        get_unstaged_changed_paths(repo)
    )

    paths.update(
        get_staged_changed_paths(repo)
    )

    paths.update(
        get_untracked_paths(repo)
    )

    for raw_path in paths:
        path = (repo / raw_path).resolve()

        if not path.exists():
            continue

        if not path.is_file():
            continue

        if _is_context_internal_artifact(path, repo):
            continue

        if _is_internal_workspace_path(path, repo):
            continue

        if is_ignored(path, repo):
            continue

        changed.add(path)

    return changed


def score_file(
    path: Path,
    repo: Path,
    task_words: set[str],
    changed_files: set[Path],
) -> int:
    score = 0

    relative = str(
        path.relative_to(repo)
    ).lower()

    filename = path.name.lower()

    if path in changed_files:
        score += 1000

    if filename in IMPORTANT_FILES:
        score += 50

    for word in task_words:
        if word in filename:
            score += 100

        elif word in relative:
            score += 50

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).lower()

        content = content[:1_000_000]

        for word in task_words:
            occurrences = content.count(word)

            if occurrences:
                score += (
                    min(occurrences, 10) * 5
                )

    except OSError:
        pass

    return score
