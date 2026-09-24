"""Repair-context lifecycle metadata and stale-state validation."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from ...workspace import atomic_write_text, get_repair_context_file


def repair_state_file(repo: Path) -> Path:
    return get_repair_context_file(repo).with_name("patch-repair-state.json")


def write_repair_context(
    repo: Path,
    content: str,
    paths: set[str] | list[str],
    repair_targets: set[str] | frozenset[str] = frozenset(),
    *,
    get_context_task_fn: Callable[[Path], str | None],
    save_context_state_fn: Callable,
    clear_incoming_fn: Callable[[Path], None],
) -> Path:
    output_file = get_repair_context_file(repo)
    hashes: dict[str, str | None] = {}
    for raw_path in sorted(set(paths)):
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        try:
            hashes[raw_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            hashes[raw_path] = None
    state = {
        "version": 1,
        "context_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "files": hashes,
        "repair_targets": sorted(repair_targets),
    }
    atomic_write_text(
        repair_state_file(repo),
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        newline="\n",
    )
    original_task = get_context_task_fn(repo)
    save_context_state_fn(
        repo,
        task=original_task,
        context_sha256=state["context_sha256"],
        context_filename=output_file.name,
        context_kind="repair",
        repair_targets=sorted(repair_targets),
    )
    atomic_write_text(output_file, content, newline="\n")
    clear_incoming_fn(repo)
    return output_file


def get_repair_context_targets(repo: Path) -> frozenset[str]:
    try:
        state = json.loads(repair_state_file(repo).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    targets = state.get("repair_targets")
    if not isinstance(targets, list):
        return frozenset()
    return frozenset(item for item in targets if isinstance(item, str) and item)


def get_repair_context_paths(repo: Path) -> frozenset[str]:
    """Return source paths that the active repair context actually exposed."""
    try:
        state = json.loads(repair_state_file(repo).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    files = state.get("files")
    if not isinstance(files, dict):
        return frozenset()
    return frozenset(
        PurePosixPath(raw_path).as_posix()
        for raw_path in files
        if isinstance(raw_path, str) and raw_path
    )


def patch_supersedes_active_repair(
    repo: Path,
    patch_paths: set[str],
    *,
    force: bool = False,
    get_context_kind_fn: Callable[[Path], str | None],
) -> bool:
    if get_context_kind_fn(repo) != "repair":
        return False
    if force:
        return True
    repair_paths = get_repair_context_paths(repo)
    if not repair_paths:
        return False
    normalized_paths = {PurePosixPath(path).as_posix() for path in patch_paths}
    return not normalized_paths.issubset(repair_paths)


def supersede_active_repair(
    repo: Path,
    *,
    save_context_state_fn: Callable,
    clear_repair_fn: Callable[[Path], None],
) -> None:
    save_context_state_fn(repo, task=None, context_kind="superseded")
    clear_repair_fn(repo)


def get_repair_context_stale_reason(repo: Path) -> str | None:
    output_file = get_repair_context_file(repo)
    state_file = repair_state_file(repo)
    try:
        content = output_file.read_bytes()
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Repair context metadata is missing or unreadable."
    if hashlib.sha256(content).hexdigest() != state.get("context_sha256"):
        return "Repair context and its metadata belong to different generations."
    files = state.get("files")
    if not isinstance(files, dict):
        return "Repair context metadata is invalid."
    changed: list[str] = []
    for raw_path, expected in files.items():
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        try:
            current = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            current = None
        if current != expected:
            changed.append(raw_path)
    if changed:
        return (
            "Working-tree files changed after repair context creation: "
            + ", ".join(sorted(changed))
        )
    return None
