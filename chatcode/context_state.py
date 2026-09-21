from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from .file_filter import is_ignored
from .git_utils import (
    get_branch,
    run_git,
)
from .workspace import atomic_write_text, get_repo_workspace


STATE_FILENAME = "context-state.json"


def _state_file(
    repo: Path,
) -> Path:
    return (
        get_repo_workspace(repo)
        / STATE_FILENAME
    )


def _repo_path(
    repo: Path,
    relative_path: str,
) -> Path:
    path = PurePosixPath(
        relative_path
    )

    return repo.joinpath(
        *path.parts
    )


def _hash_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _collect_file_hashes(
    repo: Path,
) -> dict[str, str]:
    output = run_git(
        "ls-files",
        "-c",
        "-o",
        "--exclude-standard",
        "-z",
        "--",
        cwd=repo,
    )

    hashes: dict[str, str] = {}

    if not output:
        return hashes

    for raw_path in output.split("\0"):
        if not raw_path:
            continue

        relative_path = (
            PurePosixPath(
                raw_path
            ).as_posix()
        )

        path = _repo_path(
            repo,
            relative_path,
        )

        if not path.is_file():
            continue

        if is_ignored(
            path,
            repo,
        ):
            continue

        try:
            hashes[
                relative_path
            ] = _hash_file(path)
        except OSError:
            continue

    return hashes


def save_context_state(
    repo: Path,
    task: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
    context_sha256: str | None = None,
    generation_id: str | None = None,
    context_filename: str = "UPLOAD_TO_CHATGPT.md",
    context_kind: str = "normal",
    repair_targets: list[str] | None = None,
) -> Path:
    files = _collect_file_hashes(repo)
    if source_hashes:
        files.update(source_hashes)
    state = {
        "version": 3,
        "branch": get_branch(repo),
        "task": task,
        "files": files,
        "context_sha256": context_sha256,
        "generation_id": generation_id,
        "context_filename": context_filename,
        "context_kind": context_kind,
        "repair_targets": sorted(set(repair_targets or [])),
    }

    state_file = _state_file(repo)

    atomic_write_text(
        state_file,
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )

    return state_file


def get_active_repair_targets(repo: Path) -> frozenset[str]:
    """Return failures the currently active repair context promises to fix."""
    try:
        state = json.loads(_state_file(repo).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    if state.get("context_kind") != "repair":
        return frozenset()
    targets = state.get("repair_targets")
    if not isinstance(targets, list):
        return frozenset()
    return frozenset(
        item for item in targets
        if isinstance(item, str) and item
    )


def get_context_task(
    repo: Path,
) -> str:
    state_file = _state_file(repo)
    if not state_file.exists():
        return "[Original task unavailable]"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "[Original task unavailable]"
    task = state.get("task")
    return task if isinstance(task, str) and task else "[Original task unavailable]"


def get_context_kind(repo: Path) -> str | None:
    try:
        state = json.loads(_state_file(repo).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    kind = state.get("context_kind")
    return kind if isinstance(kind, str) else None


def get_stale_context_reason(
    repo: Path,
    patch_paths: set[str],
) -> str | None:
    state_file = _state_file(repo)

    if not state_file.exists():
        return None

    try:
        state = json.loads(
            state_file.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return (
            "Context state kunde inte "
            "läsas.\n\n"
            "Kör chatcode context igen "
            "innan du applicerar patchen."
        )

    expected_context_hash = state.get("context_sha256")
    context_kind = state.get("context_kind", "normal")
    if context_kind in {"consumed", "superseded"}:
        # Consumed contexts and abandoned repair generations are historical
        # provenance only. They must not constrain a later independent patch.
        return None
    is_repair = context_kind == "repair"
    if isinstance(expected_context_hash, str) and expected_context_hash:
        context_filename = state.get("context_filename", "UPLOAD_TO_CHATGPT.md")
        if not isinstance(context_filename, str) or Path(context_filename).name != context_filename:
            return "Context state references an invalid context filename."
        context_file = state_file.parent / context_filename
        try:
            actual_context_hash = _hash_file(context_file)
        except OSError:
            actual_context_hash = None
        if actual_context_hash != expected_context_hash:
            if is_repair:
                return (
                    "Repair context is stale. PATCH_REPAIR_CONTEXT.md and its "
                    "metadata belong to different generations. Regenerate the "
                    "repair context before applying this repair patch."
                )
            return (
                "UPLOAD_TO_CHATGPT.md och context-state hör inte till samma "
                "publicerade generation. Kör chatcode context igen innan du "
                "applicerar patchen."
            )

    expected_branch = state.get(
        "branch"
    )

    current_branch = get_branch(repo)

    if expected_branch != current_branch:
        if is_repair:
            return (
                "Repair context is stale.\n\n"
                f"It was created on branch {expected_branch}, but the current "
                f"branch is {current_branch}.\n\n"
                "Regenerate the repair context before applying this repair patch."
            )
        return (
            "Contexten är inaktuell.\n\n"
            "Den skapades på branch "
            f"{expected_branch}, men du "
            "står nu på "
            f"{current_branch}.\n\n"
            "Kör chatcode context igen "
            "för samma uppgift."
        )

    snapshot = state.get(
        "files"
    )

    if not isinstance(snapshot, dict):
        if is_repair:
            return (
                "Repair context metadata is invalid. Regenerate the repair "
                "context before applying this repair patch."
            )
        return (
            "Context state är ogiltig.\n\n"
            "Kör chatcode context igen "
            "innan du applicerar patchen."
        )

    changed: list[str] = []

    for raw_path in sorted(
        patch_paths
    ):
        relative_path = (
            PurePosixPath(
                raw_path
            ).as_posix()
        )

        expected_hash = snapshot.get(
            relative_path
        )

        path = _repo_path(
            repo,
            relative_path,
        )

        if expected_hash is None:
            if path.exists():
                changed.append(
                    relative_path
                )

            continue

        if not path.is_file():
            changed.append(
                relative_path
            )
            continue

        try:
            current_hash = _hash_file(
                path
            )
        except OSError:
            changed.append(
                relative_path
            )
            continue

        if current_hash != expected_hash:
            changed.append(
                relative_path
            )

    if not changed:
        return None

    file_list = "\n".join(
        f"- {path}"
        for path in changed
    )

    if is_repair:
        return (
            "Repair context is stale.\n\n"
            "The following files changed after PATCH_REPAIR_CONTEXT.md was created:\n"
            f"{file_list}\n\n"
            "Regenerate the repair context before applying this repair patch."
        )

    return (
        "Contexten är inaktuell.\n\n"
        "Följande filer har ändrats "
        "sedan UPLOAD_TO_CHATGPT.md "
        "skapades:\n"
        f"{file_list}\n\n"
        "Kör chatcode context igen för "
        "samma uppgift och ladda upp den "
        "nya UPLOAD_TO_CHATGPT.md innan "
        "du kör chatcode apply."
    )
