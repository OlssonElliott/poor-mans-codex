from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .context_builder import is_ignored
from .git_utils import (
    get_branch,
    run_git,
)
from .workspace import get_repo_workspace


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
) -> Path:
    state = {
        "version": 2,
        "branch": get_branch(repo),
        "task": task,
        "files": _collect_file_hashes(
            repo
        ),
    }

    state_file = _state_file(repo)

    state_file.write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return state_file


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

    expected_branch = state.get(
        "branch"
    )

    current_branch = get_branch(repo)

    if expected_branch != current_branch:
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
