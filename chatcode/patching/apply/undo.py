"""Safe reversal of the latest applied ChatCode patch."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..errors import PatchError, PatchUndoError
from ..models import UndoResult
from ...git_utils import GitError, run_git
from ...history import (
    HistoryError,
    get_history_patch_file,
    get_latest_applied_entry,
    move_entry_to_undone,
)


def verify_undo(
    repo: Path,
    undone_entry: Path,
) -> None:
    patch_file = get_history_patch_file(undone_entry)
    try:
        run_git(
            "apply",
            "--check",
            "--recount",
            str(patch_file),
            cwd=repo,
        )
    except GitError as exc:
        raise PatchUndoError(
            "Patchen backades, men återställningen "
            "kunde inte verifieras.\n"
            f"{exc}"
        ) from exc


def undo_last_patch(
    repo: Path,
    *,
    validate_patch_paths_fn: Callable[[str], set[str]],
) -> UndoResult:
    try:
        history_entry = (
            get_latest_applied_entry(
                repo
            )
        )
    except HistoryError as exc:
        raise PatchUndoError(
            str(exc)
        ) from exc

    patch_file = (
        get_history_patch_file(
            history_entry
        )
    )

    try:
        patch_text = patch_file.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except (
        OSError,
        UnicodeDecodeError,
    ) as exc:
        raise PatchUndoError(
            "Historikpatchen kunde inte "
            "läsas."
        ) from exc

    try:
        paths = validate_patch_paths_fn(
            patch_text
        )
    except PatchError as exc:
        raise PatchUndoError(
            str(exc)
        ) from exc

    ignore_space = False

    try:
        run_git(
            "apply",
            "--reverse",
            "--check",
            "--recount",
            str(patch_file),
            cwd=repo,
        )

    except GitError:
        try:
            run_git(
                "apply",
                "--reverse",
                "--check",
                "--recount",
                "--ignore-space-change",
                str(patch_file),
                cwd=repo,
            )

            ignore_space = True

        except GitError as exc:
            raise PatchUndoError(
                "Den senaste ChatCode patchen "
                "kan inte backas säkert.\n"
                "Filerna kan ha ändrats efter "
                "att patchen applicerades.\n\n"
                f"{exc}"
            ) from exc

    undo_args = [
        "apply",
        "--reverse",
        "--recount",
    ]

    if ignore_space:
        undo_args.append(
            "--ignore-space-change"
        )

    undo_args.append(
        str(patch_file)
    )

    try:
        run_git(
            *undo_args,
            cwd=repo,
        )

    except GitError as exc:
        raise PatchUndoError(
            "Git kunde inte backa "
            "patchen:\n"
            f"{exc}"
        ) from exc

    try:
        undone_entry = (
            move_entry_to_undone(
                repo,
                history_entry,
            )
        )
    except HistoryError as exc:
        raise PatchUndoError(
            str(exc)
        ) from exc

    return UndoResult(
        paths=paths,
        history_entry=undone_entry,
    )
