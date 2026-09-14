from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .context_state import (
    get_stale_context_reason,
)
from .git_utils import GitError, run_git
from .history import (
    HistoryError,
    begin_history_entry,
    discard_pending_entry,
    finalize_history_entry,
    get_history_patch_file,
    get_latest_applied_entry,
    move_entry_to_undone,
)
from .workspace import (
    get_default_patch_file,
)


class PatchError(RuntimeError):
    pass


class PatchAlreadyApplied(RuntimeError):
    pass


class PatchUndoError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApplyResult:
    paths: set[str]
    history_entry: Path


@dataclass(frozen=True)
class UndoResult:
    paths: set[str]
    history_entry: Path


def extract_patch_paths(
    patch_text: str,
) -> set[str]:
    paths: set[str] = set()

    for line in patch_text.splitlines():
        if not line.startswith(
            ("--- ", "+++ ")
        ):
            continue

        raw_path = (
            line[4:]
            .strip()
            .split("\t", 1)[0]
        )

        if raw_path == "/dev/null":
            continue

        if raw_path.startswith(
            ("a/", "b/")
        ):
            raw_path = raw_path[2:]

        paths.add(raw_path)

    return paths


def validate_patch_paths(
    patch_text: str,
) -> set[str]:
    paths = extract_patch_paths(
        patch_text
    )

    if not paths:
        raise PatchError(
            "Patchen innehåller inga filer."
        )

    for raw_path in paths:
        path = PurePosixPath(
            raw_path
        )

        if path.is_absolute():
            raise PatchError(
                "Absolut sökväg är inte "
                f"tillåten: {raw_path}"
            )

        if ".." in path.parts:
            raise PatchError(
                "Sökväg utanför repot är "
                f"inte tillåten: {raw_path}"
            )

        if ".git" in path.parts:
            raise PatchError(
                "Patchen får inte ändra "
                f".git: {raw_path}"
            )

    return paths


def strip_markdown_fence(
    patch_text: str,
) -> str:
    lines = patch_text.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) >= 2:
        opening = lines[0].strip().lower()
        closing = lines[-1].strip()

        if (
            opening in {
                "```diff",
                "```patch",
                "```",
            }
            and closing == "```"
        ):
            lines = lines[1:-1]

    return "\n".join(lines)


def normalize_patch_file(
    patch_file: Path,
) -> str:
    try:
        patch_text = patch_file.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise PatchError(
            "Patchfilen måste vara UTF-8."
        ) from exc

    patch_text = strip_markdown_fence(
        patch_text
    )

    if not patch_text.endswith("\n"):
        patch_text += "\n"

    patch_file.write_text(
        patch_text,
        encoding="utf-8",
        newline="\n",
    )

    return patch_text


def patch_is_already_applied(
    repo: Path,
    patch_file: Path,
) -> bool:
    try:
        run_git(
            "apply",
            "--reverse",
            "--check",
            "--recount",
            str(patch_file),
            cwd=repo,
        )

        return True

    except GitError:
        pass

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

        return True

    except GitError:
        return False


def apply_patch(
    repo: Path,
    patch_file: Path,
) -> ApplyResult:
    patch_file = patch_file.resolve()

    if not patch_file.exists():
        raise PatchError(
            "Patchfilen finns inte:\n"
            f"{patch_file}"
        )

    if not patch_file.is_file():
        raise PatchError(
            "Detta är inte en fil: "
            f"{patch_file}"
        )

    patch_text = normalize_patch_file(
        patch_file
    )

    paths = validate_patch_paths(
        patch_text
    )

    default_patch_file = (
        get_default_patch_file(
            repo
        ).resolve()
    )

    if patch_file == default_patch_file:
        stale_reason = (
            get_stale_context_reason(
                repo,
                paths,
            )
        )

        if stale_reason is not None:
            raise PatchError(stale_reason)

    ignore_space = False

    try:
        run_git(
            "apply",
            "--check",
            "--recount",
            str(patch_file),
            cwd=repo,
        )

    except GitError as strict_error:
        if patch_is_already_applied(
            repo,
            patch_file,
        ):
            raise PatchAlreadyApplied(
                "Patchen verkar redan vara "
                "applicerad."
            )

        try:
            run_git(
                "apply",
                "--check",
                "--recount",
                "--ignore-space-change",
                str(patch_file),
                cwd=repo,
            )

            ignore_space = True

        except GitError:
            raise PatchError(
                "Patchen kunde inte "
                "appliceras:\n"
                f"{strict_error}"
            ) from strict_error

    try:
        pending_entry = (
            begin_history_entry(
                repo,
                patch_text,
                paths,
            )
        )
    except HistoryError as exc:
        raise PatchError(
            f"Kunde inte skapa historik: {exc}"
        ) from exc

    history_patch = (
        get_history_patch_file(
            pending_entry
        )
    )

    apply_args = [
        "apply",
        "--recount",
    ]

    if ignore_space:
        apply_args.append(
            "--ignore-space-change"
        )

    apply_args.append(
        str(history_patch)
    )

    try:
        run_git(
            *apply_args,
            cwd=repo,
        )

    except GitError as exc:
        discard_pending_entry(
            pending_entry
        )

        raise PatchError(
            "Git kunde inte applicera "
            "patchen:\n"
            f"{exc}"
        ) from exc

    try:
        history_entry = (
            finalize_history_entry(
                repo,
                pending_entry,
            )
        )
    except HistoryError as exc:
        raise PatchError(
            "Patchen applicerades, men "
            "ChatCode kunde inte slutföra "
            f"historiken:\n{exc}"
        ) from exc

    return ApplyResult(
        paths=paths,
        history_entry=history_entry,
    )


def undo_last_patch(
    repo: Path,
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
        paths = validate_patch_paths(
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