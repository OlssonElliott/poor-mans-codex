"""Terminal presentation helpers for patch workflows."""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from .models import ApplyResult
from ..history import get_history_patch_file
from .repair.check_repair import CHECK_REPAIR_COMPANION_PROMPT


_ANSI = {
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}

REPAIR_COMPANION_PROMPT = (
    "Use the attached PATCH_REPAIR_CONTEXT.md as repository context and the "
    "current source of truth. The listed repair targets are the sole success "
    "criterion: each of those tests must pass after the patch. Trace the failing "
    "assertion through the supplied current implementation and do not repeat the "
    "unsuccessful attempted patch. Do not change unrelated behavior from the "
    "original task. Investigate and repair the classified failures "
    "without weakening tests or reverting unrelated working-tree changes. "
    "Return exactly one complete unified diff with at least three unchanged "
    "context lines around each hunk, and no explanation outside the diff."
)

EXISTING_FAILURE_COMPANION_PROMPT = (
    "Use the attached EXISTING_FAILURE_CONTEXT.md as the current repository "
    "context and source of truth. Fix the selected pre-existing test failure "
    "without reverting unrelated working-tree changes or the previously "
    "successful patch. Return exactly one complete unified diff with at least "
    "three unchanged context lines around each hunk and no explanation outside "
    "the diff."
)

FOLLOWUP_COMPANION_PROMPT = (
    "Use the attached FOLLOWUP_CONTEXT.md as the current repository context "
    "and source of truth. The previous patch applied successfully and automated "
    "validation passed, but the user reports that the original problem remains "
    "unresolved. Use the included user feedback to continue the fix without "
    "reverting unrelated working-tree changes or the previous patch. Return "
    "exactly one complete unified diff with at least three unchanged context "
    "lines around each hunk and no explanation outside the diff."
)


def supports_color() -> bool:
    return "NO_COLOR" not in os.environ and sys.stdout.isatty()


def status(text: str, color: str | None = None) -> str:
    if color and supports_color():
        return f"{_ANSI[color]}{text}\x1b[0m"
    return text


def show_chatgpt_upload_artifact(context: Path) -> None:
    """Make a generated ChatGPT attachment unambiguous in terminal output."""
    print(status("[UPLOAD THIS FILE]", "cyan"))
    print(status(str(context), "cyan"))
    # Compatibility seam: callers/tests patch chatcode.cli.open_folder.
    # Keep this lazy so presentation does not create an import-time cycle.
    from ..cli import open_folder
    open_folder(context.resolve().parent)


def show_repair_send_instructions(repair_context: Path) -> None:
    print(f"Repair context ready to send to ChatGPT: {repair_context}")
    show_chatgpt_upload_artifact(repair_context)
    print("Attach that file and send this message with it:")
    print(REPAIR_COMPANION_PROMPT)


def show_check_repair_send_instructions(context: Path) -> None:
    print(f"Check repair context created: {context}")
    show_chatgpt_upload_artifact(context)
    print("Attach that file and send this message with it:")
    print(CHECK_REPAIR_COMPANION_PROMPT)


def show_followup_send_instructions(context: Path) -> None:
    print("Follow-up context created.")
    show_chatgpt_upload_artifact(context)
    print("Attach that file and send this message with it:")
    print(FOLLOWUP_COMPANION_PROMPT)


def applied_patch_summary(
    result: ApplyResult,
    *,
    fallback_patch_summary_fn: Callable[[str, set[str]], str],
) -> str:
    try:
        patch_text = get_history_patch_file(result.history_entry).read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        patch_text = (
            "[Applied patch text is unavailable; changed-file summary follows.]"
        )
    return fallback_patch_summary_fn(patch_text, result.paths)


def show_existing_failure_send_instructions(context: Path) -> None:
    print(f"Existing failure context created: {context}")
    show_chatgpt_upload_artifact(context)
    print("Attach that file and send this message with it:")
    print(EXISTING_FAILURE_COMPANION_PROMPT)


def choose_existing_failures(
    failures: frozenset[str],
) -> frozenset[str]:
    ordered = sorted(failures)
    if len(ordered) == 1:
        print(f"Selected pre-existing failure: {ordered[0]}")
        return frozenset(ordered)
    print("Select pre-existing failure(s) for a new fix task:")
    for index, failure in enumerate(ordered, start=1):
        print(f"[{index}] {failure}")
    print("Enter comma-separated numbers, or A for all (one is recommended).")
    answer = input("> ").strip().lower()
    if answer == "a":
        return frozenset(ordered)
    try:
        selected = {
            ordered[int(part.strip()) - 1]
            for part in answer.split(",")
            if part.strip()
        }
    except (IndexError, ValueError):
        selected = set()
    if not selected:
        print("No valid failure selected.")
    return frozenset(selected)

def ask_yes_no(
    question: str,
    *,
    default: bool,
) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(question + suffix).strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def open_diff_window(
    repo: Path,
    patch_text: str,
    paths: set[str],
    *,
    get_repo_workspace_fn: Callable[[Path], Path],
    atomic_write_text_fn: Callable,
    run_git_fn: Callable,
    open_code_diff_fn: Callable,
    rmtree_fn: Callable,
    copy2_fn: Callable,
) -> None:
    from pathlib import PurePosixPath
    from ..git_utils import GitError
    from ..history import HistoryError
    from .errors import PatchError

    try:
        preview_root = get_repo_workspace_fn(repo) / "patch-preview"
        resolved_workspace = get_repo_workspace_fn(repo).resolve()
        resolved_preview = preview_root.resolve()
        if resolved_preview.parent != resolved_workspace:
            raise OSError("Unsafe patch preview directory.")
        if preview_root.exists():
            rmtree_fn(preview_root)

        before_root = preview_root / "before"
        after_root = preview_root / "after"
        before_root.mkdir(parents=True)
        after_root.mkdir(parents=True)

        preview_patch = preview_root / "canonical-preview.diff"
        atomic_write_text_fn(preview_patch, patch_text, newline="\n")

        for raw_path in sorted(paths):
            relative = PurePosixPath(raw_path)
            source = repo.joinpath(*relative.parts)
            before = before_root.joinpath(*relative.parts)
            after = after_root.joinpath(*relative.parts)
            before.parent.mkdir(parents=True, exist_ok=True)
            after.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                copy2_fn(source, before)
                copy2_fn(source, after)

        run_git_fn(
            "apply",
            "--unsafe-paths",
            str(preview_patch),
            cwd=after_root,
        )

        for raw_path in sorted(paths):
            relative = PurePosixPath(raw_path)
            before = before_root.joinpath(*relative.parts)
            after = after_root.joinpath(*relative.parts)
            if not before.exists():
                before.parent.mkdir(parents=True, exist_ok=True)
                before.write_bytes(b"")
            if not after.exists():
                after.parent.mkdir(parents=True, exist_ok=True)
                after.write_bytes(b"")
            open_code_diff_fn(before.resolve(), after.resolve())
    except (OSError, GitError, HistoryError) as exc:
        raise PatchError(
            f"Could not open diff window: {exc}"
        ) from exc

