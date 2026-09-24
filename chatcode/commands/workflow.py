"""CLI patch workflow, repair, follow-up, and history commands."""
from __future__ import annotations

import hashlib
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from ..history import HistoryError
from ..patching.errors import PatchAlreadyApplied, PatchError


def prompt_review(
    history_entry: Path,
    *,
    open_history_review_fn: Callable[[Path], None],
) -> None:
    if not sys.stdin.isatty():
        return

    while True:
        try:
            answer = input(
                "\nChatCode: Do you want "
                "to review changes? (y/n): "
            )
        except EOFError:
            return

        answer = (
            answer.strip().lower()
        )

        if answer in {
            "y",
            "yes",
        }:
            try:
                open_history_review_fn(
                    history_entry
                )

                print(
                    "Review opened in VS Code."
                )

            except HistoryError as exc:
                print(
                    f"Review error: {exc}",
                    file=sys.stderr,
                )

            return

        if answer in {
            "n",
            "no",
            "",
        }:
            return

        print(
            "Please answer y or n."
        )


def command_apply(
    repo: Path,
    patch_path: str | None,
    no_test: bool,
    no_review: bool,
    new_task: bool = False,
    *,
    get_default_patch_file_fn: Callable[[Path], Path],
    apply_patch_fn: Callable,
    update_history_test_result_fn: Callable,
    run_tests_fn: Callable,
    prompt_review_fn: Callable[[Path], None],
) -> int:

    if patch_path is None:
        patch_file = (
            get_default_patch_file_fn(repo)
        )
    else:
        patch_file = Path(
            patch_path
        )

    try:
        result = apply_patch_fn(
            repo,
            patch_file,
            new_task=new_task,
        )

    except PatchAlreadyApplied:
        print(
            "Patch already applied. "
            "No changes were made."
        )

        return 0

    print(
        "Patch applied successfully."
    )
    print()

    for path in sorted(
        result.paths
    ):
        print(path)

    if no_test:
        update_history_test_result_fn(
            result.history_entry,
            status="SKIPPED",
        )

        print()
        print(
            "Automatic tests skipped."
        )

        test_returncode = 0

    else:
        test_returncode = run_tests_fn(
            repo,
            result.history_entry,
        )

    if not no_review:
        prompt_review_fn(
            result.history_entry
        )

    return test_returncode


def command_undo(
    repo: Path,
    no_test: bool,
    *,
    undo_last_patch_fn: Callable[[Path], object],
    update_history_test_result_fn: Callable,
    run_tests_fn: Callable,
) -> int:

    result = undo_last_patch_fn(
        repo
    )

    print(
        "Last ChatCode patch undone "
        "successfully."
    )
    print()

    for path in sorted(
        result.paths
    ):
        print(path)

    if no_test:
        update_history_test_result_fn(
            result.history_entry,
            status="SKIPPED",
        )

        print()
        print(
            "Automatic tests skipped."
        )

        return 0

    return run_tests_fn(
        repo,
        result.history_entry,
    )


def command_repair(
    repo: Path,
    *,
    get_repair_context_file_fn: Callable[[Path], Path],
    get_repair_context_stale_reason_fn: Callable[[Path], str | None],
    refresh_test_failure_repair_context_fn: Callable[[Path], Path | None],
    save_context_state_fn: Callable,
    get_context_task_fn: Callable[[Path], str | None],
    get_repair_context_targets_fn: Callable[[Path], frozenset[str]],
    show_repair_send_instructions_fn: Callable[[Path], None],
) -> int:
    repair_context = get_repair_context_file_fn(repo)
    if not repair_context.is_file():
        print(
            "No repair context is available. Apply a patch first; ChatCode "
            "creates one automatically when it detects a new regression."
        )
        return 1
    stale_reason = get_repair_context_stale_reason_fn(repo)
    if stale_reason is not None:
        refreshed = refresh_test_failure_repair_context_fn(repo)
        if refreshed is not None:
            repair_context = refreshed
            stale_reason = get_repair_context_stale_reason_fn(repo)
    if stale_reason is not None:
        print("Repair context is stale and must not be sent to ChatGPT:")
        print(stale_reason)
        print("Run chatcode context again, or reproduce the failed apply/test flow.")
        return 1
    repair_bytes = repair_context.read_bytes()
    save_context_state_fn(
        repo,
        task=get_context_task_fn(repo),
        context_sha256=hashlib.sha256(repair_bytes).hexdigest(),
        generation_id=uuid.uuid4().hex,
        context_filename=repair_context.name,
        context_kind="repair",
        repair_targets=sorted(get_repair_context_targets_fn(repo)),
    )
    show_repair_send_instructions_fn(repair_context)
    return 0


def command_followup(
    repo: Path,
    *,
    regenerate_followup_context_fn: Callable[[Path], Path | None],
    show_followup_send_instructions_fn: Callable[[Path], None],
) -> int:
    try:
        context = regenerate_followup_context_fn(repo)
    except (OSError, PatchError) as exc:
        print(f"Could not regenerate follow-up context: {exc}", file=sys.stderr)
        return 1
    if context is None:
        print("No unresolved follow-up is available for this repository.")
        return 1
    show_followup_send_instructions_fn(context)
    return 0


def command_history(
    repo: Path,
    *,
    format_history_fn: Callable[[Path], str],
) -> None:

    print(
        format_history_fn(repo)
    )


def command_review(
    repo: Path,
    index: int,
    *,
    review_history_by_index_fn: Callable,
) -> None:

    record = review_history_by_index_fn(
        repo,
        index,
    )

    print(
        "Review opened in VS Code."
    )

    print()

    print(
        f"History entry: {index}"
    )

    print(
        f"Status: {record.get('status')}"
    )
