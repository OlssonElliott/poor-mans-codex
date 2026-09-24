from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .context_builder import (
    build_safe_status,
    build_context,
    build_patch_context,
)
from .context_state import get_context_task, save_context_state
from .git_utils import (
    get_branch,
    get_repo_root,
    get_status,
)
from .history import (
    format_history,
    open_history_review,
    review_history_by_index,
    update_history_test_result,
)
from .indexing.index_manager import (
    update_project_map,
)
from .patch import (
    apply_patch,
    build_check_repair_context,
    get_repair_context_stale_reason,
    get_repair_context_targets,
    refresh_test_failure_repair_context,
    regenerate_followup_context,
    show_followup_send_instructions,
    show_repair_send_instructions,
    show_check_repair_send_instructions,
    show_chatgpt_upload_artifact,
    _status,
    _capture_repository_snapshot,
    get_background_full_suite_status,
    ensure_background_failure_repair_context,
    save_verified_baseline,
    undo_last_patch,
)
from .test_runner import (
    run_project_tests,
)
from .commands import status as _cli_status
from .commands import testing as _cli_testing
from .commands import workflow as _cli_workflow
from .commands.progress import ConsoleIndexReporter
from .commands import parser as _cli_parser
from .commands import dispatch as _cli_dispatch
from .commands.ui import open_folder
from .workspace import (
    get_default_patch_file,
    get_repair_context_file,
)


def command_status(reindex: bool = False) -> None:
    return _cli_status.command_status(
        reindex,
        get_repo_root_fn=get_repo_root,
        update_project_map_fn=update_project_map,
        reporter_factory=ConsoleIndexReporter,
        get_branch_fn=get_branch,
        build_safe_status_fn=build_safe_status,
        get_background_full_suite_status_fn=get_background_full_suite_status,
        status_fn=_status,
        ensure_background_failure_repair_context_fn=ensure_background_failure_repair_context,
        show_repair_send_instructions_fn=show_repair_send_instructions,
    )


def command_context(
    task: str,
    patch_oriented: bool = False,
) -> None:
    return _cli_status.command_context(
        task,
        patch_oriented,
        get_repo_root_fn=get_repo_root,
        reporter_factory=ConsoleIndexReporter,
        build_patch_context_fn=build_patch_context,
        build_context_fn=build_context,
        get_default_patch_file_fn=get_default_patch_file,
        show_chatgpt_upload_artifact_fn=show_chatgpt_upload_artifact,
    )

def run_tests(
    repo: Path,
    history_entry: Path | None = None,
) -> int:
    return _cli_testing.run_tests(
        repo,
        history_entry,
        run_project_tests_fn=run_project_tests,
        update_history_test_result_fn=update_history_test_result,
    )

def prompt_review(history_entry: Path) -> None:
    return _cli_workflow.prompt_review(
        history_entry,
        open_history_review_fn=open_history_review,
    )


def command_apply(
    patch_path: str | None,
    no_test: bool,
    no_review: bool,
    new_task: bool = False,
) -> int:
    return _cli_workflow.command_apply(
        get_repo_root(),
        patch_path,
        no_test,
        no_review,
        new_task,
        get_default_patch_file_fn=get_default_patch_file,
        apply_patch_fn=apply_patch,
        update_history_test_result_fn=update_history_test_result,
        run_tests_fn=run_tests,
        prompt_review_fn=prompt_review,
    )


def command_undo(no_test: bool) -> int:
    return _cli_workflow.command_undo(
        get_repo_root(),
        no_test,
        undo_last_patch_fn=undo_last_patch,
        update_history_test_result_fn=update_history_test_result,
        run_tests_fn=run_tests,
    )

def command_test() -> int:
    return _cli_testing.command_test(
        get_repo_root_fn=get_repo_root,
        run_project_tests_fn=run_project_tests,
    )


def _select_check_failures(failures: frozenset[str]) -> frozenset[str]:
    return _cli_testing.select_check_failures(failures)


def command_check() -> int:
    return _cli_testing.command_check(
        get_repo_root_fn=get_repo_root,
        run_project_tests_fn=run_project_tests,
        capture_repository_snapshot_fn=_capture_repository_snapshot,
        save_verified_baseline_fn=save_verified_baseline,
        get_status_fn=get_status,
        build_check_repair_context_fn=build_check_repair_context,
        show_check_repair_send_instructions_fn=show_check_repair_send_instructions,
        select_check_failures_fn=_select_check_failures,
    )

def command_repair() -> int:
    return _cli_workflow.command_repair(
        get_repo_root(),
        get_repair_context_file_fn=get_repair_context_file,
        get_repair_context_stale_reason_fn=get_repair_context_stale_reason,
        refresh_test_failure_repair_context_fn=refresh_test_failure_repair_context,
        save_context_state_fn=save_context_state,
        get_context_task_fn=get_context_task,
        get_repair_context_targets_fn=get_repair_context_targets,
        show_repair_send_instructions_fn=show_repair_send_instructions,
    )


def command_followup() -> int:
    return _cli_workflow.command_followup(
        get_repo_root(),
        regenerate_followup_context_fn=regenerate_followup_context,
        show_followup_send_instructions_fn=show_followup_send_instructions,
    )


def command_history() -> None:
    return _cli_workflow.command_history(
        get_repo_root(),
        format_history_fn=format_history,
    )


def command_review(index: int) -> None:
    return _cli_workflow.command_review(
        get_repo_root(),
        index,
        review_history_by_index_fn=review_history_by_index,
    )

def create_parser() -> argparse.ArgumentParser:
    return _cli_parser.create_parser()

def main() -> None:
    return _cli_dispatch.run(
        create_parser_fn=create_parser,
        command_status_fn=command_status,
        command_context_fn=command_context,
        command_followup_fn=command_followup,
        command_apply_fn=command_apply,
        command_undo_fn=command_undo,
        command_test_fn=command_test,
        command_check_fn=command_check,
        command_repair_fn=command_repair,
        command_history_fn=command_history,
        command_review_fn=command_review,
    )


if __name__ == "__main__":
    main()
