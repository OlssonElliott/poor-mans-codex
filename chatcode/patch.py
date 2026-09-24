from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .context_state import (
    get_active_repair_targets,
    get_context_task,
    get_context_kind,
    get_stale_context_reason,
    save_context_state,
)
from .git_utils import run_git
from .history import open_code_diff
from .workspace import (
    atomic_write_text,
    get_default_patch_file,
    get_repair_context_file,
    get_repo_workspace,
)

from .patching.testing import baseline as _baseline
from .patching.testing.baseline import RepositorySnapshot
from .patching.errors import PatchAlreadyApplied, PatchError, PatchUndoError
from .patching.models import (
    ApplyResult,
    PatchPreview,
    TestValidation,
    UndoResult,
)
from .patching.testing import background_tests as _background
from .patching.testing import background_service as _background_service
from .patching.repair import repair_state as _repair_state
from .patching.repair import background_repair as _background_repair
from .patching.repair import existing_failure as _existing_failure
from .patching.repair import followup_state as _followup_state
from .patching.repair import followup as _followup
from .patching.testing import validation as _validation
from .patching import summary as _summary
from .patching.apply import apply_service as _apply_service
from .patching.apply import undo as _undo
from .patching import service as _service
from .patching import api as _patch_api
from .patching import presentation as _presentation
from .patching.repair import repair_refresh as _repair_refresh
from .patching.repair import repair_service as _repair_service
from .patching.apply.patch_validation import (
    extract_patch_paths,
    validate_patch_paths,
    _required_hunk_context,
    _old_hunk_lines,
    _unique_sequence_start,
    _expand_hunk_from_current_source,
    _nonblank_sequence_spans,
    _unique_blank_gap,
    _rebuild_hunk_context,
    _reanchor_hunk_from_current_source,
    _hunks_are_ordered_and_disjoint,
    _expand_thin_hunk_context,
    _validate_hunk_context,
    strip_markdown_fence,
    normalize_patch_file,
    patch_is_already_applied,
    _patch_hunks,
    _safe_candidate_paths,
    _hunk_anchor,
    _working_tree_section,
)
from .patching.repair.check_repair import build_check_repair_context as _build_check_repair_context


def _consume_cli_apply_flags() -> tuple[bool, bool]:
    """Consume --yes before the existing CLI parser sees the apply command."""
    is_apply = len(sys.argv) > 1 and sys.argv[1] == "apply"
    auto_yes = False
    if is_apply and "--yes" in sys.argv[2:]:
        sys.argv.remove("--yes")
        auto_yes = True
    return is_apply, auto_yes


_CLI_APPLY_INVOCATION, _CLI_APPLY_YES = _consume_cli_apply_flags()

def build_check_repair_context(
    repo: Path,
    test_result,
    selected_failures: frozenset[str],
) -> Path:
    # Compatibility seam for existing callers/tests that patch the façade.
    from .context_builder import build_context_from_test_roots

    return _build_check_repair_context(
        repo,
        test_result,
        selected_failures,
        build_context_from_test_roots_fn=build_context_from_test_roots,
    )



_capture_repository_snapshot = _baseline.capture_repository_snapshot


_test_config_identity = _baseline.test_config_identity


_snapshot_payload = _baseline.snapshot_payload


def _load_verified_baseline(repo: Path, snapshot: RepositorySnapshot):
    return _baseline.load_verified_baseline(
        repo,
        snapshot,
        test_config_identity_fn=_test_config_identity,
    )


def save_verified_baseline(repo: Path, snapshot: RepositorySnapshot, result) -> None:
    return _baseline.save_verified_baseline(
        repo,
        snapshot,
        result,
        test_config_identity_fn=_test_config_identity,
    )

_background_full_suite_status_file = _background.status_file


_background_full_suite_run_file = _background.run_file


_background_full_suite_report_file = _background.report_file


_cleanup_background_full_suite_artifacts = _background.cleanup_artifacts


def _write_background_full_suite_status(
    repo: Path,
    payload: dict,
    *,
    run_file: bool = False,
) -> None:
    return _background.write_status(
        repo,
        payload,
        run_file_status=run_file,
    )


get_background_full_suite_status = _background.get_status
_snapshot_from_payload = _background.snapshot_from_payload


def _background_hooks() -> _background_service.BackgroundHooks:
    return _background_service.BackgroundHooks(
        capture_snapshot_fn=_capture_repository_snapshot,
        save_baseline_fn=save_verified_baseline,
        get_status_fn=get_background_full_suite_status,
        cleanup_fn=_cleanup_background_full_suite_artifacts,
        snapshot_payload_fn=_snapshot_payload,
        popen_fn=subprocess.Popen,
    )


def _complete_background_full_suite(
    repo_name: str,
    snapshot_payload: dict,
    run_id: str,
    application_payload: dict | None = None,
) -> None:
    return _background_service.complete_full_suite(
        _background_hooks(),
        repo_name,
        snapshot_payload,
        run_id,
        application_payload,
    )


def _complete_background_full_suite_from_status(
    repo_name: str,
    run_id: str,
) -> None:
    return _background_service.complete_full_suite_from_status(
        repo_name,
        run_id,
        complete_fn=_complete_background_full_suite,
    )


def _start_background_full_suite(
    repo: Path,
    snapshot: RepositorySnapshot,
    application: ApplyResult | None = None,
) -> None:
    return _background_service.start_full_suite(
        _background_hooks(),
        repo,
        snapshot,
        application,
    )


REPAIR_COMPANION_PROMPT = _presentation.REPAIR_COMPANION_PROMPT
EXISTING_FAILURE_COMPANION_PROMPT = _presentation.EXISTING_FAILURE_COMPANION_PROMPT
FOLLOWUP_COMPANION_PROMPT = _presentation.FOLLOWUP_COMPANION_PROMPT


_supports_color = _presentation.supports_color


_status = _presentation.status


show_chatgpt_upload_artifact = _presentation.show_chatgpt_upload_artifact


show_repair_send_instructions = _presentation.show_repair_send_instructions

def _classify_test_validation(
    baseline,
    targeted,
    full,
    repair_targets: frozenset[str] = frozenset(),
    *,
    full_suite_pending: bool = False,
    infrastructure_error: bool = False,
) -> TestValidation:
    return _validation.classify_test_validation(
        baseline,
        targeted,
        full,
        repair_targets,
        full_suite_pending=full_suite_pending,
        infrastructure_error=infrastructure_error,
    )


def _show_test_validation(validation: TestValidation) -> None:
    return _validation.show_test_validation(validation, status_fn=_status)


_repair_state_file = _repair_state.repair_state_file


def _repair_hooks() -> _repair_service.RepairHooks:
    return _repair_service.RepairHooks(
        get_context_task_fn=get_context_task,
        save_context_state_fn=save_context_state,
        clear_incoming_fn=_clear_incoming_patch,
        clear_repair_fn=_clear_repair_context,
        get_context_kind_fn=get_context_kind,
        safe_candidate_paths_fn=_safe_candidate_paths,
        working_tree_section_fn=_working_tree_section,
        patch_hunks_fn=_patch_hunks,
    )


def _write_repair_context(
    repo: Path,
    content: str,
    paths: set[str] | list[str],
    repair_targets: set[str] | frozenset[str] = frozenset(),
) -> Path:
    return _repair_service.write_repair_context(
        _repair_hooks(),
        repo,
        content,
        paths,
        repair_targets,
    )


get_repair_context_targets = _repair_state.get_repair_context_targets
_get_repair_context_paths = _repair_state.get_repair_context_paths


def _patch_supersedes_active_repair(
    repo: Path,
    patch_paths: set[str],
    *,
    force: bool = False,
) -> bool:
    return _repair_service.patch_supersedes_active_repair(
        _repair_hooks(),
        repo,
        patch_paths,
        force=force,
    )


def _supersede_active_repair(repo: Path) -> None:
    return _repair_service.supersede_active_repair(
        _repair_hooks(),
        repo,
    )


get_repair_context_stale_reason = _repair_state.get_repair_context_stale_reason
_test_result_from_saved_report = _repair_refresh.test_result_from_saved_report


def refresh_test_failure_repair_context(repo: Path) -> Path | None:
    return _repair_refresh.refresh_test_failure_repair_context(
        repo,
        classify_test_validation_fn=_classify_test_validation,
        extract_patch_paths_fn=extract_patch_paths,
        build_test_failure_repair_context_fn=build_test_failure_repair_context,
    )


def build_syntax_repair_context(
    repo: Path,
    original_patch: str,
    error: str,
    malformed_hunk: str | None = None,
    failure_type: str = "invalid_patch_syntax",
) -> Path:
    return _repair_service.build_syntax_repair_context(
        _repair_hooks(),
        _write_repair_context,
        repo,
        original_patch,
        error,
        malformed_hunk,
        failure_type,
    )


def build_stale_repair_context(
    repo: Path,
    patch_text: str,
    stale_reason: str,
    paths: set[str],
) -> Path:
    return _repair_service.build_stale_repair_context(
        _repair_hooks(),
        _write_repair_context,
        repo,
        patch_text,
        stale_reason,
        paths,
    )


def build_patch_repair_context(
    repo: Path,
    patch_text: str,
    apply_error: str,
) -> Path:
    return _repair_service.build_patch_repair_context(
        _repair_hooks(),
        _write_repair_context,
        repo,
        patch_text,
        apply_error,
    )


def build_test_failure_repair_context(
    repo: Path,
    result: ApplyResult,
    validation: TestValidation,
) -> Path:
    return _repair_service.build_test_failure_repair_context(
        _repair_hooks(),
        _write_repair_context,
        repo,
        result,
        validation,
    )

def ensure_background_failure_repair_context(
    repo: Path,
    background: dict,
) -> tuple[Path, bool]:
    return _background_repair.ensure_background_failure_repair_context(
        repo,
        background,
        get_stale_reason_fn=get_repair_context_stale_reason,
        snapshot_from_payload_fn=_snapshot_from_payload,
        capture_snapshot_fn=_capture_repository_snapshot,
        extract_patch_paths_fn=extract_patch_paths,
        build_test_failure_context_fn=build_test_failure_repair_context,
        write_background_status_fn=_write_background_full_suite_status,
    )

def _apply_hooks() -> _apply_service.ApplyHooks:
    return _apply_service.ApplyHooks(
        get_default_patch_file_fn=get_default_patch_file,
        patch_supersedes_active_repair_fn=_patch_supersedes_active_repair,
        supersede_active_repair_fn=_supersede_active_repair,
        get_stale_context_reason_fn=get_stale_context_reason,
        build_syntax_repair_context_fn=build_syntax_repair_context,
        build_stale_repair_context_fn=build_stale_repair_context,
        build_patch_repair_context_fn=build_patch_repair_context,
        get_context_kind_fn=get_context_kind,
        get_context_task_fn=get_context_task,
        save_context_state_fn=save_context_state,
        get_repair_context_file_fn=get_repair_context_file,
        repair_state_file_fn=_repair_state_file,
    )


def _apply_patch_core(
    repo: Path,
    patch_file: Path,
    *,
    dry_run: bool = False,
    new_task: bool = False,
) -> ApplyResult | PatchPreview:
    return _apply_service.apply_patch_core(
        _apply_hooks(),
        repo,
        patch_file,
        dry_run=dry_run,
        new_task=new_task,
    )


def _clear_repair_context(repo: Path) -> None:
    return _apply_service.clear_repair_context(
        _apply_hooks(),
        repo,
    )


def _clear_incoming_patch(repo: Path) -> None:
    return _apply_service.clear_incoming_patch(
        _apply_hooks(),
        repo,
    )


_fallback_patch_summary = _summary.fallback_patch_summary


def _qwen_patch_summary(
    repo: Path,
    patch_text: str,
) -> str | None:
    return _summary.qwen_patch_summary(
        repo,
        patch_text,
        which_fn=shutil.which,
        run_fn=subprocess.run,
        getenv_fn=os.getenv,
        get_context_task_fn=get_context_task,
    )


def _build_patch_summary(
    repo: Path,
    preview: PatchPreview,
) -> str:
    return (
        _qwen_patch_summary(repo, preview.patch_text)
        or _fallback_patch_summary(preview.patch_text, preview.paths)
    )

def _ask_yes_no(
    question: str,
    *,
    default: bool,
) -> bool:
    return _presentation.ask_yes_no(question, default=default)


def _open_diff_window(
    repo: Path,
    patch_text: str,
    paths: set[str],
) -> None:
    return _presentation.open_diff_window(
        repo,
        patch_text,
        paths,
        get_repo_workspace_fn=get_repo_workspace,
        atomic_write_text_fn=atomic_write_text,
        run_git_fn=run_git,
        open_code_diff_fn=open_code_diff,
        rmtree_fn=shutil.rmtree,
        copy2_fn=shutil.copy2,
    )


def _show_test_result(test_result) -> None:
    return _validation.show_test_result(test_result, status_fn=_status)


_preserve_test_report = _validation.preserve_test_report


_clear_phase_test_reports = _validation.clear_phase_test_reports


_run_background_baseline = _validation.run_background_baseline

_verify_undo = _undo.verify_undo

show_check_repair_send_instructions = _presentation.show_check_repair_send_instructions


MAX_FOLLOWUP_ROUNDS = _followup_state.MAX_FOLLOWUP_ROUNDS


_load_followup_state = _followup_state.load_followup_state


_save_followup_state = _followup_state.save_followup_state


mark_followup_resolved = _followup_state.mark_followup_resolved


_followup_attempt = _followup_state.followup_attempt


_result_from_followup_attempt = _followup_state.result_from_followup_attempt


_attempted_python_symbols_by_path = _followup_state.attempted_python_symbols_by_path


_attempted_python_symbols = _followup_state.attempted_python_symbols

build_followup_context = _followup.build_followup_context

regenerate_followup_context = _followup.regenerate_followup_context

show_followup_send_instructions = _presentation.show_followup_send_instructions


def _applied_patch_summary(result: ApplyResult) -> str:
    return _presentation.applied_patch_summary(
        result,
        fallback_patch_summary_fn=_fallback_patch_summary,
    )


_failure_output_excerpt = _existing_failure.failure_output_excerpt


def build_existing_failure_context(
    repo: Path,
    result: ApplyResult,
    validation: TestValidation,
    selected_failures: frozenset[str],
) -> Path:
    return _existing_failure.build_existing_failure_context(
        repo,
        result,
        validation,
        selected_failures,
        working_tree_section_fn=_working_tree_section,
        fallback_patch_summary_fn=_fallback_patch_summary,
        get_context_task_fn=get_context_task,
    )

show_existing_failure_send_instructions = _presentation.show_existing_failure_send_instructions


_choose_existing_failures = _presentation.choose_existing_failures


def _post_apply_choice(
    repo: Path,
    result: ApplyResult,
    test_result,
    test_error: Exception | None,
    *,
    recommend_undo: bool = False,
    recommend_keep_for_repair: bool = False,
    validation: TestValidation | None = None,
) -> None:
    return _service.post_apply_choice(
        repo,
        result,
        test_result,
        test_error,
        recommend_undo=recommend_undo,
        recommend_keep_for_repair=recommend_keep_for_repair,
        validation=validation,
        get_context_kind_fn=get_context_kind,
        mark_followup_resolved_fn=mark_followup_resolved,
        build_followup_context_fn=build_followup_context,
        applied_patch_summary_fn=_applied_patch_summary,
        show_followup_send_instructions_fn=show_followup_send_instructions,
        choose_existing_failures_fn=_choose_existing_failures,
        build_existing_failure_context_fn=build_existing_failure_context,
        show_existing_failure_send_instructions_fn=show_existing_failure_send_instructions,
        undo_last_patch_fn=undo_last_patch,
        verify_undo_fn=_verify_undo,
        clear_repair_context_fn=_clear_repair_context,
    )


def _run_apply_flow(
    repo: Path,
    patch_file: Path,
    *,
    yes: bool = False,
    new_task: bool = False,
) -> ApplyResult:
    return _service.run_apply_flow(
        repo,
        patch_file,
        yes=yes,
        new_task=new_task,
        cli_apply_invocation=_CLI_APPLY_INVOCATION,
        apply_patch_core_fn=_apply_patch_core,
        patch_supersedes_active_repair_fn=_patch_supersedes_active_repair,
        get_active_repair_targets_fn=get_active_repair_targets,
        capture_repository_snapshot_fn=_capture_repository_snapshot,
        clear_phase_test_reports_fn=_clear_phase_test_reports,
        load_verified_baseline_fn=_load_verified_baseline,
        run_background_baseline_fn=_run_background_baseline,
        build_patch_summary_fn=_build_patch_summary,
        fallback_patch_summary_fn=_fallback_patch_summary,
        ask_yes_no_fn=_ask_yes_no,
        open_diff_window_fn=_open_diff_window,
        preserve_test_report_fn=_preserve_test_report,
        show_test_result_fn=_show_test_result,
        status_fn=_status,
        start_background_full_suite_fn=_start_background_full_suite,
        save_verified_baseline_fn=save_verified_baseline,
        classify_test_validation_fn=_classify_test_validation,
        show_test_validation_fn=_show_test_validation,
        build_test_failure_repair_context_fn=build_test_failure_repair_context,
        show_repair_send_instructions_fn=show_repair_send_instructions,
        post_apply_choice_fn=_post_apply_choice,
    )

def apply_patch(
    repo: Path,
    patch_file: Path,
    *,
    new_task: bool = False,
) -> ApplyResult:
    return _patch_api.apply_patch(
        repo,
        patch_file,
        new_task=new_task,
        cli_apply_invocation=_CLI_APPLY_INVOCATION,
        cli_apply_yes=_CLI_APPLY_YES,
        run_apply_flow_fn=_run_apply_flow,
        apply_patch_core_fn=_apply_patch_core,
    )


def undo_last_patch(
    repo: Path,
) -> UndoResult:
    return _patch_api.undo_last_patch(
        repo,
        undo_last_patch_fn=_undo.undo_last_patch,
        validate_patch_paths_fn=validate_patch_paths,
    )

