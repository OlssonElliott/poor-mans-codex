"""High-level patch application workflow orchestration."""
from __future__ import annotations

import sys
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .errors import PatchError
from .models import ApplyResult, PatchPreview, TestValidation, UndoResult
from ..history import HistoryError


def post_apply_choice(
    repo: Path,
    result: ApplyResult,
    test_result,
    test_error: Exception | None,
    *,
    recommend_undo: bool = False,
    recommend_keep_for_repair: bool = False,
    validation: TestValidation | None = None,
    get_context_kind_fn: Callable[[Path], str | None],
    mark_followup_resolved_fn: Callable[[Path], None],
    build_followup_context_fn: Callable,
    applied_patch_summary_fn: Callable[[ApplyResult], str],
    show_followup_send_instructions_fn: Callable[[Path], None],
    choose_existing_failures_fn: Callable[[frozenset[str]], frozenset[str]],
    build_existing_failure_context_fn: Callable,
    show_existing_failure_send_instructions_fn: Callable[[Path], None],
    undo_last_patch_fn: Callable[[Path], UndoResult],
    verify_undo_fn: Callable[[Path, Path], None],
    clear_repair_context_fn: Callable[[Path], None],
) -> None:
    from ..history import open_history_review

    failed = (
        test_result is not None
        and test_result.returncode != 0
    )
    pre_existing_only = validation is not None and validation.status == "existing"
    followup_eligible = validation is not None and validation.status in {"passed", "repair_passed"}
    followup_feedback: str | None = None

    if followup_eligible:
        while True:
            answer = input("Did the patch solve your problem? [Y/n] ").strip().lower()
            if answer in {"", "y", "yes"}:
                if get_context_kind_fn(repo) == "followup":
                    mark_followup_resolved_fn(repo)
                print("Changes kept.")
                return
            if answer in {"n", "no"}:
                break
            print("Choose Y or N.")
        while not followup_feedback:
            followup_feedback = input("What is still not working?\n> ").strip()
            if not followup_feedback:
                print("Please describe what is still not working.")
        try:
            context = build_followup_context_fn(
                repo, result, validation, followup_feedback, applied_patch_summary_fn(result),
            )
            show_followup_send_instructions_fn(context)
            print("Changes kept.")
            return
        except (OSError, PatchError) as exc:
            print(f"Could not create follow-up context: {exc}")

    while True:
        if failed:
            undo_label = "[U] Undo (recommended)" if recommend_undo else "[U] Undo"
            keep_label = (
                "[K] Keep changes (recommended for repair)"
                if recommend_keep_for_repair else "[K] Keep changes"
            )
            if pre_existing_only:
                print(f"{keep_label}  [F] Keep changes + create fix context for existing failure  {undo_label}  [R] Review diff  [T] Show test output")
            else:
                print(f"{keep_label}  {undo_label}  [R] Review diff  [T] Show test output")
            choice = input("> ").strip().lower()
            if not choice and recommend_undo:
                choice = "u"
            elif not choice and recommend_keep_for_repair:
                choice = "k"
        else:
            retry = "  [F] Retry follow-up context" if followup_feedback else ""
            print("[K] Keep  [U] Undo  [R] Review diff" + retry)
            choice = input("> ").strip().lower() or "k"

        if choice == "k":
            print("Changes kept.")
            return

        if choice == "r":
            try:
                open_history_review(
                    result.history_entry
                )
            except HistoryError as exc:
                print(f"Could not open review: {exc}")
            continue

        if choice == "t" and failed and test_result is not None:
            try:
                print(
                    test_result.output_file.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                )
            except OSError as exc:
                print(f"Could not read test output: {exc}")
            continue

        if choice == "f" and followup_feedback and validation is not None:
            try:
                context = build_followup_context_fn(
                    repo, result, validation, followup_feedback, applied_patch_summary_fn(result),
                )
                show_followup_send_instructions_fn(context)
                print("Changes kept.")
                return
            except (OSError, PatchError) as exc:
                print(f"Could not create follow-up context: {exc}")
                continue

        if choice == "f" and pre_existing_only and validation is not None:
            selected = choose_existing_failures_fn(validation.existing_failures)
            if not selected:
                continue
            try:
                context = build_existing_failure_context_fn(repo, result, validation, selected)
                show_existing_failure_send_instructions_fn(context)
            except OSError as exc:
                print(f"Could not create existing failure context: {exc}")
                continue
            print("Changes kept.")
            return

        if choice == "u":
            undone = undo_last_patch_fn(repo)
            verify_undo_fn(
                repo,
                undone.history_entry,
            )
            print("Changes restored successfully.")
            if recommend_keep_for_repair or (
                validation is not None and validation.status == "repair_failed"
            ):
                clear_repair_context_fn(repo)
                print("Repair context invalidated because its working-tree baseline was undone.")
            return

        if test_error is not None and choice == "t":
            print(f"No test report is available: {test_error}")
            continue

        extra = ", T, or F." if (failed and pre_existing_only) or followup_feedback else ", or T." if failed else "."
        print("Choose K, U, R" + extra)


def run_apply_flow(
    repo: Path,
    patch_file: Path,
    *,
    yes: bool = False,
    new_task: bool = False,
    cli_apply_invocation: bool,
    apply_patch_core_fn: Callable,
    patch_supersedes_active_repair_fn: Callable,
    get_active_repair_targets_fn: Callable[[Path], frozenset[str]],
    capture_repository_snapshot_fn: Callable,
    clear_phase_test_reports_fn: Callable[[Path], None],
    load_verified_baseline_fn: Callable,
    run_background_baseline_fn: Callable,
    build_patch_summary_fn: Callable,
    fallback_patch_summary_fn: Callable,
    ask_yes_no_fn: Callable,
    open_diff_window_fn: Callable,
    preserve_test_report_fn: Callable,
    show_test_result_fn: Callable,
    status_fn: Callable[[str, str | None], str],
    start_background_full_suite_fn: Callable,
    save_verified_baseline_fn: Callable,
    classify_test_validation_fn: Callable,
    show_test_validation_fn: Callable,
    build_test_failure_repair_context_fn: Callable,
    show_repair_send_instructions_fn: Callable,
    post_apply_choice_fn: Callable,
) -> ApplyResult:
    preview = apply_patch_core_fn(
        repo,
        patch_file,
        dry_run=True,
        new_task=new_task,
    )
    assert isinstance(preview, PatchPreview)
    repair_targets = (
        frozenset()
        if patch_supersedes_active_repair_fn(
            repo,
            preview.paths,
            force=new_task,
        )
        else get_active_repair_targets_fn(repo)
    )

    # Start only after the candidate has passed all non-mutating applicability
    # checks. The snapshot is rechecked immediately before mutation below.
    baseline_snapshot = capture_repository_snapshot_fn(repo)
    clear_phase_test_reports_fn(repo)
    cached_baseline = load_verified_baseline_fn(repo, baseline_snapshot)
    baseline_executor: ThreadPoolExecutor | None = None
    baseline_future: Future | None = None
    if cached_baseline is not None:
        print("Pre-patch baseline: using verified cached result.")
        print("[OK] Repository state unchanged since last full-suite validation." if cached_baseline.returncode == 0
              else f"[WARN] {len(cached_baseline.failed_tests)} known pre-existing failure(s).")
    elif cli_apply_invocation:
        print(
            "Pre-patch baseline: no verified cached result; "
            "continuing without blocking."
        )
        print(
            "Failures from this apply cannot be classified against a baseline. "
            "The post-patch full suite will establish one for the next apply."
        )
    else:
        baseline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatcode-baseline")
        baseline_future = baseline_executor.submit(run_background_baseline_fn, repo)
        print("Starting pre-patch test baseline in background...")

    summary = build_patch_summary_fn(
        repo,
        preview,
    ) if not repair_targets else "\n".join([
        "Repair candidate (summary derived from the diff, not AI):",
        fallback_patch_summary_fn(preview.patch_text, preview.paths),
        "Repair targets:",
        *(f"  - {target}" for target in sorted(repair_targets)),
    ])
    print("Patch is technically applicable.\n")
    print("Planned changes:")
    print(summary)

    interactive = (
        sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    if not interactive and not yes:
        if baseline_future is not None:
            baseline_future.cancel()
        if baseline_executor is not None:
            baseline_executor.shutdown(wait=False, cancel_futures=True)
        raise PatchError(
            "Non-interactive apply requires --yes. "
            "No repository files were changed."
        )

    if not yes:
        if ask_yes_no_fn(
            "View full diff before applying?",
            default=True,
        ):
            open_diff_window_fn(repo, preview.patch_text, preview.paths)

        if not ask_yes_no_fn(
            "Apply these changes?",
            default=False,
        ):
            if baseline_future is not None:
                baseline_future.cancel()
            if baseline_executor is not None:
                baseline_executor.shutdown(wait=False, cancel_futures=True)
            print("Apply cancelled. No repository files were changed.")
            raise PatchError(
                "Apply cancelled by user."
            )

    baseline = None
    baseline_error: Exception | None = None
    try:
        current_snapshot = capture_repository_snapshot_fn(repo)
        if cached_baseline is not None and current_snapshot == baseline_snapshot:
            baseline = cached_baseline
        elif cli_apply_invocation:
            if current_snapshot != baseline_snapshot:
                print(
                    "Repository changed during review; the cached pre-patch "
                    "baseline no longer applies."
                )
        elif current_snapshot != baseline_snapshot:
            # Do not overlap a replacement suite with the discarded suite:
            # project test tools often share caches and report locations.
            print("Repository changed during review; discarding pre-patch baseline.")
            if baseline_future is not None and baseline_executor is not None:
                baseline_future.cancel()
                try:
                    baseline_future.result()
                except Exception:
                    pass
                baseline_executor.shutdown(wait=True, cancel_futures=True)
            print("Running fresh pre-patch test baseline...")
            baseline = run_background_baseline_fn(repo)
        else:
            assert baseline_future is not None and baseline_executor is not None
            if not baseline_future.done():
                print("Waiting for pre-patch baseline...")
            baseline = baseline_future.result()
            baseline_executor.shutdown(wait=True, cancel_futures=True)
        if baseline is not None:
            baseline = preserve_test_report_fn(repo, baseline, "baseline")
            show_test_result_fn(baseline)
    except Exception as exc:
        # Match the established baseline-infrastructure-error path. Test
        # failures are TestResult values and never arrive here as exceptions.
        baseline_error = exc
        if baseline_executor is not None:
            baseline_executor.shutdown(wait=False, cancel_futures=True)
        print(f"Pre-patch tests could not be run: {exc}")

    result = apply_patch_core_fn(
        repo,
        patch_file,
        new_task=new_task,
    )
    assert isinstance(result, ApplyResult)

    print("\n" + status_fn("[OK] PATCH APPLIED", "green"))
    print("Running relevant tests, then the full suite...")

    test_result = None
    relevant_result = None
    full_suite_pending = False
    test_error: Exception | None = baseline_error
    try:
        from ..history import update_history_test_result
        from ..test_runner import (
            TestError,
            run_project_tests,
            run_relevant_tests,
            unmapped_python_source_paths,
        )

        unmapped_sources = unmapped_python_source_paths(repo, result.paths)
        if unmapped_sources:
            print(status_fn(
                "[WARN] No direct test-file match could be identified for:",
                "yellow",
            ))
            for path in unmapped_sources[:10]:
                print(f"  {path}")
            if len(unmapped_sources) > 10:
                print(f"  ... and {len(unmapped_sources) - 10} more")
            print(
                "This does not prove that test coverage is missing. "
                "The full suite will still run."
            )

        try:
            relevant_result = run_relevant_tests(repo, result.paths)
            if relevant_result is not None:
                relevant_result = preserve_test_report_fn(repo, relevant_result, "targeted")
                show_test_result_fn(relevant_result)
        except TestError as exc:
            # A narrow selection is only an optimization.  It must never
            # prevent the required full-suite validation from running.
            test_error = exc
            print(f"Relevant tests could not be run: {exc}")

        remaining_after_targeted = (
            getattr(relevant_result, "failed_tests", frozenset())
            & repair_targets
            if relevant_result is not None and relevant_result.returncode != 0
            else frozenset()
        )
        if remaining_after_targeted:
            update_history_test_result(result.history_entry, "REPAIR_FAILED")
            print(status_fn(
                "[FAIL] Repair candidate rejected by relevant tests; "
                "the full suite was skipped.",
                "red",
            ))
        else:
            try:
                full_snapshot = capture_repository_snapshot_fn(repo)
                targeted_failed = (
                    relevant_result is not None
                    and relevant_result.returncode != 0
                )
                if cli_apply_invocation and not targeted_failed:
                    start_background_full_suite_fn(repo, full_snapshot, result)
                    full_suite_pending = True
                    update_history_test_result(
                        result.history_entry,
                        "PENDING",
                    )
                    print(
                        "Full test suite started in the background. "
                        "Its verified result will be used by the next apply."
                    )
                else:
                    if cli_apply_invocation and targeted_failed:
                        print(status_fn(
                            "Relevant tests failed; waiting for the full suite "
                            "before creating repair context...",
                            "yellow",
                        ))
                    test_result = run_project_tests(repo)
                    test_result = preserve_test_report_fn(repo, test_result, "full-suite")
                    if capture_repository_snapshot_fn(repo) == full_snapshot:
                        save_verified_baseline_fn(repo, full_snapshot, test_result)
                    update_history_test_result(
                        result.history_entry,
                        "PASSED" if test_result.returncode == 0 else "FAILED",
                        command=test_result.command,
                        returncode=test_result.returncode,
                        duration_seconds=test_result.duration_seconds,
                    )
                    show_test_result_fn(test_result)
            except (TestError, OSError) as exc:
                test_error = exc
                update_history_test_result(
                    result.history_entry,
                    "ERROR",
                )
                print(f"Tests could not be run: {exc}")
    except HistoryError as exc:
        test_error = exc
        print(f"Could not save test status: {exc}")

    validation = classify_test_validation_fn(
        baseline,
        relevant_result,
        test_result,
        repair_targets,
        full_suite_pending=full_suite_pending,
        infrastructure_error=test_error is not None,
    )
    show_test_validation_fn(validation)
    validation_passed = validation.status in {
        "passed", "repair_passed", "existing", "pending",
    }
    repair_context_needed = (
        not validation_passed
        and validation.status != "infrastructure_error"
    )
    if repair_context_needed:
        try:
            repair_context = build_test_failure_repair_context_fn(
                repo, result, validation
            )
            color = "red" if validation.status == "regressions" else "yellow"
            print(status_fn("Repair context created.", color))
            show_repair_send_instructions_fn(repair_context)
        except OSError as exc:
            print(status_fn(f"Could not create repair context: {exc}", "yellow"))

    print("\nWhat changed:")
    print(summary)

    if interactive:
        post_apply_choice_fn(
            repo,
            result,
            (
                test_result
                if test_result is not None and test_result.returncode != 0
                else relevant_result
            ),
            test_error,
            recommend_keep_for_repair=repair_context_needed,
            validation=validation,
        )
    elif not validation_passed:
        print(
            "Functional validation did not pass during non-interactive "
            "--yes apply. Changes were kept; review the test report above."
        )
    elif test_error is not None:
        print(
            "Tests were unavailable during non-interactive --yes apply. "
            "Changes were kept."
        )

    return result
