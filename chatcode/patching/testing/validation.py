"""Test validation classification, rendering, and report helpers."""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..models import TestValidation
from ...workspace import get_test_results_dir


def classify_test_validation(
    baseline,
    targeted,
    full,
    repair_targets: frozenset[str] = frozenset(),
    *,
    full_suite_pending: bool = False,
    infrastructure_error: bool = False,
) -> TestValidation:
    """Classify post-patch failures without claiming more than the output shows."""
    if full_suite_pending:
        return TestValidation(
            baseline, targeted, full, "pending",
            repair_targets=repair_targets,
        )
    if (
        infrastructure_error
        and full is None
        and (targeted is None or targeted.returncode == 0)
    ):
        return TestValidation(
            baseline, targeted, full, "infrastructure_error",
            repair_targets=repair_targets,
        )
    if targeted is None and full is None:
        return TestValidation(
            baseline, targeted, full, "unavailable",
            repair_targets=repair_targets,
        )
    # Targeted tests are the explicit success criterion for this patch.  They
    # remain authoritative even when their failing test was already red in the
    # baseline; the baseline only determines full-suite regressions.
    if targeted is not None and targeted.returncode != 0:
        targeted_failures = getattr(targeted, "failed_tests", frozenset())
        before = getattr(baseline, "failed_tests", frozenset())
        full_failures = getattr(full, "failed_tests", frozenset()) if full else frozenset()
        remaining_targets = targeted_failures & repair_targets
        new_failures = (
            full_failures - before
            if baseline is not None else frozenset()
        )
        existing_failures = (
            full_failures & before
            if baseline is not None else frozenset()
        )
        fixed_failures = (
            before - full_failures
            if baseline is not None and full is not None else frozenset()
        )
        return TestValidation(
            baseline, targeted, full,
            "repair_failed" if remaining_targets else "targeted_failed",
            new_failures=new_failures,
            existing_failures=existing_failures,
            fixed_failures=fixed_failures,
            repair_targets=repair_targets,
            remaining_repair_failures=remaining_targets,
        )

    if full is not None and full.returncode == 0:
        before = getattr(baseline, "failed_tests", frozenset())
        return TestValidation(
            baseline, targeted, full,
            "repair_passed" if repair_targets else "passed",
            fixed_failures=before,
            repair_targets=repair_targets,
        )
    if baseline is None:
        return TestValidation(baseline, targeted, full, "unclear")

    before = getattr(baseline, "failed_tests", frozenset())
    # Regressions are deliberately the full-suite delta, B - A.  A targeted
    # failure is handled above as a separate, patch-specific criterion.
    after = getattr(full, "failed_tests", frozenset()) if full is not None else frozenset()
    remaining_targets = after & repair_targets
    if remaining_targets:
        return TestValidation(
            baseline, targeted, full, "repair_failed",
            new_failures=after - before,
            existing_failures=after & before,
            fixed_failures=before - after,
            repair_targets=repair_targets,
            remaining_repair_failures=remaining_targets,
        )
    if baseline.returncode != 0 and not before:
        # We know the baseline failed, but not *which* test failed.  No
        # post-patch identifier can safely be called new in that situation.
        return TestValidation(baseline, targeted, full, "unclear")
    if full is None or not before and not after:
        return TestValidation(baseline, targeted, full, "unclear")
    new = after - before
    existing = after & before
    if new:
        return TestValidation(
            baseline, targeted, full, "regressions", new, existing,
            fixed_failures=before - after,
        )
    if existing:
        return TestValidation(
            baseline, targeted, full, "existing", frozenset(), existing,
            fixed_failures=before - after,
        )
    return TestValidation(baseline, targeted, full, "unclear")


def show_test_validation(
    validation: TestValidation,
    *,
    status_fn: Callable[[str, str | None], str],
) -> None:
    functional_status = {
        "passed": ("PASSED", "green"),
        "existing": ("PASS WITH PRE-EXISTING FAILURES", "yellow"),
        "repair_passed": ("REPAIR SUCCESSFUL", "green"),
        "regressions": ("FAILED", "red"),
        "repair_failed": ("REPAIR UNSUCCESSFUL", "red"),
        "targeted_failed": ("FAILED", "red"),
        "pending": ("FULL SUITE RUNNING IN BACKGROUND", "cyan"),
        "infrastructure_error": ("TEST INFRASTRUCTURE ERROR", "yellow"),
    }.get(validation.status, ("REVIEW REQUIRED", "yellow"))
    print("\nFunctional validation: " + status_fn(*functional_status))
    for label, result in (
        ("Baseline", validation.baseline),
        ("Relevant tests", validation.targeted),
        ("Full suite", validation.full),
    ):
        if result is None:
            if label == "Full suite" and validation.status == "pending":
                print(status_fn("[WAIT] Full suite: running in background", "cyan"))
            else:
                print(status_fn(f"[WARN] {label}: not run", "yellow"))
        else:
            expected_repair_baseline = (
                label == "Baseline"
                and validation.status == "repair_passed"
                and result.returncode != 0
                and bool(
                    getattr(result, "failed_tests", frozenset())
                    & validation.repair_targets
                )
            )
            if expected_repair_baseline:
                print(status_fn(
                    f"[EXPECTED] {label}: failed before repair; target reproduced ({result.command})",
                    "yellow",
                ))
                continue
            if label == "Full suite" and validation.status == "existing":
                count = len(validation.existing_failures)
                plural = "failure remains" if count == 1 else "failures remain"
                print(status_fn(
                    f"[WARN] {label}: same {count} pre-existing {plural} ({result.command})",
                    "yellow",
                ))
                continue
            if label == "Baseline" and validation.status == "existing":
                count = len(validation.existing_failures)
                print(status_fn(f"[FAIL] {label}: {count} existing failure(s) ({result.command})", "red"))
                continue
            status = "passed" if result.returncode == 0 else "failed"
            color = "green" if result.returncode == 0 else "red"
            symbol = "[OK]" if result.returncode == 0 else "[FAIL]"
            print(status_fn(f"{symbol} {label}: {status} ({result.command})", color))

    if validation.status == "repair_passed":
        print(status_fn(
            "[OK] Assessment: all repair targets now pass and no regressions were detected.",
            "green",
        ))
    elif validation.status == "passed":
        print(status_fn("[OK] Assessment: no new regressions detected.", "green"))
    elif validation.status == "repair_failed":
        print(status_fn("[FAIL] Repair target failures remain: " + ", ".join(
            sorted(validation.remaining_repair_failures)
        ), "red"))
        print("Recommended action: review the diff/test output, then undo this unsuccessful repair and create a fresh repair context.")
    elif validation.status == "regressions":
        print(status_fn("REGRESSION DETECTED", "red"))
        print(status_fn("Failed: " + ", ".join(sorted(validation.new_failures)), "red"))
        print("Recommended action: run `chatcode repair` to create repair context, then send it to ChatGPT.")
    elif validation.status == "existing":
        print(status_fn(
            f"[WARN] {len(validation.existing_failures)} pre-existing failure(s) remain: "
            + ", ".join(sorted(validation.existing_failures)),
            "yellow",
        ))
        print("[OK] Assessment: no new regressions detected; existing failures predate this patch.")
    elif validation.status == "targeted_failed":
        print(status_fn("[FAIL] Relevant tests failed; targeted validation remains required for this patch.", "red"))
        print("Recommended action: repair the targeted failures before keeping this patch.")
    elif validation.status == "pending":
        print(status_fn(
            "[WAIT] Assessment: immediate validation passed; the full suite "
            "continues in the background.",
            "cyan",
        ))
        print("No repair action is needed unless the background suite reports a failure.")
    elif validation.status == "infrastructure_error":
        print(status_fn(
            "[WARN] Assessment: no failing test was observed, but full "
            "validation could not be started.",
            "yellow",
        ))
        print("Recommended action: run `chatcode test` later; no repair context was created.")
    else:
        print(status_fn("[WARN] Assessment: failures are unclear and require review.", "yellow"))
        print("Recommended action: run `chatcode repair` and send the context to ChatGPT for review.")

    targeted = "PASS" if validation.targeted and validation.targeted.returncode == 0 else "FAIL" if validation.targeted else "N/A"
    full = "PASS" if validation.full and validation.full.returncode == 0 else "FAIL" if validation.full else "N/A"
    regression_count = len(validation.new_failures)
    targeted_color = "green" if targeted == "PASS" else "red" if targeted == "FAIL" else "yellow"
    full_color = "green" if full == "PASS" else "red" if full == "FAIL" else "yellow"
    regression_color = "green" if regression_count == 0 and validation.status in {"passed", "repair_passed", "existing"} else "red" if regression_count else "yellow"
    print(" | ".join([
        status_fn("APPLY: PASS", "green"),
        status_fn(f"TARGETED: {targeted}", targeted_color),
        status_fn(f"FULL SUITE: {full}", full_color),
        status_fn(f"REGRESSIONS: {regression_count}", regression_color),
        *(
            [status_fn(f"PRE-EXISTING: {len(validation.existing_failures)}", "yellow")]
            if validation.existing_failures else []
        ),
        *(
            [status_fn("REPAIR: PASS | TARGET FAILURES REMAIN: 0", "green")]
            if validation.status == "repair_passed" else []
        ),
        *(
            [status_fn(
                f"REPAIR: FAIL | TARGET FAILURES REMAIN: {len(validation.remaining_repair_failures)}",
                "red",
            )]
            if validation.status == "repair_failed" else []
        ),
    ]))


def show_test_result(
    test_result,
    *,
    status_fn: Callable[[str, str | None], str],
) -> None:
    status = "PASSED" if test_result.returncode == 0 else "FAILED"
    symbol = "[OK]" if test_result.returncode == 0 else "[FAIL]"
    color = "green" if test_result.returncode == 0 else "red"
    print(status_fn(
        f"{symbol} TESTS {status}: {test_result.command} "
        f"({test_result.duration_seconds:.2f}s)",
        color,
    ))
    print(f"Test report: {test_result.output_file}")


def preserve_test_report(repo: Path, test_result, phase: str):
    """Keep phase output for repair context while ``latest.md`` is replaced."""
    destination = get_test_results_dir(repo) / f"{phase}.md"
    try:
        shutil.copyfile(test_result.output_file, destination)
        return replace(test_result, output_file=destination)
    except (OSError, TypeError):
        return test_result


def clear_phase_test_reports(repo: Path) -> None:
    for name in ("baseline.md", "targeted.md", "full-suite.md"):
        try:
            (get_test_results_dir(repo) / name).unlink(missing_ok=True)
        except OSError:
            pass


def run_background_baseline(repo: Path):
    """Run the existing project-wide test command without terminal output."""
    from ...test_runner import run_project_tests
    return run_project_tests(repo)
