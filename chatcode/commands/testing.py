"""CLI test execution and repository health checks."""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from ..git_utils import GitError
from ..patching.errors import PatchError
from ..test_runner import TestError


def run_tests(
    repo: Path,
    history_entry: Path | None = None,
    *,
    run_project_tests_fn: Callable,
    update_history_test_result_fn: Callable,
) -> int:
    print()
    print("Running tests...")

    try:
        result = run_project_tests_fn(
            repo
        )

    except TestError as exc:
        print()
        print(
            "Tests could not be run "
            "automatically:"
        )
        print(exc)

        if history_entry is not None:
            update_history_test_result_fn(
                history_entry,
                status="NOT_AVAILABLE",
            )

        return 0

    print(
        f"Command: {result.command}"
    )

    print(
        "Duration: "
        f"{result.duration_seconds:.2f}s"
    )

    print()

    if result.returncode == 0:
        test_status = "PASSED"
        print("Tests passed.")
    else:
        test_status = "FAILED"

        print(
            "Tests failed with exit code "
            f"{result.returncode}."
        )

    if history_entry is not None:
        update_history_test_result_fn(
            history_entry,
            status=test_status,
            command=result.command,
            returncode=result.returncode,
            duration_seconds=(
                result.duration_seconds
            ),
        )

    print()
    print("Test results:")
    print(result.output_file)

    if result.returncode != 0:
        print()
        print(
            'Run: chatcode context '
            '"fixa testerna"'
        )

    return result.returncode


def command_test(
    *,
    get_repo_root_fn: Callable[[], Path],
    run_project_tests_fn: Callable,
) -> int:
    repo = get_repo_root_fn()

    result = run_project_tests_fn(
        repo
    )

    print(
        f"Command: {result.command}"
    )

    print(
        "Duration: "
        f"{result.duration_seconds:.2f}s"
    )

    print()

    if result.returncode == 0:
        print("Tests passed.")
    else:
        print(
            "Tests failed with exit code "
            f"{result.returncode}."
        )

    print()
    print("Test results:")
    print(result.output_file)

    return result.returncode


def select_check_failures(failures: frozenset[str]) -> frozenset[str]:
    ordered = sorted(failures)
    if len(ordered) == 1:
        return frozenset(ordered)
    print("Select failure [number(s), or A for all]:")
    for number, failure in enumerate(ordered, start=1):
        print(f"[{number}] {failure}")
    answer = input("> ").strip().lower()
    if answer == "a":
        return frozenset(ordered)
    try:
        return frozenset(ordered[int(item.strip()) - 1] for item in answer.split(",") if item.strip())
    except (IndexError, ValueError):
        print("No valid failure selected.")
        return frozenset()


def command_check(
    *,
    get_repo_root_fn: Callable[[], Path],
    run_project_tests_fn: Callable,
    capture_repository_snapshot_fn: Callable,
    save_verified_baseline_fn: Callable,
    get_status_fn: Callable[[Path], str],
    build_check_repair_context_fn: Callable,
    show_check_repair_send_instructions_fn: Callable[[Path], None],
    select_check_failures_fn: Callable[[frozenset[str]], frozenset[str]],
) -> int:
    """Report current working-tree health without creating a task by default."""
    repo = get_repo_root_fn()
    print("Running repository health check...")
    snapshot = None
    try:
        snapshot = capture_repository_snapshot_fn(repo)
    except GitError:
        pass
    try:
        result = run_project_tests_fn(repo)
    except TestError as exc:
        print("\nRepository health: ERROR")
        print("Test command could not complete:")
        print(exc)
        return 2
    if snapshot is not None:
        try:
            if capture_repository_snapshot_fn(repo) == snapshot:
                save_verified_baseline_fn(repo, snapshot, result)
        except GitError:
            pass

    try:
        dirty = get_status_fn(repo)
    except GitError:
        dirty = ""
    if dirty:
        lines = dirty.splitlines()
        print(f"Working tree: {len(lines)} changed file(s)")

    if result.returncode == 0:
        print("\nRepository health: PASS")
        print("[OK] Tests passed")
        print("[OK] No failing test cases detected")
        return 0

    failures = result.failed_tests
    if not failures:
        print("\nRepository health: ERROR")
        print("Test command completed unsuccessfully without stable test IDs.")
        print(f"Test output: {result.output_file}")
        if sys.stdin.isatty():
            while True:
                choice = input("[T] Show test output  [X] Exit\n> ").strip().lower()
                if choice in {"", "x"}:
                    break
                if choice == "t":
                    try:
                        print(result.output_file.read_text(encoding="utf-8", errors="replace"))
                    except OSError as exc:
                        print(f"Could not read test output: {exc}")
                    continue
                print("Choose T or X.")
        return 2

    print("\nRepository health: FAIL")
    print(f"{len(failures)} failing test(s):")
    for number, failure in enumerate(sorted(failures), start=1):
        print(f"{number}. {failure}")
    if not sys.stdin.isatty():
        return 1
    while True:
        choice = input("[F] Create fix context  [T] Show test output  [X] Exit\n> ").strip().lower()
        if choice in {"", "x"}:
            return 1
        if choice == "t":
            try:
                print(result.output_file.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                print(f"Could not read test output: {exc}")
            continue
        if choice == "f":
            selected = select_check_failures_fn(failures)
            if selected:
                try:
                    context = build_check_repair_context_fn(repo, result, selected)
                    show_check_repair_send_instructions_fn(context)
                    # The check remains failed (exit 1), but a successful
                    # context handoff completes this interaction.
                    return 1
                except (OSError, PatchError) as exc:
                    print(f"Could not create check repair context: {exc}")
            continue
        print("Choose F, T, or X.")
