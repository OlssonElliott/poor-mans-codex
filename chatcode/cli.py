from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .context_builder import (
    build_safe_status,
    build_context,
    build_patch_context,
)
from .context_state import get_context_task, save_context_state
from .git_utils import (
    GitError,
    get_branch,
    get_repo_root,
    get_status,
)
from .history import (
    HistoryError,
    format_history,
    open_history_review,
    review_history_by_index,
    update_history_test_result,
)
from .indexing.index_manager import (
    IndexProgress,
    SemanticIndexInterrupted,
    update_project_map,
)
from .patch import (
    PatchAlreadyApplied,
    PatchError,
    PatchUndoError,
    apply_patch,
    build_check_repair_context,
    get_repair_context_stale_reason,
    get_repair_context_targets,
    refresh_test_failure_repair_context,
    show_repair_send_instructions,
    show_check_repair_send_instructions,
    undo_last_patch,
)
from .test_runner import (
    TestError,
    run_project_tests,
)
from .workspace import (
    get_default_patch_file,
    get_repair_context_file,
)


def open_folder(
    path: Path,
) -> None:
    path = path.resolve()

    try:
        if sys.platform == "win32":
            os.startfile(path)

        elif sys.platform == "darwin":
            subprocess.Popen([
                "open",
                str(path),
            ])

        else:
            subprocess.Popen([
                "xdg-open",
                str(path),
            ])

    except OSError as exc:
        print(
            "Could not open folder "
            f"automatically: {exc}",
            file=sys.stderr,
        )


def command_status(reindex: bool = False) -> None:
    repo = get_repo_root()

    if reindex:
        print("Forcing a full project-index rebuild...")
        update_project_map(
            repo,
            progress=ConsoleIndexReporter(),
            force_rebuild=True,
        )
        print()

    print(f"Repository: {repo}")
    print(
        f"Branch:     {get_branch(repo)}"
    )
    print()

    status = build_safe_status(repo)

    if status:
        print("Changes:")
        print(status)
    else:
        print("Working tree clean.")


class ConsoleIndexReporter:
    def __init__(self) -> None:
        self._progress_line = False
        self._initial = False

    def _finish_progress_line(self) -> None:
        if self._progress_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_line = False

    def __call__(self, event: IndexProgress) -> None:
        if event.phase == "index_mode" and event.status == "selected":
            print(f"Index mode: {event.reason}.")
        elif event.phase == "index" and event.status == "initial":
            self._initial = True
            print("Project map not found. Building initial project index...")
        elif event.phase == "index" and event.status == "forced":
            self._initial = True
            print("Cached project map discarded. Rebuilding from source...")
        elif event.phase == "scan" and event.status == "started":
            print("Scanning project files...")
        elif event.phase == "index" and event.status == "updating":
            print("Project changes detected. Updating project index...")
        elif event.phase == "index" and event.status == "up_to_date":
            print("Project index up to date.")
        elif event.phase == "static" and event.status == "complete":
            print(f"Static analysis complete: {event.total} source files.")
        elif event.phase == "static" and event.status == "saved":
            print("Project map saved.")
            if self._initial:
                print(f"Project index created: {event.total} source files.")
            else:
                print(
                    "Project index updated: "
                    f"{event.changed} changed, {event.added} new, "
                    f"{event.deleted} removed."
                )
        elif event.phase == "semantic" and event.status == "started":
            print(
                "Semantic cache: "
                f"{event.reused} reused, {event.invalidated} invalidated, "
                f"{event.pending} pending."
            )
            print(f"Running semantic analysis with {event.model}...")
        elif event.phase == "semantic" and event.status == "progress":
            width = 20
            filled = round(width * event.completed / event.total) if event.total else width
            bar = "█" * filled + "░" * (width - filled)
            details = (
                f"Semantic indexing [{bar}] "
                f"{event.completed}/{event.total}"
            )
            if event.current_file:
                details += f" | {event.current_file}"
            if event.eta_seconds is not None:
                if event.eta_seconds < 60:
                    details += f" | ~{round(event.eta_seconds)}s remaining"
                else:
                    details += f" | ~{round(event.eta_seconds / 60)}m remaining"
            sys.stdout.write("\r" + details.ljust(120))
            sys.stdout.flush()
            self._progress_line = True
        elif event.phase == "semantic" and event.status == "complete":
            self._finish_progress_line()
            if self._initial:
                print(f"Semantic analysis complete: {event.completed} files.")
            else:
                print(f"Semantic analysis updated: {event.processed} files.")
            if event.failed:
                labels = {
                    "timeout": "timeout",
                    "ollama_not_found": "Ollama not found",
                    "model_not_found": "model not found",
                    "ollama_process_error": "Ollama process error",
                    "empty_response": "empty response",
                    "invalid_json": "invalid JSON",
                    "schema_validation_error": "schema validation",
                    "unexpected_exception": "unexpected exception",
                }
                print("Qwen semantic analysis failures:")
                for reason, count in event.failure_counts:
                    print(f"  {labels.get(reason, reason)}: {count}")
                print("Example semantic failures:")
                for example in event.failure_examples:
                    print(f"  file: {example.file}")
                    print(f"  reason: {example.reason}")
                    print(f"  error: {example.error}")
                    if example.response:
                        response = example.response.replace("\n", "\\n")
                        print(f"  response: {response}")
                print("Continuing with static index.")
            if event.stopped_early:
                print(
                    "Semantic circuit breaker stopped indexing after "
                    f"{event.processed} files ({event.reason}); "
                    f"{event.remaining} files remain pending."
                )
                print("Continuing with static retrieval for this run.")
        elif event.phase == "semantic_preflight" and event.status == "started":
            print(f"Checking semantic model {event.model}...")
        elif event.phase == "semantic_preflight" and event.status == "complete":
            print("Semantic model check passed.")
        elif event.phase == "semantic_preflight" and event.status == "failed":
            print(f"Semantic model check failed ({event.reason}): {event.error}")
            print("Skipping AI semantic indexing.")
            print("Continuing with static retrieval.")
        elif event.phase == "semantic" and event.status == "interrupted":
            self._finish_progress_line()
            print("Semantic indexing interrupted by user.")
            print(
                f"Progress saved: {event.completed}/{event.total} files completed."
            )


def command_context(
    task: str,
    patch_oriented: bool = False,
) -> None:
    repo = get_repo_root()
    reporter = ConsoleIndexReporter()

    if patch_oriented:
        output = build_patch_context(
            repo,
            task,
            index_progress=reporter,
        )
    else:
        output = build_context(
            repo,
            task,
            index_progress=reporter,
        )

    patch_file = get_default_patch_file(
        repo
    )

    patch_file.write_text(
        "",
        encoding="utf-8",
    )

    print("Context created:")
    print(output)
    print()
    print("Workspace directory:")
    print(output.parent)

    open_folder(
        output.parent
    )


def run_tests(
    repo: Path,
    history_entry: Path | None = None,
) -> int:
    print()
    print("Running tests...")

    try:
        result = run_project_tests(
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
            update_history_test_result(
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
        update_history_test_result(
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


def prompt_review(
    history_entry: Path,
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
                open_history_review(
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
    patch_path: str | None,
    no_test: bool,
    no_review: bool,
) -> int:
    repo = get_repo_root()

    if patch_path is None:
        patch_file = (
            get_default_patch_file(repo)
        )
    else:
        patch_file = Path(
            patch_path
        )

    try:
        result = apply_patch(
            repo,
            patch_file,
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
        update_history_test_result(
            result.history_entry,
            status="SKIPPED",
        )

        print()
        print(
            "Automatic tests skipped."
        )

        test_returncode = 0

    else:
        test_returncode = run_tests(
            repo,
            result.history_entry,
        )

    if not no_review:
        prompt_review(
            result.history_entry
        )

    return test_returncode


def command_undo(
    no_test: bool,
) -> int:
    repo = get_repo_root()

    result = undo_last_patch(
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
        update_history_test_result(
            result.history_entry,
            status="SKIPPED",
        )

        print()
        print(
            "Automatic tests skipped."
        )

        return 0

    return run_tests(
        repo,
        result.history_entry,
    )


def command_test() -> int:
    repo = get_repo_root()

    result = run_project_tests(
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


def _select_check_failures(failures: frozenset[str]) -> frozenset[str]:
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


def command_check() -> int:
    """Report current working-tree health without creating a task by default."""
    repo = get_repo_root()
    print("Running repository health check...")
    try:
        result = run_project_tests(repo)
    except TestError as exc:
        print("\nRepository health: ERROR")
        print("Test command could not complete:")
        print(exc)
        return 2

    try:
        dirty = get_status(repo)
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
            selected = _select_check_failures(failures)
            if selected:
                try:
                    context = build_check_repair_context(repo, result, selected)
                    show_check_repair_send_instructions(context)
                    # The check remains failed (exit 1), but a successful
                    # context handoff completes this interaction.
                    return 1
                except (OSError, PatchError) as exc:
                    print(f"Could not create check repair context: {exc}")
            continue
        print("Choose F, T, or X.")


def command_repair() -> int:
    repo = get_repo_root()
    repair_context = get_repair_context_file(repo)
    if not repair_context.is_file():
        print(
            "No repair context is available. Apply a patch first; ChatCode "
            "creates one automatically when it detects a new regression."
        )
        return 1
    stale_reason = get_repair_context_stale_reason(repo)
    if stale_reason is not None:
        refreshed = refresh_test_failure_repair_context(repo)
        if refreshed is not None:
            repair_context = refreshed
            stale_reason = get_repair_context_stale_reason(repo)
    if stale_reason is not None:
        print("Repair context is stale and must not be sent to ChatGPT:")
        print(stale_reason)
        print("Run chatcode context again, or reproduce the failed apply/test flow.")
        return 1
    repair_bytes = repair_context.read_bytes()
    save_context_state(
        repo,
        task=get_context_task(repo),
        context_sha256=hashlib.sha256(repair_bytes).hexdigest(),
        generation_id=uuid.uuid4().hex,
        context_filename=repair_context.name,
        context_kind="repair",
        repair_targets=sorted(get_repair_context_targets(repo)),
    )
    show_repair_send_instructions(repair_context)
    return 0


def command_history() -> None:
    repo = get_repo_root()

    print(
        format_history(repo)
    )


def command_review(
    index: int,
) -> None:
    repo = get_repo_root()

    record = review_history_by_index(
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatcode",
        description=(
            "Local bridge between your "
            "Git repository and ChatGPT."
        ),
    )

    subparsers = (
        parser.add_subparsers(
            dest="command"
        )
    )

    status_parser = subparsers.add_parser(
        "status",
        help=(
            "Show current repository "
            "status."
        ),
    )

    context_parser = (
        subparsers.add_parser(
            "context",
            help=(
                "Build ChatGPT context "
                "for the current "
                "repository."
            ),
        )
    )

    context_parser.add_argument(
        "task",
        help=(
            "Describe what you want "
            "ChatGPT to work on."
        ),
    )
    status_parser.add_argument(
        "--reindex",
        action="store_true",
        help="Discard and rebuild the cached project index before showing status.",
    )

    patch_context_parser = (
        subparsers.add_parser(
            "patch-context",
            help=(
                "Build patch-oriented context from exact current working-tree files."
            ),
        )
    )

    patch_context_parser.add_argument(
        "task",
        help="Describe the patch ChatGPT should generate.",
    )

    apply_parser = (
        subparsers.add_parser(
            "apply",
            help=(
                "Safely apply a unified "
                "diff and run project "
                "tests."
            ),
        )
    )

    apply_parser.add_argument(
        "patch",
        nargs="?",
        help=(
            "Optional path to a .diff "
            "or .patch file. If omitted, "
            "ChatCode uses the repository's "
            "workspace patches/"
            "incoming.diff."
        ),
    )

    apply_parser.add_argument(
        "--no-test",
        action="store_true",
        help=(
            "Apply the patch without "
            "running tests afterwards."
        ),
    )

    apply_parser.add_argument(
        "--no-review",
        action="store_true",
        help=(
            "Apply the patch without "
            "asking to open review."
        ),
    )

    undo_parser = (
        subparsers.add_parser(
            "undo",
            help=(
                "Safely reverse the latest "
                "ChatCode patch."
            ),
        )
    )

    undo_parser.add_argument(
        "--no-test",
        action="store_true",
        help=(
            "Undo the patch without "
            "running tests afterwards."
        ),
    )

    subparsers.add_parser(
        "test",
        help=(
            "Automatically detect and run "
            "the project's test suite."
        ),
    )

    subparsers.add_parser(
        "check",
        help="Run a standalone repository health check.",
    )

    subparsers.add_parser(
        "repair",
        help="Show a freshness-validated context for the latest failed patch or test flow.",
    )

    subparsers.add_parser(
        "history",
        help=(
            "Show ChatCode patch history."
        ),
    )

    review_parser = (
        subparsers.add_parser(
            "review",
            help=(
                "Open a ChatCode history "
                "entry as a VS Code diff."
            ),
        )
    )

    review_parser.add_argument(
        "index",
        nargs="?",
        type=int,
        default=1,
        help=(
            "History entry number to "
            "review. Defaults to the "
            "latest entry."
        ),
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    try:
        match args.command:
            case "status":
                command_status(args.reindex)

            case "context":
                command_context(
                    args.task,
                    patch_oriented=False,
                )

            case "patch-context":
                command_context(
                    args.task,
                    patch_oriented=True,
                )

            case "apply":
                returncode = command_apply(
                    args.patch,
                    args.no_test,
                    args.no_review,
                )

                if returncode != 0:
                    sys.exit(
                        returncode
                    )

            case "undo":
                returncode = command_undo(
                    args.no_test,
                )

                if returncode != 0:
                    sys.exit(
                        returncode
                    )

            case "test":
                returncode = (
                    command_test()
                )

                if returncode != 0:
                    sys.exit(
                        returncode
                    )

            case "check":
                returncode = command_check()
                if returncode != 0:
                    sys.exit(returncode)

            case "repair":
                returncode = command_repair()
                if returncode != 0:
                    sys.exit(returncode)

            case "history":
                command_history()

            case "review":
                command_review(
                    args.index
                )

            case _:
                parser.print_help()

    except SemanticIndexInterrupted:
        sys.exit(130)

    except KeyboardInterrupt:
        print("\nChatCode interrupted by user.", file=sys.stderr)
        sys.exit(130)

    except (
        GitError,
        HistoryError,
        PatchError,
        PatchUndoError,
        TestError,
    ) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
