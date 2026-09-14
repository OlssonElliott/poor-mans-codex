from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .context_builder import build_context
from .context_state import save_context_state
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
from .patch import (
    PatchAlreadyApplied,
    PatchError,
    PatchUndoError,
    apply_patch,
    undo_last_patch,
)
from .test_runner import (
    TestError,
    run_project_tests,
)
from .workspace import (
    get_default_patch_file,
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


def command_status() -> None:
    repo = get_repo_root()

    print(f"Repository: {repo}")
    print(
        f"Branch:     {get_branch(repo)}"
    )
    print()

    status = get_status(repo)

    if status:
        print("Changes:")
        print(status)
    else:
        print("Working tree clean.")


def command_context(
    task: str,
) -> None:
    repo = get_repo_root()

    output = build_context(
        repo,
        task,
    )

    save_context_state(
        repo
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

    subparsers.add_parser(
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
                command_status()

            case "context":
                command_context(
                    args.task
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

            case "history":
                command_history()

            case "review":
                command_review(
                    args.index
                )

            case _:
                parser.print_help()

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