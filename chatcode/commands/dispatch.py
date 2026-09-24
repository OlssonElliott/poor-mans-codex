"""Top-level ChatCode CLI command dispatch."""
from __future__ import annotations

import sys
from collections.abc import Callable

from ..git_utils import GitError
from ..history import HistoryError
from ..indexing.index_manager import SemanticIndexInterrupted
from ..patching.errors import PatchError, PatchUndoError
from ..test_runner import TestError


def run(
    *,
    create_parser_fn: Callable,
    command_status_fn: Callable,
    command_context_fn: Callable,
    command_followup_fn: Callable,
    command_apply_fn: Callable,
    command_undo_fn: Callable,
    command_test_fn: Callable,
    command_check_fn: Callable,
    command_repair_fn: Callable,
    command_history_fn: Callable,
    command_review_fn: Callable,
) -> None:
    parser = create_parser_fn()
    args = parser.parse_args()

    try:
        match args.command:
            case "status":
                command_status_fn(args.reindex)

            case "context":
                command_context_fn(
                    args.task,
                    patch_oriented=False,
                )

            case "patch-context":
                command_context_fn(
                    args.task,
                    patch_oriented=True,
                )

            case "followup":
                raise SystemExit(command_followup_fn())

            case "apply":
                returncode = command_apply_fn(
                    args.patch,
                    args.no_test,
                    args.no_review,
                    args.new_task,
                )
                if returncode != 0:
                    sys.exit(returncode)

            case "undo":
                returncode = command_undo_fn(args.no_test)
                if returncode != 0:
                    sys.exit(returncode)

            case "test":
                returncode = command_test_fn()
                if returncode != 0:
                    sys.exit(returncode)

            case "check":
                returncode = command_check_fn()
                if returncode != 0:
                    sys.exit(returncode)

            case "repair":
                returncode = command_repair_fn()
                if returncode != 0:
                    sys.exit(returncode)

            case "history":
                command_history_fn()

            case "review":
                command_review_fn(args.index)

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
