"""Argument parser construction for the ChatCode CLI."""
from __future__ import annotations

import argparse


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

    apply_parser.add_argument(
        "--new-task",
        action="store_true",
        help=(
            "Treat incoming.diff as a new independent task and supersede "
            "any active repair generation. Normally ChatCode detects this "
            "automatically when the patch leaves the repair context's file "
            "scope; use this flag when both tasks touch exactly the same files."
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
        "followup",
        help="Regenerate the latest unresolved follow-up using current repository source.",
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
