from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from chatcode.cli import ConsoleIndexReporter, main
from chatcode.indexing.index_manager import (
    IndexProgress,
    SemanticFailureExample,
    SemanticIndexInterrupted,
)


class IndexProgressCliTests(unittest.TestCase):
    def test_preflight_failure_reports_static_fallback(self) -> None:
        output = io.StringIO()
        reporter = ConsoleIndexReporter()
        with redirect_stdout(output):
            reporter(IndexProgress(
                "semantic_preflight", "started", model="missing-model"
            ))
            reporter(IndexProgress(
                "semantic_preflight",
                "failed",
                model="missing-model",
                reason="model_not_found",
                error="model not found",
            ))
        rendered = output.getvalue()
        self.assertIn("Checking semantic model missing-model", rendered)
        self.assertIn("model_not_found", rendered)
        self.assertIn("Continuing with static retrieval", rendered)

    def test_failure_summary_shows_classification_and_bounded_examples(self) -> None:
        output = io.StringIO()
        reporter = ConsoleIndexReporter()
        event = IndexProgress(
            "semantic",
            "complete",
            completed=3,
            total=3,
            processed=3,
            failed=3,
            failure_counts=(("invalid_json", 2), ("timeout", 1)),
            failure_examples=(
                SemanticFailureExample(
                    "app.py", "invalid_json", "bad JSON", "Here is JSON:\n..."
                ),
            ),
        )

        with redirect_stdout(output):
            reporter(event)

        rendered = output.getvalue()
        self.assertIn("invalid JSON: 2", rendered)
        self.assertIn("timeout: 1", rendered)
        self.assertIn("file: app.py", rendered)
        self.assertIn("response: Here is JSON:\\n...", rendered)

    def test_interrupted_progress_has_clean_user_message(self) -> None:
        output = io.StringIO()
        reporter = ConsoleIndexReporter()

        with redirect_stdout(output):
            reporter(IndexProgress(
                "semantic",
                "interrupted",
                completed=12,
                total=30,
                current_file="bot/commands/give.py",
                model="qwen2.5-coder:7b",
            ))

        rendered = output.getvalue()
        self.assertIn("Semantic indexing interrupted by user.", rendered)
        self.assertIn("Progress saved: 12/30 files completed.", rendered)
        self.assertNotIn("Traceback", rendered)

    def test_cli_uses_interrupt_exit_code_without_traceback(self) -> None:
        output = io.StringIO()
        errors = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys,
            "argv",
            ["chatcode", "context", "task"],
        ), patch(
            "chatcode.cli.command_context",
            side_effect=SemanticIndexInterrupted(1, 2, Path(directory) / "project-map.json"),
        ), redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as stopped:
                main()

        self.assertEqual(stopped.exception.code, 130)
        self.assertNotIn("Traceback", output.getvalue() + errors.getvalue())


if __name__ == "__main__":
    unittest.main()
