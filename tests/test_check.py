from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.cli import command_check
from chatcode.test_runner import TestError, TestResult
from chatcode.workspace import get_check_repair_context_file


class CheckCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        (self.repo / ".git").mkdir()
        self.report = self.repo / "test-results.md"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def result(self, returncode: int, failures: frozenset[str] = frozenset()) -> TestResult:
        self.report.write_text(
            "FAIL: test_broken (test_broken.BrokenTests.test_broken)\nAssertionError\n",
            encoding="utf-8",
        )
        return TestResult("python -m unittest", returncode, 0.1, self.report, failures)

    def test_healthy_check_passes_without_creating_context(self) -> None:
        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "chatcode.cli.run_project_tests", return_value=self.result(0)
        ), patch("builtins.print") as output:
            code = command_check()

        self.assertEqual(code, 0)
        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("Repository health: PASS", rendered)
        self.assertFalse(get_check_repair_context_file(self.repo).exists())

    def test_failed_check_reports_stable_id_and_returns_one(self) -> None:
        failure = "test_broken.BrokenTests.test_broken"
        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "chatcode.cli.run_project_tests", return_value=self.result(1, frozenset({failure}))
        ), patch("chatcode.cli.sys.stdin.isatty", return_value=False), patch("builtins.print") as output:
            code = command_check()

        self.assertEqual(code, 1)
        self.assertIn(failure, "\n".join(str(call.args[0]) for call in output.call_args_list if call.args))

    def test_infrastructure_error_returns_two(self) -> None:
        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "chatcode.cli.run_project_tests", side_effect=TestError("interpreter missing")
        ), patch("builtins.print") as output:
            code = command_check()

        self.assertEqual(code, 2)
        self.assertIn("Repository health: ERROR", "\n".join(str(call.args[0]) for call in output.call_args_list if call.args))

    def test_successful_fix_context_is_terminal_but_preserves_failed_exit_code(self) -> None:
        failure = "test_broken.BrokenTests.test_broken"
        context = self.repo / "CHECK_REPAIR_CONTEXT.md"
        context.write_text("context", encoding="utf-8")
        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "chatcode.cli.run_project_tests", return_value=self.result(1, frozenset({failure}))
        ), patch("chatcode.cli.sys.stdin.isatty", return_value=True), patch(
            "chatcode.cli.build_check_repair_context", return_value=context
        ) as build, patch("builtins.input", side_effect=["f"]) as prompt, patch("builtins.print") as output:
            code = command_check()

        self.assertEqual(code, 1)
        build.assert_called_once()
        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertEqual(prompt.call_count, 1)
        self.assertIn("[F] Create fix context", prompt.call_args.args[0])
        self.assertEqual(rendered.count("Check repair context created:"), 1)

    def test_check_context_delegates_source_capture_to_shared_test_root_pipeline(self) -> None:
        from chatcode.patch import build_check_repair_context

        failure = "test_broken.BrokenTests.test_broken"
        source = "===== REQUIRED SOURCE: tests/test_broken.py::test_broken [patch target] =====\ndef test_broken(): pass\n"
        with patch("chatcode.context_builder.build_context_from_test_roots", return_value=(source, [])) as shared:
            # The function imports the helper lazily, so patch its defining module.
            context = build_check_repair_context(self.repo, self.result(1, frozenset({failure})), frozenset({failure}))

        shared.assert_called_once()
        self.assertIn(source, context.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
