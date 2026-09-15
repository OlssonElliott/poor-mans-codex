from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chatcode.patch import (
    PatchError,
    PatchPreview,
    TestValidation,
    _classify_test_validation,
    _show_test_validation,
    _status,
    build_test_failure_repair_context,
    get_repair_context_stale_reason,
    show_repair_send_instructions,
    _build_patch_summary,
    _clear_incoming_patch,
    _clear_repair_context,
    _consume_cli_apply_flags,
    _open_diff_window,
    _run_apply_flow,
    apply_patch,
)
from chatcode.test_runner import TestResult
from chatcode.unified_diff import canonicalize_unified_diff
from chatcode.workspace import (
    get_default_patch_file,
    get_repair_context_file,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


class ApplyFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.workspace = self.root / "workspace"
        self.workspace_patch = patch(
            "chatcode.workspace.get_workspace_root",
            return_value=self.workspace,
        )
        self.workspace_patch.start()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "ChatCode Tests")
        git(self.repo, "config", "user.email", "chatcode@example.test")
        self.source = self.repo / "app.py"
        self.source.write_text("value = 1\n", encoding="utf-8", newline="\n")
        git(self.repo, "add", "app.py")
        git(self.repo, "commit", "-qm", "fixture")
        self.incoming = self.root / "incoming.diff"
        self.incoming.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n",
            encoding="utf-8",
            newline="\n",
        )

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temporary.cleanup()

    def result(self, returncode: int = 0) -> TestResult:
        report = self.root / "test-result.md"
        report.write_text("saved test output", encoding="utf-8")
        return TestResult(
            command="pytest",
            returncode=returncode,
            duration_seconds=0.1,
            output_file=report,
        )

    def interactive(self):
        return (
            patch("chatcode.patch.sys.stdin.isatty", return_value=True),
            patch("chatcode.patch.sys.stdout.isatty", return_value=True),
        )

    def test_confirmation_no_leaves_repo_unchanged(self) -> None:
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             stdin, stdout, patch("builtins.input", side_effect=["n", "n"]), \
             self.assertRaisesRegex(PatchError, "cancelled"):
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 1\n")

    def test_view_diff_opens_window_instead_of_printing_patch(self) -> None:
        stdin, stdout = self.interactive()
        patch_text = self.incoming.read_text(encoding="utf-8").rstrip()
        canonical, _ = canonicalize_unified_diff(patch_text)

        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             patch("chatcode.patch._open_diff_window") as open_diff, \
             stdin, stdout, \
             patch("builtins.input", side_effect=["y", "n"]), \
             patch("builtins.print") as output, \
             self.assertRaisesRegex(PatchError, "cancelled"):
            _run_apply_flow(self.repo, self.incoming)

        open_diff.assert_called_once_with(self.repo, canonical, {"app.py"})
        printed = "\n".join(
            str(call.args[0])
            for call in output.call_args_list
            if call.args
        )
        self.assertNotIn(patch_text, printed)
        self.assertEqual(
            self.source.read_text(encoding="utf-8"),
            "value = 1\n",
        )

    def test_preview_uses_the_same_reused_vscode_diff_route_as_review(self) -> None:
        with patch("chatcode.patch.open_code_diff") as open_diff:
            _open_diff_window(
                self.repo,
                self.incoming.read_text(encoding="utf-8"),
                {"app.py"},
            )

        before, after = open_diff.call_args.args
        self.assertEqual(before.read_text(encoding="utf-8"), "value = 1\n")
        self.assertEqual(after.read_text(encoding="utf-8"), "value = 2\n")
        self.assertIn("patch-preview", before.parts)
        self.assertIn("patch-preview", after.parts)

    def test_preview_uses_canonical_counts_instead_of_raw_incoming_patch(self) -> None:
        raw = (
            "--- a/app.py\n+++ b/app.py\n"
            "@@ -1,50 +1,70 @@\n-value = 1\n+value = 2\n"
        )
        canonical, _ = canonicalize_unified_diff(raw)

        with patch("chatcode.patch.open_code_diff") as open_diff:
            _open_diff_window(self.repo, canonical, {"app.py"})

        _before, after = open_diff.call_args.args
        self.assertEqual(after.read_text(encoding="utf-8"), "value = 2\n")

    def test_successful_apply_removes_repair_context(self) -> None:
        repair = get_repair_context_file(
            self.repo
        )
        repair.write_text(
            "old unresolved repair",
            encoding="utf-8",
        )
        sibling = repair.parent / "keep-me.txt"
        sibling.write_text(
            "keep",
            encoding="utf-8",
        )

        apply_patch(
            self.repo,
            self.incoming,
        )

        self.assertFalse(repair.exists())
        self.assertTrue(sibling.is_file())
        self.assertEqual(
            sibling.read_text(encoding="utf-8"),
            "keep",
        )

    def test_successful_apply_without_repair_context_is_safe(self) -> None:
        repair = get_repair_context_file(
            self.repo
        )
        self.assertFalse(repair.exists())

        apply_patch(
            self.repo,
            self.incoming,
        )

        self.assertEqual(
            self.source.read_text(
                encoding="utf-8"
            ),
            "value = 2\n",
        )
        self.assertFalse(repair.exists())

    def test_cancelled_apply_keeps_repair_context(self) -> None:
        repair = get_repair_context_file(
            self.repo
        )
        repair.write_text(
            "still unresolved",
            encoding="utf-8",
        )
        stdin, stdout = self.interactive()

        with patch(
            "chatcode.patch._qwen_patch_summary",
            return_value=None,
        ), stdin, stdout, patch(
            "builtins.input",
            side_effect=["n", "n"],
        ), self.assertRaisesRegex(
            PatchError,
            "cancelled",
        ):
            _run_apply_flow(
                self.repo,
                self.incoming,
            )

        self.assertTrue(repair.is_file())
        self.assertEqual(
            repair.read_text(encoding="utf-8"),
            "still unresolved",
        )
        self.assertEqual(
            self.source.read_text(
                encoding="utf-8"
            ),
            "value = 1\n",
        )

    def test_failed_validation_keeps_current_repair_state(self) -> None:
        repair = get_repair_context_file(
            self.repo
        )
        repair.write_text(
            "old repair",
            encoding="utf-8",
        )
        invalid = self.root / "invalid.diff"
        invalid.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-missing = True\n"
            "+value = 2\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(PatchError):
            apply_patch(
                self.repo,
                invalid,
            )

        self.assertTrue(repair.is_file())
        self.assertNotEqual(
            repair.read_text(encoding="utf-8"),
            "",
        )
        self.assertEqual(
            self.source.read_text(
                encoding="utf-8"
            ),
            "value = 1\n",
        )

    def test_repair_cleanup_failure_does_not_fail_apply(self) -> None:
        repair = get_repair_context_file(
            self.repo
        )
        repair.write_text(
            "old repair",
            encoding="utf-8",
        )

        with patch(
            "chatcode.patch.Path.unlink",
            side_effect=OSError("locked"),
        ):
            result = apply_patch(
                self.repo,
                self.incoming,
            )

        self.assertIn("app.py", result.paths)
        self.assertEqual(
            self.source.read_text(
                encoding="utf-8"
            ),
            "value = 2\n",
        )

    def test_clear_repair_context_only_targets_repair_file(self) -> None:
        repair = get_repair_context_file(
            self.repo
        )
        repair.write_text(
            "repair",
            encoding="utf-8",
        )
        sibling = repair.parent / "project-map.json"
        sibling.write_text(
            "{}",
            encoding="utf-8",
        )

        _clear_repair_context(
            self.repo
        )

        self.assertFalse(repair.exists())
        self.assertTrue(sibling.is_file())

    def test_successful_apply_clears_canonical_incoming_patch(self) -> None:
        canonical = get_default_patch_file(
            self.repo
        )
        canonical.write_text(
            self.incoming.read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
        sibling = canonical.parent.parent / "project-map.json"
        sibling.write_text(
            "{}",
            encoding="utf-8",
        )

        apply_patch(
            self.repo,
            canonical,
        )

        self.assertTrue(canonical.is_file())
        self.assertEqual(
            canonical.read_text(encoding="utf-8"),
            "",
        )
        self.assertEqual(
            sibling.read_text(encoding="utf-8"),
            "{}",
        )

    def test_cancelled_apply_keeps_canonical_incoming_patch(self) -> None:
        canonical = get_default_patch_file(
            self.repo
        )
        original = self.incoming.read_text(
            encoding="utf-8"
        )
        canonical.write_text(
            original,
            encoding="utf-8",
            newline="\n",
        )
        stdin, stdout = self.interactive()

        with patch(
            "chatcode.patch._qwen_patch_summary",
            return_value=None,
        ), stdin, stdout, patch(
            "builtins.input",
            side_effect=["n", "n"],
        ), self.assertRaisesRegex(
            PatchError,
            "cancelled",
        ):
            _run_apply_flow(
                self.repo,
                canonical,
            )

        self.assertEqual(
            canonical.read_text(encoding="utf-8"),
            original,
        )

    def test_failed_validation_keeps_canonical_incoming_patch(self) -> None:
        canonical = get_default_patch_file(
            self.repo
        )
        invalid = (
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-missing = True\n"
            "+value = 2\n"
        )
        canonical.write_text(
            invalid,
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(PatchError):
            apply_patch(
                self.repo,
                canonical,
            )

        self.assertTrue(
            canonical.read_text(encoding="utf-8")
        )

    def test_incoming_cleanup_failure_is_nonfatal(self) -> None:
        fake_incoming = Mock()
        fake_incoming.exists.return_value = True
        fake_incoming.write_text.side_effect = OSError(
            "locked"
        )

        with patch(
            "chatcode.patch.get_default_patch_file",
            return_value=fake_incoming,
        ):
            _clear_incoming_patch(
                self.repo
            )

        fake_incoming.write_text.assert_called_once()

    def test_yes_skips_pre_apply_prompts_and_runs_tests(self) -> None:
        test_result = self.result()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", return_value=test_result,
        ) as run, patch("builtins.input") as prompt:
            _run_apply_flow(self.repo, self.incoming, yes=True)

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")
        self.assertEqual(run.call_count, 2)
        run.assert_called_with(self.repo)
        prompt.assert_not_called()

    def test_failure_classification_distinguishes_regressions_from_baseline(self) -> None:
        baseline = self.result(1)
        baseline = TestResult(**{**baseline.__dict__, "failed_tests": frozenset({"tests.test_old"})})
        after = self.result(1)
        after = TestResult(**{**after.__dict__, "failed_tests": frozenset({"tests.test_old", "tests.test_new"})})

        validation = _classify_test_validation(baseline, None, after)

        self.assertEqual(validation.status, "regressions")
        self.assertEqual(validation.new_failures, frozenset({"tests.test_new"}))
        self.assertEqual(validation.existing_failures, frozenset({"tests.test_old"}))

    def test_unparseable_failed_output_is_classified_as_unclear(self) -> None:
        validation = _classify_test_validation(
            self.result(1), None, self.result(1)
        )

        self.assertEqual(validation.status, "unclear")

    def test_unidentified_failed_baseline_never_proves_a_regression(self) -> None:
        baseline = self.result(1)
        after = self.result(1)
        after = TestResult(
            **{**after.__dict__, "failed_tests": frozenset({"tests.test_visible"})}
        )

        validation = _classify_test_validation(baseline, None, after)

        self.assertEqual(validation.status, "unclear")
        self.assertEqual(validation.new_failures, frozenset())

    def test_repair_is_unsuccessful_when_target_failure_remains(self) -> None:
        target = "test_dashboard_api.DashboardAPITests.test_traps"
        baseline = TestResult(**{
            **self.result(1).__dict__,
            "failed_tests": frozenset({target}),
        })
        after = TestResult(**{
            **self.result(1).__dict__,
            "failed_tests": frozenset({target}),
        })

        validation = _classify_test_validation(
            baseline, None, after, frozenset({target})
        )

        self.assertEqual(validation.status, "repair_failed")
        self.assertEqual(validation.remaining_repair_failures, frozenset({target}))
        with patch("builtins.print") as output:
            _show_test_validation(validation)
        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("REPAIR UNSUCCESSFUL", rendered)
        self.assertIn("REPAIR: FAIL | TARGET FAILURES REMAIN: 1", rendered)
        self.assertIn("undo this unsuccessful repair", rendered)

    def test_repair_target_that_disappears_is_not_reported_as_failed(self) -> None:
        target = "tests.test_fixed"
        validation = _classify_test_validation(
            TestResult(**{
                **self.result(1).__dict__,
                "failed_tests": frozenset({target}),
            }),
            None,
            self.result(),
            frozenset({target}),
        )

        self.assertEqual(validation.status, "repair_passed")
        self.assertFalse(validation.remaining_repair_failures)
        with patch("builtins.print") as output:
            _show_test_validation(validation)
        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("REPAIR SUCCESSFUL", rendered)
        self.assertIn("REPAIR: PASS | TARGET FAILURES REMAIN: 0", rendered)

    def test_failed_repair_stops_after_targeted_test_and_defaults_to_undo(self) -> None:
        target = "test_dashboard_api.DashboardAPITests.test_traps"
        baseline = TestResult(**{
            **self.result(1).__dict__,
            "failed_tests": frozenset({target}),
        })
        targeted = TestResult(**{
            **self.result(1).__dict__,
            "failed_tests": frozenset({target}),
        })
        stdin, stdout = self.interactive()

        with patch(
            "chatcode.patch.get_active_repair_targets",
            return_value=frozenset({target}),
        ), patch(
            "chatcode.test_runner.run_project_tests",
            return_value=baseline,
        ) as full, patch(
            "chatcode.test_runner.run_relevant_tests",
            return_value=targeted,
        ), stdin, stdout, patch(
            "builtins.input",
            side_effect=["n", "y", ""],
        ), patch("builtins.print") as output:
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(full.call_count, 1)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 1\n")
        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("Repair candidate rejected by relevant tests", rendered)
        self.assertIn("[U] Undo (recommended)", rendered)
        self.assertIn("Repair context rebuilt against the restored working tree", rendered)

    def test_terminal_status_is_readable_when_color_is_disabled(self) -> None:
        with patch.dict("os.environ", {"NO_COLOR": "1"}, clear=False), patch(
            "chatcode.patch.sys.stdout.isatty", return_value=True
        ):
            rendered = _status("[FAIL] FULL SUITE", "red")

        self.assertEqual(rendered, "[FAIL] FULL SUITE")

    def test_terminal_status_uses_ansi_color_when_supported(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "chatcode.patch.sys.stdout.isatty", return_value=True
        ):
            rendered = _status("[OK] PATCH APPLIED", "green")

        self.assertIn("\x1b[32m", rendered)
        self.assertIn("[OK] PATCH APPLIED", rendered)

    def test_regression_repair_context_uses_current_source_and_patch(self) -> None:
        applied = apply_patch(self.repo, self.incoming)
        self.source.write_text("value = 3\n", encoding="utf-8", newline="\n")
        failed = self.result(1)
        failed = TestResult(
            **{**failed.__dict__, "failed_tests": frozenset({"tests.test_app"})}
        )
        validation = TestValidation(
            baseline=self.result(),
            targeted=None,
            full=failed,
            status="regressions",
            new_failures=frozenset({"tests.test_app"}),
        )

        output = build_test_failure_repair_context(self.repo, applied, validation)
        content = output.read_text(encoding="utf-8")

        self.assertIn("value = 3", content)
        self.assertIn("+value = 2", content)
        self.assertIn("regression: tests.test_app", content)
        self.assertIn("Do not restore or overwrite unrelated user changes.", content)
        self.assertIsNone(get_repair_context_stale_reason(self.repo))

        self.source.write_text("value = 4\n", encoding="utf-8", newline="\n")
        self.assertIn(
            "Working-tree files changed",
            get_repair_context_stale_reason(self.repo),
        )

    def test_repair_context_includes_failure_trace_sources_and_compacts_noise(self) -> None:
        applied = apply_patch(self.repo, self.incoming)
        test_file = self.repo / "tests" / "test_broken.py"
        source_file = self.repo / "pkg" / "service.py"
        test_file.parent.mkdir()
        source_file.parent.mkdir()
        test_file.write_text("def test_failure():\n    assert False\n", encoding="utf-8")
        source_file.write_text("def broken():\n    return False\n", encoding="utf-8")
        report = self.root / "failure-report.md"
        report.write_text(
            "Status: FAILED\nExit code: 1\nCommand: python -m unittest\n"
            "Duration: 1.00 seconds\nNOISY SUCCESS OUTPUT\n"
            "======================================================================\n"
            "FAIL: test_failure (test_broken.BrokenTests.test_failure)\n"
            "----------------------------------------------------------------------\n"
            f"Traceback:\n  File \"{source_file}\", line 2, in broken\n"
            "AssertionError\nFAILED (failures=1)\n",
            encoding="utf-8",
        )
        failed = TestResult(
            command="python -m unittest",
            returncode=1,
            duration_seconds=1.0,
            output_file=report,
            failed_tests=frozenset({"test_broken.BrokenTests.test_failure"}),
        )
        validation = TestValidation(
            baseline=failed,
            targeted=None,
            full=failed,
            status="existing",
            existing_failures=failed.failed_tests,
        )

        output = build_test_failure_repair_context(self.repo, applied, validation)
        content = output.read_text(encoding="utf-8")

        self.assertIn("tests/test_broken.py", content)
        self.assertIn("pkg/service.py", content)
        self.assertIn("def broken", content)
        self.assertNotIn("NOISY SUCCESS OUTPUT", content)

    def test_repair_send_instructions_require_a_companion_user_prompt(self) -> None:
        with patch("builtins.print") as output:
            show_repair_send_instructions(Path("PATCH_REPAIR_CONTEXT.md"))

        printed = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn("Attach that file and send this message with it", printed)
        self.assertIn("Use the attached PATCH_REPAIR_CONTEXT.md", printed)

    def test_passed_tests_offer_review_then_keep(self) -> None:
        canonical = get_default_patch_file(
            self.repo
        )
        canonical.write_text(
            self.incoming.read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             patch("chatcode.test_runner.run_project_tests", return_value=self.result()), \
             patch("chatcode.history.open_history_review") as review, \
             stdin, stdout, patch("builtins.input", side_effect=["n", "y", "r", "k"]):
            result = _run_apply_flow(self.repo, canonical)

        review.assert_called_once_with(
            result.history_entry
        )
        self.assertEqual(
            canonical.read_text(encoding="utf-8"),
            "",
        )
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_review_then_undo_still_restores_file(self) -> None:
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             patch("chatcode.test_runner.run_project_tests", return_value=self.result()), \
             patch("chatcode.history.open_history_review") as review, \
             stdin, stdout, patch("builtins.input", side_effect=["n", "y", "r", "u"]):
            _run_apply_flow(self.repo, self.incoming)

        review.assert_called_once()
        self.assertEqual(
            self.source.read_text(encoding="utf-8"),
            "value = 1\n",
        )

    def test_failed_tests_can_show_saved_output_then_keep(self) -> None:
        canonical = get_default_patch_file(
            self.repo
        )
        canonical.write_text(
            self.incoming.read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             patch("chatcode.test_runner.run_project_tests", return_value=self.result(1)), \
             stdin, stdout, patch("builtins.input", side_effect=["n", "y", "t", "k"]), \
             patch("builtins.print") as output:
            _run_apply_flow(self.repo, canonical)

        printed = "\n".join(
            str(call.args[0])
            for call in output.call_args_list
            if call.args
        )
        self.assertIn(
            "Assessment: failures are unclear and require review.",
            printed,
        )
        self.assertIn(
            "[K] Keep changes  [U] Undo  [R] Review diff  [T] Show test output",
            printed,
        )
        self.assertNotIn("Keep changes anyway", printed)
        self.assertIn("saved test output", printed)
        self.assertEqual(
            canonical.read_text(encoding="utf-8"),
            "",
        )
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_undo_choice_restores_file(self) -> None:
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             patch("chatcode.test_runner.run_project_tests", return_value=self.result()), \
             stdin, stdout, patch("builtins.input", side_effect=["n", "y", "u"]):
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 1\n")

    def test_non_interactive_without_yes_refuses_to_apply(self) -> None:
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             self.assertRaisesRegex(PatchError, "requires --yes"):
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 1\n")

    def test_qwen_summary_is_used_when_available(self) -> None:
        preview = PatchPreview({"app.py"}, self.incoming.read_text(encoding="utf-8"))
        completed = Mock(
            returncode=0,
            stdout=json.dumps({"bullets": ["Changes the visible app value."]}),
            stderr="",
        )
        with patch("chatcode.patch.shutil.which", return_value="ollama"), patch(
            "chatcode.patch.subprocess.run", return_value=completed,
        ), patch("chatcode.patch.get_context_task", return_value="change value"):
            summary = _build_patch_summary(self.repo, preview)

        self.assertIn("Changes the visible app value.", summary)

    def test_fallback_summary_is_used_when_qwen_fails(self) -> None:
        preview = PatchPreview({"app.py"}, self.incoming.read_text(encoding="utf-8"))
        with patch("chatcode.patch._qwen_patch_summary", return_value=None):
            summary = _build_patch_summary(self.repo, preview)

        self.assertIn("app.py", summary)
        self.assertIn("+1 / -1", summary)


    def test_interactive_apply_runs_tests_once_and_reports_success_once(self) -> None:
        stdin, stdout = self.interactive()
        test_result = self.result()

        with patch(
            "chatcode.patch._qwen_patch_summary",
            return_value=None,
        ), patch(
            "chatcode.test_runner.run_project_tests",
            return_value=test_result,
        ) as run, stdin, stdout, patch(
            "builtins.input",
            side_effect=["n", "y", "k"],
        ), patch("builtins.print") as output:
            _run_apply_flow(
                self.repo,
                self.incoming,
            )

        self.assertEqual(run.call_count, 2)
        run.assert_called_with(self.repo)
        success_messages = [
            call
            for call in output.call_args_list
            if call.args
            and call.args[0]
            == "\n[OK] PATCH APPLIED"
        ]
        self.assertEqual(
            len(success_messages),
            1,
        )

    def test_cli_apply_exits_after_new_workflow(self) -> None:
        result = Mock()

        with patch(
            "chatcode.patch._CLI_APPLY_INVOCATION",
            True,
        ), patch(
            "chatcode.patch._CLI_APPLY_YES",
            False,
        ), patch(
            "chatcode.patch._run_apply_flow",
            return_value=result,
        ) as flow:
            with self.assertRaises(SystemExit) as stopped:
                apply_patch(
                    self.repo,
                    self.incoming,
                )

        self.assertEqual(
            stopped.exception.code,
            0,
        )
        flow.assert_called_once_with(
            self.repo,
            self.incoming,
            yes=False,
        )

    def test_cli_yes_flag_is_consumed_before_parser(self) -> None:
        import sys

        with patch.object(sys, "argv", ["chatcode", "apply", "--yes"]):
            is_apply, yes = _consume_cli_apply_flags()
            self.assertTrue(is_apply)
            self.assertTrue(yes)
            self.assertEqual(sys.argv, ["chatcode", "apply"])


if __name__ == "__main__":
    unittest.main()
