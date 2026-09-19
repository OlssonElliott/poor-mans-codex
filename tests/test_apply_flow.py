from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chatcode.context_state import get_stale_context_reason, save_context_state
from chatcode.patch import (
    PatchError,
    ApplyResult,
    PatchPreview,
    _expand_thin_hunk_context,
    TestValidation,
    _classify_test_validation,
    _show_test_validation,
    _status,
    build_test_failure_repair_context,
    build_existing_failure_context,
    build_followup_context,
    mark_followup_resolved,
    get_repair_context_stale_reason,
    show_repair_send_instructions,
    show_chatgpt_upload_artifact,
    _build_patch_summary,
    _clear_incoming_patch,
    _clear_repair_context,
    _consume_cli_apply_flags,
    _open_diff_window,
    _run_apply_flow,
    _capture_repository_snapshot,
    save_verified_baseline,
    apply_patch,
)
from chatcode.cli import command_followup
from chatcode.test_runner import TestResult
from chatcode.unified_diff import canonicalize_unified_diff, parse_unified_diff
from chatcode.workspace import (
    get_default_patch_file,
    get_repair_context_file,
    get_existing_failure_context_file,
    get_followup_context_file,
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


def _run_in_thread(repo: Path, patch_file: Path, errors: list[BaseException]) -> None:
    try:
        _run_apply_flow(repo, patch_file)
    except BaseException as exc:
        errors.append(exc)


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

    def canonical_patch(self, old: int, new: int) -> Path:
        incoming = get_default_patch_file(self.repo)
        incoming.parent.mkdir(parents=True, exist_ok=True)
        incoming.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            f"-value = {old}\n"
            f"+value = {new}\n",
            encoding="utf-8",
            newline="\n",
        )
        return incoming

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

    def test_unique_contextless_hunk_is_expanded_before_apply(self) -> None:
        self.source.write_text(
            "alpha = 1\n"
            "beta = 2\n"
            "gamma = 3\n"
            "value = 1\n"
            "delta = 4\n"
            "epsilon = 5\n"
            "zeta = 6\n",
            encoding="utf-8",
            newline="\n",
        )
        git(self.repo, "add", "app.py")
        git(self.repo, "commit", "-qm", "larger fixture")
        self.incoming.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -4 +4 @@\n"
            "-value = 1\n"
            "+value = 2\n",
            encoding="utf-8",
            newline="\n",
        )

        apply_patch(self.repo, self.incoming)

        parsed = parse_unified_diff(
            self.incoming.read_text(encoding="utf-8")
        )
        hunk = parsed.files[0].hunks[0]
        self.assertGreaterEqual(
            sum(line.kind == " " for line in hunk.lines),
            3,
        )
        self.assertIn("value = 2", self.source.read_text(encoding="utf-8"))

    def test_hunk_expansion_uses_uncommitted_working_tree_context(self) -> None:
        self.source.write_text(
            "alpha = 1\n"
            "beta = 2\n"
            "gamma = 3\n"
            "value = 1\n"
            "delta = 4\n"
            "epsilon = 5\n"
            "zeta = 6\n",
            encoding="utf-8",
            newline="\n",
        )
        git(self.repo, "add", "app.py")
        git(self.repo, "commit", "-qm", "larger fixture")
        self.source.write_text(
            self.source.read_text(encoding="utf-8").replace(
                "beta = 2",
                "beta = 99",
            ),
            encoding="utf-8",
            newline="\n",
        )
        patch_text = (
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -4 +4 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        )

        expanded = _expand_thin_hunk_context(self.repo, patch_text)

        self.assertIn(" beta = 99\n", expanded)
        self.assertNotIn(" beta = 2\n", expanded)

    def test_ambiguous_contextless_hunk_falls_back_to_repair(self) -> None:
        self.source.write_text(
            "alpha = 1\n"
            "value = 1\n"
            "middle = 0\n"
            "value = 1\n"
            "omega = 9\n",
            encoding="utf-8",
            newline="\n",
        )
        git(self.repo, "add", "app.py")
        git(self.repo, "commit", "-qm", "ambiguous fixture")
        self.incoming.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -2 +2 @@\n"
            "-value = 1\n"
            "+value = 2\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, self.incoming)

        self.assertEqual(
            raised.exception.failure_type,
            "insufficient_patch_context",
        )
        self.assertEqual(
            self.source.read_text(encoding="utf-8").count("value = 1"),
            2,
        )

    def test_blank_line_context_drift_is_reanchored_before_apply(self) -> None:
        self.source.write_text(
            "alpha = 1\n"
            "beta = 2\n"
            "}\n"
            "\n"
            "function target() {\n"
            "  return 1;\n"
            "}\n"
            "omega = 9\n",
            encoding="utf-8",
            newline="\n",
        )
        git(self.repo, "add", "app.py")
        git(self.repo, "commit", "-qm", "reanchor fixture")
        self.incoming.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,9 +1,10 @@\n"
            " alpha = 1\n"
            " beta = 2\n"
            " }\n"
            " \n"
            " \n"
            "+inserted = True\n"
            " function target() {\n"
            "   return 1;\n"
            " }\n"
            " omega = 9\n",
            encoding="utf-8",
            newline="\n",
        )

        apply_patch(self.repo, self.incoming)

        self.assertEqual(
            self.source.read_text(encoding="utf-8"),
            "alpha = 1\n"
            "beta = 2\n"
            "}\n"
            "\n"
            "inserted = True\n"
            "function target() {\n"
            "  return 1;\n"
            "}\n"
            "omega = 9\n",
        )
        normalized = self.incoming.read_text(encoding="utf-8")
        self.assertIn(
            " }\n \n+inserted = True\n function target() {\n",
            normalized,
        )

    def test_ambiguous_blank_line_reanchor_falls_back_to_repair(self) -> None:
        self.source.write_text(
            "section\n"
            "alpha\n"
            "}\n"
            "\n"
            "function target() {\n"
            "  return 1;\n"
            "}\n"
            "separator\n"
            "section\n"
            "alpha\n"
            "}\n"
            "\n"
            "function target() {\n"
            "  return 1;\n"
            "}\n",
            encoding="utf-8",
            newline="\n",
        )
        git(self.repo, "add", "app.py")
        git(self.repo, "commit", "-qm", "ambiguous reanchor fixture")
        self.incoming.write_text(
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,8 +1,9 @@\n"
            " section\n"
            " alpha\n"
            " }\n"
            " \n"
            " \n"
            "+inserted = True\n"
            " function target() {\n"
            "   return 1;\n"
            " }\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, self.incoming)

        self.assertEqual(
            raised.exception.failure_type,
            "patch_target_mismatch",
        )
        self.assertNotIn(
            "inserted = True",
            self.source.read_text(encoding="utf-8"),
        )

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

    def test_unchanged_baseline_failure_passes_with_visible_pre_existing_status(self) -> None:
        failure = "tests.test_old"
        baseline = TestResult(**{**self.result(1).__dict__, "failed_tests": frozenset({failure})})
        after = TestResult(**{**self.result(1).__dict__, "failed_tests": frozenset({failure})})

        validation = _classify_test_validation(baseline, self.result(), after)

        self.assertEqual(validation.status, "existing")
        self.assertFalse(validation.new_failures)
        self.assertEqual(validation.existing_failures, frozenset({failure}))
        with patch("builtins.print") as output:
            _show_test_validation(validation)
        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("PASS WITH PRE-EXISTING FAILURES", rendered)
        self.assertIn("REGRESSIONS: 0", rendered)
        self.assertIn("PRE-EXISTING: 1", rendered)

    def test_unchanged_baseline_failure_does_not_create_repair_context(self) -> None:
        failure = "tests.test_old"
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({failure}),
        })
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", side_effect=[failed, failed],
        ), patch("chatcode.patch.build_test_failure_repair_context") as repair:
            _run_apply_flow(self.repo, self.incoming, yes=True)

        repair.assert_not_called()

    def test_pre_existing_failure_menu_offers_fix_context_action(self) -> None:
        failure = "tests.test_old"
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({failure}),
        })
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", side_effect=[failed, failed],
        ), stdin, stdout, patch("builtins.input", side_effect=["n", "y", "f"]) as prompt, patch(
            "chatcode.cli.open_folder"
        ) as open_folder, patch(
            "builtins.print"
        ) as output:
            _run_apply_flow(self.repo, self.incoming)

        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("[F] Keep changes + create fix context for existing failure", rendered)
        self.assertIn("Existing failure context created:", rendered)
        self.assertIn("[UPLOAD THIS FILE]", rendered)
        context = get_existing_failure_context_file(self.repo)
        self.assertTrue(context.is_file())
        self.assertIn(failure, context.read_text(encoding="utf-8"))
        open_folder.assert_called_once_with(context.parent)
        self.assertEqual(prompt.call_count, 3)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")
        self.assertEqual(
            rendered.count("[F] Keep changes + create fix context for existing failure"),
            1,
        )

    def test_pre_existing_failure_keep_does_not_create_or_open_context(self) -> None:
        failure = "tests.test_old"
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({failure}),
        })
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", side_effect=[failed, failed],
        ), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "k"]
        ), patch("chatcode.cli.open_folder") as open_folder:
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")
        self.assertFalse(get_existing_failure_context_file(self.repo).exists())
        open_folder.assert_not_called()

    def test_failed_existing_failure_context_generation_returns_to_menu(self) -> None:
        failure = "tests.test_old"
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({failure}),
        })
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", side_effect=[failed, failed],
        ), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "f", "k"]
        ) as prompt, patch(
            "chatcode.patch.build_existing_failure_context",
            side_effect=OSError("disk full"),
        ), patch("chatcode.cli.open_folder") as open_folder, patch(
            "builtins.print"
        ) as output:
            _run_apply_flow(self.repo, self.incoming)

        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("Could not create existing failure context: disk full", rendered)
        self.assertEqual(prompt.call_count, 4)
        self.assertEqual(
            rendered.count("[F] Keep changes + create fix context for existing failure"),
            2,
        )
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")
        open_folder.assert_not_called()

    def test_existing_failure_context_is_a_separate_focused_new_task(self) -> None:
        applied = apply_patch(self.repo, self.incoming)
        test_file = self.repo / "tests" / "test_broken.py"
        implementation = self.repo / "pkg" / "service.py"
        test_file.parent.mkdir(exist_ok=True)
        implementation.parent.mkdir()
        test_file.write_text("from pkg.service import broken\n\ndef test_failure():\n    assert broken()\n", encoding="utf-8")
        implementation.write_text("def broken():\n    return False\n", encoding="utf-8")
        report = self.root / "existing-failure.md"
        failure = "test_broken.BrokenTests.test_failure"
        report.write_text(
            f"Status: FAILED\nFAIL: test_failure ({failure})\nTraceback\nAssertionError\n",
            encoding="utf-8",
        )
        failed = TestResult(**{
            **self.result(1).__dict__, "output_file": report,
            "failed_tests": frozenset({failure}),
        })
        validation = TestValidation(
            baseline=failed, targeted=self.result(), full=failed, status="existing",
            existing_failures=frozenset({failure}),
        )

        context = build_existing_failure_context(
            self.repo, applied, validation, frozenset({failure})
        )
        content = context.read_text(encoding="utf-8")

        self.assertEqual(context, get_existing_failure_context_file(self.repo))
        self.assertIn("# ChatCode Existing Failure Context", content)
        self.assertIn(failure, content)
        self.assertIn("failed before the previous patch", content)
        self.assertIn("introduced zero new regressions", content)
        self.assertIn("def broken", content)
        self.assertIn("from pkg.service import broken", content)
        self.assertNotIn("PATCH_REPAIR_CONTEXT", str(context))

    def test_existing_failure_context_contains_only_selected_failures(self) -> None:
        applied = apply_patch(self.repo, self.incoming)
        first, second = "tests.test_first", "tests.test_second"
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({first, second}),
        })
        validation = TestValidation(
            baseline=failed, targeted=self.result(), full=failed, status="existing",
            existing_failures=frozenset({first, second}),
        )

        context = build_existing_failure_context(self.repo, applied, validation, frozenset({first}))
        content = context.read_text(encoding="utf-8")

        self.assertIn(first, content)
        self.assertNotIn(second, content)

    def test_fixed_baseline_failure_is_recorded_without_a_regression(self) -> None:
        baseline = TestResult(**{
            **self.result(1).__dict__,
            "failed_tests": frozenset({"tests.test_old", "tests.test_fixed"}),
        })
        after = TestResult(**{
            **self.result(1).__dict__,
            "failed_tests": frozenset({"tests.test_old"}),
        })

        validation = _classify_test_validation(baseline, self.result(), after)

        self.assertEqual(validation.status, "existing")
        self.assertEqual(validation.fixed_failures, frozenset({"tests.test_fixed"}))
        self.assertFalse(validation.new_failures)

    def test_targeted_failure_is_authoritative_even_when_it_existed_at_baseline(self) -> None:
        failure = "tests.test_required"
        baseline = TestResult(**{**self.result(1).__dict__, "failed_tests": frozenset({failure})})
        targeted = TestResult(**{**self.result(1).__dict__, "failed_tests": frozenset({failure})})
        after = TestResult(**{**self.result(1).__dict__, "failed_tests": frozenset({failure})})

        validation = _classify_test_validation(baseline, targeted, after)

        self.assertEqual(validation.status, "targeted_failed")
        self.assertFalse(validation.new_failures)

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

    def test_failed_repair_stops_after_targeted_test_and_defaults_to_keep(self) -> None:
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
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")
        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("Repair candidate rejected by relevant tests", rendered)
        self.assertIn("[K] Keep changes (recommended for repair)", rendered)
        self.assertNotIn("[U] Undo (recommended)", rendered)

    def test_undo_invalidates_post_patch_repair_context(self) -> None:
        baseline = self.result()
        failed = TestResult(**{
            **self.result(1).__dict__,
            "failed_tests": frozenset({"tests.test_regression"}),
        })
        stdin, stdout = self.interactive()

        with patch(
            "chatcode.test_runner.run_project_tests",
            side_effect=[baseline, failed],
        ), patch(
            "chatcode.test_runner.run_relevant_tests", return_value=None,
        ), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "u"],
        ), patch("builtins.print") as output:
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 1\n")
        self.assertFalse(get_repair_context_file(self.repo).exists())
        self.assertIsNotNone(get_stale_context_reason(self.repo, {"app.py"}))
        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, self.canonical_patch(2, 3))
        self.assertEqual(raised.exception.failure_type, "stale_context")
        self.assertIn("Repair context is stale", str(raised.exception))
        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("Repair context invalidated", rendered)

    def test_baseline_starts_before_apply_approval(self) -> None:
        started = threading.Event()

        def baseline(_repo: Path):
            started.set()
            return self.result()

        def answer(prompt: str) -> str:
            if prompt.startswith("View full diff"):
                return "n"
            self.assertTrue(started.is_set(), "baseline must start before approval")
            return "n"

        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", side_effect=baseline
        ) as run, stdin, stdout, patch("builtins.input", side_effect=answer), self.assertRaisesRegex(
            PatchError, "cancelled"
        ):
            _run_apply_flow(self.repo, self.incoming)

        run.assert_called_once_with(self.repo)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 1\n")

    def test_verified_cache_skips_new_pre_patch_baseline(self) -> None:
        snapshot = _capture_repository_snapshot(self.repo)
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._test_config_identity", return_value="pytest"), patch(
            "chatcode.patch._qwen_patch_summary", return_value=None
        ), patch(
            "chatcode.test_runner.run_project_tests", return_value=self.result()
        ) as run, patch("chatcode.test_runner.run_relevant_tests", return_value=None), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "y"]
        ), patch("builtins.print") as output:
            save_verified_baseline(self.repo, snapshot, self.result())
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(run.call_count, 1)
        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("using verified cached result", rendered)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_changed_repository_during_review_replaces_background_baseline(self) -> None:
        first = self.result()
        replacement = self.result()
        full = self.result()

        def answer(prompt: str) -> str:
            if prompt.startswith("View full diff"):
                return "n"
            (self.repo / "review-change.txt").write_text("changed during review\n", encoding="utf-8")
            return "y"

        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", side_effect=[first, replacement, full]
        ) as run, patch("chatcode.test_runner.run_relevant_tests", return_value=None), stdin, stdout, patch(
            "builtins.input", side_effect=answer
        ):
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(run.call_count, 3)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_fast_approval_waits_for_the_single_running_baseline(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def tests(_repo: Path):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                self.assertTrue(release.wait(5))
            return self.result()

        stdin, stdout = self.interactive()
        errors: list[BaseException] = []
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", side_effect=tests
        ), patch("chatcode.test_runner.run_relevant_tests", return_value=None), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "y"]
        ):
            runner = threading.Thread(
                target=lambda: _run_in_thread(self.repo, self.incoming, errors)
            )
            runner.start()
            self.assertTrue(started.wait(5))
            self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 1\n")
            release.set()
            runner.join(5)

        self.assertFalse(runner.is_alive())
        self.assertFalse(errors)
        self.assertEqual(calls, 2)  # one baseline and one post-patch full suite
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_normal_patch_uses_normal_context_baseline(self) -> None:
        save_context_state(self.repo, task="normal task")

        apply_patch(self.repo, self.canonical_patch(1, 2))

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_normal_patch_rejects_manual_edit_after_context(self) -> None:
        save_context_state(self.repo, task="normal task")
        self.source.write_text("value = 9\n", encoding="utf-8", newline="\n")

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, self.canonical_patch(1, 2))

        self.assertEqual(raised.exception.failure_type, "stale_context")
        self.assertIn("UPLOAD_TO_CHATGPT.md", str(raised.exception))
        self.assertNotIn("Repair context is stale", str(raised.exception))

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

    def test_chatgpt_upload_artifact_is_labeled_and_color_safe(self) -> None:
        artifact = Path("EXISTING_FAILURE_CONTEXT.md")
        with patch("builtins.print") as output:
            show_chatgpt_upload_artifact(artifact)

        printed = [call.args[0] for call in output.call_args_list]
        self.assertEqual(printed, ["[UPLOAD THIS FILE]", str(artifact)])

        with patch.dict("os.environ", {}, clear=True), patch(
            "chatcode.patch.sys.stdout.isatty", return_value=True
        ), patch("builtins.print") as output:
            show_chatgpt_upload_artifact(artifact)

        colored = [call.args[0] for call in output.call_args_list]
        self.assertTrue(all("\x1b[36m" in value for value in colored))
        self.assertIn("[UPLOAD THIS FILE]", colored[0])
        self.assertIn(str(artifact), colored[1])

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

    def test_repair_patch_uses_post_patch_context_baseline(self) -> None:
        save_context_state(self.repo, task="normal task")
        first = apply_patch(self.repo, self.canonical_patch(1, 2))
        failure = "tests.test_drop_autocomplete_lists_only_unequipped_inventory_items"
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({failure}),
        })
        validation = TestValidation(
            baseline=self.result(),
            targeted=None,
            full=failed,
            status="regressions",
            new_failures=frozenset({failure}),
        )
        build_test_failure_repair_context(self.repo, first, validation)

        self.assertIsNone(get_stale_context_reason(self.repo, {"app.py"}))
        second = apply_patch(self.repo, self.canonical_patch(2, 3))

        self.assertIn("app.py", second.paths)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 3\n")

    def test_repair_patch_rejects_manual_edit_after_repair_context(self) -> None:
        save_context_state(self.repo, task="normal task")
        first = apply_patch(self.repo, self.canonical_patch(1, 2))
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({"tests.test_regression"}),
        })
        build_test_failure_repair_context(
            self.repo,
            first,
            TestValidation(
                baseline=self.result(), targeted=None, full=failed,
                status="regressions", new_failures=failed.failed_tests,
            ),
        )
        self.source.write_text("value = 99\n", encoding="utf-8", newline="\n")

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, self.canonical_patch(2, 3))

        self.assertEqual(raised.exception.failure_type, "stale_context")
        self.assertIn("Repair context is stale", str(raised.exception))
        self.assertIn("PATCH_REPAIR_CONTEXT.md", str(raised.exception))
        self.assertNotIn("run chatcode context", str(raised.exception).lower())

    def test_new_normal_context_replaces_completed_repair_baseline(self) -> None:
        save_context_state(self.repo, task="normal task")
        first = apply_patch(self.repo, self.canonical_patch(1, 2))
        failed = TestResult(**{
            **self.result(1).__dict__, "failed_tests": frozenset({"tests.test_regression"}),
        })
        build_test_failure_repair_context(
            self.repo,
            first,
            TestValidation(
                baseline=self.result(), targeted=None, full=failed,
                status="regressions", new_failures=failed.failed_tests,
            ),
        )
        apply_patch(self.repo, self.canonical_patch(2, 3))

        save_context_state(self.repo, task="later normal task")
        apply_patch(self.repo, self.canonical_patch(3, 4))

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 4\n")

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
        self.assertIn("[UPLOAD THIS FILE]", printed)
        self.assertIn("PATCH_REPAIR_CONTEXT.md", printed)
        self.assertIn("Attach that file and send this message with it", printed)
        self.assertIn("Use the attached PATCH_REPAIR_CONTEXT.md", printed)

    def test_passed_tests_yes_keeps_without_post_apply_menu(self) -> None:
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
             stdin, stdout, patch("builtins.input", side_effect=["n", "y", "y"]):
            result = _run_apply_flow(self.repo, canonical)

        review.assert_not_called()
        self.assertEqual(
            canonical.read_text(encoding="utf-8"),
            "",
        )
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_passed_tests_yes_does_not_offer_undo_menu(self) -> None:
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             patch("chatcode.test_runner.run_project_tests", return_value=self.result()), \
             patch("chatcode.history.open_history_review") as review, \
             stdin, stdout, patch("builtins.input", side_effect=["n", "y", "yes"]):
            _run_apply_flow(self.repo, self.incoming)

        review.assert_not_called()
        self.assertEqual(
            self.source.read_text(encoding="utf-8"),
            "value = 2\n",
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
            "[K] Keep changes (recommended for repair)  [U] Undo  [R] Review diff  [T] Show test output",
            printed,
        )
        self.assertNotIn("Keep changes anyway", printed)
        self.assertIn("saved test output", printed)
        self.assertEqual(
            canonical.read_text(encoding="utf-8"),
            "",
        )
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

    def test_default_yes_keeps_successful_patch(self) -> None:
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), \
             patch("chatcode.test_runner.run_project_tests", return_value=self.result()), \
             stdin, stdout, patch("builtins.input", side_effect=["n", "y", ""]):
            _run_apply_flow(self.repo, self.incoming)

        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")

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
            side_effect=["n", "y", "y"],
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

    def test_successful_patch_no_creates_fresh_followup_context(self) -> None:
        save_context_state(self.repo, task="update app value")
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", return_value=self.result(),
        ), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "n", "the visible value is still wrong"]
        ) as prompt, patch("chatcode.cli.open_folder") as open_folder, patch(
            "builtins.print"
        ) as output:
            _run_apply_flow(self.repo, self.incoming)

        context = get_followup_context_file(self.repo)
        content = context.read_text(encoding="utf-8")
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")
        self.assertIn("## Original task\nupdate app value", content)
        self.assertIn("the visible value is still wrong", content)
        self.assertIn("value = 2", content)
        self.assertNotIn("value = 1", content)
        self.assertIn("Previous patch summary", content)
        self.assertIn("[UPLOAD THIS FILE]", "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        ))
        open_folder.assert_called_once_with(context.parent)
        self.assertEqual(prompt.call_count, 4)

    def test_empty_followup_feedback_reprompts(self) -> None:
        save_context_state(self.repo, task="update app value")
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", return_value=self.result(),
        ), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "n", "", "still wrong"]
        ) as prompt, patch("chatcode.cli.open_folder"):
            _run_apply_flow(self.repo, self.incoming)

        self.assertTrue(get_followup_context_file(self.repo).is_file())
        self.assertEqual(prompt.call_count, 5)

    def test_followup_feedback_adds_a_fresh_retrieval_surface(self) -> None:
        secondary = self.repo / "secondary.py"
        secondary.write_text(
            "def render_filter():\n    return 'current filter source'\n",
            encoding="utf-8", newline="\n",
        )
        save_context_state(self.repo, task="change app value")
        context = build_followup_context(
            self.repo,
            ApplyResult({"app.py"}, self.root / "history"),
            TestValidation(None, None, self.result(), "passed"),
            "The render_filter() control still does not update.",
            "Changed files:\n  - app.py",
        )

        content = context.read_text(encoding="utf-8")
        self.assertIn("render_filter", content)
        self.assertIn("current filter source", content)

    def test_followup_materializes_alternative_to_attempted_callback(self) -> None:
        current = (
            "def attempted_suggestion():\n    return 'PATCHED path A'\n\n"
            "def alternative_suggestion():\n    return f'Visible item ({1} nearby)'\n\n"
            "@ui.autocomplete(item=attempted_suggestion)\n"
            "def store_item(): pass\n\n"
            "@ui.autocomplete(item=alternative_suggestion)\n"
            "def collect_item(): pass\n"
        )
        source = self.repo / "commands.py"
        source.write_text(current, encoding="utf-8", newline="\n")
        git(self.repo, "add", "commands.py")
        git(self.repo, "commit", "-qm", "callbacks")
        history = self.root / "attempted-history"
        (history / "before").mkdir(parents=True)
        (history / "after").mkdir(parents=True)
        (history / "before" / "commands.py").write_text(
            current.replace("PATCHED path A", "original path A"), encoding="utf-8"
        )
        (history / "after" / "commands.py").write_text(current, encoding="utf-8")
        save_context_state(self.repo, task="change the visible item suggestion")

        context = build_followup_context(
            self.repo,
            ApplyResult({"commands.py"}, history),
            TestValidation(None, None, self.result(), "passed"),
            "The visible item output is still wrong.",
            "Changed attempted_suggestion in commands.py",
        ).read_text(encoding="utf-8")

        self.assertIn("def alternative_suggestion()", context)
        self.assertIn("return f'Visible item ({1} nearby)'", context)

    def test_followup_keeps_previous_patch_path_as_explicit_evidence(self) -> None:
        history = self.root / "followup-history"
        (history / "before").mkdir(parents=True)
        (history / "after").mkdir(parents=True)
        (history / "before" / "app.py").write_text(
            "value = 0\n",
            encoding="utf-8",
        )
        (history / "after" / "app.py").write_text(
            "value = 1\n",
            encoding="utf-8",
        )
        self.source.write_text(
            "value = 1\n",
            encoding="utf-8",
            newline="\n",
        )

        context = build_followup_context(
            self.repo,
            ApplyResult({"app.py"}, history),
            TestValidation(None, None, self.result(), "passed"),
            "The problem is still visible at runtime.",
            "Changed app.py",
            original_task="fix unrelated runtime behavior",
            persist_state=False,
        ).read_text(encoding="utf-8")

        self.assertIn("app.py", context)
        self.assertIn("value = 1", context)

    def test_followup_command_regenerates_from_current_source_and_saved_history(self) -> None:
        history = self.root / "history"
        (history / "before").mkdir(parents=True)
        (history / "after").mkdir(parents=True)
        before = "def attempted():\n    return 'old'\n"
        after = "def attempted():\n    return 'patched'\n"
        (history / "before" / "app.py").write_text(before, encoding="utf-8")
        (history / "after" / "app.py").write_text(after, encoding="utf-8")
        self.source.write_text(after, encoding="utf-8", newline="\n")
        save_context_state(self.repo, task="fix visible value")
        result = ApplyResult({"app.py"}, history)
        validation = TestValidation(None, None, self.result(), "passed")
        build_followup_context(
            self.repo, result, validation, "the visible value is still wrong", "Changed attempted",
        )
        self.source.write_text(
            "def attempted():\n    return 'CURRENT STATE C'\n",
            encoding="utf-8", newline="\n",
        )

        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "chatcode.cli.open_folder"
        ) as opened, patch("builtins.print") as output:
            status = command_followup()

        content = get_followup_context_file(self.repo).read_text(encoding="utf-8")
        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertEqual(status, 0)
        self.assertIn("CURRENT STATE C", content)
        self.assertNotIn("return 'patched'", content)
        self.assertIn("fix visible value", content)
        self.assertIn("the visible value is still wrong", content)
        self.assertIn("Changed attempted", content)
        self.assertIn("[UPLOAD THIS FILE]", rendered)
        opened.assert_called_once_with(get_followup_context_file(self.repo).parent)

    def test_followup_command_rejects_missing_and_resolved_state(self) -> None:
        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "chatcode.cli.open_folder"
        ) as opened, patch("builtins.print") as output:
            self.assertEqual(command_followup(), 1)
        self.assertIn(
            "No unresolved follow-up",
            "\n".join(str(call.args[0]) for call in output.call_args_list if call.args),
        )
        opened.assert_not_called()

        save_context_state(self.repo, task="fix value")
        history = self.root / "resolved-history"
        (history / "before").mkdir(parents=True)
        (history / "after").mkdir(parents=True)
        (history / "before" / "app.py").write_text("value = 1\n", encoding="utf-8")
        (history / "after" / "app.py").write_text("value = 2\n", encoding="utf-8")
        build_followup_context(
            self.repo, ApplyResult({"app.py"}, history),
            TestValidation(None, None, self.result(), "passed"), "still wrong", "Changed app.py",
        )
        mark_followup_resolved(self.repo)
        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "chatcode.cli.open_folder"
        ) as opened:
            self.assertEqual(command_followup(), 1)
        opened.assert_not_called()

    def test_failed_followup_context_keeps_patch_and_allows_keep(self) -> None:
        stdin, stdout = self.interactive()
        with patch("chatcode.patch._qwen_patch_summary", return_value=None), patch(
            "chatcode.test_runner.run_project_tests", return_value=self.result(),
        ), stdin, stdout, patch(
            "builtins.input", side_effect=["n", "y", "n", "still wrong", "k"]
        ) as prompt, patch(
            "chatcode.patch.build_followup_context", side_effect=OSError("disk full"),
        ), patch("chatcode.cli.open_folder") as open_folder, patch(
            "builtins.print"
        ) as output:
            _run_apply_flow(self.repo, self.incoming)

        rendered = "\n".join(str(call.args[0]) for call in output.call_args_list if call.args)
        self.assertIn("Could not create follow-up context: disk full", rendered)
        self.assertIn("[K] Keep  [U] Undo  [R] Review diff  [F] Retry follow-up context", rendered)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "value = 2\n")
        self.assertEqual(prompt.call_count, 5)
        open_folder.assert_not_called()

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
