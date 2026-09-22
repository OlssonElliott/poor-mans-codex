from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from chatcode.config import get_boolean_setting, get_index_mode, get_setting
from chatcode.cli import command_status, create_parser


class ConfigTests(unittest.TestCase):
    def test_status_accepts_force_reindex_option(self) -> None:
        args = create_parser().parse_args(["status", "--reindex"])
        self.assertEqual(args.command, "status")
        self.assertTrue(args.reindex)

    def test_followup_command_parses(self) -> None:
        args = create_parser().parse_args(["followup"])
        self.assertEqual(args.command, "followup")

    def test_status_shows_completed_background_suite(self) -> None:
        background = {
            "state": "completed",
            "returncode": 0,
            "command": "python -m pytest -n auto",
            "duration_seconds": 12.5,
            "failed_tests": [],
            "report": "report.md",
        }
        with patch(
            "chatcode.cli.get_repo_root", return_value="repo"
        ), patch(
            "chatcode.cli.get_branch", return_value="main"
        ), patch(
            "chatcode.cli.build_safe_status", return_value=""
        ), patch(
            "chatcode.cli.get_background_full_suite_status",
            return_value=background,
        ), patch("builtins.print") as output:
            command_status()

        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("Background full suite: PASSED", rendered)
        self.assertIn("Duration: 12.50s", rendered)

    def test_status_colors_passed_background_suite_green(self) -> None:
        background = {
            "state": "completed",
            "returncode": 0,
        }
        with patch.dict(os.environ, {}, clear=True), patch(
            "chatcode.patch.sys.stdout.isatty", return_value=True
        ), patch(
            "chatcode.cli.get_repo_root", return_value="repo"
        ), patch(
            "chatcode.cli.get_branch", return_value="main"
        ), patch(
            "chatcode.cli.build_safe_status", return_value=""
        ), patch(
            "chatcode.cli.get_background_full_suite_status",
            return_value=background,
        ), patch("builtins.print") as output:
            command_status()

        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("Background full suite: \x1b[32mPASSED\x1b[0m", rendered)

    def test_status_colors_running_background_suite_yellow(self) -> None:
        background = {
            "state": "running",
        }
        with patch.dict(os.environ, {}, clear=True), patch(
            "chatcode.patch.sys.stdout.isatty", return_value=True
        ), patch(
            "chatcode.cli.get_repo_root", return_value="repo"
        ), patch(
            "chatcode.cli.get_branch", return_value="main"
        ), patch(
            "chatcode.cli.build_safe_status", return_value=""
        ), patch(
            "chatcode.cli.get_background_full_suite_status",
            return_value=background,
        ), patch("builtins.print") as output:
            command_status()

        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("Background full suite: \x1b[33mRUNNING\x1b[0m", rendered)

    def test_status_shows_background_suite_error(self) -> None:
        with patch(
            "chatcode.cli.get_repo_root", return_value="repo"
        ), patch(
            "chatcode.cli.get_branch", return_value="main"
        ), patch(
            "chatcode.cli.build_safe_status", return_value=""
        ), patch(
            "chatcode.cli.get_background_full_suite_status",
            return_value={"state": "error", "error": "worker failed"},
        ), patch("builtins.print") as output:
            command_status()

        rendered = "\n".join(
            str(call.args[0]) for call in output.call_args_list if call.args
        )
        self.assertIn("Background full suite: ERROR", rendered)
        self.assertIn("worker failed", rendered)

    def test_process_environment_overrides_dotenv(self) -> None:
        with patch.dict(os.environ, {"CHATCODE_QWEN_MODEL": "session-model"}):
            self.assertEqual(get_setting("CHATCODE_QWEN_MODEL"), "session-model")

    def test_boolean_setting_accepts_explicit_toggle_values(self) -> None:
        with patch.dict(os.environ, {"CHATCODE_QWEN_ENABLED": "true"}):
            self.assertTrue(get_boolean_setting("CHATCODE_QWEN_ENABLED"))
        with patch.dict(os.environ, {"CHATCODE_QWEN_ENABLED": "off"}):
            self.assertFalse(get_boolean_setting("CHATCODE_QWEN_ENABLED", default=True))

    def test_explicit_index_mode_wins(self) -> None:
        with patch("chatcode.config.get_setting") as setting:
            setting.side_effect = lambda name: {
                "CHATCODE_INDEX_MODE": "static",
                "CHATCODE_QWEN_MODEL": "qwen",
                "CHATCODE_QWEN_ENABLED": "true",
            }.get(name)
            self.assertEqual(get_index_mode(), "static")

    def test_model_selects_ai_when_mode_and_legacy_toggle_are_absent(self) -> None:
        with patch("chatcode.config.get_setting") as setting:
            setting.side_effect = lambda name: {
                "CHATCODE_QWEN_MODEL": "qwen",
            }.get(name)
            self.assertEqual(get_index_mode(), "ai")

    def test_legacy_disabled_toggle_selects_static(self) -> None:
        with patch("chatcode.config.get_setting") as setting:
            setting.side_effect = lambda name: {
                "CHATCODE_QWEN_MODEL": "qwen",
                "CHATCODE_QWEN_ENABLED": "false",
            }.get(name)
            self.assertEqual(get_index_mode(), "static")

    def test_invalid_explicit_index_mode_is_rejected(self) -> None:
        with patch("chatcode.config.get_setting", return_value="hybrid"):
            with self.assertRaisesRegex(ValueError, "ai.*static"):
                get_index_mode()


if __name__ == "__main__":
    unittest.main()
