from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.test_runner import (
    TestError as ChatCodeTestError,
    _failed_test_ids,
    _project_python,
    detect_test_command,
    run_relevant_tests,
    unmapped_python_source_paths,
)


class TestCommandDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_python_project_with_tests_directory_uses_unittest_without_pytest_config(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            "[project]\nname = \"demo\"\n",
            encoding="utf-8",
        )
        (self.repo / "tests").mkdir()

        with patch(
            "chatcode.test_runner._project_python",
            return_value="python-test",
        ):
            command = detect_test_command(
                self.repo
            )

        self.assertEqual(
            command.args,
            [
                "python-test",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
            ],
        )

    def test_project_python_skips_candidate_without_required_module(self) -> None:
        with patch(
            "chatcode.test_runner._python_candidates",
            return_value=["tool-python", "project-python"],
        ), patch(
            "chatcode.test_runner._python_can_import",
            side_effect=lambda python, module: (
                python == "project-python"
                and module == "pytest"
            ),
        ):
            selected = _project_python(
                self.repo,
                required_module="pytest",
            )

        self.assertEqual(selected, "project-python")

    def test_project_python_reports_missing_required_module(self) -> None:
        with patch(
            "chatcode.test_runner._python_candidates",
            return_value=["tool-python"],
        ), patch(
            "chatcode.test_runner._python_can_import",
            return_value=False,
        ):
            with self.assertRaisesRegex(
                ChatCodeTestError,
                "kan importera pytest",
            ):
                _project_python(
                    self.repo,
                    required_module="pytest",
                )

    def test_setup_cfg_and_tox_ini_can_signal_pytest(self) -> None:
        configs = {
            "setup.cfg": (
                "[tool:pytest]\n"
                "addopts = -q\n"
            ),
            "tox.ini": (
                "[testenv]\n"
                "commands = pytest\n"
            ),
        }

        for name, content in configs.items():
            with self.subTest(name=name):
                config = self.repo / name
                config.write_text(
                    content,
                    encoding="utf-8",
                )

                with patch(
                    "chatcode.test_runner._project_python",
                    return_value="python-test",
                ), patch(
                    "chatcode.test_runner._python_can_import",
                    return_value=True,
                ):
                    command = detect_test_command(
                        self.repo
                    )

                self.assertEqual(
                    command.args,
                    [
                        "python-test",
                        "-m",
                        "pytest",
                        "-n",
                        "auto",
                    ],
                )
                config.unlink()

    def test_pytest_runs_serially_when_xdist_is_unavailable(self) -> None:
        (self.repo / "pytest.ini").write_text(
            "[pytest]\n",
            encoding="utf-8",
        )

        with patch(
            "chatcode.test_runner._project_python",
            return_value="python-test",
        ), patch(
            "chatcode.test_runner._python_can_import",
            return_value=False,
        ):
            command = detect_test_command(
                self.repo
            )

        self.assertEqual(
            command.args,
            [
                "python-test",
                "-m",
                "pytest",
            ],
        )

    def test_project_without_test_signals_still_raises(self) -> None:
        with self.assertRaisesRegex(
            ChatCodeTestError,
            "kunde inte hitta något testkommando",
        ):
            detect_test_command(
                self.repo
            )

    def test_configured_full_command_wins_over_autodetection(self) -> None:
        config = self.repo / ".chatcode" / "tests.toml"
        config.parent.mkdir()
        config.write_text(
            '[tests]\nfull = ["custom-test", "--all"]\n',
            encoding="utf-8",
        )
        (self.repo / "package.json").write_text(
            '{"scripts":{"test":"ignored"}}',
            encoding="utf-8",
        )

        command = detect_test_command(self.repo)

        self.assertEqual(command.args, ["custom-test", "--all"])

    def test_configured_fast_command_runs_without_filename_mapping(self) -> None:
        config = self.repo / ".chatcode" / "tests.toml"
        config.parent.mkdir()
        config.write_text(
            '[tests]\nfast = ["custom-test", "--unit"]\n',
            encoding="utf-8",
        )

        with patch("chatcode.test_runner._run_test_command") as run:
            run_relevant_tests(self.repo, {"unmapped/source.file"})

        command = run.call_args.args[1]
        self.assertEqual(command.args, ["custom-test", "--unit"])

    def test_configured_command_string_preserves_grouped_arguments(self) -> None:
        config = self.repo / ".chatcode" / "tests.toml"
        config.parent.mkdir()
        config.write_text(
            '[tests]\nfast = "python -m pytest -m \\"not integration\\""\n',
            encoding="utf-8",
        )

        with patch("chatcode.test_runner._run_test_command") as run:
            run_relevant_tests(self.repo, set())

        command = run.call_args.args[1]
        self.assertEqual(
            command.args,
            ["python", "-m", "pytest", "-m", "not integration"],
        )

    def test_invalid_configured_command_is_rejected(self) -> None:
        config = self.repo / ".chatcode" / "tests.toml"
        config.parent.mkdir()
        config.write_text("[tests]\nfull = []\n", encoding="utf-8")

        with self.assertRaisesRegex(ChatCodeTestError, "tests.full"):
            detect_test_command(self.repo)

    def test_relevant_unittest_file_is_run_before_full_suite(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            "[project]\nname = \"demo\"\n",
            encoding="utf-8",
        )
        tests = self.repo / "tests"
        tests.mkdir()
        (tests / "test_widget.py").write_text(
            "# fixture\n",
            encoding="utf-8",
        )

        with patch(
            "chatcode.test_runner._project_python",
            return_value="python-test",
        ), patch(
            "chatcode.test_runner._run_test_command",
        ) as run:
            run_relevant_tests(self.repo, {"package/widget.py"})

        command = run.call_args.args[1]
        self.assertEqual(
            command.args,
            [
                "python-test", "-m", "unittest", "discover",
                "-s", "tests", "-p", "test_widget.py",
            ],
        )

    def test_relevant_pytest_file_uses_parallel_workers(self) -> None:
        (self.repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        tests = self.repo / "tests"
        tests.mkdir()
        (tests / "test_widget.py").write_text("", encoding="utf-8")

        with patch(
            "chatcode.test_runner._project_python",
            return_value="python-test",
        ), patch(
            "chatcode.test_runner._python_can_import",
            return_value=True,
        ), patch(
            "chatcode.test_runner._run_test_command",
        ) as run:
            run_relevant_tests(self.repo, {"package/widget.py"})

        command = run.call_args.args[1]
        self.assertEqual(
            command.args,
            [
                "python-test", "-m", "pytest", "-n", "auto",
                "tests/test_widget.py",
            ],
        )

    def test_relevant_tests_do_not_guess_ambiguous_mapping(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            "[project]\nname = \"demo\"\n",
            encoding="utf-8",
        )
        tests = self.repo / "tests"
        (tests / "one").mkdir(parents=True)
        (tests / "two").mkdir()
        (tests / "one" / "test_widget.py").write_text("", encoding="utf-8")
        (tests / "two" / "test_widget.py").write_text("", encoding="utf-8")

        self.assertIsNone(
            run_relevant_tests(self.repo, {"package/widget.py"})
        )

    def test_reports_python_source_without_identifiable_test(self) -> None:
        (self.repo / "pyproject.toml").write_text(
            "[project]\nname = \"demo\"\n",
            encoding="utf-8",
        )
        (self.repo / "tests").mkdir()

        self.assertEqual(
            unmapped_python_source_paths(
                self.repo,
                {"package/widget.py", "README.md", "tests/test_other.py"},
            ),
            ["package/widget.py"],
        )

    def test_source_with_one_conventional_test_is_not_reported(self) -> None:
        (self.repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        tests = self.repo / "tests"
        tests.mkdir()
        (tests / "test_widget.py").write_text("", encoding="utf-8")

        self.assertEqual(
            unmapped_python_source_paths(self.repo, {"package/widget.py"}),
            [],
        )

    def test_failure_parser_ignores_unittest_aggregate_summary(self) -> None:
        failures = _failed_test_ids(
            "FAIL: test_drop (tests.test_world.WorldTests.test_drop)\n"
            "FAILED (failures=1, errors=0)\n",
            "",
        )

        self.assertEqual(
            failures,
            frozenset({"tests.test_world.WorldTests.test_drop"}),
        )


if __name__ == "__main__":
    unittest.main()
