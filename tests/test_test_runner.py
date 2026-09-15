from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.test_runner import (
    TestError as ChatCodeTestError,
    detect_test_command,
)


class TestCommandDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_python_project_with_tests_directory_uses_pytest(self) -> None:
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
                "pytest",
            ],
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
                config.unlink()

    def test_project_without_test_signals_still_raises(self) -> None:
        with self.assertRaisesRegex(
            ChatCodeTestError,
            "kunde inte hitta något testkommando",
        ):
            detect_test_command(
                self.repo
            )


if __name__ == "__main__":
    unittest.main()
