from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from chatcode.config import get_boolean_setting, get_index_mode, get_setting


class ConfigTests(unittest.TestCase):
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
