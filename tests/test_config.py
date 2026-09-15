from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from chatcode.config import get_boolean_setting, get_setting


class ConfigTests(unittest.TestCase):
    def test_process_environment_overrides_dotenv(self) -> None:
        with patch.dict(os.environ, {"CHATCODE_QWEN_MODEL": "session-model"}):
            self.assertEqual(get_setting("CHATCODE_QWEN_MODEL"), "session-model")

    def test_boolean_setting_accepts_explicit_toggle_values(self) -> None:
        with patch.dict(os.environ, {"CHATCODE_QWEN_ENABLED": "true"}):
            self.assertTrue(get_boolean_setting("CHATCODE_QWEN_ENABLED"))
        with patch.dict(os.environ, {"CHATCODE_QWEN_ENABLED": "off"}):
            self.assertFalse(get_boolean_setting("CHATCODE_QWEN_ENABLED", default=True))


if __name__ == "__main__":
    unittest.main()
