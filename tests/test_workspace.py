from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chatcode.workspace import atomic_write_text


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_write_replaces_complete_context_without_temp_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "UPLOAD_TO_CHATGPT.md"
            destination.write_text("old", encoding="utf-8")

            atomic_write_text(destination, "complete context", newline="\n")

            self.assertEqual(destination.read_text(encoding="utf-8"), "complete context")
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
