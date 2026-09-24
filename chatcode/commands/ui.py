"""Small platform-specific CLI UI helpers."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..config import get_boolean_setting


def open_folder(
    path: Path,
) -> None:
    if get_boolean_setting("CHATCODE_DISABLE_AUTO_OPEN"):
        return

    path = path.resolve()

    try:
        if sys.platform == "win32":
            os.startfile(path)

        elif sys.platform == "darwin":
            subprocess.Popen([
                "open",
                str(path),
            ])

        else:
            subprocess.Popen([
                "xdg-open",
                str(path),
            ])

    except OSError as exc:
        print(
            "Could not open folder "
            f"automatically: {exc}",
            file=sys.stderr,
        )
