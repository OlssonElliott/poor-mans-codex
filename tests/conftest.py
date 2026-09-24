from __future__ import annotations

import os
import tempfile
from pathlib import Path


# Prevent ChatCode from opening Explorer/Finder/xdg-open during tests,
# including detached subprocesses.
os.environ.setdefault("CHATCODE_DISABLE_AUTO_OPEN", "1")

# Never let test repositories write artifacts into ChatCode's real workspace.
# The environment variable is inherited by subprocesses and pytest-xdist
# workers, so even code paths without unittest.mock isolation stay contained.
_TEST_WORKSPACE = tempfile.TemporaryDirectory(
    prefix="chatcode-pytest-"
)
os.environ["CHATCODE_WORKSPACE_ROOT"] = str(
    Path(_TEST_WORKSPACE.name)
    / "workspace"
)


def pytest_sessionfinish(session, exitstatus) -> None:
    _TEST_WORKSPACE.cleanup()
