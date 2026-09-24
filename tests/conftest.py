from __future__ import annotations

import os


# Test repositories live under temporary workspaces. Prevent ChatCode from
# opening Explorer/Finder/xdg-open during tests, including detached subprocesses
# that do not inherit unittest.mock patches from the parent process.
os.environ.setdefault("CHATCODE_DISABLE_AUTO_OPEN", "1")
