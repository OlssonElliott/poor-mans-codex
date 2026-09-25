from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from chatcode.workspace import (
    TEMPORARY_WORKSPACE_MAX_COUNT,
    TEMPORARY_WORKSPACE_MAX_AGE_SECONDS,
    atomic_write_text,
    cleanup_stale_temporary_workspaces,
    get_workspace_root,
)


class WorkspaceRootTests(unittest.TestCase):
    def test_environment_can_redirect_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configured = (
                Path(temporary)
                / "isolated-workspace"
            )

            previous = os.environ.get(
                "CHATCODE_WORKSPACE_ROOT"
            )
            os.environ[
                "CHATCODE_WORKSPACE_ROOT"
            ] = str(configured)
            try:
                workspace_root = get_workspace_root()
            finally:
                if previous is None:
                    os.environ.pop(
                        "CHATCODE_WORKSPACE_ROOT",
                        None,
                    )
                else:
                    os.environ[
                        "CHATCODE_WORKSPACE_ROOT"
                    ] = previous

            self.assertEqual(
                workspace_root,
                configured.resolve(),
            )
            self.assertTrue(
                workspace_root.is_dir(),
            )


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_write_replaces_complete_context_without_temp_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "UPLOAD_TO_CHATGPT.md"
            destination.write_text("old", encoding="utf-8")

            atomic_write_text(destination, "complete context", newline="\n")

            self.assertEqual(destination.read_text(encoding="utf-8"), "complete context")
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])


class TemporaryWorkspaceCleanupTests(unittest.TestCase):
    def test_removes_only_stale_temporary_repository_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = root / "tmpabc12345"
            recent = root / "tmpdef67890"
            ordinary = root / "my-project"
            for directory in (stale, recent, ordinary):
                (directory / "_workspace").mkdir(parents=True)

            now = time.time()
            old = now - TEMPORARY_WORKSPACE_MAX_AGE_SECONDS - 1
            os.utime(stale, (old, old))

            cleanup_stale_temporary_workspaces(root, now=now)

            self.assertFalse(stale.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(ordinary.exists())

    def test_preserves_the_active_workspace_even_when_its_parent_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "tmpabc12345" / "_workspace"
            workspace.mkdir(parents=True)
            old = time.time() - TEMPORARY_WORKSPACE_MAX_AGE_SECONDS - 1
            os.utime(workspace.parent, (old, old))

            cleanup_stale_temporary_workspaces(root, active_workspace=workspace)

            self.assertTrue(workspace.exists())

    def test_removes_a_temporary_workspace_after_one_day(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = root / "tmpabc12345"
            stale.mkdir()
            now = time.time()
            os.utime(
                stale,
                (now - TEMPORARY_WORKSPACE_MAX_AGE_SECONDS - 1,) * 2,
            )

            cleanup_stale_temporary_workspaces(root, now=now)

            self.assertFalse(stale.exists())

    def test_limits_number_of_recent_temporary_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = time.time()
            total = (
                TEMPORARY_WORKSPACE_MAX_COUNT
                + 3
            )

            directories: list[Path] = []
            for index in range(total):
                directory = (
                    root
                    / f"tmp{index:08d}"
                )
                directory.mkdir()
                os.utime(
                    directory,
                    (
                        now - index - 1,
                        now - index - 1,
                    ),
                )
                directories.append(
                    directory
                )

            cleanup_stale_temporary_workspaces(
                root,
                now=now,
            )

            remaining = [
                directory
                for directory in directories
                if directory.exists()
            ]

            self.assertEqual(
                len(remaining),
                TEMPORARY_WORKSPACE_MAX_COUNT,
            )
            self.assertFalse(
                directories[-1].exists()
            )
            self.assertFalse(
                directories[-2].exists()
            )
            self.assertFalse(
                directories[-3].exists()
            )

    def test_count_limit_never_removes_active_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = time.time()
            total = (
                TEMPORARY_WORKSPACE_MAX_COUNT
                + 1
            )

            directories = [
                root / f"tmp{index:08d}"
                for index in range(total)
            ]
            for index, directory in enumerate(directories):
                (directory / "_workspace").mkdir(
                    parents=True
                )
                modified = now - index - 1
                os.utime(
                    directory,
                    (modified, modified),
                )

            active = directories[-1]

            cleanup_stale_temporary_workspaces(
                root,
                active_workspace=active / "_workspace",
                now=now,
            )

            self.assertTrue(active.exists())
            self.assertEqual(
                sum(
                    directory.exists()
                    for directory in directories
                ),
                TEMPORARY_WORKSPACE_MAX_COUNT,
            )


if __name__ == "__main__":
    unittest.main()
