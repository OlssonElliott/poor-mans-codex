from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.history import (
    MAX_HISTORY_ENTRIES,
    _prune_history_dir,
    begin_history_entry,
    finalize_history_entry,
    get_latest_applied_entry,
    move_entry_to_undone,
)
from chatcode.workspace import (
    get_applied_history_dir,
    get_repo_workspace,
    get_undone_history_dir,
)


class HistoryRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.workspace = self.root / "workspace"
        self.workspace_patch = patch(
            "chatcode.workspace.get_workspace_root",
            return_value=self.workspace,
        )
        self.workspace_patch.start()
        self.source = self.repo / "app.py"
        self.source.write_text(
            "value = 1\n",
            encoding="utf-8",
            newline="\n",
        )

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temporary.cleanup()

    def create_applied_entry(self) -> Path:
        pending = begin_history_entry(
            self.repo,
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n",
            {"app.py"},
        )
        return finalize_history_entry(
            self.repo,
            pending,
        )

    def history_entries(
        self,
        directory: Path,
    ) -> list[Path]:
        return sorted(
            path
            for path in directory.iterdir()
            if (
                path.is_dir()
                and (path / "metadata.json").is_file()
            )
        )

    def test_up_to_five_applied_entries_are_kept(self) -> None:
        created = [
            self.create_applied_entry()
            for _ in range(MAX_HISTORY_ENTRIES)
        ]

        remaining = self.history_entries(
            get_applied_history_dir(self.repo)
        )

        self.assertEqual(
            [path.name for path in remaining],
            [path.name for path in created],
        )

    def test_sixth_applied_entry_removes_oldest(self) -> None:
        created = [
            self.create_applied_entry()
            for _ in range(MAX_HISTORY_ENTRIES + 1)
        ]

        remaining = self.history_entries(
            get_applied_history_dir(self.repo)
        )

        self.assertEqual(
            len(remaining),
            MAX_HISTORY_ENTRIES,
        )
        self.assertFalse(created[0].exists())
        self.assertEqual(
            {path.name for path in remaining},
            {path.name for path in created[-MAX_HISTORY_ENTRIES:]},
        )

    def test_sixth_undone_entry_removes_oldest(self) -> None:
        moved: list[Path] = []

        for _ in range(MAX_HISTORY_ENTRIES + 1):
            applied = self.create_applied_entry()
            moved.append(
                move_entry_to_undone(
                    self.repo,
                    applied,
                )
            )

        remaining = self.history_entries(
            get_undone_history_dir(self.repo)
        )

        self.assertEqual(
            len(remaining),
            MAX_HISTORY_ENTRIES,
        )
        self.assertFalse(moved[0].exists())
        self.assertEqual(
            {path.name for path in remaining},
            {path.name for path in moved[-MAX_HISTORY_ENTRIES:]},
        )

    def test_latest_applied_entry_still_works_after_cleanup(self) -> None:
        created = [
            self.create_applied_entry()
            for _ in range(MAX_HISTORY_ENTRIES + 2)
        ]

        latest = get_latest_applied_entry(
            self.repo
        )

        self.assertEqual(
            latest.name,
            created[-1].name,
        )

    def test_cleanup_does_not_touch_other_workspace_files(self) -> None:
        repo_workspace = get_repo_workspace(
            self.repo
        )
        project_map = repo_workspace / "project-map.json"
        context_state = repo_workspace / "context-state.json"
        incoming = (
            repo_workspace
            / "patches"
            / "incoming.diff"
        )
        incoming.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        project_map.write_text(
            "{}",
            encoding="utf-8",
        )
        context_state.write_text(
            "{}",
            encoding="utf-8",
        )
        incoming.write_text(
            "patch",
            encoding="utf-8",
        )

        for _ in range(MAX_HISTORY_ENTRIES + 2):
            self.create_applied_entry()

        self.assertEqual(
            project_map.read_text(encoding="utf-8"),
            "{}",
        )
        self.assertEqual(
            context_state.read_text(encoding="utf-8"),
            "{}",
        )
        self.assertEqual(
            incoming.read_text(encoding="utf-8"),
            "patch",
        )

    def test_missing_history_directory_is_safe(self) -> None:
        missing = self.root / "does-not-exist"

        _prune_history_dir(
            missing,
            "APPLIED",
        )

        self.assertFalse(missing.exists())

    def test_unexpected_history_contents_are_ignored(self) -> None:
        applied = get_applied_history_dir(
            self.repo
        )
        unexpected_file = applied / "notes.txt"
        unexpected_dir = applied / "manual-backup"

        unexpected_file.write_text(
            "keep me",
            encoding="utf-8",
        )
        unexpected_dir.mkdir()
        (unexpected_dir / "data.txt").write_text(
            "keep me too",
            encoding="utf-8",
        )

        _prune_history_dir(
            applied,
            "APPLIED",
        )

        self.assertTrue(unexpected_file.is_file())
        self.assertTrue(unexpected_dir.is_dir())

    def test_broken_old_history_entry_does_not_crash_cleanup(self) -> None:
        applied = get_applied_history_dir(
            self.repo
        )
        broken = (
            applied
            / "20000101_000000_000001"
        )
        broken.mkdir()
        (broken / "metadata.json").write_text(
            "{not valid json",
            encoding="utf-8",
        )

        for _ in range(MAX_HISTORY_ENTRIES + 1):
            self.create_applied_entry()

        self.assertFalse(broken.exists())


if __name__ == "__main__":
    unittest.main()
