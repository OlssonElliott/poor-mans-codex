from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chatcode.context_builder import (
    PATCH_CONTEXT_HEADER,
    build_patch_context,
    collect_relevant_files,
)
from chatcode.context_state import save_context_state
from chatcode.patch import PatchError, apply_patch
from chatcode.workspace import (
    get_default_patch_file,
    get_repair_context_file,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


class PatchContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.workspace = self.root / "workspace"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "ChatCode Tests")
        git(self.repo, "config", "user.email", "chatcode@example.test")
        self.workspace_patch = patch(
            "chatcode.workspace.get_workspace_root",
            return_value=self.workspace,
        )
        self.workspace_patch.start()

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temp.cleanup()

    def write(self, relative: str, content: str) -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def test_ai_mode_does_not_run_legacy_repository_content_ranker(self) -> None:
        source = self.write("src/widget.py", "def widget():\n    return 1\n")
        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map", return_value=update
        ), patch(
            "chatcode.context_builder.retrieve_files", return_value=[source]
        ) as retrieve, patch(
            "chatcode.context_builder.get_changed_files", return_value=set()
        ), patch(
            "chatcode.context_builder.iter_repository_files"
        ) as legacy_scan:
            selected = collect_relevant_files(self.repo, "change widget")

        self.assertEqual(selected, [source])
        retrieve.assert_called_once_with(
            self.repo, "change widget", max_files=12, index_mode="ai"
        )
        legacy_scan.assert_not_called()

    def commit_all(self) -> None:
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "fixture")

    def test_clean_working_tree_uses_exact_current_file_and_hash(self) -> None:
        source = self.write("src/widget.py", "def widget():\n    return 1\n")
        self.commit_all()

        output = build_patch_context(self.repo, "change widget")
        context = output.read_text(encoding="utf-8")

        self.assertIn(PATCH_CONTEXT_HEADER, context)
        self.assertIn("Working tree clean", context)
        self.assertIn("src/widget.py", context)
        self.assertIn(source.read_text(encoding="utf-8"), context)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), context)
        self.assertIn(self.repo.resolve().as_posix(), context)
        self.assertNotIn("[File truncated by ChatCode]", context)
        self.assertNotIn("## Unstaged changes", context)

    def test_dirty_likely_target_is_included_in_full(self) -> None:
        source = self.write("src/handler.py", "def handler():\n    return 'base'\n")
        self.commit_all()
        source.write_text(
            "def handler():\n    return 'intentional dirty value'\n",
            encoding="utf-8",
            newline="\n",
        )

        context = build_patch_context(
            self.repo,
            "update handler",
        ).read_text(encoding="utf-8")

        self.assertIn("===== FULL FILE: src/handler.py =====", context)
        self.assertIn("intentional dirty value", context)
        self.assertIn(" M src/handler.py", context)

    def test_file_modified_after_context_generation_is_rejected(self) -> None:
        source = self.write("app.py", "value = 1\n")
        self.commit_all()
        build_patch_context(self.repo, "change app value")
        save_context_state(self.repo)
        source.write_text("value = 99\n", encoding="utf-8", newline="\n")
        incoming = get_default_patch_file(self.repo)
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaisesRegex(PatchError, "inaktuell") as raised:
            apply_patch(self.repo, incoming)

        self.assertEqual(raised.exception.failure_type, "stale_context")
        repair = get_repair_context_file(self.repo).read_text(encoding="utf-8")
        self.assertIn("Failure type: `stale_context`", repair)
        self.assertIn("value = 99", repair)
        self.assertEqual(source.read_text(encoding="utf-8"), "value = 99\n")

    def test_large_file_uses_labeled_symbol_aware_current_excerpt(self) -> None:
        padding = "".join(f"padding_{index} = {index}\n" for index in range(6000))
        source = self.write(
            "src/large.py",
            "import os\n\n" + padding + "def wanted_symbol():\n    return os.getcwd()\n",
        )
        self.commit_all()

        context = build_patch_context(
            self.repo,
            "change wanted_symbol",
        ).read_text(encoding="utf-8")

        self.assertIn("EXCERPTS FROM CURRENT FILE: src/large.py", context)
        self.assertIn("EXCERPT src/large.py lines ", context)
        self.assertIn("import os", context)
        self.assertIn("def wanted_symbol", context)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), context)
        self.assertNotIn("[File truncated by ChatCode]", context)

    def test_apply_check_success_preserves_existing_uncommitted_change(self) -> None:
        source = self.write("app.py", "first\nbase\n")
        self.commit_all()
        source.write_text("first\ndirty\n", encoding="utf-8", newline="\n")
        incoming = self.root / "success.diff"
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,3 @@\n first\n dirty\n+added\n",
            encoding="utf-8",
            newline="\n",
        )

        apply_patch(self.repo, incoming)

        self.assertEqual(
            source.read_text(encoding="utf-8"),
            "first\ndirty\nadded\n",
        )

    def test_count_metadata_is_canonicalized_before_apply(self) -> None:
        source = self.write("app.py", "old\n")
        self.commit_all()
        incoming = self.root / "wrong-counts.diff"
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n@@ -1,40 +1,70 @@\n-old\n+new\n",
            encoding="utf-8",
            newline="\n",
        )

        apply_patch(self.repo, incoming)

        self.assertEqual(source.read_text(encoding="utf-8"), "new\n")
        self.assertIn(
            "@@ -1,1 +1,1 @@",
            incoming.read_text(encoding="utf-8"),
        )

    def test_unrelated_uncommitted_change_is_preserved(self) -> None:
        source = self.write("app.py", "old\n")
        unrelated = self.write("notes.txt", "committed\n")
        self.commit_all()
        unrelated.write_text("dirty and unrelated\n", encoding="utf-8", newline="\n")
        incoming = self.root / "app-only.diff"
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
            encoding="utf-8",
            newline="\n",
        )

        apply_patch(self.repo, incoming)

        self.assertEqual(source.read_text(encoding="utf-8"), "new\n")
        self.assertEqual(
            unrelated.read_text(encoding="utf-8"),
            "dirty and unrelated\n",
        )

    def test_corrupt_patch_gets_syntax_specific_repair_context(self) -> None:
        source = self.write("app.py", "current = True\n")
        self.commit_all()
        save_context_state(self.repo, task="change current flag")
        incoming = self.root / "corrupt.diff"
        original = "--- a/app.py\n+++ b/app.py\n-current = True\n+current = False\n"
        incoming.write_text(original, encoding="utf-8", newline="\n")

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, incoming)

        self.assertEqual(raised.exception.failure_type, "invalid_patch_syntax")
        repair = get_repair_context_file(self.repo).read_text(encoding="utf-8")
        self.assertIn("Failure type: `invalid_patch_syntax`", repair)
        self.assertIn("change current flag", repair)
        self.assertIn("## Complete generated patch", repair)
        self.assertIn(original.rstrip(), repair)
        self.assertIn("current = True", repair)
        self.assertEqual(source.read_text(encoding="utf-8"), "current = True\n")

    def test_failed_apply_check_writes_compact_repair_context(self) -> None:
        source = self.write("app.py", "current = True\n")
        self.commit_all()
        incoming = self.root / "failure.diff"
        failed_hunk = (
            "@@ -1 +1 @@\n"
            "-stale = True\n"
            "+current = False"
        )
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n" + failed_hunk + "\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaisesRegex(PatchError, "Repair context") as raised:
            apply_patch(self.repo, incoming)

        self.assertEqual(raised.exception.failure_type, "patch_target_mismatch")
        repair = get_repair_context_file(self.repo).read_text(encoding="utf-8")
        self.assertIn("git apply error", repair)
        self.assertIn("current = True", repair)
        self.assertIn("-stale = True\n+current = False", repair)
        self.assertIn("only a corrected unified diff", repair)
        self.assertEqual(source.read_text(encoding="utf-8"), "current = True\n")

    def test_out_of_range_hunk_recovers_matching_context_without_invalid_range(self) -> None:
        lines = [f"padding_{index} = {index}" for index in range(7000)]
        lines[120] = "needle = True"
        source = self.write("large.py", "\n".join(lines) + "\n")
        self.commit_all()
        incoming = self.root / "out-of-range.diff"
        incoming.write_text(
            "--- a/large.py\n+++ b/large.py\n"
            "@@ -99999,2 +99999,2 @@\n"
            " needle = True\n"
            "-missing_after_needle = True\n"
            "+replacement = True\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, incoming)

        self.assertEqual(raised.exception.failure_type, "patch_target_mismatch")
        repair = get_repair_context_file(self.repo).read_text(encoding="utf-8")
        self.assertIn("needle = True", repair)
        self.assertNotRegex(repair, r"lines (\d{3,})-(\d{1,2})(?:\D|$)")
        self.assertNotIn("```text\n\n```", repair)
        self.assertEqual(source.read_text(encoding="utf-8"), "\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
