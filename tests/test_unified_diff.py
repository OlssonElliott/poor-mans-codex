from __future__ import annotations

import unittest

from chatcode.unified_diff import (
    UnifiedDiffError,
    canonicalize_unified_diff,
    count_mismatches,
    parse_unified_diff,
)


class UnifiedDiffTests(unittest.TestCase):
    def test_valid_unified_diff(self) -> None:
        patch = (
            "--- a/app.py\n+++ b/app.py\n"
            "@@ -1,2 +1,2 @@\n unchanged\n-old\n+new\n"
        )
        parsed = parse_unified_diff(patch)
        self.assertEqual(len(parsed.files), 1)
        self.assertEqual(count_mismatches(parsed), [])

    def test_incorrect_old_count_is_detected_and_normalized(self) -> None:
        patch = "--- a/app.py\n+++ b/app.py\n@@ -1,9 +1,1 @@\n-old\n+new\n"
        parsed = parse_unified_diff(patch)
        self.assertIn("-9/+1", count_mismatches(parsed)[0])
        canonical, _ = canonicalize_unified_diff(patch)
        self.assertIn("@@ -1,1 +1,1 @@", canonical)

    def test_incorrect_new_count_is_detected_and_normalized(self) -> None:
        patch = "--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,7 @@\n-old\n+new\n"
        parsed = parse_unified_diff(patch)
        self.assertIn("-1/+7", count_mismatches(parsed)[0])
        canonical, _ = canonicalize_unified_diff(patch)
        self.assertIn("@@ -1,1 +1,1 @@", canonical)

    def test_rename_metadata_survives_canonicalization(self) -> None:
        patch = (
            "diff --git a/old.py b/pkg/new.py\n"
            "similarity index 80%\n"
            "rename from old.py\n"
            "rename to pkg/new.py\n"
            "--- a/old.py\n"
            "+++ b/pkg/new.py\n"
            "@@ -1 +1 @@\n"
            "-old = True\n"
            "+old = False\n"
        )

        canonical, mismatches = canonicalize_unified_diff(patch)

        self.assertEqual(mismatches, [])
        self.assertIn(
            "diff --git a/old.py b/pkg/new.py\n",
            canonical,
        )
        self.assertIn(
            "similarity index 80%\n"
            "rename from old.py\n"
            "rename to pkg/new.py\n",
            canonical,
        )
        self.assertIn("--- a/old.py\n", canonical)
        self.assertIn("+++ b/pkg/new.py\n", canonical)

    def test_metadata_only_rename_is_valid_and_preserved(self) -> None:
        patch = (
            "diff --git a/old.py b/pkg/new.py\n"
            "similarity index 100%\n"
            "rename from old.py\n"
            "rename to pkg/new.py\n"
        )

        canonical, mismatches = canonicalize_unified_diff(patch)
        parsed = parse_unified_diff(canonical)

        self.assertEqual(mismatches, [])
        self.assertEqual(canonical, patch)
        self.assertEqual(len(parsed.files), 1)
        self.assertEqual(parsed.files[0].old_path, "old.py")
        self.assertEqual(parsed.files[0].new_path, "pkg/new.py")
        self.assertEqual(parsed.files[0].hunks, ())

    def test_metadata_only_copy_is_valid_and_preserved(self) -> None:
        patch = (
            "diff --git a/source.py b/pkg/source.py\n"
            "similarity index 100%\n"
            "copy from source.py\n"
            "copy to pkg/source.py\n"
        )

        canonical, mismatches = canonicalize_unified_diff(patch)

        self.assertEqual(mismatches, [])
        self.assertEqual(canonical, patch)

    def test_missing_hunk_header_is_rejected(self) -> None:
        patch = "--- a/app.py\n+++ b/app.py\n-old\n+new\n"
        with self.assertRaisesRegex(UnifiedDiffError, "hunk header"):
            parse_unified_diff(patch)

    def test_context_only_hunk_is_rejected_before_git(self) -> None:
        patch = (
            "--- a/rpg_bot/commands/world.py\n"
            "+++ b/rpg_bot/commands/world.py\n"
            "@@ -229,6 +229,7 @@\n"
            "     async def inventory_item_autocomplete(\n"
            "         self, interaction: discord.Interaction, current: str\n"
            "     ) -> list[app_commands.Choice[str]]:\n"
            "         \"\"\"Suggest droppable items.\"\"\"\n"
            "         try:\n"
            "             character_id = self._active_character_id(interaction.user.id)\n"
            "         except CharacterNotFoundError:\n"
            "             return []\n"
            " \n"
            "         inventory = self.database.get_character_inventory(character_id)\n"
        )
        with self.assertRaisesRegex(UnifiedDiffError, "hunk contains no changes"):
            canonicalize_unified_diff(patch)

    def test_no_newline_metadata_does_not_affect_counts(self) -> None:
        patch = (
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"
            "-old\n\\ No newline at end of file\n"
            "+new\n\\ No newline at end of file\n"
        )
        canonical, mismatches = canonicalize_unified_diff(patch)
        self.assertEqual(mismatches, [])
        self.assertIn("@@ -1,1 +1,1 @@", canonical)


if __name__ == "__main__":
    unittest.main()
