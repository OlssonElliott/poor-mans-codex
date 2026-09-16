from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chatcode.context_builder import (
    CONTEXT_PURPOSE,
    MAX_FILES,
    PATCH_CONTEXT_HEADER,
    PATCH_RESPONSE_INSTRUCTIONS,
    UPLOAD_INSTRUCTIONS,
    build_context,
    build_patch_context,
    build_patch_source_context,
    build_tree,
    collect_relevant_files,
)
from chatcode.context_state import get_stale_context_reason, save_context_state
from chatcode.cli import command_repair
from chatcode.indexing.project_graph import load_map, save_map
from chatcode.retrieval.hybrid_retriever import CompletenessResult
from chatcode.patch import PatchError, apply_patch, build_patch_repair_context
from chatcode.workspace import (
    get_default_patch_file,
    get_repair_context_file,
    get_repo_workspace,
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

    def test_completeness_symbol_resolution_file_reaches_final_context(self) -> None:
        command = self.write("commands/transfer.py", "def transfer():\n    pass\n")
        service = self.write("services/inventory_transfer.py", "class InventoryTransferService:\n    pass\n")
        update = Mock(effective_mode="ai")
        completion = CompletenessResult(
            [service], {service: "Qwen completeness symbol resolution"}
        )
        with patch("chatcode.context_builder.update_project_map", return_value=update), patch(
            "chatcode.context_builder.retrieve_files", return_value=[command]
        ), patch("chatcode.context_builder.QwenCompletenessChecker.check", return_value=completion):
            selected = collect_relevant_files(self.repo, "transfer an inventory item")
        self.assertIn(command, selected)
        self.assertIn(service, selected)

    def test_real_split_take_drop_tests_promote_both_command_handlers(self) -> None:
        service_test = self.write(
            "tests/test_world.py",
            "def test_takes_and_drops_loose_stacked_items_without_duplication():\n"
            "    world.take_loose_item()\n"
            "    world.drop_item()\n",
        )
        command_test = self.write(
            "tests/test_world_commands.py",
            "async def test_take_names_character_and_refreshes_open_inventory():\n"
            "    await self.cog.take.callback(self.cog.take.binding, interaction, 'potion', 1)\n\n"
            "async def test_drop_posts_character_action_in_game():\n"
            "    await self.cog.drop.callback(self.cog.drop.binding, interaction, 'potion', 1)\n\n"
            "async def test_take_autocomplete_lists_loose_items_with_quantities():\n"
            "    await self.cog.loose_item_autocomplete(interaction, '')\n",
        )
        commands = self.write(
            "rpg_bot/commands/world.py",
            "async def take(): world.take_loose_item()\n"
            "async def drop(): world.drop_item()\n"
            "async def loose_item_autocomplete(): pass\n",
        )
        service = self.write(
            "rpg_bot/world_service.py",
            "def take_loose_item(self): self.place_item()\n"
            "def drop_item(self): self.place_catalog_item()\n"
            "def place_item(self): pass\n"
            "def place_catalog_item(self): pass\n",
        )
        save_map(self.repo, {"version": 3, "files": {
            "tests/test_world.py": {"symbols": [], "dependencies": []},
            "tests/test_world_commands.py": {"symbols": [], "dependencies": []},
            "rpg_bot/commands/world.py": {"symbols": [
                {"name": "take"}, {"name": "drop"}, {"name": "loose_item_autocomplete"},
            ], "dependencies": []},
            "rpg_bot/world_service.py": {"symbols": [
                {"name": "take_loose_item"}, {"name": "drop_item"},
                {"name": "place_item"}, {"name": "place_catalog_item"},
            ], "dependencies": []},
        }})
        task = (
            "When you drop many of the same thing into a room, they should stack. "
            "when you pick up a stack, you should be asked how many you want to pick up of the items"
        )
        update = Mock(effective_mode="static")
        with patch("chatcode.context_builder.update_project_map", return_value=update), patch(
            "chatcode.context_builder.retrieve_files", return_value=[service_test, command_test]
        ):
            _files, required = collect_relevant_files(
                self.repo, task, include_target_symbols=True
            )
        self.assertIn("take", required[commands])
        self.assertIn("drop", required[commands])
        self.assertIn("loose_item_autocomplete", required[commands])
        for symbol in ("take_loose_item", "drop_item", "place_item", "place_catalog_item"):
            self.assertIn(symbol, required[service])

    def test_required_drop_chain_outranks_supporting_context_in_final_export(self) -> None:
        command = self.write(
            "rpg_bot/commands/world.py",
            "async def drop(self, interaction):\n"
            "    return self.world.drop_item(interaction.user.id, 'potion')\n",
        )
        service = self.write(
            "rpg_bot/world_service.py",
            "def drop_item(self, character_id, item):\n"
            "    return self.database.transfer_to_room(character_id, item)\n",
        )
        database = self.write(
            "rpg_bot/database.py",
            "def generic_helper(self):\n    return 'low priority'\n\n"
            "def transfer_to_room(self, character_id, item):\n"
            "    return 'dirty current transfer implementation'\n",
        )
        support = self.write("docs/support.md", "irrelevant support\n" * 300)
        self.commit_all()
        database.write_text(
            database.read_text(encoding="utf-8").replace(
                "dirty current transfer implementation", "dirty working-tree transfer implementation"
            ), encoding="utf-8", newline="\n",
        )
        save_map(self.repo, {"version": 3, "files": {
            "rpg_bot/commands/world.py": {"symbols": [{"name": "drop"}], "dependencies": []},
            "rpg_bot/world_service.py": {"symbols": [{"name": "drop_item"}], "dependencies": []},
            # Deliberately omit transfer_to_room: materialization must resolve
            # the direct callee from fresh required-file source.
            "rpg_bot/database.py": {"symbols": [{"name": "generic_helper"}], "dependencies": []},
            "docs/support.md": {"symbols": [], "dependencies": []},
        }})
        required = {
            command: ["drop"], service: ["drop_item"], database: ["generic_helper"],
        }
        with patch(
            "chatcode.context_builder.collect_relevant_files",
            return_value=([command, service, database, support], required),
        ), patch.dict("os.environ", {"CHATCODE_CONTEXT_BUDGET_CHARS": "700"}, clear=False):
            context = build_context(self.repo, "drop an item into the room").read_text(encoding="utf-8")
        self.assertIn("async def drop(self, interaction):", context)
        self.assertIn("def drop_item(self, character_id, item):", context)
        self.assertIn("def transfer_to_room(self, character_id, item):", context)
        self.assertIn("dirty working-tree transfer implementation", context)
        self.assertNotIn("irrelevant support", context)

    def test_tiny_budget_marks_each_required_symbol_non_patchable(self) -> None:
        source = self.write(
            "service.py",
            "def required_operation():\n    return 'complete but too large for budget'\n",
        )
        save_map(self.repo, {"version": 3, "files": {
            "service.py": {"symbols": [{"name": "required_operation"}], "dependencies": []},
        }})
        with patch.dict("os.environ", {"CHATCODE_CONTEXT_BUDGET_CHARS": "10"}, clear=False):
            context = build_patch_source_context(
                self.repo, "change operation", [source], {source: ["required_operation"]}
            )
        self.assertIn(
            "REQUIRED SOURCE UNAVAILABLE: service.py::required_operation", context
        )
        self.assertIn("reason: budget; do not patch", context)

    def test_constrained_budget_materializes_dirty_selected_patch_targets_first(self) -> None:
        database = self.write(
            "rpg_bot/database.py",
            "class Database:\n" + "    value = 1\n" * 120,
        )
        test_file = self.write(
            "tests/test_world_commands.py",
            "def test_trap_disarm_kit():\n"
            "    confirmation = 'committed'\n"
            + "    padding = 1\n" * 120,
        )
        supporting = self.write("docs/architecture.md", "support\n" * 200)
        self.commit_all()
        test_file.write_text(
            "def test_trap_disarm_kit():\n"
            "    confirmation = 'dirty current working tree'\n"
            "    durability = 0\n"
            + "    padding = 1\n" * 120,
            encoding="utf-8", newline="\n",
        )
        save_map(self.repo, {"version": 3, "files": {
            "rpg_bot/database.py": {"dependencies": ["docs/architecture.md"]},
            "tests/test_world_commands.py": {"dependencies": []},
            "docs/architecture.md": {"dependencies": []},
        }})
        with patch("chatcode.context_builder.collect_relevant_files", return_value=[database, test_file]), patch.dict(
            "os.environ", {"CHATCODE_CONTEXT_BUDGET_CHARS": "1800"}, clear=False,
        ):
            context = build_context(self.repo, "Trap disarm kit durability").read_text(encoding="utf-8")
        self.assertIn("rpg_bot/database.py", context)
        self.assertIn("tests/test_world_commands.py", context)
        self.assertIn("dirty current working tree", context)
        self.assertNotIn("docs/architecture.md =====", context)
        self.assertNotIn("tests/test_world_commands.py [source unavailable", context)

    def test_unmaterialized_selection_is_explicitly_marked_non_patchable(self) -> None:
        source = self.write("src/large.py", "value = 1\n")
        with patch("chatcode.context_builder.collect_relevant_files", return_value=[source]), patch.dict(
            "os.environ", {"CHATCODE_CONTEXT_BUDGET_CHARS": "1"}, clear=False,
        ):
            context = build_context(self.repo, "change value").read_text(encoding="utf-8")
        self.assertIn("src/large.py [source unavailable; do not patch]", context)
        self.assertNotIn("===== FULL FILE: src/large.py =====", context)

    def test_large_explicit_command_handlers_materialize_complete_fresh_nodes(self) -> None:
        command = self.write(
            "rpg_bot/commands/world.py",
            "def unrelated_prefix():\n    return 'not a target'\n\n" + "# padding\n" * 2500
            + "@app_commands.command(name='take')\nasync def take(interaction):\n    return 'dirty take handler'\n\n"
            + "@app_commands.command(name='drop')\nasync def drop(interaction):\n    return 'dirty drop handler'\n",
        )
        service = self.write(
            "rpg_bot/world_service.py",
            "def unrelated_service():\n    return 0\n\n" + "# padding\n" * 2500
            + "class WorldService:\n    def take_loose_item(self):\n        return 'take target'\n\n    def drop_item(self):\n        return 'drop target'\n",
        )
        save_map(self.repo, {"version": 3, "files": {
            "rpg_bot/commands/world.py": {"symbols": [], "dependencies": []},
            "rpg_bot/world_service.py": {"symbols": [{"name": "take_loose_item"}, {"name": "drop_item"}], "dependencies": []},
        }})
        context = build_patch_source_context(
            self.repo, "make /take and /drop use take_loose_item() and drop_item()", files=[command, service],
        )
        self.assertIn("dirty take handler", context)
        self.assertIn("dirty drop handler", context)
        self.assertIn("return 'take target'", context)
        self.assertIn("return 'drop target'", context)
        self.assertNotIn("unrelated_prefix", context)

    def test_explicit_readme_filename_is_mandatory_context(self) -> None:
        readme = self.write("README.md", "# Current README\n")
        selected = collect_relevant_files(
            self.repo,
            "update README.md",
        )
        self.assertIn(readme.resolve(), selected)

    def test_bare_readme_resolves_unambiguous_readme_md(self) -> None:
        readme = self.write("README.md", "# Current README\n")
        selected = collect_relevant_files(
            self.repo,
            "update the readme",
        )
        self.assertIn(readme.resolve(), selected)

    def test_exact_and_windows_style_paths_select_exact_file(self) -> None:
        source = self.write("chatcode/patch.py", "value = 1\n")
        for task in (
            "update chatcode/patch.py",
            r"update chatcode\patch.py",
        ):
            with self.subTest(task=task):
                selected = collect_relevant_files(
                    self.repo,
                    task,
                )
                self.assertIn(source.resolve(), selected)

    def test_explicit_file_is_not_dropped_at_normal_cap(self) -> None:
        readme = self.write("README.md", "# Current README\n")
        graph_files = [
            self.write(
                f"src/result_{index}.py",
                f"value = {index}\n",
            )
            for index in range(MAX_FILES)
        ]
        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ), patch(
            "chatcode.context_builder.retrieve_files",
            return_value=graph_files,
        ):
            selected = collect_relevant_files(
                self.repo,
                "update README.md",
            )

        self.assertIn(readme.resolve(), selected)
        self.assertGreaterEqual(len(selected), MAX_FILES)

    def test_explicit_file_is_kept_alongside_many_dirty_files(self) -> None:
        readme = self.write("README.md", "# Current README\n")
        dirty = [
            self.write(
                f"src/dirty_{index}.py",
                f"value = {index}\n",
            )
            for index in range(MAX_FILES + 1)
        ]
        self.commit_all()
        for index, source in enumerate(dirty):
            source.write_text(
                f"value = {index + 100}\n",
                encoding="utf-8",
                newline="\n",
            )

        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ), patch(
            "chatcode.context_builder.retrieve_files",
            return_value=[readme],
        ):
            selected = collect_relevant_files(
                self.repo,
                "update README.md",
            )

        self.assertIn(readme.resolve(), selected)
        for source in dirty:
            self.assertIn(source.resolve(), selected)

    def test_explicit_file_is_deduplicated_from_retrieval(self) -> None:
        readme = self.write("README.md", "# Current README\n")
        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ), patch(
            "chatcode.context_builder.retrieve_files",
            return_value=[readme, readme],
        ):
            selected = collect_relevant_files(
                self.repo,
                "update README.md",
            )

        self.assertEqual(
            selected.count(readme.resolve()),
            1,
        )

    def test_explicit_dirty_file_uses_current_contents_and_hash(self) -> None:
        readme = self.write("README.md", "# committed\n")
        self.commit_all()
        readme.write_text(
            "# dirty current README\n",
            encoding="utf-8",
            newline="\n",
        )

        context = build_context(
            self.repo,
            "update the readme",
        ).read_text(encoding="utf-8")

        self.assertIn("===== FULL FILE: README.md =====", context)
        self.assertIn("# dirty current README", context)
        self.assertIn(
            hashlib.sha256(readme.read_bytes()).hexdigest(),
            context,
        )

    def test_ambiguous_basename_is_not_selected_arbitrarily(self) -> None:
        first = self.write("one/config.py", "value = 1\n")
        second = self.write("two/config.py", "value = 2\n")
        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ), patch(
            "chatcode.context_builder.retrieve_files",
            return_value=[],
        ):
            selected = collect_relevant_files(
                self.repo,
                "update config.py",
            )

        self.assertNotIn(first.resolve(), selected)
        self.assertNotIn(second.resolve(), selected)

    def test_nonexistent_and_parent_traversal_references_are_ignored(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("secret = True\n", encoding="utf-8")
        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ), patch(
            "chatcode.context_builder.retrieve_files",
            return_value=[],
        ):
            selected = collect_relevant_files(
                self.repo,
                "update missing.py and ../outside.py",
            )

        self.assertNotIn(outside.resolve(), selected)
        self.assertEqual(selected, [])

    def test_internal_artifact_is_not_selected_by_explicit_ordinary_task(self) -> None:
        self.write("chatcode/context_builder.py", "# source\n")
        self.write("chatcode/workspace.py", "# source\n")
        internal = self.write(
            "PATCH_REPAIR_CONTEXT.md",
            "generated repair context\n",
        )
        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ), patch(
            "chatcode.context_builder.retrieve_files",
            return_value=[],
        ):
            selected = collect_relevant_files(
                self.repo,
                "summarize PATCH_REPAIR_CONTEXT.md",
            )

        self.assertNotIn(internal.resolve(), selected)

    def test_context_instructions_use_source_file_line_numbers(self) -> None:
        self.assertIn(
            CONTEXT_PURPOSE,
            UPLOAD_INSTRUCTIONS,
        )
        self.assertIn(
            "never from Markdown/document line numbers",
            PATCH_RESPONSE_INSTRUCTIONS,
        )

    def test_normal_context_reads_current_working_tree_directly(self) -> None:
        source = self.write(
            "src/widget.py",
            "value = 'committed'\n",
        )
        self.commit_all()
        source.write_text(
            "value = 'current working tree'\n",
            encoding="utf-8",
            newline="\n",
        )

        with patch(
            "chatcode.context_builder.collect_relevant_files",
            return_value=[source],
        ):
            context = build_context(
                self.repo,
                "change widget",
            ).read_text(encoding="utf-8")

        self.assertIn(
            "value = 'current working tree'",
            context,
        )
        self.assertIn(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            context,
        )
        self.assertIn(
            "Source line range: 1-1",
            context,
        )

    def test_chatcode_internal_artifacts_are_hidden_from_normal_tree(self) -> None:
        self.write(
            "chatcode/context_builder.py",
            "# source\n",
        )
        self.write(
            "chatcode/workspace.py",
            "# source\n",
        )
        self.write(
            "UPLOAD_TO_CHATGPT.md",
            "old generated context\n",
        )
        self.write(
            "PATCH_REPAIR_CONTEXT.md",
            "old repair context\n",
        )
        self.write(
            "history/applied/old/metadata.json",
            "{}\n",
        )
        user_file = self.write(
            "docs/context.md",
            "legitimate user documentation\n",
        )

        tree = build_tree(self.repo)

        self.assertNotIn(
            "UPLOAD_TO_CHATGPT.md",
            tree,
        )
        self.assertNotIn(
            "PATCH_REPAIR_CONTEXT.md",
            tree,
        )
        self.assertNotIn(
            "history",
            tree,
        )
        self.assertIn(
            str(user_file.relative_to(self.repo)),
            tree,
        )

    def test_changed_files_are_not_lost_when_they_exceed_normal_cap(self) -> None:
        sources = [
            self.write(
                f"src/changed_{index}.py",
                f"value = {index}\n",
            )
            for index in range(MAX_FILES + 2)
        ]
        self.commit_all()

        for index, source in enumerate(sources):
            source.write_text(
                f"value = {index + 100}\n",
                encoding="utf-8",
                newline="\n",
            )

        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ), patch(
            "chatcode.context_builder.retrieve_files",
            return_value=[],
        ):
            selected = collect_relevant_files(
                self.repo,
                "update changed values",
            )

        self.assertEqual(
            set(selected),
            set(sources),
        )

    def test_internal_chatcode_workspace_is_not_sent_to_indexer(self) -> None:
        source = self.write("src/widget.py", "def widget():\n    return 1\n")
        internal_workspace = self.repo / "workspace"
        internal_file = internal_workspace / "other-project" / "app.py"
        internal_file.parent.mkdir(parents=True, exist_ok=True)
        internal_file.write_text("def other():\n    pass\n", encoding="utf-8")

        update = Mock(effective_mode="ai")
        with patch(
            "chatcode.workspace.get_workspace_root",
            return_value=internal_workspace,
        ):
            # Seed the same active workspace that ChatCode will purge and
            # query; changing the provider after writing would create a
            # different index file rather than a stale entry in this one.
            save_map(self.repo, {
                "version": 3,
                "files": {
                    "workspace/other-project/app.py": {
                        "path": "workspace/other-project/app.py",
                        "hash": "stale",
                        "language": "python",
                    }
                },
            })
        with patch(
            "chatcode.workspace.get_workspace_root",
            return_value=internal_workspace,
        ), patch(
            "chatcode.context_builder.update_project_map",
            return_value=update,
        ) as update_map, patch(
            "chatcode.context_builder.retrieve_files",
            return_value=[source, internal_file],
        ):
            selected = collect_relevant_files(self.repo, "change widget")

        indexed_paths = update_map.call_args.kwargs["paths"]
        self.assertIn("src/widget.py", indexed_paths)
        self.assertNotIn("workspace/other-project/app.py", indexed_paths)
        with patch(
            "chatcode.workspace.get_workspace_root",
            return_value=internal_workspace,
        ):
            self.assertNotIn(
                "workspace/other-project/app.py",
                load_map(self.repo).get("files", {}),
            )
        self.assertEqual(selected, [source])

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

    def test_staged_file_uses_working_tree_contents(self) -> None:
        source = self.write("src/staged.py", "value = 'A'\n")
        self.commit_all()
        source.write_text("value = 'B staged'\n", encoding="utf-8", newline="\n")
        git(self.repo, "add", "src/staged.py")

        context = build_patch_context(self.repo, "change src/staged.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("value = 'B staged'", context)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), context)

    def test_published_context_and_state_generation_must_match(self) -> None:
        self.write("app.py", "value = 1\n")
        self.commit_all()
        output = build_patch_context(self.repo, "change app.py")

        self.assertIsNone(get_stale_context_reason(self.repo, {"app.py"}))
        output.write_text(
            output.read_text(encoding="utf-8") + "\npartial newer generation",
            encoding="utf-8",
            newline="\n",
        )

        self.assertIn(
            "inte till samma publicerade generation",
            get_stale_context_reason(self.repo, {"app.py"}),
        )

    def test_staged_then_unstaged_file_uses_latest_working_tree_contents(self) -> None:
        source = self.write("src/three_versions.py", "value = 'A'\n")
        self.commit_all()
        source.write_text("value = 'B index'\n", encoding="utf-8", newline="\n")
        git(self.repo, "add", "src/three_versions.py")
        source.write_text("value = 'C working tree'\n", encoding="utf-8", newline="\n")

        context = build_patch_context(
            self.repo, "change src/three_versions.py"
        ).read_text(encoding="utf-8")

        self.assertIn("value = 'C working tree'", context)
        self.assertNotIn("value = 'B index'", context)
        self.assertNotIn("value = 'A'", context)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), context)

    def test_selected_untracked_file_uses_working_tree_contents(self) -> None:
        source = self.write("src/new_file.py", "def newly_created():\n    return 'disk'\n")

        context = build_patch_context(self.repo, "change src/new_file.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("def newly_created", context)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), context)

    def test_regeneration_and_symbol_context_do_not_reuse_stale_source(self) -> None:
        padding = "".join(f"padding_{index} = {index}\n" for index in range(3000))
        source = self.write(
            "src/fresh_large.py", padding + "def old_symbol():\n    return 'old'\n"
        )
        self.commit_all()
        first = build_patch_source_context(
            self.repo, "change old_symbol", files=[source]
        )
        source.write_text(
            padding + "def new_symbol():\n    return 'new'\n",
            encoding="utf-8",
            newline="\n",
        )

        second = build_patch_source_context(
            self.repo, "change new_symbol", files=[source]
        )

        self.assertIn("def old_symbol", first)
        self.assertIn("def new_symbol", second)
        self.assertNotIn("def old_symbol", second)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), second)

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

    def test_captured_source_hash_is_not_overwritten_by_later_state_scan(self) -> None:
        source = self.write("app.py", "value = 'context version'\n")
        self.commit_all()
        captured_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        source.write_text("value = 'later version'\n", encoding="utf-8", newline="\n")

        save_context_state(
            self.repo,
            task="change app",
            source_hashes={"app.py": captured_hash},
        )

        self.assertIsNotNone(get_stale_context_reason(self.repo, {"app.py"}))

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

        self.assertIn("===== SYMBOL CONTEXT: src/large.py =====", context)
        self.assertIn("EXCERPT src/large.py source lines ", context)
        self.assertIn("import os", context)
        self.assertIn("def wanted_symbol", context)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), context)
        self.assertNotIn("[File truncated by ChatCode]", context)

    def test_small_relevant_file_is_included_in_full(self) -> None:
        source = self.write("src/small.py", "def wanted():\n    return 1\n")

        context = build_patch_source_context(
            self.repo,
            "change wanted",
            files=[source],
        )

        self.assertIn("===== FULL FILE: src/small.py =====", context)
        self.assertIn("Source line range: 1-2", context)

    def test_large_file_uses_symbol_context_instead_of_full_file(self) -> None:
        padding = "".join(f"padding_{index} = {index}\n" for index in range(3000))
        source = self.write(
            "src/large_service.py",
            "import os\n\n"
            + padding
            + "def wanted_feature():\n    return os.getcwd()\n",
        )

        context = build_patch_source_context(
            self.repo,
            "change wanted_feature",
            files=[source],
        )

        self.assertIn("===== SYMBOL CONTEXT: src/large_service.py =====", context)
        self.assertIn("EXCERPT src/large_service.py source lines", context)
        self.assertIn("def wanted_feature", context)
        self.assertNotIn("padding_1500 = 1500", context)

    def test_large_test_file_selects_relevant_test_excerpt(self) -> None:
        unrelated = "".join(
            f"def test_unrelated_{index}():\n    assert {index} >= 0\n\n"
            for index in range(500)
        )
        source = self.write(
            "tests/test_transfer.py",
            unrelated
            + "def test_transfer_confirmation():\n"
            + "    assert confirm_transfer()\n",
        )

        context = build_patch_source_context(
            self.repo,
            "fix transfer confirmation",
            files=[source],
        )

        self.assertIn("test_transfer_confirmation", context)
        self.assertNotIn("test_unrelated_250", context)

    def test_context_budget_prioritizes_explicit_file(self) -> None:
        target = self.write(
            "src/target.py",
            "def target_feature():\n"
            + "".join(f"    value_{index} = {index}\n" for index in range(100))
            + "    return value_99\n",
        )
        other = self.write(
            "src/other.py",
            "def other_feature():\n"
            + "".join(f"    other_{index} = {index}\n" for index in range(100))
            + "    return other_99\n",
        )

        with patch.dict(
            "os.environ",
            {"CHATCODE_CONTEXT_BUDGET_CHARS": "5000"},
            clear=False,
        ):
            context = build_patch_source_context(
                self.repo,
                "update src/target.py",
                files=[other, target],
            )

        self.assertIn("src/target.py", context)

    def test_dependency_context_is_added_when_budget_allows(self) -> None:
        source = self.write("src/service.py", "def service():\n    return helper()\n")
        dependency = self.write("src/helper.py", "def helper():\n    return 1\n")
        save_map(self.repo, {
            "version": 3,
            "files": {
                "src/service.py": {
                    "path": "src/service.py",
                    "dependencies": ["src/helper.py"],
                },
                "src/helper.py": {
                    "path": "src/helper.py",
                    "dependencies": [],
                },
            },
        })

        context = build_patch_source_context(
            self.repo,
            "change service",
            files=[source],
        )

        self.assertIn("src/service.py", context)
        self.assertIn("src/helper.py", context)
        self.assertIn(dependency.read_text(encoding="utf-8"), context)

    def test_source_export_does_not_insert_blank_lines(self) -> None:
        source = self.write("src/compact.py", "first = 1\nsecond = 2\n")

        context = build_patch_source_context(
            self.repo,
            "change compact",
            files=[source],
        )

        self.assertIn("first = 1\nsecond = 2", context)
        self.assertNotIn("first = 1\n\nsecond = 2", context)

    def test_symbol_fallback_keeps_bounded_start_of_unstructured_file(self) -> None:
        source = self.write(
            "src/fallback.java",
            "".join(f"line_{index};\n" for index in range(3000)),
        )

        context = build_patch_source_context(
            self.repo,
            "change completely unrelated concept",
            files=[source],
        )

        self.assertIn("===== SYMBOL CONTEXT: src/fallback.java =====", context)
        self.assertIn("line_0;", context)
        self.assertNotIn("line_2000;", context)

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

    def test_context_free_hunk_for_large_existing_file_is_rejected_before_apply(self) -> None:
        source = self.write("app.py", "first\nold\nlast\n")
        self.commit_all()
        incoming = self.root / "fragile.diff"
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n"
            "@@ -2 +2 @@\n-old\n+new\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, incoming)

        self.assertEqual(
            raised.exception.failure_type,
            "insufficient_patch_context",
        )
        repair = get_repair_context_file(self.repo).read_text(encoding="utf-8")
        self.assertIn("Failure type: `insufficient_patch_context`", repair)
        self.assertIn("hunk has only 0 unchanged context line(s)", repair)
        self.assertEqual(source.read_text(encoding="utf-8"), "first\nold\nlast\n")

    def test_hunk_with_fewer_than_three_available_context_lines_is_rejected(self) -> None:
        source = self.write("app.py", "first\nsecond\nold\nfourth\nfifth\n")
        self.commit_all()
        incoming = self.root / "too-little-context.diff"
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n"
            "@@ -1,3 +1,3 @@\n first\n second\n-old\n+new\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(PatchError) as raised:
            apply_patch(self.repo, incoming)

        self.assertEqual(raised.exception.failure_type, "insufficient_patch_context")
        self.assertIn("only 2 unchanged context line(s)", str(raised.exception))

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
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), repair)
        self.assertIn("-stale = True\n+current = False", repair)
        self.assertIn("only a corrected unified diff", repair)
        self.assertEqual(source.read_text(encoding="utf-8"), "current = True\n")

    def test_repair_context_matches_windows_style_error_path(self) -> None:
        source = self.write("pkg/app.py", "current = True\n")
        self.commit_all()
        patch_text = (
            "--- a/pkg/app.py\n"
            "+++ b/pkg/app.py\n"
            "@@ -1 +1 @@\n"
            "-stale = True\n"
            "+current = False\n"
        )

        output = build_patch_repair_context(
            self.repo,
            patch_text,
            "error: patch failed: pkg\\app.py:1\n"
            "error: pkg\\app.py: patch does not apply",
        )
        repair = output.read_text(encoding="utf-8")

        self.assertIn("### pkg/app.py", repair)
        self.assertIn("current = True", repair)
        self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), repair)

    def test_repair_command_activates_repair_generation_for_next_apply(self) -> None:
        source = self.write("app.py", "current = True\n")
        self.commit_all()
        build_patch_repair_context(
            self.repo,
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-stale = True\n+current = False\n",
            "error: patch failed: app.py:1",
        )

        with patch("chatcode.cli.get_repo_root", return_value=self.repo), patch(
            "builtins.print"
        ):
            returncode = command_repair()

        state = json.loads(
            (get_repo_workspace(self.repo) / "context-state.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(returncode, 0)
        self.assertEqual(state["context_filename"], "PATCH_REPAIR_CONTEXT.md")
        self.assertEqual(state["context_kind"], "repair")
        self.assertIsNone(get_stale_context_reason(self.repo, {"app.py"}))

        source.write_text("later = True\n", encoding="utf-8", newline="\n")
        self.assertIsNotNone(get_stale_context_reason(self.repo, {"app.py"}))

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

        self.assertEqual(raised.exception.failure_type, "insufficient_patch_context")
        repair = get_repair_context_file(self.repo).read_text(encoding="utf-8")
        self.assertIn("needle = True", repair)
        self.assertNotRegex(repair, r"lines (\d{3,})-(\d{1,2})(?:\D|$)")
        self.assertNotIn("```text\n\n```", repair)
        self.assertEqual(source.read_text(encoding="utf-8"), "\n".join(lines) + "\n")


if __name__ == "__main__":
    unittest.main()
