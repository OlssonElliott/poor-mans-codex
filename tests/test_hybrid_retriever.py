from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.indexing.project_graph import save_map
from chatcode.retrieval.hybrid_retriever import (
    RetrievalResult,
    QwenCompletenessChecker,
    QwenTaskHintAnalyzer,
    expand_candidates,
    implementation_closure,
    test_callsite_closure,
    resolve_explicit_targets,
    resolve_alternative_callback_roots,
    resolve_task_surface_roots,
    resolve_transport_roots,
    resolve_semantic_hints,
)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.workspace = self.repo / "workspace"
        self.workspace.mkdir()
        self.workspace_patch = patch("chatcode.workspace.get_workspace_root", return_value=self.workspace)
        self.workspace_patch.start()

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temp.cleanup()

    def write(self, name: str, text: str = "") -> Path:
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "pass\n", encoding="utf-8")
        return path

    def map(self, entries: dict[str, dict]) -> None:
        save_map(self.repo, {"version": 3, "files": entries})

    def test_seed_expands_to_direct_service_and_related_test(self) -> None:
        command = self.write("commands/inventory.py")
        service = self.write("services/inventory_service.py")
        test = self.write("tests/test_inventory.py")
        self.map({
            "commands/inventory.py": {"dependencies": ["services/inventory_service.py"], "symbols": []},
            "services/inventory_service.py": {"dependencies": [], "symbols": [{"name": "InventoryService"}]},
            "tests/test_inventory.py": {"dependencies": [], "symbols": []},
        })
        result = expand_candidates(self.repo, "fix inventory command", [command])
        self.assertIn(service, result.files)
        self.assertIn(test, result.files)

    def test_missing_http_method_resolves_indexed_sibling_handler_owner(self) -> None:
        frontend = self.write(
            "web/editor.tsx", "fetch('/api/items/1', { method: 'PUT' })\n"
        )
        api = self.write("app/item_api.py", "def update_item_template(): pass\n")
        server = self.write(
            "infra/local_transport.py",
            "class LocalRequestHandler:\n"
            "    def do_GET(self): pass\n"
            "    def do_POST(self): pass\n"
            "    def do_PATCH(self): pass\n"
            "    def do_DELETE(self): pass\n"
            "    def do_OPTIONS(self): pass\n",
        )
        unrelated = self.write(
            "other/server.py",
            "class OtherHandler:\n"
            "    def do_GET(self): pass\n"
            "    def do_POST(self): pass\n"
            "    def do_PATCH(self): pass\n",
        )
        self.map({
            "web/editor.tsx": {"dependencies": [], "symbols": []},
            "app/item_api.py": {"dependencies": [], "symbols": [{"name": "update_item_template"}]},
            "infra/local_transport.py": {"dependencies": ["app/item_api.py"], "symbols": [
                {"name": "LocalRequestHandler"}, {"name": "do_GET"}, {"name": "do_POST"},
                {"name": "do_PATCH"}, {"name": "do_DELETE"}, {"name": "do_OPTIONS"},
            ]},
            "other/server.py": {"dependencies": [], "symbols": [{"name": "OtherHandler"}]},
        })

        result = resolve_transport_roots(self.repo, [frontend, api])

        self.assertEqual(result.files, [server])
        self.assertEqual(result.required_symbols[server], ["LocalRequestHandler"])
        self.assertNotIn(unrelated, result.files)

    def test_disconnected_explicit_task_surfaces_receive_separate_roots(self) -> None:
        command = self.write("commands/world.py")
        inventory = self.write("ui/inventory_ui.py")
        self.map({
            "commands/world.py": {"dependencies": [], "symbols": [{"name": "drop"}]},
            "ui/inventory_ui.py": {"dependencies": [], "symbols": [
                {"name": "inventory_embed"}, {"name": "InventoryView"},
            ]},
        })

        result = resolve_task_surface_roots(
            self.repo,
            "right now stackable items are not displayed as stacked in the character inventory. "
            "when you write /drop they are displayed as different items.",
        )

        self.assertIn(command, result.files)
        self.assertIn(inventory, result.files)
        self.assertIn("drop", result.required_symbols[command])
        self.assertIn("inventory_embed", result.required_symbols[inventory])
        self.assertIn("InventoryView", result.required_symbols[inventory])

    def test_runtime_literal_promotes_owning_symbol_over_similar_siblings(self) -> None:
        commands = self.write(
            "commands/world.py",
            "class WorldCommands:\n"
            "    async def inventory_item_autocomplete(self):\n"
            "        return f'Inventory item x{2}'\n\n"
            "    async def loose_item_autocomplete(self):\n"
            "        return f'Great Axe ({1} here)'\n\n"
            "    async def room_item_display(self): pass\n"
            "    async def item_list_display(self): pass\n"
            "    async def item_choice_display(self): pass\n",
        )
        self.map({
            "commands/world.py": {"dependencies": [], "symbols": [
                {"name": "inventory_item_autocomplete"},
                {"name": "loose_item_autocomplete"},
                {"name": "room_item_display"},
                {"name": "item_list_display"},
                {"name": "item_choice_display"},
            ]},
        })

        result = resolve_task_surface_roots(
            self.repo,
            "The item suggestion currently says Great Axe (1 here). "
            "Stackable items should display as Health Potion x2.",
        )

        self.assertIn("loose_item_autocomplete", result.required_symbols[commands])

    def test_typescript_jsx_literal_promotes_owning_component(self) -> None:
        editor = self.write(
            "dashboard/app/dungeon-editor.tsx",
            "export function DungeonEditor() {\n"
            "  return <button type=\"button\">Item library</button>;\n"
            "}\n\n"
            "function ItemLibraryDialog() {\n"
            "  return <div>Catalog</div>;\n"
            "}\n",
        )
        self.map({
            "dashboard/app/dungeon-editor.tsx": {
                "dependencies": [],
                "symbols": [
                    {"name": "DungeonEditor", "kind": "function"},
                    {"name": "ItemLibraryDialog", "kind": "function"},
                ],
            },
        })

        result = resolve_task_surface_roots(
            self.repo,
            "Add a feature library button next to Item library.",
        )

        self.assertIn(editor, result.files)
        self.assertIn("DungeonEditor", result.required_symbols[editor])

    def test_unresolved_followup_uses_callback_binding_to_find_alternative(self) -> None:
        commands = self.write(
            "commands/items.py",
            "def attempted_suggestion(): return 'still unavailable'\n"
            "def alternative_suggestion(): return f'Visible ({1} nearby)'\n"
            "def incidental_suggestion(): return 'nearby error'\n\n"
            "@ui.autocomplete(item=attempted_suggestion)\n"
            "def store_item(): pass\n\n"
            "@ui.autocomplete(item=alternative_suggestion)\n"
            "def collect_item(): pass\n\n"
            "@ui.autocomplete(item=incidental_suggestion)\n"
            "def inspect_item(): pass\n",
        )
        self.map({"commands/items.py": {"dependencies": [], "symbols": [
            {"name": "attempted_suggestion"}, {"name": "alternative_suggestion"},
            {"name": "incidental_suggestion"},
        ]}})

        result = resolve_alternative_callback_roots(
            self.repo,
            "The visible item still says Visible (1 nearby).",
            {"attempted_suggestion"},
            limit=1,
        )

        self.assertEqual(result.required_symbols[commands], ["alternative_suggestion"])

    def test_explicit_command_follows_only_the_resolved_entry_point_body(self) -> None:
        command = self.write(
            "commands/world.py",
            "class WorldCommands:\n"
            "    @app.command(name='drop')\n"
            "    async def drop(self):\n        return self.drop_item()\n\n"
            "    async def unrelated(self):\n        return self.erase_everything()\n",
        )
        service = self.write(
            "services/world.py",
            "def drop_item(): pass\n\ndef erase_everything(): pass\n",
        )
        self.map({
            "commands/world.py": {"dependencies": [], "symbols": [
                {"name": "drop", "definition_name": "drop"},
                {"name": "unrelated"},
            ]},
            "services/world.py": {"dependencies": [], "symbols": [
                {"name": "drop_item"}, {"name": "erase_everything"},
            ]},
        })

        result = resolve_explicit_targets(self.repo, "fix /drop")

        self.assertIn("drop", result.required_symbols[command])
        self.assertIn("drop_item", result.required_symbols[service])
        self.assertNotIn("erase_everything", result.required_symbols[service])

    def test_generic_dependency_does_not_force_large_expansion(self) -> None:
        command = self.write("commands/payments.py")
        generic = self.write("utils/helpers.py")
        self.map({
            "commands/payments.py": {"dependencies": ["utils/helpers.py"], "symbols": []},
            "utils/helpers.py": {"dependencies": ["models/one.py", "models/two.py"], "symbols": []},
            "models/one.py": {"dependencies": [], "symbols": []},
            "models/two.py": {"dependencies": [], "symbols": []},
        })
        result = expand_candidates(self.repo, "fix payments", [command], limit=2)
        self.assertEqual(result.files, [command, generic])

    def test_complex_task_keeps_later_requirements_and_support_symbols(self) -> None:
        model = self.write("world/models.py", "def create_container_template(): pass\n")
        database = self.write(
            "storage/database.py",
            "def initialize(): pass\n"
            "def create_container_instance(): pass\n"
            "def load_container_contents(): pass\n",
        )
        api = self.write("api/container_api.py", "def update_container(): pass\n")
        ui = self.write(
            "dashboard/container-editor.tsx",
            "export function ContainerEditor() { return null }\n",
        )
        locks = self.write("world/locks.py", "def lockpick_container(): pass\n")
        service = self.write("world/service.py", "def place_container(): pass\n")
        self.map({
            "world/models.py": {
                "dependencies": [],
                "symbols": [{"name": "create_container_template"}],
            },
            "storage/database.py": {
                "dependencies": [],
                "symbols": [
                    {"name": "initialize"},
                    {"name": "create_container_instance"},
                    {"name": "load_container_contents"},
                ],
            },
            "api/container_api.py": {
                "dependencies": [],
                "symbols": [{"name": "update_container"}],
            },
            "dashboard/container-editor.tsx": {
                "dependencies": [],
                "symbols": [{"name": "ContainerEditor"}],
            },
            "world/locks.py": {
                "dependencies": [],
                "symbols": [{"name": "lockpick_container"}],
            },
            "world/service.py": {
                "dependencies": [],
                "symbols": [{"name": "place_container"}],
            },
        })
        task = (
            "Create reusable container templates. "
            "Save container instances in the database. "
            "Expose container editing through the API. "
            "Render a container editor in the dashboard. "
            "Support lockpicking for containers. "
            "Update world service placement."
        )

        result = resolve_task_surface_roots(self.repo, task)

        for expected in (model, database, api, ui, locks, service):
            self.assertIn(expected, result.files)
        self.assertIn("initialize", result.required_symbols[database])
        self.assertIn("create_container_instance", result.required_symbols[database])
        self.assertTrue(
            any(
                "Support lockpicking for containers" in diagnostic
                for diagnostic in result.diagnostics
            )
        )

    def test_completeness_accepts_only_known_unique_candidates(self) -> None:
        command = self.write("commands/run.py")
        repository = self.write("repositories/items.py")
        result = expand_candidates(self.repo, "change run", [command])
        result.candidates.append(repository)
        checker = QwenCompletenessChecker(model="test")
        completed = type("Run", (), {"returncode": 0, "stdout": '{"missing_files":["repositories/items.py","nope.py","repositories/items.py"]}'})()
        with patch.object(checker, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=completed):
            self.assertEqual(checker.missing_files(self.repo, "change run", result), [repository])

    def test_completeness_resolves_missing_symbol_outside_candidate_pool(self) -> None:
        command = self.write("commands/transfer.py")
        service = self.write("services/inventory_transfer.py")
        self.map({
            "commands/transfer.py": {"dependencies": [], "symbols": []},
            "services/inventory_transfer.py": {"dependencies": [], "symbols": [{"name": "InventoryTransferService"}]},
        })
        result = RetrievalResult([command], {command: ["Qwen seed"]}, [command])
        checker = QwenCompletenessChecker(model="test")
        completed = type("Run", (), {"returncode": 0, "stdout": '{"missing_files":[],"missing_symbols":["InventoryTransferService"],"missing_concepts":[]}'})()
        with patch.object(checker, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=completed) as run:
            outcome = checker.check(self.repo, "transfer inventory", result)
        self.assertEqual(outcome.files, [service])
        self.assertEqual(outcome.reasons[service], "Qwen completeness symbol resolution")
        self.assertEqual(run.call_count, 1)

    def test_hallucinated_symbol_adds_no_file_and_matches_are_bounded(self) -> None:
        command = self.write("commands/run.py")
        matches = [self.write(f"services/service_{number}.py") for number in range(5)]
        entries = {"commands/run.py": {"dependencies": [], "symbols": []}}
        entries.update({path.relative_to(self.repo).as_posix(): {"dependencies": [], "symbols": [{"name": "RequiredService"}]} for path in matches})
        self.map(entries)
        result = RetrievalResult([command], candidates=[command])
        checker = QwenCompletenessChecker(model="test")
        completed = type("Run", (), {"returncode": 0, "stdout": '{"missing_symbols":["RequiredService","ImaginaryService"]}'})()
        with patch.object(checker, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=completed):
            outcome = checker.check(self.repo, "run", result)
        self.assertEqual(outcome.files, sorted(matches, key=lambda path: str(path).lower())[:3])
        self.assertNotIn(self.repo / "imaginary.py", outcome.files)

    def test_explicit_commands_and_implementation_calls_resolve_deterministically(self) -> None:
        commands = self.write(
            "rpg_bot/commands/world.py",
            "@app_commands.command(name='drop')\n"
            "async def drop():\n    await world.drop_item()\n"
            "@app_commands.command(name='take')\n"
            "async def take():\n    await world.take_loose_item()\n",
        )
        service = self.write("rpg_bot/world_service.py", "class WorldService:\n    def drop_item(self): database.drop_character_inventory_item()\n    def take_loose_item(self): pass\n")
        database = self.write("rpg_bot/database.py", "class Database:\n    def drop_character_inventory_item(self): pass\n")
        self.map({
            "rpg_bot/commands/world.py": {"dependencies": [], "symbols": []},
            "rpg_bot/world_service.py": {"dependencies": [], "symbols": [{"name": "WorldService"}, {"name": "drop_item"}, {"name": "take_loose_item"}]},
            "rpg_bot/database.py": {"dependencies": [], "symbols": [{"name": "drop_character_inventory_item"}]},
        })
        result = resolve_explicit_targets(self.repo, "Make /drop and /take use WorldService.drop_item()")
        self.assertIn(commands, result.files)
        self.assertIn(service, result.files)
        self.assertIn(database, result.files)
        self.assertIn("explicit command /drop", result.reasons[commands])
        self.assertEqual(result.required_symbols[commands], ["drop", "take"])
        self.assertIn("drop_item", result.required_symbols[service])

    def test_common_words_are_not_explicit_targets_and_same_name_is_bounded(self) -> None:
        first = self.write("one.py")
        second = self.write("two.py")
        self.map({
            "one.py": {"dependencies": [], "symbols": [{"name": "drop_item"}]},
            "two.py": {"dependencies": [], "symbols": [{"name": "drop_item"}]},
        })
        self.assertEqual(resolve_explicit_targets(self.repo, "items should stack").files, [])
        result = resolve_explicit_targets(self.repo, "fix drop_item()")
        self.assertEqual(result.files, [first, second])

    def test_semantic_hints_use_repository_vocabulary_and_keep_symbols(self) -> None:
        service = self.write("world_service.py", "class WorldService:\n    def take_loose_item(self): pass\n")
        self.map({"world_service.py": {"dependencies": [], "symbols": [{"name": "take_loose_item"}]}})
        analyzer = QwenTaskHintAnalyzer(model="test")
        completed = type("Run", (), {"returncode": 0, "stdout": '{"symbol_hints":["take_loose_item","invented"]}'})()
        with patch.object(analyzer, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=completed) as run:
            hints = analyzer.hints(self.repo, "pick up part of a stack")
        self.assertEqual(hints, ["take_loose_item"])
        self.assertIn("take_loose_item", run.call_args.kwargs["input"])
        resolved = resolve_semantic_hints(self.repo, hints)
        self.assertEqual(resolved.files, [service])
        self.assertEqual(resolved.required_symbols[service], ["take_loose_item"])

    def test_task_hint_vocabulary_prioritizes_retrieved_file_symbols(self) -> None:
        command = self.write("commands/world.py")
        self.map({"commands/world.py": {"dependencies": [], "symbols": [{"name": "drop"}, {"name": "take"}]}})
        analyzer = QwenTaskHintAnalyzer(model="test")
        completed = type("Run", (), {"returncode": 0, "stdout": '{"symbol_hints":["take"]}'})()
        with patch.object(analyzer, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=completed) as run:
            self.assertEqual(analyzer.hints(self.repo, "pick up a stack", [command]), ["take"])
        self.assertIn('"take"', run.call_args.kwargs["input"])

    def test_task_hint_status_distinguishes_empty_failed_and_tolerant_json(self) -> None:
        symbol = self.write("world.py")
        self.map({"world.py": {"dependencies": [], "symbols": [{"name": "take"}]}})
        analyzer = QwenTaskHintAnalyzer(model="test")
        empty = type("Run", (), {"returncode": 0, "stdout": '{"symbol_hints":[]}'})()
        with patch.object(analyzer, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=empty):
            self.assertEqual(analyzer.hints(self.repo, "pick up", [symbol]), [])
        self.assertEqual(analyzer.last_status, "empty")
        malformed = type("Run", (), {"returncode": 0, "stdout": "not json"})()
        with patch.object(analyzer, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=malformed):
            self.assertEqual(analyzer.hints(self.repo, "pick up", [symbol]), [])
        self.assertEqual(analyzer.last_status, "failed")
        fenced = type("Run", (), {"returncode": 0, "stdout": '```json\n{"hints":["take"]}\n```'})()
        with patch.object(analyzer, "is_available", return_value=True), patch("chatcode.retrieval.hybrid_retriever.subprocess.run", return_value=fenced):
            self.assertEqual(analyzer.hints(self.repo, "pick up", [symbol]), ["take"])
        self.assertEqual(analyzer.last_status, "complete")

    def test_selected_test_call_closure_finds_bounded_project_implementation(self) -> None:
        test = self.write("tests/test_world.py", "def test_drop_stack():\n    self.cog.drop.callback()\n")
        command = self.write("commands/world.py", "def drop():\n    service.drop_item()\n")
        service = self.write("services/world.py", "def drop_item():\n    database.drop_record()\n")
        database = self.write("database.py", "def drop_record():\n    pass\n")
        self.map({
            "tests/test_world.py": {"dependencies": [], "symbols": []},
            "commands/world.py": {"dependencies": [], "symbols": [{"name": "drop"}]},
            "services/world.py": {"dependencies": [], "symbols": [{"name": "drop_item"}]},
            "database.py": {"dependencies": [], "symbols": [{"name": "drop_record"}]},
        })
        result = implementation_closure(self.repo, "dropped items should stack", [test])
        self.assertEqual(result.files, [command, service])
        self.assertEqual(result.required_symbols[command], ["drop"])
        self.assertEqual(result.required_symbols[service], ["drop_item"])

    def test_rooted_closure_follows_same_class_generic_persistence_chain(self) -> None:
        catalog = self.write(
            "catalog.py",
            "class ItemCatalog:\n"
            "    def get(self):\n"
            "        self._reload_if_changed()\n"
            "        return 'item'\n\n"
            "    def _reload_if_changed(self):\n"
            "        self.load()\n\n"
            "    def load(self):\n"
            "        return 'disk'\n",
        )
        self.map({"catalog.py": {"dependencies": [], "symbols": [
            {"name": "ItemCatalog"}, {"name": "get"},
            {"name": "_reload_if_changed"}, {"name": "load"},
        ]}})

        result = implementation_closure(
            self.repo,
            "stackable setting should survive restart",
            [catalog],
            root_symbols={catalog: ["get"]},
        )

        self.assertEqual(
            result.required_symbols[catalog],
            ["_reload_if_changed", "load"],
        )

    def test_rooted_closure_follows_generic_call_through_direct_dependency(self) -> None:
        api = self.write(
            "api.py",
            "class ItemAPI:\n"
            "    def update_item(self):\n"
            "        self.store.save()\n",
        )
        store = self.write(
            "store.py",
            "class ItemStore:\n"
            "    def save(self):\n"
            "        return True\n",
        )
        self.map({
            "api.py": {
                "dependencies": ["store.py"],
                "symbols": [{"name": "ItemAPI"}, {"name": "update_item"}],
            },
            "store.py": {
                "dependencies": [],
                "symbols": [{"name": "ItemStore"}, {"name": "save"}],
            },
        })

        result = implementation_closure(
            self.repo,
            "save edited item",
            [api],
            root_symbols={api: ["update_item"]},
        )

        self.assertEqual(result.required_symbols[store], ["save"])

    def test_rooted_closure_does_not_guess_ambiguous_generic_owner(self) -> None:
        api = self.write("api.py", "def update_item():\n    backend.save()\n")
        first = self.write("first_store.py", "def save():\n    return 1\n")
        second = self.write("second_store.py", "def save():\n    return 2\n")
        self.map({
            "api.py": {"dependencies": [], "symbols": [{"name": "update_item"}]},
            "first_store.py": {"dependencies": [], "symbols": [{"name": "save"}]},
            "second_store.py": {"dependencies": [], "symbols": [{"name": "save"}]},
        })

        result = implementation_closure(
            self.repo,
            "save edited item",
            [api],
            root_symbols={api: ["update_item"]},
        )

        self.assertNotIn(first, result.files)
        self.assertNotIn(second, result.files)

    def test_semantic_hint_promotes_only_the_exact_same_file_symbol(self) -> None:
        service = self.write("world_service.py", "def drop_item(): pass\ndef take_loose_item(): pass\ndef create_room(): pass\n")
        self.map({"world_service.py": {"dependencies": [], "symbols": [
            {"name": "drop_item"}, {"name": "take_loose_item"}, {"name": "create_room"},
        ]}})
        result = resolve_semantic_hints(self.repo, ["drop_item"])
        self.assertEqual(result.files, [service])
        self.assertEqual(result.required_symbols[service], ["drop_item"])
        self.assertEqual(result.reasons[service], ["semantic symbol hint drop_item"])

    def test_relevant_test_recovers_implementations_without_semantic_hints(self) -> None:
        test = self.write(
            "tests/test_world.py",
            "def test_takes_and_drops_loose_stacked_items_without_duplication():\n"
            "    self.cog.drop.callback()\n    self.cog.take.callback()\n    fixture.setup()\n",
        )
        command = self.write(
            "commands/world.py",
            "async def drop():\n    world.drop_item()\n\nasync def take():\n    world.take_loose_item()\n",
        )
        service = self.write(
            "world_service.py",
            "def drop_item():\n    database.place_item()\n\ndef take_loose_item():\n    database.place_catalog_item()\n",
        )
        database = self.write("database.py", "def place_item(): pass\ndef place_catalog_item(): pass\n")
        self.map({
            "tests/test_world.py": {"symbols": [], "dependencies": []},
            "commands/world.py": {"symbols": [{"name": "drop"}, {"name": "take"}], "dependencies": []},
            "world_service.py": {"symbols": [{"name": "drop_item"}, {"name": "take_loose_item"}], "dependencies": []},
            "database.py": {"symbols": [{"name": "place_item"}, {"name": "place_catalog_item"}], "dependencies": []},
        })
        task = "When you drop many of the same thing into a room, they should stack. when you pick up a stack, you should be asked how many you want to pick up of the items"
        result = test_callsite_closure(self.repo, task, [test])
        self.assertEqual(result.required_symbols[command], ["drop", "take"])
        self.assertEqual(result.required_symbols[service], ["drop_item", "take_loose_item"])
        self.assertEqual(result.required_symbols[database], ["place_item", "place_catalog_item"])
        self.assertNotIn("setup", {symbol for symbols in result.required_symbols.values() for symbol in symbols})

    def test_command_callback_alias_resolves_actual_handler_definition(self) -> None:
        test = self.write(
            "tests/test_commands.py",
            "def test_takes_and_drops_items():\n    self.cog.take.callback()\n    self.cog.drop.callback()\n",
        )
        commands = self.write(
            "commands/world.py",
            "async def take_item(): pass\nasync def drop(): pass\n",
        )
        self.map({
            "tests/test_commands.py": {"symbols": [], "dependencies": []},
            "commands/world.py": {"symbols": [
                {"name": "take", "kind": "command", "definition_name": "take_item"},
                {"name": "take_item", "kind": "function"},
                {"name": "drop", "kind": "function"},
            ], "dependencies": []},
        })
        result = test_callsite_closure(self.repo, "taking and dropping items", [test])
        self.assertEqual(result.required_symbols[commands], ["take_item", "drop"])

    def test_service_action_recovers_separately_named_command_callback_test(self) -> None:
        test = self.write(
            "tests/test_world_commands.py",
            "async def test_stacked_items_drop_without_duplication():\n"
            "    world.take_loose_item()\n"
            "    world.drop_item()\n\n"
            "async def test_take_names_character():\n"
            "    room = world.get_character_room()\n"
            "    world.place_item(room, 'potion')\n"
            "    await self.cog.take.callback()\n\n"
            "async def test_unrelated_lockpick():\n"
            "    room = world.get_room()\n"
            "    await self.cog.lockpick.callback()\n",
        )
        commands = self.write(
            "commands/world.py",
            "async def take(): pass\nasync def lockpick(): pass\n",
        )
        self.map({
            "tests/test_world_commands.py": {"symbols": [], "dependencies": []},
            "commands/world.py": {"symbols": [{"name": "take"}, {"name": "lockpick"}], "dependencies": []},
        })
        task = "When you drop many of the same thing into a room, they should stack. when you pick up a stack, you should be asked how many you want to pick up of the items"
        result = test_callsite_closure(self.repo, task, [test])
        self.assertIn("take", result.required_symbols[commands])
        self.assertTrue(any("test_take_names_character" in line for line in result.diagnostics))
        self.assertFalse(any("test_unrelated_lockpick" in line for line in result.diagnostics))
