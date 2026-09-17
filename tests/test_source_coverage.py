from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.context_builder import build_patch_source_context
from chatcode.retrieval.source_coverage import plan_source_coverage


CONTAINER_TASK = """\
Implementera om container-systemet i DM dashboarden så att containers blir riktiga återanvändbara loot-objekt, ungefär på samma sätt som vårt befintliga item library fungerar.
Inspect item library and room placement.
Inspect doors and current lock mechanics.
Refactor door lock mechanics into general lock mechanics shared by doors and containers.
Create a reusable Container Library with id name type description default lock default hidden and discovery DC.
Room instances need independent runtime state and contents from the item library with quantities.
Keep loose room items and contained items distinct.
Dashboard Add container should pick a library definition or create one and place it in the current room.
Editor needs contents quantity lock hidden and description.
Migrate old world-object containers safely.
Add tests for template reuse independent instances contents quantity room links hidden locks door regression and edit delete independence.
"""


class SourceCoverageTests(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temporary.cleanup()

    def write(self, relative: str, content: str) -> Path:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return path

    def fixture_files(self) -> dict[str, Path]:
        return {
            "database": self.write(
                "rpg_bot/database.py",
                "class Database:\n"
                "    def ensure_character_location_knowledge(self):\n"
                "        return None\n"
                "    def initialize_schema(self):\n"
                "        self.db.execute(\"CREATE TABLE world_entities "
                "(id TEXT, kind TEXT, room_id TEXT)\")\n"
                "        self.db.execute(\"CREATE TABLE entity_inventory "
                "(entity_id TEXT, item_id TEXT, quantity INTEGER)\")\n"
                "    def create_world_entity(self, room_id, kind):\n"
                "        return self.db.execute(\"INSERT INTO world_entities VALUES (?, ?)\", "
                "(room_id, kind))\n"
                "    def load_world_entity(self, entity_id):\n"
                "        return self.db.execute(\"SELECT * FROM world_entities WHERE id=?\", "
                "(entity_id,))\n"
                "    def update_world_entity_lock(self, entity_id, locked):\n"
                "        return self.db.execute(\"UPDATE world_entities SET locked=? WHERE id=?\", "
                "(locked, entity_id))\n"
                "    def delete_world_entity(self, entity_id):\n"
                "        return self.db.execute(\"DELETE FROM world_entities WHERE id=?\", "
                "(entity_id,))\n",
            ),
            "api": self.write(
                "rpg_bot/dashboard_api.py",
                "class DashboardAPI:\n"
                "    def _validate_door_lock(self, data):\n"
                "        return data\n"
                "    def handle(self, method, path, payload):\n"
                "        if path.startswith('/world/entities'):\n"
                "            return self.world_service.update_world_entity(payload)\n"
                "        if path.startswith('/items'):\n"
                "            return self.world_service.item_library(payload)\n"
                "    def _node_data(self, room):\n"
                "        return {'room': room.id, 'entities': room.entities, "
                "'doors': room.doors}\n",
            ),
            "world": self.write(
                "rpg_bot/world.py",
                "from enum import Enum\n"
                "class EntityKind(Enum):\n"
                "    CONTAINER = 'container'\n"
                "    CORPSE = 'corpse'\n"
                "class WorldEntity:\n"
                "    def __init__(self, entity_id, kind, room_id, hidden=False, lock=None):\n"
                "        self.id = entity_id\n"
                "        self.kind = kind\n"
                "        self.room_id = room_id\n"
                "        self.hidden = hidden\n"
                "        self.lock = lock\n"
                "class InventoryHolder:\n"
                "    def inventory(self):\n"
                "        return []\n"
                "class Room:\n"
                "    def containers(self):\n"
                "        return []\n",
            ),
            "service": self.write(
                "rpg_bot/world_service.py",
                "class WorldService:\n"
                "    def item_library(self):\n"
                "        return self.item_catalog.templates()\n"
                "    def create_item_template(self):\n"
                "        return self.item_catalog.create()\n"
                "    def create_world_entity(self, room_id, kind):\n"
                "        return self.database.create_world_entity(room_id, kind)\n"
                "    def update_world_entity(self, entity_id, changes):\n"
                "        return self.database.update_world_entity_lock("
                "entity_id, changes.get('locked'))\n"
                "    def place_entity_in_room(self, room_id, entity_id):\n"
                "        return self.database.move_world_entity(entity_id, room_id)\n",
            ),
            "commands": self.write(
                "rpg_bot/commands/world.py",
                "class WorldCommands:\n"
                "    async def lockpick(self, interaction, target):\n"
                "        lock = self.world_service.get_lock(target)\n"
                "        return self.world_service.attempt_lockpick(lock)\n"
                "    async def character_portrait(self):\n"
                "        return None\n",
            ),
            "editor": self.write(
                "dashboard/app/dungeon-editor.tsx",
                "export function ItemLibraryDialog() {\n"
                "  return <div>Item library template editor</div>;\n"
                "}\n"
                "export function RoomInspector() {\n"
                "  return <section>Room contents items doors hidden discovery</section>;\n"
                "}\n"
                "export function DoorEditor() {\n"
                "  return <div>Door lock editor</div>;\n"
                "}\n"
                "export const AddRoomObjectDialog: React.FC = () => {\n"
                "  return <div>Add item to room contents</div>;\n"
                "};\n",
            ),
            "chart": self.write(
                "dashboard/components/chart.tsx",
                "export function ChartContainer() { return <div>container</div>; }\n",
            ),
            "portraits": self.write(
                "rpg_bot/portraits.py",
                "def create_character_portrait():\n"
                "    return 'portrait'\n",
            ),
        }

    def test_complex_task_promotes_architectural_source_not_single_noun_false_positives(self) -> None:
        files = self.fixture_files()
        selected = list(files.values())
        plan = plan_source_coverage(
            self.repo,
            CONTAINER_TASK,
            selected,
            {
                files["database"]: ["ensure_character_location_knowledge"],
                files["api"]: ["_validate_door_lock"],
            },
        )

        self.assertIn("initialize_schema", plan.symbols[files["database"]])
        self.assertIn("create_world_entity", plan.symbols[files["database"]])
        self.assertIn("handle", plan.symbols[files["api"]])
        self.assertIn("_node_data", plan.symbols[files["api"]])
        self.assertIn("WorldEntity", plan.symbols[files["world"]])
        self.assertIn("EntityKind", plan.symbols[files["world"]])
        self.assertIn("InventoryHolder", plan.symbols[files["world"]])
        self.assertIn("item_library", plan.symbols[files["service"]])
        self.assertIn("create_world_entity", plan.symbols[files["service"]])
        self.assertIn("lockpick", plan.symbols[files["commands"]])
        self.assertIn("ItemLibraryDialog", plan.symbols[files["editor"]])
        self.assertIn("RoomInspector", plan.symbols[files["editor"]])
        self.assertNotIn(files["chart"], plan.symbols)
        self.assertNotIn(files["portraits"], plan.symbols)

    def test_small_task_does_not_run_broad_source_planning(self) -> None:
        source = self.write("service.py", "def save_item():\n    return True\n")
        plan = plan_source_coverage(
            self.repo,
            "fix save_item()",
            [source],
            {source: ["save_item"]},
        )
        self.assertEqual(plan.symbols, {})

    def test_patch_context_materializes_coverage_before_unrelated_selected_source(self) -> None:
        files = self.fixture_files()
        padding = "".join(
            f"export const chartNoise{index} = {index};\n"
            for index in range(1800)
        )
        files["chart"].write_text(
            padding + files["chart"].read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )

        with patch(
            "chatcode.context_builder.get_changed_files", return_value=set()
        ), patch.dict(
            os.environ,
            {"CHATCODE_CONTEXT_BUDGET_CHARS": "26000"},
            clear=False,
        ):
            context = build_patch_source_context(
                self.repo,
                CONTAINER_TASK,
                files=list(files.values()),
                target_symbols={
                    files["database"]: ["ensure_character_location_knowledge"],
                    files["api"]: ["_validate_door_lock"],
                },
            )

        self.assertIn("CREATE TABLE world_entities", context)
        self.assertIn("def create_world_entity", context)
        self.assertIn("def handle", context)
        self.assertIn("def _node_data", context)
        self.assertIn("class WorldEntity", context)
        self.assertIn("class EntityKind", context)
        self.assertIn("def lockpick", context)
        self.assertIn("function ItemLibraryDialog", context)
        self.assertIn("function RoomInspector", context)
        self.assertNotIn("chartNoise900", context)


if __name__ == "__main__":
    unittest.main()
