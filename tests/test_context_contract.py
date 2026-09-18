from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.context_builder import (
    ContextBuildError,
    build_context,
    build_patch_context,
    build_patch_source_context,
)
from chatcode.indexing.project_graph import save_map
from chatcode.retrieval.context_contract import plan_context_contract
from chatcode.retrieval.source_coverage import plan_source_coverage
from chatcode.workspace import get_repo_workspace


BROAD_TASK = """\
Add dashboard API routes for reusable room containers.
Update DungeonEditor state and callbacks for creating and editing containers.
Render container contents and lock state in RoomInspector.
Persist container edits and keep room instances independent.
"""


ROOM_FEATURE_TASK = """\
Room Features backend already exists in WorldService and persistence.
Add Dashboard API routes for create read update and delete Room Features.
Update DungeonEditor and RoomInspector so Room Features can be listed edited and deleted.
Add an Add Room Feature dialog using the existing dashboard UI style.
Keep Room Features separate from inventory items and containers.
Add relevant tests for the Dashboard API flow.
"""


class ContextContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
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

    def fixture_files(self) -> tuple[Path, Path, Path]:
        api = self.write(
            "rpg_bot/dashboard_api.py",
            "class DashboardAPI:\n"
            "    def _validate_door_lock(self, data):\n"
            "        return data\n"
            "    def handle(self, method, path, payload):\n"
            "        if path.startswith('/containers'):\n"
            "            return self.world.update_container(payload)\n"
            "    def _node_data(self, room):\n"
            "        return {'containers': room.containers()}\n",
        )
        editor = self.write(
            "dashboard/app/dungeon-editor.tsx",
            "export function DungeonEditor() {\n"
            "  const [containers, setContainers] = useState([]);\n"
            "  return <RoomInspector containers={containers} />;\n"
            "}\n"
            "export function RoomInspector() {\n"
            "  return <section>container contents lock state</section>;\n"
            "}\n"
            "export function CharacterPortraitPanel() {\n"
            "  return <div>portrait</div>;\n"
            "}\n",
        )
        database = self.write(
            "rpg_bot/database.py",
            "class Database:\n"
            "    def initialize_schema(self):\n"
            "        self.db.execute('CREATE TABLE containers (id TEXT)')\n"
            "    def update_container(self, container_id, payload):\n"
            "        return self.db.execute('UPDATE containers SET id=?', (container_id,))\n",
        )
        save_map(self.repo, {"version": 3, "files": {
            "rpg_bot/dashboard_api.py": {
                "symbols": [
                    {"name": "DashboardAPI"}, {"name": "_validate_door_lock"},
                    {"name": "handle"}, {"name": "_node_data"},
                ],
                "dependencies": [],
            },
            "dashboard/app/dungeon-editor.tsx": {
                "symbols": [
                    {"name": "DungeonEditor"}, {"name": "RoomInspector"},
                    {"name": "CharacterPortraitPanel"},
                ],
                "dependencies": [],
            },
            "rpg_bot/database.py": {
                "symbols": [
                    {"name": "Database"}, {"name": "initialize_schema"},
                    {"name": "update_container"},
                ],
                "dependencies": [],
            },
        }})
        return api, editor, database

    def test_requirement_contract_promotes_transport_and_editor_owners(self) -> None:
        api, editor, database = self.fixture_files()
        files = [api, editor, database]
        existing = {api: ["_validate_door_lock"]}
        coverage = plan_source_coverage(self.repo, BROAD_TASK, files, existing)

        contract = plan_context_contract(
            self.repo,
            BROAD_TASK,
            files,
            existing,
            coverage,
        )

        self.assertTrue(contract.active)
        self.assertIn("handle", contract.symbols[api])
        self.assertIn("_node_data", contract.symbols[api])
        self.assertIn("DungeonEditor", contract.symbols[editor])
        self.assertIn("RoomInspector", contract.symbols[editor])
        self.assertNotIn("CharacterPortraitPanel", contract.symbols.get(editor, []))
        self.assertTrue(any(contract.requirement_symbols.values()))

    def test_noisy_retrieval_roots_are_support_not_publication_blockers(self) -> None:
        api = self.write(
            "rpg_bot/dashboard_api.py",
            "class DashboardAPI:\n"
            "    def handle(self, method, path, payload):\n"
            "        if path.startswith('/room-features'):\n"
            "            return self.world.create_room_feature(payload)\n"
            "    def _node_data(self, room):\n"
            "        return {'room_features': room.features}\n",
        )
        editor = self.write(
            "dashboard/app/dungeon-editor.tsx",
            "export function DungeonEditor() {\n"
            "  const [roomFeatures, setRoomFeatures] = useState([]);\n"
            "  return <RoomInspector roomFeatures={roomFeatures} />;\n"
            "}\n"
            "export function RoomInspector() {\n"
            "  return <section>Room Features</section>;\n"
            "}\n"
            "export function RoomFeatureDialog() {\n"
            "  return <div>Name Type Description Delete</div>;\n"
            "}\n",
        )
        service = self.write(
            "rpg_bot/world_service.py",
            "class WorldService:\n"
            "    def create_room_feature(self, payload):\n"
            "        return self.database.create_room_feature(payload)\n"
            "    def update_room_feature(self, feature_id, payload):\n"
            "        return self.database.update_room_feature(feature_id, payload)\n"
            "    def delete_room_feature(self, feature_id):\n"
            "        return self.database.delete_room_feature(feature_id)\n",
        )
        noisy_names = [
            "inventory",
            "read_button",
            "drop_button",
            "character",
            "_item_label",
            "_item_label_for_quantity",
            "_inventory_lines",
            "__init__",
        ]
        inventory = self.write(
            "rpg_bot/commands/inventory.py",
            "".join(
                f"def {name}():\n"
                f"    marker = {name!r}\n"
                f"    padding = {'x' * 3200!r}\n"
                "    return marker, padding\n\n"
                for name in noisy_names
            ),
        )
        files = [api, editor, service, inventory]
        save_map(self.repo, {"version": 3, "files": {
            "rpg_bot/dashboard_api.py": {
                "symbols": [
                    {"name": "DashboardAPI"}, {"name": "handle"},
                    {"name": "_node_data"},
                ],
                "dependencies": [],
            },
            "dashboard/app/dungeon-editor.tsx": {
                "symbols": [
                    {"name": "DungeonEditor"}, {"name": "RoomInspector"},
                    {"name": "RoomFeatureDialog"},
                ],
                "dependencies": [],
            },
            "rpg_bot/world_service.py": {
                "symbols": [
                    {"name": "WorldService"}, {"name": "create_room_feature"},
                    {"name": "update_room_feature"}, {"name": "delete_room_feature"},
                ],
                "dependencies": [],
            },
            "rpg_bot/commands/inventory.py": {
                "symbols": [{"name": name} for name in noisy_names],
                "dependencies": [],
            },
        }})
        retrieval_targets = {
            api: ["handle"],
            editor: ["RoomInspector"],
            inventory: noisy_names,
        }
        coverage = plan_source_coverage(
            self.repo, ROOM_FEATURE_TASK, files, retrieval_targets
        )
        contract = plan_context_contract(
            self.repo,
            ROOM_FEATURE_TASK,
            files,
            retrieval_targets,
            coverage,
        )

        self.assertIn("handle", contract.mandatory_symbols.get(api, []))
        self.assertIn("RoomInspector", contract.mandatory_symbols.get(editor, []))
        for symbol in noisy_names:
            self.assertNotIn(
                symbol, contract.mandatory_symbols.get(inventory, [])
            )
            self.assertIn(symbol, contract.support_symbols.get(inventory, []))

        with patch(
            "chatcode.context_builder.collect_relevant_files",
            return_value=(files, retrieval_targets),
        ), patch.dict(
            os.environ,
            {
                "CHATCODE_CONTEXT_BUDGET_CHARS": "8000",
                "CHATCODE_CONTEXT_HARD_BUDGET_CHARS": "18000",
            },
            clear=False,
        ):
            output = build_context(self.repo, ROOM_FEATURE_TASK)

        context = output.read_text(encoding="utf-8")
        self.assertNotIn("===== CONTEXT CONTRACT INCOMPLETE =====", context)
        self.assertNotIn(
            "REQUIRED SOURCE UNAVAILABLE: rpg_bot/commands/inventory.py",
            context,
        )
        self.assertIn("def handle", context)
        self.assertIn("function RoomInspector", context)

    def test_requirement_owner_expansion_materializes_patchable_surfaces(self) -> None:
        api = self.write(
            "rpg_bot/dashboard_api.py",
            "def _container_template_data(value):\n"
            "    return {'wrong_support_marker': value}\n\n"
            "def _node_data(room):\n"
            "    return {'OWNER_NODE_SERIALIZER': room}\n\n"
            "def _graph_data(graph):\n"
            "    return {'OWNER_GRAPH_SERIALIZER': graph}\n\n"
            "class DashboardAPI:\n"
            "    def handle(self, method, path, body=None):\n"
            "        marker = 'OWNER_HANDLE'\n"
            "        return self._handle(method, path, body or {})\n\n"
            "    def _handle(self, method, path, body):\n"
            "        marker = 'OWNER_ROUTER'\n"
            "        if path.startswith('/api/rooms/'):\n"
            "            return 200, body\n"
            "        return 404, {}\n",
        )
        editor = self.write(
            "dashboard/app/dungeon-editor.tsx",
            "export function DungeonEditor() {\n"
            "  const [graph, setGraph] = useState(null);\n"
            "  const ownerStateMarker = 'OWNER_DUNGEON_STATE';\n"
            "  const saveRoom = async () => setGraph(graph);\n"
            "  return <RoomInspector room={{ id: 'hall' }} onSave={saveRoom} />;\n"
            "}\n\n"
            "function RoomInspector({ room, onSave }: {\n"
            "  room: { id: string };\n"
            "  onSave: () => Promise<void>;\n"
            "}) {\n"
            "  const ownerContentsMarker = 'OWNER_ROOM_CONTENTS';\n"
            "  return <section>Room Contents {room.id}</section>;\n"
            "}\n\n"
            "function ItemLibraryDialog() {\n"
            "  return <div>Existing item library reference</div>;\n"
            "}\n",
        )
        api_tests = self.write(
            "tests/test_dashboard_api.py",
            "import unittest\n\n"
            "class DashboardAPITests(unittest.TestCase):\n"
            "    def setUp(self):\n"
            "        self.setup_marker = 'OWNER_API_SETUP'\n"
            "        self.api = object()\n\n"
            "    def test_room_create_route(self):\n"
            "        marker = 'OWNER_CREATE_ROUTE_TEST'\n"
            "        self.assertIn('room', marker.lower())\n\n"
            "    def test_room_delete_route(self):\n"
            "        marker = 'OWNER_DELETE_ROUTE_TEST'\n"
            "        self.assertIn('room', marker.lower())\n\n"
            "    def test_unrelated_portrait_route(self):\n"
            "        marker = 'portrait'\n"
            "        self.assertTrue(marker)\n",
        )
        files = [api, editor, api_tests]
        save_map(self.repo, {"version": 3, "files": {
            "rpg_bot/dashboard_api.py": {
                "symbols": [
                    {"name": "_container_template_data"},
                    {"name": "_node_data"},
                    {"name": "_graph_data"},
                    {"name": "DashboardAPI"},
                    {"name": "handle"},
                    {"name": "_handle"},
                ],
                "dependencies": [],
            },
            "dashboard/app/dungeon-editor.tsx": {
                "symbols": [
                    {"name": "DungeonEditor"},
                    {"name": "RoomInspector"},
                    {"name": "ItemLibraryDialog"},
                ],
                "dependencies": [],
            },
            "tests/test_dashboard_api.py": {
                "symbols": [
                    {"name": "DashboardAPITests"},
                    {"name": "setUp"},
                    {"name": "test_room_create_route"},
                    {"name": "test_room_delete_route"},
                    {"name": "test_unrelated_portrait_route"},
                ],
                "dependencies": [],
            },
        }})
        retrieval_targets = {
            api: ["_container_template_data"],
            editor: ["RoomInspector"],
        }

        coverage = plan_source_coverage(
            self.repo, ROOM_FEATURE_TASK, files, retrieval_targets
        )
        contract = plan_context_contract(
            self.repo,
            ROOM_FEATURE_TASK,
            files,
            retrieval_targets,
            coverage,
        )

        self.assertIn("handle", contract.mandatory_symbols.get(api, []))
        self.assertIn("_handle", contract.mandatory_symbols.get(api, []))
        self.assertIn("_node_data", contract.mandatory_symbols.get(api, []))
        self.assertIn("_graph_data", contract.mandatory_symbols.get(api, []))
        self.assertIn("DungeonEditor", contract.mandatory_symbols.get(editor, []))
        self.assertIn("RoomInspector", contract.mandatory_symbols.get(editor, []))
        self.assertIn("setUp", contract.mandatory_symbols.get(api_tests, []))
        self.assertTrue(
            any(
                name.startswith("test_room_")
                for name in contract.mandatory_symbols.get(api_tests, [])
            )
        )

        with patch(
            "chatcode.context_builder.collect_relevant_files",
            return_value=(files, retrieval_targets),
        ):
            output = build_context(self.repo, ROOM_FEATURE_TASK)

        context = output.read_text(encoding="utf-8")
        for marker in (
            "OWNER_HANDLE",
            "OWNER_ROUTER",
            "OWNER_NODE_SERIALIZER",
            "OWNER_GRAPH_SERIALIZER",
            "OWNER_DUNGEON_STATE",
            "OWNER_ROOM_CONTENTS",
            "OWNER_API_SETUP",
        ):
            self.assertIn(marker, context)
        self.assertNotIn(
            "tests/test_dashboard_api.py [source unavailable; do not patch]",
            context,
        )

    def test_priority_coverage_cannot_displace_mandatory_owner_bundle(self) -> None:
        api = self.write(
            "rpg_bot/dashboard_api.py",
            "def _container_template_data(value):\n"
            f"    padding = {'x' * 40000!r}\n"
            "    return {'inventory_template': value, 'padding': padding}\n\n"
            "def _node_data(room):\n"
            "    return {'OWNER_NODE_SERIALIZER': room}\n\n"
            "def _graph_data(graph):\n"
            "    return {'OWNER_GRAPH_SERIALIZER': graph}\n\n"
            "class DashboardAPI:\n"
            "    def handle(self, method, path, body=None):\n"
            "        marker = 'OWNER_HANDLE'\n"
            "        return self._handle(method, path, body or {})\n\n"
            "    def _handle(self, method, path, body):\n"
            "        marker = 'OWNER_ROUTER'\n"
            "        if path.startswith('/api/rooms/'):\n"
            "            return 200, body\n"
            "        return 404, {}\n",
        )
        editor = self.write(
            "dashboard/app/dungeon-editor.tsx",
            "type RoomNodeData = { id: string; name: string };\n"
            "type ContentKind = 'enemy' | 'item' | 'container';\n\n"
            "export function DungeonEditor() {\n"
            "  const ownerStateMarker = 'OWNER_DUNGEON_STATE';\n"
            "  const [selected, setSelected] = useState<RoomNodeData | null>(null);\n"
            "  const [kind, setKind] = useState<ContentKind | null>(null);\n"
            "  return <RoomInspector room={selected} />;\n"
            "}\n\n"
            "function RoomInspector({ room }: { room: RoomNodeData | null }) {\n"
            "  const ownerContentsMarker = 'OWNER_ROOM_CONTENTS';\n"
            "  return <section>{room?.name ?? 'Room Contents'}</section>;\n"
            "}\n",
        )
        api_tests = self.write(
            "tests/test_dashboard_api.py",
            "import unittest\n\n"
            "class DashboardAPITests(unittest.TestCase):\n"
            "    def setUp(self):\n"
            "        self.setup_marker = 'OWNER_API_SETUP'\n\n"
            "    def test_room_create_route(self):\n"
            "        marker = 'OWNER_CREATE_ROUTE_TEST'\n"
            "        self.assertIn('room', marker.lower())\n",
        )
        files = [api, editor, api_tests]
        save_map(self.repo, {"version": 3, "files": {
            "rpg_bot/dashboard_api.py": {
                "symbols": [
                    {"name": "_container_template_data"},
                    {"name": "_node_data"},
                    {"name": "_graph_data"},
                    {"name": "DashboardAPI"},
                    {"name": "handle"},
                    {"name": "_handle"},
                ],
                "dependencies": [],
            },
            "dashboard/app/dungeon-editor.tsx": {
                "symbols": [
                    {"name": "RoomNodeData"},
                    {"name": "ContentKind"},
                    {"name": "DungeonEditor"},
                    {"name": "RoomInspector"},
                ],
                "dependencies": [],
            },
            "tests/test_dashboard_api.py": {
                "symbols": [
                    {"name": "DashboardAPITests"},
                    {"name": "setUp"},
                    {"name": "test_room_create_route"},
                ],
                "dependencies": [],
            },
        }})
        retrieval_targets = {
            api: ["_container_template_data"],
            editor: ["RoomInspector"],
        }
        coverage = plan_source_coverage(
            self.repo, ROOM_FEATURE_TASK, files, retrieval_targets
        )
        coverage.symbols.setdefault(api, [])
        if "_container_template_data" not in coverage.symbols[api]:
            coverage.symbols[api].append("_container_template_data")

        contract = plan_context_contract(
            self.repo,
            ROOM_FEATURE_TASK,
            files,
            retrieval_targets,
            coverage,
        )
        self.assertNotIn(
            "_container_template_data",
            contract.mandatory_symbols.get(api, []),
        )
        self.assertIn(
            "_container_template_data",
            contract.priority_symbols.get(api, []),
        )
        self.assertIn("RoomNodeData", contract.mandatory_symbols.get(editor, []))
        self.assertIn("ContentKind", contract.mandatory_symbols.get(editor, []))

        with patch(
            "chatcode.context_builder.collect_relevant_files",
            return_value=(files, retrieval_targets),
        ), patch(
            "chatcode.context_builder.plan_source_coverage",
            return_value=coverage,
        ), patch.dict(
            os.environ,
            {
                "CHATCODE_CONTEXT_BUDGET_CHARS": "10000",
                "CHATCODE_CONTEXT_HARD_BUDGET_CHARS": "70000",
            },
            clear=False,
        ):
            output = build_context(self.repo, ROOM_FEATURE_TASK)

        context = output.read_text(encoding="utf-8")
        self.assertNotIn("===== CONTEXT CONTRACT INCOMPLETE =====", context)
        self.assertIn("OWNER_HANDLE", context)
        self.assertIn("OWNER_ROUTER", context)
        self.assertIn("OWNER_DUNGEON_STATE", context)
        self.assertIn("OWNER_ROOM_CONTENTS", context)
        self.assertIn("OWNER_API_SETUP", context)
        self.assertIn("type RoomNodeData", context)
        self.assertIn("type ContentKind", context)

    def test_contract_budget_expands_to_fit_large_required_handler(self) -> None:
        marker = "CURRENT_REQUIRED_HANDLER_"
        api = self.write(
            "rpg_bot/dashboard_api.py",
            "class DashboardAPI:\n"
            "    def handle(self, method, path, payload):\n"
            f"        data = '{marker + ('x' * 24000)}'\n"
            "        return data\n",
        )
        with patch.dict(
            os.environ,
            {
                "CHATCODE_CONTEXT_BUDGET_CHARS": "5000",
                "CHATCODE_CONTEXT_HARD_BUDGET_CHARS": "50000",
            },
            clear=False,
        ):
            context = build_patch_source_context(
                self.repo,
                BROAD_TASK,
                files=[api],
                target_symbols={api: ["handle"]},
            )

        self.assertIn(marker, context)
        self.assertNotIn("REQUIRED SOURCE UNAVAILABLE", context)
        self.assertNotIn("CONTEXT CONTRACT INCOMPLETE", context)

    def test_contract_keeps_more_than_six_required_symbols_in_one_file(self) -> None:
        source = self.write(
            "rpg_bot/dashboard_api.py",
            "".join(
                f"def required_route_{index}():\n    return {index}\n\n"
                for index in range(8)
            ),
        )
        required = {source: [f"required_route_{index}" for index in range(8)]}
        context = build_patch_source_context(
            self.repo,
            BROAD_TASK,
            files=[source],
            target_symbols=required,
        )
        for index in range(8):
            self.assertIn(f"def required_route_{index}", context)
        self.assertNotIn("REQUIRED SOURCE UNAVAILABLE", context)

    def test_publish_gate_removes_incomplete_upload_when_hard_cap_is_exceeded(self) -> None:
        api = self.write(
            "rpg_bot/dashboard_api.py",
            "class DashboardAPI:\n"
            "    def handle(self, method, path, payload):\n"
            f"        data = '{'x' * 30000}'\n"
            "        return data\n",
        )
        output = get_repo_workspace(self.repo) / "UPLOAD_TO_CHATGPT.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("stale old context\n", encoding="utf-8")

        with patch(
            "chatcode.context_builder.collect_relevant_files",
            return_value=([api], {api: ["handle"]}),
        ), patch.dict(
            os.environ,
            {
                "CHATCODE_CONTEXT_BUDGET_CHARS": "5000",
                "CHATCODE_CONTEXT_HARD_BUDGET_CHARS": "12000",
            },
            clear=False,
        ):
            with self.assertRaises(ContextBuildError):
                build_context(self.repo, BROAD_TASK)

        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
