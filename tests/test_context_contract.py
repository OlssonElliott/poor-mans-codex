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
