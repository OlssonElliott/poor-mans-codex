from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatcode.indexing.project_graph import save_map
from chatcode.retrieval.hybrid_retriever import (
    RetrievalResult,
    QwenCompletenessChecker,
    expand_candidates,
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
        save_map(self.repo, {"version": 2, "files": entries})

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
