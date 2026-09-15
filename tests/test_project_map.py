from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chatcode.indexing.index_manager import (
    IndexProgress,
    SemanticIndexInterrupted,
    update_project_map,
)
from chatcode.indexing.project_graph import load_map, map_path, save_map
from chatcode.indexing.semantic_analyzer import (
    QwenSemanticAnalyzer,
    SemanticAnalysis,
    validate_semantic_response,
)
from chatcode.patch import apply_patch
from chatcode.retrieval.graph_retriever import retrieve_files


class NoSemantics:
    def analyze(self, path: Path, repo: Path, static_result: dict) -> list[dict]:
        return []


class FakeSemanticAnalyzer:
    def __init__(
        self,
        model: str = "test-qwen",
        interrupt_at: int | None = None,
        analyzer_version: int = 2,
    ) -> None:
        self.model = model
        self.interrupt_at = interrupt_at
        self.analyzer_version = analyzer_version
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return True

    def is_eligible(self, path: Path, static_result: dict) -> bool:
        return bool(static_result.get("symbols"))

    def analyze(self, path: Path, repo: Path, static_result: dict) -> SemanticAnalysis:
        relative = path.relative_to(repo).as_posix()
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        self.calls.append(relative)
        return SemanticAnalysis(
            "complete",
            summary=f"Handles the {path.stem} feature.",
            tags=(path.stem, "feature"),
            important_symbols=(static_result["symbols"][0]["name"],),
        )


class ProjectMapTests(unittest.TestCase):
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

    def test_first_index_builds_compact_file_metadata(self) -> None:
        self.write("src/helper.py", "def helper():\n    return 1\n")
        self.write("src/app.py", "from src.helper import helper\n\ndef run():\n    return helper()\n")

        update = update_project_map(self.repo, semantic_analyzer=NoSemantics())
        graph = load_map(self.repo)

        self.assertEqual(set(update.added), {"src/app.py", "src/helper.py"})
        self.assertTrue(map_path(self.repo).is_file())
        app = graph["files"]["src/app.py"]
        self.assertEqual(app["path"], "src/app.py")
        self.assertEqual(app["language"], "python")
        self.assertIn("run", [symbol["name"] for symbol in app["symbols"]])
        self.assertIn("src/helper.py", app["dependencies"])
        self.assertTrue(app["summary"])
        self.assertTrue(app["tags"])
        self.assertNotIn("relations", graph)
        self.assertNotIn("symbols", graph)
        self.assertNotIn("semantic_relations", app)
        self.assertFalse(map_path(self.repo).is_relative_to(self.repo))

    def test_low_level_ast_operations_are_not_persisted(self) -> None:
        calls = "\n".join(f"    helper_{number}()" for number in range(200))
        self.write(
            "service.py",
            "class Service:\n"
            "    def run(self):\n"
            "        self.state = self.value\n"
            f"{calls}\n",
        )

        update_project_map(self.repo, semantic_analyzer=NoSemantics())
        raw_index = map_path(self.repo).read_text(encoding="utf-8")

        self.assertNotIn("FILE_CONTAINS_SYMBOL", raw_index)
        self.assertNotIn('"CALLS"', raw_index)
        self.assertNotIn("READS_STATE", raw_index)
        self.assertNotIn("WRITES_STATE", raw_index)
        symbols = load_map(self.repo)["files"]["service.py"]["symbols"]
        self.assertEqual([symbol["qualified_name"] for symbol in symbols], ["Service", "Service.run"])

    def test_qwen_summary_tags_and_symbols_are_hard_bounded(self) -> None:
        payload = {
            "summary": "x" * 1_000,
            "tags": [f"tag-{number}-" + "y" * 100 for number in range(30)],
            "important_symbols": [f"symbol_{number}_" + "z" * 150 for number in range(40)],
        }

        metadata = validate_semantic_response(json.dumps(payload))

        self.assertEqual(len(metadata["summary"]), 600)
        self.assertEqual(len(metadata["tags"]), 10)
        self.assertTrue(all(len(tag) <= 80 for tag in metadata["tags"]))
        self.assertEqual(len(metadata["important_symbols"]), 20)
        self.assertTrue(all(len(name) <= 120 for name in metadata["important_symbols"]))

    def test_retrieval_ranks_summary_tags_and_symbols(self) -> None:
        self.write("inventory.py", "def accept_transfer():\n    pass\n")
        self.write("unrelated.py", "def heartbeat():\n    pass\n")
        update_project_map(self.repo, semantic_analyzer=NoSemantics())

        selected = retrieve_files(self.repo, "player transfer confirmation")

        self.assertEqual(selected[0].name, "inventory.py")

    def test_ai_and_static_retrieval_use_separate_metadata(self) -> None:
        self.write("app.py", "def run():\n    pass\n")
        update_project_map(self.repo, semantic_analyzer=NoSemantics())
        graph = load_map(self.repo)
        entry = graph["files"]["app.py"]
        entry["summary"] = "Controls the obsidian dragon ritual."
        entry["tags"] = ["obsidian-dragon"]
        entry["important_symbols"] = ["summon_dragon"]
        entry["semantic"] = {
            "status": "complete",
            "model": "qwen",
            "analyzed_hash": entry["hash"],
            "analyzer_version": 3,
        }
        save_map(self.repo, graph)

        ai_files = retrieve_files(self.repo, "obsidian dragon", index_mode="ai")
        static_files = retrieve_files(
            self.repo, "obsidian dragon", index_mode="static"
        )

        self.assertEqual([path.name for path in ai_files], ["app.py"])
        self.assertEqual(static_files, [])

    def test_legacy_target_repo_graph_is_ignored_and_rebuilt_in_workspace(self) -> None:
        legacy = self.write(
            ".chatcode/project-map.json",
            json.dumps({"version": 1, "files": {"runtime.py": {}}, "relations": [1, 2, 3]}),
        )
        self.write("src/app.py", "def app():\n    pass\n")

        update_project_map(self.repo, semantic_analyzer=NoSemantics())

        self.assertEqual(set(load_map(self.repo)["files"]), {"src/app.py"})
        self.assertTrue(legacy.is_file())
        self.assertTrue(map_path(self.repo).is_relative_to(self.workspace))

    def test_representative_index_size_scales_with_file_metadata(self) -> None:
        for number in range(150):
            self.write(
                f"src/module_{number}.py",
                f"def feature_{number}():\n    return {number}\n",
            )

        update_project_map(self.repo, semantic_analyzer=NoSemantics())
        index_file = map_path(self.repo)

        self.assertEqual(len(load_map(self.repo)["files"]), 150)
        self.assertLess(index_file.stat().st_size, 150_000)

    def test_runtime_dependencies_caches_and_build_outputs_are_excluded(self) -> None:
        excluded = (
            ".runtime/python312/tools/Lib/random.py",
            ".venv/Lib/site-packages/package.py",
            "venv/lib/python/package.py",
            "node_modules/tool/index.js",
            ".git/hooks/helper.py",
            ".chatcode/internal.py",
            "__pycache__/cached.py",
            "dist/generated.py",
            "build/generated.py",
        )
        for relative in excluded:
            self.write(relative, "def dependency():\n    pass\n")
        self.write("src/app.py", "def app():\n    return 1\n")
        self.write("bot/commands/ping.py", "def ping():\n    return 'pong'\n")

        semantic = Mock()
        semantic.analyze.return_value = []
        update_project_map(self.repo, semantic_analyzer=semantic)
        graph = load_map(self.repo)

        self.assertEqual(set(graph["files"]), {"bot/commands/ping.py", "src/app.py"})
        analyzed = {
            call.args[0].relative_to(self.repo).as_posix()
            for call in semantic.analyze.call_args_list
        }
        self.assertEqual(analyzed, {"bot/commands/ping.py", "src/app.py"})
        self.assertTrue(all(not path.startswith(".runtime/") for path in graph["files"]))

    def test_nested_excluded_directory_is_pruned(self) -> None:
        self.write("packages/game/.runtime/python/Lib/runtime.py", "def runtime():\n    pass\n")
        self.write("packages/game/src/game.py", "def game():\n    pass\n")

        update_project_map(self.repo, semantic_analyzer=NoSemantics())

        self.assertEqual(set(load_map(self.repo)["files"]), {"packages/game/src/game.py"})

    def test_previously_indexed_excluded_entry_is_removed_on_incremental_update(self) -> None:
        runtime_path = ".runtime/python312/tools/Lib/random.py"
        source = self.write(runtime_path, "def random():\n    pass\n")
        save_map(self.repo, {
            "version": 2,
            "files": {
                runtime_path: {
                    "hash": "old",
                    "language": "python",
                    "imports": [],
                    "symbols": [{
                        "name": "random",
                        "qualified_name": "random",
                        "kind": "function",
                    }],
                    "summary": "old runtime",
                    "tags": ["runtime"],
                    "dependencies": [],
                }
            },
        })
        self.write("src/app.py", "def app():\n    pass\n")

        semantic = Mock()
        semantic.analyze.return_value = []
        result = update_project_map(
            self.repo,
            paths=["src/app.py"],
            semantic_analyzer=semantic,
        )

        self.assertIn(runtime_path, result.deleted)
        self.assertNotIn(runtime_path, load_map(self.repo)["files"])
        self.assertTrue(source.is_file())
        self.assertEqual(semantic.analyze.call_count, 1)

    def test_changed_new_and_deleted_files_are_updated_incrementally(self) -> None:
        changed = self.write("changed.py", "def old():\n    pass\n")
        deleted = self.write("deleted.py", "def gone():\n    pass\n")
        stable = self.write("stable.py", "def stable():\n    pass\n")
        update_project_map(self.repo, semantic_analyzer=NoSemantics())

        changed.write_text("def new():\n    pass\n", encoding="utf-8")
        deleted.unlink()
        self.write("added.py", "def added():\n    pass\n")
        result = update_project_map(self.repo, semantic_analyzer=NoSemantics())
        graph = load_map(self.repo)

        self.assertEqual(result.added, ("added.py",))
        self.assertEqual(result.changed, ("changed.py",))
        self.assertEqual(result.deleted, ("deleted.py",))
        self.assertIn("stable.py", result.unchanged)
        names = [symbol["name"] for symbol in graph["files"]["changed.py"]["symbols"]]
        self.assertNotIn("old", names)
        self.assertIn("new", names)
        self.assertNotIn("deleted.py", graph["files"])
        self.assertEqual(stable.read_text(encoding="utf-8"), "def stable():\n    pass\n")

    def test_unchanged_files_are_not_reanalyzed(self) -> None:
        self.write("one.py", "def one():\n    return 1\n")
        update_project_map(self.repo, semantic_analyzer=NoSemantics())

        with patch("chatcode.indexing.index_manager.analyze_file") as analyze:
            result = update_project_map(self.repo, semantic_analyzer=NoSemantics())

        analyze.assert_not_called()
        self.assertEqual(result.unchanged, ("one.py",))

    def test_rename_is_detected_by_matching_hash(self) -> None:
        original = self.write("old_name.py", "def feature():\n    return 1\n")
        update_project_map(self.repo, semantic_analyzer=NoSemantics())
        renamed = self.repo / "new_name.py"
        original.rename(renamed)

        result = update_project_map(self.repo, semantic_analyzer=NoSemantics())
        graph = load_map(self.repo)

        self.assertEqual(result.renamed, (("old_name.py", "new_name.py"),))
        self.assertEqual(graph["files"]["new_name.py"]["renamed_from"], "old_name.py")

    def test_manual_edit_replaces_only_changed_file_metadata(self) -> None:
        source = self.write("app.py", "def first():\n    return target()\n")
        self.write("target.py", "def target():\n    return 1\n")
        update_project_map(self.repo, semantic_analyzer=NoSemantics())
        source.write_text("def second():\n    return 2\n", encoding="utf-8")

        result = update_project_map(self.repo, semantic_analyzer=NoSemantics())
        graph = load_map(self.repo)

        self.assertEqual(result.changed, ("app.py",))
        names = [symbol["name"] for symbol in graph["files"]["app.py"]["symbols"]]
        self.assertNotIn("first", names)
        self.assertIn("second", names)
        self.assertNotIn("relations", graph)

    def test_apply_updates_only_patch_paths(self) -> None:
        # apply_patch requires a Git repository and history, but the index itself
        # is exercised through the real post-apply hook.
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=self.repo, check=True)
        source = self.write("app.py", "value = 1\n")
        subprocess.run(["git", "add", "app.py"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True)
        incoming = self.repo / "change.diff"
        incoming.write_text("--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n", encoding="utf-8")

        apply_patch(self.repo, incoming)

        self.assertEqual(source.read_text(encoding="utf-8"), "value = 2\n")
        self.assertIn("app.py", load_map(self.repo)["files"])

    def test_apply_invalidates_only_changed_source_semantics(self) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=self.repo, check=True)
        self.write("app.py", "def app():\n    return 1\n")
        self.write("stable.py", "def stable():\n    return 2\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True)
        update_project_map(self.repo, semantic_analyzer=FakeSemanticAnalyzer())
        stable_before = load_map(self.repo)["files"]["stable.py"]
        incoming = self.root / "source-change.diff"
        incoming.write_text(
            "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def app():\n-    return 1\n+    return 9\n",
            encoding="utf-8",
        )

        apply_patch(self.repo, incoming)

        after_apply = load_map(self.repo)
        self.assertEqual(after_apply["files"]["app.py"]["semantic"]["status"], "pending")
        self.assertEqual(after_apply["files"]["stable.py"], stable_before)
        analyzer = FakeSemanticAnalyzer()
        result = update_project_map(self.repo, semantic_analyzer=analyzer)
        self.assertEqual(analyzer.calls, ["app.py"])
        self.assertEqual(result.semantic_reused, 1)
        self.assertEqual(result.semantic_pending, 1)

    def test_apply_to_non_source_file_preserves_all_semantic_cache(self) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=self.repo, check=True)
        self.write("app.py", "def app():\n    return 1\n")
        self.write("README.md", "before\n")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True)
        update_project_map(self.repo, semantic_analyzer=FakeSemanticAnalyzer())
        cached = load_map(self.repo)["files"]["app.py"]
        incoming = self.root / "docs-change.diff"
        incoming.write_text(
            "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-before\n+after\n",
            encoding="utf-8",
        )

        apply_patch(self.repo, incoming)

        self.assertEqual(load_map(self.repo)["files"]["app.py"], cached)
        analyzer = FakeSemanticAnalyzer()
        result = update_project_map(self.repo, semantic_analyzer=analyzer)
        self.assertEqual(analyzer.calls, [])
        self.assertEqual(result.semantic_reused, 1)

    def test_file_retrieval_expands_direct_source_dependency(self) -> None:
        self.write(
            "doors.py",
            "from movement import collision_check\n\ndef open_door():\n    return collision_check()\n",
        )
        self.write("movement.py", "def collision_check():\n    return True\n")
        update_project_map(self.repo, semantic_analyzer=NoSemantics())

        paths = retrieve_files(self.repo, "door reopens", depth=2)
        relative = [path.relative_to(self.repo).as_posix() for path in paths]

        self.assertIn("doors.py", relative)
        self.assertIn("movement.py", relative)

    def test_invalid_and_low_confidence_qwen_output_is_ignored(self) -> None:
        self.assertEqual(validate_semantic_response("not json"), {})
        self.assertEqual(validate_semantic_response(json.dumps({"relations": []})), {})

    def test_unavailable_qwen_is_not_an_error(self) -> None:
        source = self.write("app.py", "def app():\n    pass\n")
        analyzer = QwenSemanticAnalyzer(model="qwen-test")
        analyzer.enabled = True
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value=None):
            result = analyzer.analyze(source, self.repo, {"symbols": [{"name": "app"}]})
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.summary, "")

    def test_disabled_qwen_does_not_invoke_ollama(self) -> None:
        source = self.write("app.py", "def app():\n    pass\n")
        analyzer = QwenSemanticAnalyzer(model="qwen-test")
        analyzer.enabled = False
        with patch("chatcode.indexing.semantic_analyzer.shutil.which") as available:
            result = analyzer.analyze(source, self.repo, {"symbols": [{"name": "app"}]})
        self.assertEqual(result.status, "failed")
        available.assert_not_called()

    def test_semantic_analyzer_exception_falls_back_to_static_graph(self) -> None:
        source = self.write("app.py", "def app():\n    pass\n")
        broken = Mock()
        broken.analyze.side_effect = TimeoutError("model timed out")

        update_project_map(self.repo, semantic_analyzer=broken)

        self.assertIn("app", [
            symbol["name"] for symbol in load_map(self.repo)["files"]["app.py"]["symbols"]
        ])
        broken.analyze.assert_called_once()

    def test_static_map_is_saved_before_semantic_analysis_starts(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")

        class InspectingAnalyzer(FakeSemanticAnalyzer):
            def analyze(inner_self, path: Path, repo: Path, static_result: dict) -> SemanticAnalysis:
                saved = load_map(repo)
                self.assertIn("app.py", saved["files"])
                self.assertEqual(saved["files"]["app.py"]["semantic"]["status"], "pending")
                raise KeyboardInterrupt

        with self.assertRaises(SemanticIndexInterrupted):
            update_project_map(self.repo, semantic_analyzer=InspectingAnalyzer())

        self.assertIn("app.py", load_map(self.repo)["files"])

    def test_interrupted_semantic_index_resumes_without_repeating_completed_files(self) -> None:
        for name in ("a", "b", "c"):
            self.write(f"{name}.py", f"def {name}():\n    return 1\n")

        first = FakeSemanticAnalyzer(interrupt_at=1)
        with self.assertRaises(SemanticIndexInterrupted) as interrupted:
            update_project_map(self.repo, semantic_analyzer=first)

        self.assertEqual(interrupted.exception.completed, 1)
        checkpoint = load_map(self.repo)
        self.assertEqual(checkpoint["files"]["a.py"]["semantic"]["status"], "complete")
        self.assertEqual(checkpoint["files"]["b.py"]["semantic"]["status"], "pending")
        self.assertEqual(checkpoint["files"]["a.py"]["summary"], "Handles the a feature.")

        resumed = FakeSemanticAnalyzer()
        result = update_project_map(self.repo, semantic_analyzer=resumed)

        self.assertEqual(resumed.calls, ["b.py", "c.py"])
        self.assertEqual(result.semantic_completed, 3)
        self.assertTrue(all(
            entry["semantic"]["status"] == "complete"
            for entry in load_map(self.repo)["files"].values()
        ))

    def test_changed_file_invalidates_only_its_semantic_cache(self) -> None:
        first_path = self.write("first.py", "def first():\n    return 1\n")
        self.write("second.py", "def second():\n    return 2\n")
        update_project_map(self.repo, semantic_analyzer=FakeSemanticAnalyzer())
        first_path.write_text("def first():\n    return 99\n", encoding="utf-8")

        analyzer = FakeSemanticAnalyzer()
        result = update_project_map(self.repo, semantic_analyzer=analyzer)

        self.assertEqual(analyzer.calls, ["first.py"])
        self.assertEqual(result.semantic_reused, 1)
        self.assertEqual(result.semantic_invalidated, 1)
        self.assertEqual(result.semantic_pending, 1)
        self.assertEqual(load_map(self.repo)["files"]["second.py"]["semantic"]["status"], "complete")

    def test_unchanged_semantic_files_are_not_reprocessed(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")
        update_project_map(self.repo, semantic_analyzer=FakeSemanticAnalyzer())
        analyzer = FakeSemanticAnalyzer()

        result = update_project_map(self.repo, semantic_analyzer=analyzer)

        self.assertEqual(analyzer.calls, [])
        self.assertEqual(result.semantic_completed, 1)
        self.assertEqual(result.semantic_reused, 1)
        self.assertEqual(result.semantic_invalidated, 0)
        self.assertEqual(result.semantic_pending, 0)

    def test_three_changed_files_reuse_remaining_semantic_cache(self) -> None:
        paths = [
            self.write(f"module_{number}.py", f"def feature_{number}():\n    return {number}\n")
            for number in range(117)
        ]
        update_project_map(self.repo, semantic_analyzer=FakeSemanticAnalyzer())
        for number in (2, 55, 116):
            paths[number].write_text(
                f"def feature_{number}():\n    return 'changed'\n",
                encoding="utf-8",
            )

        analyzer = FakeSemanticAnalyzer()
        result = update_project_map(self.repo, semantic_analyzer=analyzer)

        self.assertEqual(
            analyzer.calls,
            ["module_116.py", "module_2.py", "module_55.py"],
        )
        self.assertEqual(result.semantic_reused, 114)
        self.assertEqual(result.semantic_invalidated, 3)
        self.assertEqual(result.semantic_pending, 3)

    def test_unchanged_static_refresh_preserves_semantic_metadata_exactly(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")
        update_project_map(self.repo, semantic_analyzer=FakeSemanticAnalyzer())
        before = load_map(self.repo)["files"]["app.py"]
        preserved = {
            key: before[key]
            for key in ("summary", "tags", "important_symbols", "semantic")
        }

        analyzer = FakeSemanticAnalyzer()
        update_project_map(
            self.repo,
            paths=["app.py"],
            semantic_analyzer=analyzer,
        )
        after = load_map(self.repo)["files"]["app.py"]

        self.assertEqual(analyzer.calls, [])
        self.assertEqual(
            {key: after[key] for key in preserved},
            preserved,
        )

    def test_semantic_timeout_keeps_static_graph_and_continues(self) -> None:
        self.write("a.py", "def a():\n    return 1\n")
        self.write("b.py", "def b():\n    return 2\n")

        analyzer = QwenSemanticAnalyzer(model="qwen-test")
        analyzer.enabled = True
        successful = Mock(
            returncode=0,
            stdout='{"summary":"Handles B.","tags":["b"],"important_symbols":["b"]}',
            stderr="",
        )
        with patch.object(
            QwenSemanticAnalyzer, "preflight", return_value=SemanticAnalysis("complete")
        ), patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run",
            side_effect=[
                subprocess.TimeoutExpired(["ollama", "run", "qwen-test"], 30),
                successful,
            ],
        ) as run:
            result = update_project_map(self.repo, semantic_analyzer=analyzer)
        graph = load_map(self.repo)

        self.assertEqual(run.call_count, 2)
        self.assertEqual(result.semantic_failed, 1)
        self.assertEqual(graph["files"]["a.py"]["semantic"]["status"], "failed")
        self.assertEqual(graph["files"]["b.py"]["semantic"]["status"], "complete")
        self.assertIn("a", [symbol["name"] for symbol in graph["files"]["a.py"]["symbols"]])
        self.assertIn("b", [symbol["name"] for symbol in graph["files"]["b.py"]["symbols"]])

    def test_invalid_qwen_json_keeps_static_graph(self) -> None:
        source = self.write("app.py", "def app():\n    return 1\n")
        analyzer = QwenSemanticAnalyzer(model="qwen-test")
        analyzer.enabled = True
        completed = Mock(returncode=0, stdout="not json", stderr="")
        with patch.object(
            QwenSemanticAnalyzer, "preflight", return_value=SemanticAnalysis("complete")
        ), patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run",
            return_value=completed,
        ):
            result = update_project_map(self.repo, semantic_analyzer=analyzer)

        graph = load_map(self.repo)
        self.assertEqual(result.semantic_failed, 1)
        self.assertIn("app", [symbol["name"] for symbol in graph["files"]["app.py"]["symbols"]])
        self.assertEqual(graph["files"]["app.py"]["semantic"]["status"], "failed")
        self.assertEqual(
            graph["files"]["app.py"]["semantic"]["failure_reason"],
            "invalid_json",
        )
        self.assertTrue(source.is_file())

    def test_format_failure_is_classified_and_not_retried_next_run(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")

        class InvalidAnalyzer(FakeSemanticAnalyzer):
            def analyze(inner_self, path: Path, repo: Path, static_result: dict) -> SemanticAnalysis:
                inner_self.calls.append(path.relative_to(repo).as_posix())
                return SemanticAnalysis(
                    "failed",
                    error="bad output",
                    failure_reason="invalid_json",
                    raw_response="Certainly! not-json",
                )

        first = InvalidAnalyzer(analyzer_version=3)
        events: list[IndexProgress] = []
        result = update_project_map(self.repo, semantic_analyzer=first, progress=events.append)

        self.assertEqual(result.semantic_failure_counts, (("invalid_json", 1),))
        self.assertEqual(result.semantic_failure_examples[0].file, "app.py")
        self.assertEqual(result.semantic_failure_examples[0].response, "Certainly! not-json")
        completed = [event for event in events if event.phase == "semantic" and event.status == "complete"][-1]
        self.assertEqual(completed.failure_counts, (("invalid_json", 1),))

        second = InvalidAnalyzer(analyzer_version=3)
        repeated = update_project_map(self.repo, semantic_analyzer=second)
        self.assertEqual(second.calls, [])
        self.assertEqual(repeated.semantic_pending, 0)

    def test_transient_failure_retries_next_run_and_other_files_continue(self) -> None:
        self.write("a.py", "def a():\n    return 1\n")
        self.write("b.py", "def b():\n    return 2\n")

        class MixedAnalyzer(FakeSemanticAnalyzer):
            def analyze(inner_self, path: Path, repo: Path, static_result: dict) -> SemanticAnalysis:
                relative = path.relative_to(repo).as_posix()
                inner_self.calls.append(relative)
                if relative == "a.py":
                    return SemanticAnalysis(
                        "failed", error="slow", failure_reason="timeout"
                    )
                return SemanticAnalysis(
                    "complete", summary="B works.", tags=("b",),
                    important_symbols=("b",),
                )

        first = MixedAnalyzer(analyzer_version=3)
        result = update_project_map(self.repo, semantic_analyzer=first)
        graph = load_map(self.repo)
        self.assertEqual(result.semantic_failure_counts, (("timeout", 1),))
        self.assertEqual(graph["files"]["a.py"]["semantic"]["status"], "failed")
        self.assertEqual(graph["files"]["b.py"]["semantic"]["status"], "complete")

        retry = FakeSemanticAnalyzer(analyzer_version=3)
        update_project_map(self.repo, semantic_analyzer=retry)
        self.assertEqual(retry.calls, ["a.py"])

    def test_static_mode_never_invokes_semantic_analyzer(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")
        analyzer = FakeSemanticAnalyzer()

        result = update_project_map(
            self.repo, semantic_analyzer=analyzer, index_mode="static"
        )

        self.assertEqual(analyzer.calls, [])
        self.assertEqual(result.requested_mode, "static")
        self.assertEqual(result.effective_mode, "static")
        self.assertEqual(
            load_map(self.repo)["files"]["app.py"]["semantic"],
            {"status": "disabled"},
        )

    def test_failed_preflight_falls_back_without_analyzing_files(self) -> None:
        for name in ("a", "b", "c"):
            self.write(f"{name}.py", f"def {name}():\n    return 1\n")

        class PreflightFailure(FakeSemanticAnalyzer):
            def preflight(inner_self, repo: Path) -> SemanticAnalysis:
                return SemanticAnalysis(
                    "failed",
                    error="configured model does not exist",
                    failure_reason="model_not_found",
                )

        analyzer = PreflightFailure()
        events: list[IndexProgress] = []
        result = update_project_map(
            self.repo, semantic_analyzer=analyzer, progress=events.append
        )

        self.assertEqual(analyzer.calls, [])
        self.assertEqual(result.requested_mode, "ai")
        self.assertEqual(result.effective_mode, "static")
        self.assertEqual(result.semantic_preflight_failure, "model_not_found")
        self.assertFalse(any(
            event.phase == "semantic" and event.status == "started"
            for event in events
        ))
        self.assertTrue(any(
            event.phase == "semantic_preflight" and event.status == "failed"
            for event in events
        ))

    def test_circuit_breaker_stops_after_four_matching_failures_in_first_five(self) -> None:
        for number in range(8):
            self.write(
                f"module_{number}.py",
                f"def feature_{number}():\n    return {number}\n",
            )

        class SystemicFailure(FakeSemanticAnalyzer):
            def analyze(inner_self, path: Path, repo: Path, static_result: dict) -> SemanticAnalysis:
                relative = path.relative_to(repo).as_posix()
                inner_self.calls.append(relative)
                if len(inner_self.calls) <= 4:
                    return SemanticAnalysis(
                        "failed", error="bad JSON", failure_reason="invalid_json"
                    )
                return SemanticAnalysis(
                    "complete", summary="Works.", tags=("works",),
                    important_symbols=(),
                )

        analyzer = SystemicFailure(analyzer_version=3)
        events: list[IndexProgress] = []
        result = update_project_map(
            self.repo, semantic_analyzer=analyzer, progress=events.append
        )

        self.assertEqual(len(analyzer.calls), 5)
        self.assertTrue(result.semantic_stopped_early)
        self.assertEqual(result.effective_mode, "static")
        self.assertEqual(result.semantic_failure_counts, (("invalid_json", 4),))
        graph = load_map(self.repo)
        self.assertEqual(sum(
            entry["semantic"]["status"] == "pending"
            for entry in graph["files"].values()
        ), 3)
        completed = [
            event for event in events
            if event.phase == "semantic" and event.status == "complete"
        ][-1]
        self.assertTrue(completed.stopped_early)
        self.assertEqual(completed.remaining, 3)

    def test_unavailable_qwen_still_saves_usable_static_map(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")
        analyzer = QwenSemanticAnalyzer(model="qwen-test")
        analyzer.enabled = True
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value=None):
            result = update_project_map(self.repo, semantic_analyzer=analyzer)

        graph = load_map(self.repo)
        self.assertIn("app", [symbol["name"] for symbol in graph["files"]["app.py"]["symbols"]])
        self.assertEqual(graph["files"]["app.py"]["semantic"]["status"], "pending")
        self.assertEqual(result.effective_mode, "static")
        self.assertEqual(result.semantic_preflight_failure, "ollama_not_found")

    def test_atomic_save_preserves_previous_valid_map_when_replace_fails(self) -> None:
        original = {"version": 2, "files": {}}
        output = save_map(self.repo, original)
        with patch("pathlib.Path.replace", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                save_map(self.repo, {**original, "files": {"new.py": {}}})

        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), original)

    def test_progress_reports_completion_and_eta(self) -> None:
        for name in ("a", "b", "c"):
            self.write(f"{name}.py", f"def {name}():\n    return 1\n")
        events: list[IndexProgress] = []

        update_project_map(
            self.repo,
            semantic_analyzer=FakeSemanticAnalyzer(),
            progress=events.append,
        )

        semantic_progress = [
            event for event in events
            if event.phase == "semantic" and event.status == "progress"
        ]
        self.assertEqual(semantic_progress[-1].completed, 3)
        self.assertEqual(semantic_progress[-1].total, 3)
        self.assertTrue(any(event.eta_seconds is not None for event in semantic_progress))

    def test_model_change_invalidates_only_semantic_cache(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")
        update_project_map(self.repo, semantic_analyzer=FakeSemanticAnalyzer("qwen-a"))
        replacement = FakeSemanticAnalyzer("qwen-b")

        with patch("chatcode.indexing.index_manager.analyze_file") as static_analysis:
            result = update_project_map(self.repo, semantic_analyzer=replacement)

        static_analysis.assert_not_called()
        self.assertEqual(replacement.calls, ["app.py"])
        semantic = load_map(self.repo)["files"]["app.py"]["semantic"]
        self.assertEqual(semantic["model"], "qwen-b")
        self.assertEqual(result.semantic_reused, 0)
        self.assertEqual(result.semantic_invalidated, 1)

    def test_analyzer_version_change_invalidates_semantics_only(self) -> None:
        self.write("app.py", "def app():\n    return 1\n")
        update_project_map(
            self.repo,
            semantic_analyzer=FakeSemanticAnalyzer(analyzer_version=2),
        )
        replacement = FakeSemanticAnalyzer(analyzer_version=3)

        with patch("chatcode.indexing.index_manager.analyze_file") as static_analysis:
            result = update_project_map(self.repo, semantic_analyzer=replacement)

        static_analysis.assert_not_called()
        self.assertEqual(replacement.calls, ["app.py"])
        self.assertEqual(result.semantic_invalidated, 1)
        semantic = load_map(self.repo)["files"]["app.py"]["semantic"]
        self.assertEqual(semantic["analyzer_version"], 3)


if __name__ == "__main__":
    unittest.main()
