from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from chatcode.indexing.semantic_analyzer import (
    MAX_IMPORTANT_SYMBOLS,
    MAX_TAGS,
    QwenSemanticAnalyzer,
    parse_semantic_response,
    parse_semantic_response_detailed,
)


VALID = {
    "summary": "Handles inventory items.",
    "tags": ["inventory", "items"],
    "important_symbols": ["give_item"],
}


class SemanticResponseTests(unittest.TestCase):
    def test_valid_pure_json(self) -> None:
        self.assertEqual(parse_semantic_response(json.dumps(VALID)), VALID)

    def test_json_code_fence(self) -> None:
        response = f"```json\n{json.dumps(VALID)}\n```"
        self.assertEqual(parse_semantic_response(response), VALID)

    def test_explanatory_prefix(self) -> None:
        response = f"Here is the result:\n{json.dumps(VALID)}"
        self.assertEqual(parse_semantic_response(response), VALID)

    def test_multiple_json_objects_are_rejected_as_ambiguous(self) -> None:
        parsed, reason = parse_semantic_response_detailed("{} then {}")
        self.assertIsNone(parsed)
        self.assertEqual(reason, "invalid_json")

    def test_empty_and_malformed_responses_are_classified(self) -> None:
        self.assertEqual(parse_semantic_response_detailed(""), (None, "empty_response"))
        self.assertEqual(
            parse_semantic_response_detailed('{"summary":'),
            (None, "invalid_json"),
        )

    def test_missing_summary_uses_empty_static_fallback_signal(self) -> None:
        parsed = parse_semantic_response('{"tags":["inventory"]}')
        self.assertEqual(parsed["summary"], "")
        self.assertEqual(parsed["tags"], ["inventory"])

    def test_imperfect_values_are_normalized(self) -> None:
        parsed = parse_semantic_response(json.dumps({
            "summary": None,
            "tags": " inventory ",
            "important_symbols": [" give_item ", 123, "give_item"],
        }))
        self.assertEqual(parsed, {
            "summary": "",
            "tags": ["inventory"],
            "important_symbols": ["give_item"],
        })

    def test_output_bounds_are_enforced(self) -> None:
        parsed = parse_semantic_response(json.dumps({
            "summary": "x" * 700,
            "tags": [f"tag-{number}-" + "x" * 100 for number in range(20)],
            "important_symbols": [f"symbol_{number}" for number in range(30)],
        }))
        self.assertEqual(len(parsed["summary"]), 600)
        self.assertEqual(len(parsed["tags"]), MAX_TAGS)
        self.assertTrue(all(len(tag) <= 80 for tag in parsed["tags"]))
        self.assertEqual(len(parsed["important_symbols"]), MAX_IMPORTANT_SYMBOLS)

    def test_unrecognized_schema_is_classified(self) -> None:
        parsed, reason = parse_semantic_response_detailed('{"relations":[]}')
        self.assertIsNone(parsed)
        self.assertEqual(reason, "schema_validation_error")


class QwenSemanticAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.source = self.repo / "app.py"
        self.source.write_text("def app():\n    return 1\n", encoding="utf-8")
        self.static = {"symbols": [{"name": "app"}], "language": "python"}
        self.analyzer = QwenSemanticAnalyzer(model="qwen2.5-coder:1.5b")
        self.analyzer.enabled = True

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prompt_is_json_only_and_directly_inspectable(self) -> None:
        prompt = self.analyzer.build_prompt(self.source, self.repo, self.static)
        self.assertIn("Return ONLY a valid JSON object", prompt)
        self.assertIn("Do not use ```json fences", prompt)
        self.assertIn('"important_symbols":["string"]', prompt)

    def test_timeout_is_not_retried(self) -> None:
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["ollama"], 30),
        ) as run:
            result = self.analyzer.analyze(self.source, self.repo, self.static)
        self.assertEqual(result.failure_reason, "timeout")
        self.assertEqual(run.call_count, 1)

    def test_process_error_is_classified_and_not_retried(self) -> None:
        failed = Mock(returncode=1, stdout="", stderr="model unavailable")
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run", return_value=failed,
        ) as run:
            result = self.analyzer.analyze(self.source, self.repo, self.static)
        self.assertEqual(result.failure_reason, "ollama_process_error")
        self.assertEqual(run.call_count, 1)

    def test_missing_model_is_classified(self) -> None:
        failed = Mock(
            returncode=1,
            stdout="",
            stderr="Error: model 'missing' not found",
        )
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run", return_value=failed,
        ):
            result = self.analyzer.analyze(self.source, self.repo, self.static)
        self.assertEqual(result.failure_reason, "model_not_found")

    def test_preflight_uses_real_prompt_invocation_and_parser(self) -> None:
        completed = Mock(returncode=0, stdout=json.dumps(VALID), stderr="")
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run", return_value=completed,
        ) as run:
            result = self.analyzer.preflight(self.repo)
        self.assertEqual(result.status, "complete")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["ollama", "show", "qwen2.5-coder:1.5b"],
        )
        self.assertIn("chatcode_semantic_preflight", run.call_args.kwargs["input"])
        self.assertEqual(
            run.call_args.args[0],
            ["ollama", "run", "qwen2.5-coder:1.5b", "--format", "json"],
        )

    def test_one_format_retry_succeeds(self) -> None:
        bad = Mock(returncode=0, stdout="not json", stderr="")
        good = Mock(returncode=0, stdout=json.dumps(VALID), stderr="")
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run", side_effect=[bad, good],
        ) as run:
            result = self.analyzer.analyze(self.source, self.repo, self.static)
        self.assertEqual(result.status, "complete")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["ollama", "run", "qwen2.5-coder:1.5b", "--format", "json"],
        )
        self.assertIn("previous response was invalid", run.call_args_list[1].kwargs["input"])

    def test_one_format_retry_also_fails(self) -> None:
        bad = Mock(returncode=0, stdout="not json", stderr="")
        with patch("chatcode.indexing.semantic_analyzer.shutil.which", return_value="ollama"), patch(
            "chatcode.indexing.semantic_analyzer.subprocess.run", return_value=bad,
        ) as run:
            result = self.analyzer.analyze(self.source, self.repo, self.static)
        self.assertEqual(result.failure_reason, "invalid_json")
        self.assertEqual(result.raw_response, "not json")
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
