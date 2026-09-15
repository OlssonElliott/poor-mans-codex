from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..config import get_index_mode, get_setting


SEMANTIC_SCHEMA_VERSION = 3
MAX_SUMMARY_CHARS = 600
MAX_TAGS = 10
MAX_TAG_CHARS = 80
MAX_IMPORTANT_SYMBOLS = 20
MAX_SYMBOL_CHARS = 120
MAX_FAILURE_RESPONSE_CHARS = 500
FORMAT_FAILURES = frozenset({
    "empty_response", "invalid_json", "schema_validation_error",
})


@dataclass(frozen=True)
class SemanticAnalysis:
    status: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    important_symbols: tuple[str, ...] = ()
    error: str | None = None
    failure_reason: str | None = None
    raw_response: str | None = None


class SemanticAnalyzer(Protocol):
    def analyze(self, path: Path, repo: Path, static_result: dict[str, Any]) -> SemanticAnalysis: ...


class QwenSemanticAnalyzer:
    """Best-effort compact file descriptions through the Ollama CLI."""

    def __init__(self, model: str | None = None, timeout: float = 30.0) -> None:
        self.model = model or get_setting("CHATCODE_QWEN_MODEL")
        self.enabled = get_index_mode() == "ai"
        self.timeout = timeout
        self.analyzer_version = SEMANTIC_SCHEMA_VERSION

    def is_available(self) -> bool:
        return bool(self.enabled and self.model and shutil.which("ollama"))

    def is_eligible(self, path: Path, static_result: dict[str, Any]) -> bool:
        try:
            size = path.stat().st_size
        except OSError:
            return False
        return bool(static_result.get("symbols")) and 0 < size <= 60_000

    def build_prompt(
        self,
        path: Path,
        repo: Path,
        static_result: dict[str, Any],
        source: str | None = None,
    ) -> str:
        """Build the exact prompt, providing a one-file diagnostic/test hook."""
        text = source if source is not None else path.read_text(
            encoding="utf-8", errors="replace"
        )
        return (
            "Analyze this source file for code retrieval. The task is semantic "
            "indexing, not code generation.\n\n"
            "Return ONLY a valid JSON object.\n"
            "Do not use Markdown.\n"
            "Do not use ```json fences.\n"
            "Do not explain your answer.\n"
            "Do not include text before or after the JSON.\n\n"
            "Schema:\n"
            '{"summary":"string","tags":["string"],'
            '"important_symbols":["string"]}\n\n'
            "Constraints: summary <= 600 characters; at most 10 tags; at most "
            "20 important_symbols; use only supplied symbol names. Do not return "
            "relationships, calls, state reads, or state writes.\n"
            f"File: {path.relative_to(repo).as_posix()}\n"
            f"Static metadata: {json.dumps(_prompt_metadata(static_result), ensure_ascii=False)}\n"
            f"Code:\n{text}"
        )

    def analyze(
        self, path: Path, repo: Path, static_result: dict[str, Any]
    ) -> SemanticAnalysis:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self._analyze_source(path, repo, static_result, text)

    def preflight(self, repo: Path) -> SemanticAnalysis:
        """Exercise the real prompt, Ollama invocation, parser, and schema path."""
        if not self.enabled or not self.model:
            return _failure("ollama_process_error", "Qwen semantic analysis is disabled")
        if not shutil.which("ollama"):
            return _failure("ollama_not_found", "Ollama executable was not found")
        try:
            model_check = subprocess.run(
                ["ollama", "show", self.model],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return _failure("timeout", f"model check timed out after {self.timeout:g}s")
        except FileNotFoundError as exc:
            return _failure("ollama_not_found", str(exc))
        except OSError as exc:
            return _failure("ollama_process_error", str(exc))
        if model_check.returncode != 0:
            return _failure(
                "model_not_found",
                model_check.stderr.strip() or f"model {self.model!r} was not found",
            )
        path = repo / "chatcode_semantic_preflight.py"
        source = "def chatcode_semantic_preflight():\n    return 'ok'\n"
        static_result = {
            "language": "python",
            "symbols": [{
                "name": "chatcode_semantic_preflight",
                "qualified_name": "chatcode_semantic_preflight",
                "kind": "function",
            }],
            "dependencies": [],
        }
        return self._analyze_source(path, repo, static_result, source)

    def _analyze_source(
        self,
        path: Path,
        repo: Path,
        static_result: dict[str, Any],
        text: str,
    ) -> SemanticAnalysis:
        if not self.enabled or not self.model:
            return _failure("ollama_process_error", "Qwen semantic analysis is disabled")
        if not shutil.which("ollama"):
            return _failure("ollama_not_found", "Ollama executable was not found")
        if not text.strip() or len(text) > 60_000 or not static_result.get("symbols"):
            return SemanticAnalysis("complete")

        first = _restrict_important_symbols(
            self._run(self.build_prompt(path, repo, static_result, text)),
            static_result,
        )
        if first.status == "complete" or first.failure_reason not in FORMAT_FAILURES:
            return first
        repair_prompt = (
            "Your previous response was invalid. Reformat it as ONLY one valid JSON "
            "object with exactly these fields: summary (string), tags (array of "
            "strings), important_symbols (array of strings). No Markdown, fences, "
            "explanation, prefix, or suffix.\nPrevious response:\n"
            + (first.raw_response or "")
        )
        return _restrict_important_symbols(self._run(repair_prompt), static_result)

    def _run(self, prompt: str) -> SemanticAnalysis:
        try:
            completed = subprocess.run(
                ["ollama", "run", self.model, "--format", "json"],
                input=prompt, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return _failure("timeout", f"timed out after {self.timeout:g}s")
        except FileNotFoundError as exc:
            return _failure("ollama_not_found", str(exc))
        except OSError as exc:
            return _failure("ollama_process_error", str(exc))
        if completed.returncode != 0:
            error = completed.stderr.strip() or "Ollama returned a non-zero exit code"
            lowered_error = error.lower()
            missing_model = "model" in lowered_error and any(
                marker in lowered_error
                for marker in ("not found", "does not exist", "manifest")
            )
            reason = "model_not_found" if missing_model else "ollama_process_error"
            return _failure(
                reason,
                error,
            )
        metadata, reason = parse_semantic_response_detailed(completed.stdout)
        if metadata is None:
            messages = {
                "empty_response": "Ollama returned an empty response",
                "invalid_json": "response did not contain one valid JSON object",
                "schema_validation_error": "JSON object did not match the semantic schema",
            }
            return _failure(
                reason or "schema_validation_error",
                messages.get(reason or "", "invalid semantic response"),
                completed.stdout,
            )
        return SemanticAnalysis(
            "complete", summary=metadata["summary"], tags=tuple(metadata["tags"]),
            important_symbols=tuple(metadata["important_symbols"]),
            raw_response=_truncate_response(completed.stdout),
        )


def _failure(reason: str, error: str, response: str | None = None) -> SemanticAnalysis:
    return SemanticAnalysis(
        "failed", error=error, failure_reason=reason,
        raw_response=_truncate_response(response),
    )


def _restrict_important_symbols(
    result: SemanticAnalysis,
    static_result: dict[str, Any],
) -> SemanticAnalysis:
    if result.status != "complete":
        return result
    allowed = {
        str(symbol.get(key))
        for symbol in static_result.get("symbols", [])
        if isinstance(symbol, dict)
        for key in ("name", "qualified_name")
        if symbol.get(key)
    }
    return SemanticAnalysis(
        result.status,
        summary=result.summary,
        tags=result.tags,
        important_symbols=tuple(
            symbol for symbol in result.important_symbols if symbol in allowed
        ),
        error=result.error,
        failure_reason=result.failure_reason,
        raw_response=result.raw_response,
    )


def _truncate_response(response: str | None) -> str | None:
    if response is None:
        return None
    cleaned = response.strip()
    return cleaned[:MAX_FAILURE_RESPONSE_CHARS] if cleaned else None


def _prompt_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "language": entry.get("language"),
        "symbols": entry.get("symbols", [])[:100],
        "dependencies": entry.get("dependencies", [])[:50],
    }


def validate_semantic_response(text: str) -> dict[str, Any]:
    return parse_semantic_response(text) or {}


def parse_semantic_response(text: str) -> dict[str, Any] | None:
    metadata, _ = parse_semantic_response_detailed(text)
    return metadata


def parse_semantic_response_detailed(
    text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response"
    payload = _load_json_object(text)
    if payload is None:
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "schema_validation_error"
    if not {"summary", "tags", "important_symbols"}.intersection(payload):
        return None, "schema_validation_error"
    summary_value = payload.get("summary")
    summary = (
        re.sub(r"\s+", " ", summary_value).strip()[:MAX_SUMMARY_CHARS]
        if isinstance(summary_value, str) else ""
    )
    tags_value = payload.get("tags", [])
    if isinstance(tags_value, str):
        tags_value = [tags_value]
    elif not isinstance(tags_value, list):
        tags_value = []
    symbols_value = payload.get("important_symbols", [])
    if isinstance(symbols_value, str):
        symbols_value = [symbols_value]
    elif not isinstance(symbols_value, list):
        symbols_value = []
    return {
        "summary": summary,
        "tags": _bounded_strings(tags_value, MAX_TAGS, MAX_TAG_CHARS),
        "important_symbols": _bounded_strings(
            symbols_value, MAX_IMPORTANT_SYMBOLS, MAX_SYMBOL_CHARS
        ),
    }, None


def _load_json_object(text: str) -> Any | None:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE
    )
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    objects = _complete_json_objects(stripped)
    if len(objects) != 1:
        return None
    try:
        return json.loads(objects[0])
    except json.JSONDecodeError:
        return None


def _complete_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start:index + 1])
                start = None
    return objects


def _bounded_strings(values: list[Any], limit: int, max_chars: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = re.sub(r"\s+", " ", value).strip()[:max_chars]
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
        if len(output) >= limit:
            break
    return output
