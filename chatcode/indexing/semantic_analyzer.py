from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..config import get_boolean_setting, get_setting


SEMANTIC_SCHEMA_VERSION = 2
MAX_SUMMARY_CHARS = 600
MAX_TAGS = 10
MAX_TAG_CHARS = 80
MAX_IMPORTANT_SYMBOLS = 20
MAX_SYMBOL_CHARS = 120


@dataclass(frozen=True)
class SemanticAnalysis:
    status: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    important_symbols: tuple[str, ...] = ()
    error: str | None = None


class SemanticAnalyzer(Protocol):
    def analyze(
        self,
        path: Path,
        repo: Path,
        static_result: dict[str, Any],
    ) -> SemanticAnalysis: ...


class QwenSemanticAnalyzer:
    """Best-effort compact file descriptions through the Ollama CLI."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or get_setting("CHATCODE_QWEN_MODEL")
        self.enabled = get_boolean_setting("CHATCODE_QWEN_ENABLED")
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

    def analyze(
        self,
        path: Path,
        repo: Path,
        static_result: dict[str, Any],
    ) -> SemanticAnalysis:
        if not self.is_available():
            return SemanticAnalysis("failed", error="Ollama/Qwen is unavailable")
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip() or len(text) > 60_000 or not static_result.get("symbols"):
            return SemanticAnalysis("complete")
        prompt = (
            "Describe this source file for code-retrieval only. Do not write or suggest code. "
            "Return only strict JSON with this shape: "
            '{"summary":"max 2-3 concise sentences","tags":["max 10 short concepts"],'
            '"important_symbols":["max 20 names from the supplied symbols"]}. '
            "Do not return relationships, calls, state reads, or state writes.\n"
            f"File: {path.relative_to(repo).as_posix()}\n"
            f"Static metadata: {json.dumps(_prompt_metadata(static_result), ensure_ascii=False)}\n"
            f"Code:\n{text}"
        )
        try:
            completed = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return SemanticAnalysis("failed", error=f"timed out after {self.timeout:g}s")
        except OSError as exc:
            return SemanticAnalysis("failed", error=str(exc))
        if completed.returncode != 0:
            return SemanticAnalysis(
                "failed",
                error=completed.stderr.strip() or "Ollama returned a non-zero exit code",
            )
        metadata = parse_semantic_response(completed.stdout)
        if metadata is None:
            return SemanticAnalysis("failed", error="invalid semantic JSON response")
        return SemanticAnalysis(
            "complete",
            summary=metadata["summary"],
            tags=tuple(metadata["tags"]),
            important_symbols=tuple(metadata["important_symbols"]),
        )


def _prompt_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "language": entry.get("language"),
        "symbols": entry.get("symbols", [])[:100],
        "dependencies": entry.get("dependencies", [])[:50],
    }


def validate_semantic_response(text: str) -> dict[str, Any]:
    return parse_semantic_response(text) or {}


def parse_semantic_response(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), str):
        return None
    if not isinstance(payload.get("tags", []), list):
        return None
    if not isinstance(payload.get("important_symbols", []), list):
        return None
    summary = re.sub(r"\s+", " ", payload["summary"]).strip()[:MAX_SUMMARY_CHARS]
    tags = _bounded_strings(payload.get("tags", []), MAX_TAGS, MAX_TAG_CHARS)
    symbols = _bounded_strings(
        payload.get("important_symbols", []),
        MAX_IMPORTANT_SYMBOLS,
        MAX_SYMBOL_CHARS,
    )
    return {"summary": summary, "tags": tags, "important_symbols": symbols}


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
