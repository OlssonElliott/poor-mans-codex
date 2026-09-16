"""Conservative deterministic expansion around semantic retrieval seeds.

The project map is intentionally a lightweight index.  This module uses it as
evidence, not as a compiler: direct dependencies, reverse dependencies,
symbol definitions and tests are useful signals, but never automatic closure.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from chatcode.config import get_setting
from chatcode.indexing.project_graph import load_map


MAX_CANDIDATES = 24
MAX_COMPLETENESS_RESOLVED = 3
GENERIC_PATH_PARTS = {"utils", "util", "common", "config", "settings", "constants", "helpers"}


@dataclass
class RetrievalResult:
    files: list[Path]
    reasons: dict[Path, list[str]] = field(default_factory=dict)
    candidates: list[Path] = field(default_factory=list)


@dataclass
class CompletenessResult:
    files: list[Path] = field(default_factory=list)
    reasons: dict[Path, str] = field(default_factory=dict)


def expand_candidates(repo: Path, task: str, seeds: list[Path], limit: int = 12) -> RetrievalResult:
    """Rank direct, local evidence around seeds without transitive expansion."""
    index = load_map(repo).get("files", {})
    seed_names = {path.relative_to(repo).as_posix() for path in seeds if path.is_file()}
    words = {word.lower() for word in re.findall(r"[A-Za-z0-9_]+", task) if len(word) >= 3}
    reverse: dict[str, set[str]] = defaultdict(set)
    symbol_owners: dict[str, set[str]] = defaultdict(set)
    for relative, entry in index.items():
        for dependency in entry.get("dependencies", []):
            reverse[str(dependency)].add(relative)
        for symbol in entry.get("symbols", []):
            if isinstance(symbol, dict):
                for key in ("name", "qualified_name"):
                    name = str(symbol.get(key, "")).lower()
                    if name:
                        symbol_owners[name].add(relative)

    scores: dict[str, int] = defaultdict(int)
    reasons: dict[str, list[str]] = defaultdict(list)
    def add(relative: str, score: int, reason: str) -> None:
        if relative not in index or not (repo / relative).is_file():
            return
        scores[relative] += score
        if reason not in reasons[relative]:
            reasons[relative].append(reason)

    for seed in sorted(seed_names):
        # Keep a valid seed even when a caller supplied it before the cache was
        # populated (for example during a first-run fallback).
        scores[seed] += 100
        reasons[seed].append("Qwen seed")
        entry = index.get(seed, {})
        for dependency in entry.get("dependencies", []):
            # Direct imports are evidence, but generic plumbing is deliberately weak.
            weight = 8 if any(part in GENERIC_PATH_PARTS for part in Path(dependency).parts) else 24
            add(str(dependency), weight, "direct dependency of seed")
        for importer in reverse.get(seed, set()):
            add(importer, 18, "imports seed")
        for symbol in entry.get("symbols", []):
            if not isinstance(symbol, dict):
                continue
            name = str(symbol.get("name", "")).lower()
            if not name or len(name) < 4:
                continue
            for owner in symbol_owners.get(name, set()):
                if owner != seed:
                    add(owner, 14, "defines seed symbol")

    for relative, entry in index.items():
        lower = relative.lower()
        for word in words:
            if word in lower:
                add(relative, 20, "lexical path match")
            for symbol in entry.get("symbols", []):
                if isinstance(symbol, dict) and word in str(symbol.get("qualified_name") or symbol.get("name", "")).lower():
                    add(relative, 28, "symbol match")
        # Tests named after a seed module are a strong, bounded relation.
        if _is_test(relative):
            stem = Path(relative).stem.lower().removeprefix("test_")
            if any(stem and stem in Path(seed).stem.lower() for seed in seed_names):
                add(relative, 22, "related test")

    # rg is much cheaper and more accurate than reopening every indexed file
    # in Python.  It is optional: the map-based signals remain fully usable on
    # installations without ripgrep.
    if shutil.which("rg"):
        for word in sorted(words)[:12]:
            try:
                search = subprocess.run(
                    ["rg", "-l", "-i", "--no-messages", "--glob", "!*.lock", word, str(repo)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3,
                )
            except (OSError, subprocess.TimeoutExpired):
                break
            for raw_path in search.stdout.splitlines()[:80]:
                try:
                    relative = Path(raw_path).resolve().relative_to(repo.resolve()).as_posix()
                except ValueError:
                    continue
                add(relative, 12, "lexical text match")

    ordered = sorted(scores, key=lambda value: (-scores[value], value.lower()))
    selected_names = ordered[:max(limit, len(seed_names))]
    paths = [repo / name for name in selected_names]
    return RetrievalResult(paths, {repo / name: reasons[name] for name in selected_names}, [repo / name for name in ordered[:MAX_CANDIDATES]])


def _is_test(relative: str) -> bool:
    path = Path(relative)
    return path.name.startswith("test_") or path.name.endswith(("_test.py", ".test.ts", ".spec.ts", ".test.js", ".spec.js"))


class QwenCompletenessChecker:
    """One compact check; symbol/concept requests are resolved locally once."""
    def __init__(self, model: str | None = None, timeout: float = 30.0) -> None:
        self.model = model or get_setting("CHATCODE_QWEN_MODEL")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.model and shutil.which("ollama"))

    def check(self, repo: Path, task: str, result: RetrievalResult) -> CompletenessResult:
        if not self.is_available():
            return CompletenessResult()
        selected = {path.relative_to(repo).as_posix() for path in result.files}
        candidate_rows = []
        index = load_map(repo).get("files", {})
        selected_rows = []
        for relative in sorted(selected):
            entry = index.get(relative, {})
            selected_rows.append({"path": relative, "symbols": [str(item.get("qualified_name") or item.get("name")) for item in entry.get("symbols", [])[:8] if isinstance(item, dict)], "dependencies": entry.get("dependencies", [])[:6]})
        for path in result.candidates:
            relative = path.relative_to(repo).as_posix()
            if relative in selected:
                continue
            entry = index.get(relative, {})
            symbols = [str(item.get("qualified_name") or item.get("name")) for item in entry.get("symbols", [])[:8] if isinstance(item, dict)]
            candidate_rows.append({"path": relative, "symbols": symbols, "dependencies": entry.get("dependencies", [])[:6]})
        prompt = (
            "You check whether code-retrieval context is complete. Return ONLY JSON: "
            '{"missing_files":["candidate/path"],"missing_symbols":["ClassOrFunction"],'
            '"missing_concepts":["short implementation concept"]}. missing_files may ONLY be paths in Candidates. '
            "For a required file absent from Candidates, name its project symbol or a short concept; do not invent paths.\n"
            f"Task: {task}\nSelected: {json.dumps(selected_rows, ensure_ascii=False)}\nCandidates: {json.dumps(candidate_rows, ensure_ascii=False)}"
        )
        try:
            run = subprocess.run(["ollama", "run", self.model, "--format", "json"], input=prompt,
                text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=self.timeout)
            payload = json.loads(run.stdout) if run.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return CompletenessResult()
        requested = payload.get("missing_files", []) if isinstance(payload, dict) else []
        available = {path.relative_to(repo).as_posix(): path for path in result.candidates}
        output = CompletenessResult()
        for name in requested:
            if not isinstance(name, str) or name not in available or name in selected:
                continue
            path = available[name]
            if path not in output.files:
                output.files.append(path)
                output.reasons[path] = "Qwen completeness known-path addition"
        symbols = _bounded_strings(payload.get("missing_symbols", []) if isinstance(payload, dict) else [], 8)
        concepts = _bounded_strings(payload.get("missing_concepts", []) if isinstance(payload, dict) else [], 4)
        # The initial pool is intentionally broad but bounded (task rg hits plus
        # one-hop graph evidence). These requests cover the rare case where it
        # contains no route to a required implementation, without trusting an
        # invented filename or asking Qwen a third question.
        for path, reason in _resolve_missing_requests(repo, index, selected | {p.relative_to(repo).as_posix() for p in output.files}, symbols, concepts):
            if path not in output.files:
                output.files.append(path)
                output.reasons[path] = reason
        return output

    def missing_files(self, repo: Path, task: str, result: RetrievalResult) -> list[Path]:
        """Compatibility helper for callers that only need the additions."""
        return self.check(repo, task, result).files


def _bounded_strings(value: object, limit: int) -> list[str]:
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    output: list[str] = []
    for item in values:
        cleaned = re.sub(r"\s+", " ", item).strip()[:120] if isinstance(item, str) else ""
        if cleaned and cleaned.casefold() not in {item.casefold() for item in output}:
            output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _resolve_missing_requests(repo: Path, index: dict, selected: set[str], symbols: list[str], concepts: list[str]) -> list[tuple[Path, str]]:
    """Resolve Qwen's non-path requests solely through real index/search hits."""
    scores: dict[str, int] = defaultdict(int)
    reasons: dict[str, str] = {}
    normalized_symbols = {re.sub(r"[^a-z0-9]", "", value.lower()) for value in symbols}
    concept_words = [word.lower() for concept in concepts for word in re.findall(r"[A-Za-z0-9_]+", concept) if len(word) >= 3]
    for relative, entry in index.items():
        if relative in selected or not (repo / relative).is_file():
            continue
        names = [str(symbol.get("qualified_name") or symbol.get("name", "")) for symbol in entry.get("symbols", []) if isinstance(symbol, dict)]
        for name in names:
            if re.sub(r"[^a-z0-9]", "", name.lower()) in normalized_symbols:
                scores[relative] += 100
                reasons[relative] = "Qwen completeness symbol resolution"
        haystack = " ".join([relative, str(entry.get("summary", "")), *entry.get("tags", []), *names]).lower()
        matched = sum(word in haystack for word in concept_words)
        if matched >= 2:
            scores[relative] += matched * 12
            reasons.setdefault(relative, "Qwen completeness concept resolution")
    # A concept can name behavior rather than a declared symbol. Search only
    # its significant terms, and require corroboration from index evidence.
    if concept_words and shutil.which("rg"):
        for word in sorted(set(concept_words))[:8]:
            try:
                run = subprocess.run(["rg", "-l", "-i", "--no-messages", word, str(repo)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                break
            for raw in run.stdout.splitlines()[:50]:
                try:
                    relative = Path(raw).resolve().relative_to(repo.resolve()).as_posix()
                except ValueError:
                    continue
                if relative in index and relative not in selected:
                    scores[relative] += 6
                    reasons.setdefault(relative, "Qwen completeness concept resolution")
    ranked = sorted((name for name, score in scores.items() if score >= 24), key=lambda name: (-scores[name], name.lower()))
    return [(repo / name, reasons[name]) for name in ranked[:MAX_COMPLETENESS_RESOLVED]]
