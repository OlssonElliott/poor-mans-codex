from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any

from chatcode.config import get_index_mode
from chatcode.indexing.project_graph import load_map


MAX_INITIAL_FILES = 8
MAX_DEPENDENCY_DEPTH = 2


def retrieve_files(
    repo: Path,
    query: str,
    max_files: int = 12,
    depth: int = MAX_DEPENDENCY_DEPTH,
    index_mode: str | None = None,
) -> list[Path]:
    mode = index_mode or get_index_mode()
    if mode not in {"ai", "static"}:
        raise ValueError("index_mode must be either 'ai' or 'static'")
    index = load_map(repo)
    words = {
        word.lower()
        for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", query)
        if len(word) >= 3
    }
    if not words:
        return []

    scored: list[tuple[float, str]] = []
    for relative, metadata in index.get("files", {}).items():
        score = _score_file(relative, metadata, words, mode)
        if score > 0:
            scored.append((score, relative))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))

    selected = [relative for _, relative in scored[:min(MAX_INITIAL_FILES, max_files)]]
    seen = set(selected)
    queue = deque((relative, 0) for relative in selected)
    while queue and len(selected) < max_files:
        relative, level = queue.popleft()
        if level >= min(depth, MAX_DEPENDENCY_DEPTH):
            continue
        metadata = index["files"].get(relative, {})
        for dependency in metadata.get("dependencies", [])[:20]:
            if dependency in seen or dependency not in index["files"]:
                continue
            seen.add(dependency)
            selected.append(dependency)
            queue.append((dependency, level + 1))
            if len(selected) >= max_files:
                break

    return [repo / relative for relative in selected if (repo / relative).is_file()]


def _score_file(
    relative: str,
    metadata: dict[str, Any],
    words: set[str],
    mode: str,
) -> float:
    path = relative.lower()
    symbols = [
        str(symbol.get("qualified_name") or symbol.get("name", "")).lower()
        for symbol in metadata.get("symbols", [])
        if isinstance(symbol, dict)
    ]
    imports = [str(name).lower() for name in metadata.get("imports", [])]
    score = 0.0
    for word in words:
        if word in path:
            score += 6.0
        score += sum(5.0 for symbol in symbols if word in symbol)
        semantic = metadata.get("semantic")
        semantic_complete = bool(
            isinstance(semantic, dict) and semantic.get("status") == "complete"
        )
        if mode == "ai" and semantic_complete:
            summary = str(metadata.get("summary", "")).lower()
            tags = [str(tag).lower() for tag in metadata.get("tags", [])]
            important = [
                str(name).lower() for name in metadata.get("important_symbols", [])
            ]
            score += sum(8.0 for tag in tags if word in tag)
            score += sum(6.0 for symbol in important if word in symbol)
            if word in summary:
                score += 3.0
        else:
            score += sum(3.0 for imported in imports if word in imported)
    return score
