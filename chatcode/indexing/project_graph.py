from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from ..workspace import get_repo_workspace


SCHEMA_VERSION = 2
INDEX_FILENAME = "project-map.json"


def map_path(repo: Path) -> Path:
    return get_repo_workspace(repo) / INDEX_FILENAME


def empty_map() -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "files": {}}


def load_map(repo: Path) -> dict[str, Any]:
    path = map_path(repo)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_map()
    if (
        not isinstance(data, dict)
        or data.get("version") != SCHEMA_VERSION
        or not isinstance(data.get("files"), dict)
    ):
        return empty_map()
    return data


def save_map(repo: Path, index: dict[str, Any]) -> Path:
    path = map_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    content = json.dumps(index, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    return path


def normalize_compact_index(index: dict[str, Any]) -> None:
    """Normalize compact file metadata and cheaply resolve direct imports."""
    index["version"] = SCHEMA_VERSION
    index.pop("symbols", None)
    index.pop("relations", None)
    files = index.setdefault("files", {})
    for relative, entry in files.items():
        entry["path"] = relative
        entry.pop("relations", None)
        entry.pop("semantic_relations", None)
    _resolve_dependencies(files)


def rebuild_graph(index: dict[str, Any]) -> None:
    """Backward-compatible name for the compact index normalizer."""
    normalize_compact_index(index)


def _resolve_dependencies(files: dict[str, dict[str, Any]]) -> None:
    module_paths: dict[str, str] = {}
    for relative in files:
        if relative.endswith(".py"):
            module = relative[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[:-9]
            module_paths[module] = relative

    for relative, entry in files.items():
        dependencies: set[str] = set()
        for imported in entry.get("imports", []):
            dependency = _resolve_import(relative, imported, files, module_paths)
            if dependency and dependency != relative:
                dependencies.add(dependency)
        entry["dependencies"] = sorted(dependencies)[:50]


def _resolve_import(
    source: str,
    imported: str,
    files: dict[str, dict[str, Any]],
    module_paths: dict[str, str],
) -> str | None:
    if source.endswith(".py"):
        leading = len(imported) - len(imported.lstrip("."))
        module = imported.lstrip(".")
        if leading:
            package = source[:-3].split("/")[:-1]
            keep = max(0, len(package) - (leading - 1))
            module = ".".join([*package[:keep], *module.split(".")])
        parts = module.split(".")
        for length in range(len(parts), 0, -1):
            match = module_paths.get(".".join(parts[:length]))
            if match:
                return match
        return None

    if imported.startswith("."):
        source_parent = PurePosixPath(source).parent
        candidate = source_parent.joinpath(imported).as_posix()
        normalized = PurePosixPath(candidate).as_posix()
        choices = [
            normalized,
            *(normalized + suffix for suffix in (".ts", ".tsx", ".js", ".jsx")),
            *(f"{normalized}/index{suffix}" for suffix in (".ts", ".tsx", ".js", ".jsx")),
        ]
        for choice in choices:
            # PurePosixPath preserves '..'; resolve it lexically without touching disk.
            parts: list[str] = []
            for part in PurePosixPath(choice).parts:
                if part == ".." and parts:
                    parts.pop()
                elif part not in {".", ""}:
                    parts.append(part)
            resolved = "/".join(parts)
            if resolved in files:
                return resolved
    return None
