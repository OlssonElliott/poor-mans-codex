"""Stable current-source capture and publication guards."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from .errors import ContextBuildError
from .materializer import _dependency_paths, _materialization_targets


CONTEXT_CAPTURE_RETRIES = 3


def _current_file_hashes(
    files: list[Path],
) -> dict[Path, str | None]:
    hashes: dict[Path, str | None] = {}
    for path in files:
        try:
            hashes[path] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        except OSError:
            hashes[path] = None
    return hashes


def _captured_source_paths(repo: Path, files: list[Path]) -> list[Path]:
    """Return every working-tree file that may be emitted in source context.

    Dependencies are selected from cached graph metadata, but their contents
    are loaded from disk.  They must therefore participate in the same
    before/after integrity check as primary files.
    """
    return list(dict.fromkeys([
        *files,
        *sorted(_dependency_paths(repo, files)),
    ]))


def build_stable_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path],
    target_symbols: dict[Path, list[str]] | None = None,
    *,
    build_patch_source_context_fn: Callable,
) -> tuple[str, dict[str, str]]:
    """Read source from disk and retry if it changes during context capture."""
    for _attempt in range(CONTEXT_CAPTURE_RETRIES):
        expanded_targets = _materialization_targets(repo, target_symbols or {})
        captured_paths = list(dict.fromkeys([
            *_captured_source_paths(repo, files),
            *(path for path, _symbol, _priority in expanded_targets),
        ]))
        before = _current_file_hashes(captured_paths)
        expanded_metadata: dict[Path, list[str]] = {
            path: list(symbols) for path, symbols in (target_symbols or {}).items()
        }
        for owner, symbol, _priority in expanded_targets:
            expanded_metadata.setdefault(owner, [])
            if symbol not in expanded_metadata[owner]:
                expanded_metadata[owner].append(symbol)
        source_context = build_patch_source_context_fn(
            repo,
            task,
            files=files,
            target_symbols=expanded_metadata,
            critical_paths=list((target_symbols or {}).keys()),
        )
        # Re-resolve dependencies in case graph metadata was refreshed while
        # rendering, then verify every path that could have been emitted.
        captured_paths = list(dict.fromkeys([
            *captured_paths,
            *_captured_source_paths(repo, files),
        ]))
        after = _current_file_hashes(captured_paths)
        if before == after:
            return source_context, {
                path.relative_to(repo).as_posix(): digest
                for path, digest in after.items()
                if digest is not None
            }

    raise RuntimeError(
        "Source files changed while ChatCode was building context. "
        "Run chatcode context again."
    )


def _format_selected_files(files: list[Path], repo: Path, source_context: str) -> str:
    """Make exceptional non-materialization explicit and non-patchable."""
    materialized: set[str] = set()
    for line in source_context.splitlines():
        for prefix in (
            "===== FULL FILE: ",
            "===== SYMBOL CONTEXT: ",
            "===== EXCERPT: ",
            "===== SELECTED SOURCE: ",
        ):
            if not line.startswith(prefix) or not line.endswith(" ====="):
                continue
            label = line[len(prefix):-len(" =====")]
            if prefix == "===== SYMBOL CONTEXT: " and "::" in label:
                label = label.split("::", 1)[0]
            if " source lines " in label:
                label = label.split(" source lines ", 1)[0]
            materialized.add(label)
            break
    lines = []
    for path in files:
        relative = path.relative_to(repo).as_posix()
        suffix = "" if relative in materialized else " [source unavailable; do not patch]"
        lines.append(f"- {relative}{suffix}")
    return "\n".join(lines) or "None"


def _assert_publishable_context_contract(
    output_file: Path, source_context: str,
) -> None:
    marker = "===== CONTEXT CONTRACT INCOMPLETE ====="
    if marker not in source_context:
        return
    try:
        output_file.unlink(missing_ok=True)
    except OSError:
        pass
    tail = source_context.split(marker, 1)[1]
    missing = [
        line[2:]
        for line in tail.splitlines()
        if line.startswith("- ") and "::" in line
    ]
    details = ", ".join(missing[:8]) or "required source"
    raise ContextBuildError(
        "ChatCode could not materialize all mandatory current source "
        f"within the hard context budget: {details}. "
        "No UPLOAD_TO_CHATGPT.md was published."
    )
