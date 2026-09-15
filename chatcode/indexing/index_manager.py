from __future__ import annotations

import time
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashes import hash_file
from .project_graph import load_map, map_path, rebuild_graph, save_map
from .scanner import is_indexable, scan_project
from .semantic_analyzer import QwenSemanticAnalyzer, SemanticAnalysis, SemanticAnalyzer
from .static_analyzer import analyze_file


@dataclass(frozen=True)
class IndexProgress:
    phase: str
    status: str
    completed: int = 0
    total: int = 0
    current_file: str | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    model: str | None = None
    failed: int = 0
    added: int = 0
    changed: int = 0
    deleted: int = 0
    processed: int = 0


ProgressCallback = Callable[[IndexProgress], None]


class SemanticIndexInterrupted(KeyboardInterrupt):
    def __init__(self, completed: int, total: int, map_file: Path) -> None:
        super().__init__("Semantic indexing interrupted by user")
        self.completed = completed
        self.total = total
        self.map_file = map_file


@dataclass(frozen=True)
class IndexUpdate:
    added: tuple[str, ...]
    changed: tuple[str, ...]
    deleted: tuple[str, ...]
    unchanged: tuple[str, ...]
    renamed: tuple[tuple[str, str], ...]
    map_file: Path
    semantic_completed: int = 0
    semantic_total: int = 0
    semantic_failed: int = 0


def _emit(progress: ProgressCallback | None, event: IndexProgress) -> None:
    if progress is not None:
        progress(event)


def _analyzer_identity(analyzer: SemanticAnalyzer) -> tuple[str, int]:
    model = getattr(analyzer, "model", None)
    if not isinstance(model, str) or not model:
        model = analyzer.__class__.__name__
    version = getattr(analyzer, "analyzer_version", 1)
    if not isinstance(version, int):
        version = 1
    return model, version


def _analyzer_available(analyzer: SemanticAnalyzer) -> bool:
    check = getattr(analyzer, "is_available", None)
    return bool(check()) if callable(check) else True


def _semantic_eligible(
    analyzer: SemanticAnalyzer,
    path: Path,
    entry: dict[str, Any],
) -> bool:
    check = getattr(analyzer, "is_eligible", None)
    return bool(check(path, entry)) if callable(check) else True


def _semantic_is_current(entry: dict[str, Any], model: str, version: int) -> bool:
    semantic = entry.get("semantic")
    return bool(
        isinstance(semantic, dict)
        and semantic.get("status") == "complete"
        and semantic.get("analyzed_hash") == entry.get("hash")
        and semantic.get("model") == model
        and semantic.get("analyzer_version") == version
    )


def _pending_semantic(model: str, version: int) -> dict[str, Any]:
    return {
        "status": "pending",
        "analyzed_hash": None,
        "model": model,
        "analyzer_version": version,
    }


def _normalize_analysis(result: Any) -> SemanticAnalysis:
    if isinstance(result, SemanticAnalysis):
        return result
    if isinstance(result, list):
        return SemanticAnalysis("complete")
    return SemanticAnalysis("failed", error="semantic analyzer returned an unsupported result")


def _static_summary(relative: str, entry: dict[str, Any]) -> str:
    language = str(entry.get("language", "source")).replace("_", "/")
    names = [
        symbol.get("qualified_name") or symbol.get("name")
        for symbol in entry.get("symbols", [])[:8]
        if isinstance(symbol, dict)
    ]
    summary = f"{language.title()} source file {relative}."
    if names:
        summary += " Defines " + ", ".join(str(name) for name in names) + "."
    return summary[:600]


def _static_tags(relative: str, entry: dict[str, Any]) -> list[str]:
    values: list[str] = [str(entry.get("language", "source"))]
    values.extend(re.split(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9]+", relative.rsplit(".", 1)[0]))
    for symbol in entry.get("symbols", [])[:20]:
        if isinstance(symbol, dict):
            values.extend(re.split(r"[_\W]+", str(symbol.get("name", ""))))
    output: list[str] = []
    for value in values:
        cleaned = value.strip().lower()[:80]
        if len(cleaned) >= 2 and cleaned not in output:
            output.append(cleaned)
        if len(output) >= 10:
            break
    return output


def _reset_semantic_metadata(relative: str, entry: dict[str, Any]) -> None:
    entry["summary"] = _static_summary(relative, entry)
    entry["tags"] = _static_tags(relative, entry)
    entry["important_symbols"] = []


def update_project_map(
    repo: Path,
    paths: Iterable[str | Path] | None = None,
    semantic_analyzer: SemanticAnalyzer | None = None,
    progress: ProgressCallback | None = None,
    run_semantic: bool = True,
) -> IndexUpdate:
    repo = repo.resolve()
    existed = map_path(repo).is_file()
    graph = load_map(repo)
    old_files = graph["files"]
    analyzer = semantic_analyzer if semantic_analyzer is not None else QwenSemanticAnalyzer()
    model, analyzer_version = _analyzer_identity(analyzer)

    if not existed:
        _emit(progress, IndexProgress("index", "initial"))
        _emit(progress, IndexProgress("scan", "started"))

    newly_excluded = {
        relative
        for relative in old_files
        if not is_indexable((repo / relative).resolve(), repo)
    }
    renamed: list[tuple[str, str]] = []
    if paths is None:
        current_paths = scan_project(repo)
        current = {path.relative_to(repo).as_posix(): path for path in current_paths}
        deleted = sorted((set(old_files) - set(current)) | newly_excluded)
    else:
        requested: dict[str, Path] = {}
        for raw in paths:
            candidate = Path(raw)
            absolute = candidate if candidate.is_absolute() else repo / candidate
            try:
                relative = absolute.resolve().relative_to(repo).as_posix()
            except ValueError:
                continue
            requested[relative] = absolute
        current = {
            relative: path
            for relative, path in requested.items()
            if path.is_file() and is_indexable(path, repo)
        }
        deleted = sorted(
            {
                relative
                for relative in requested
                if relative in old_files and relative not in current
            }
            | newly_excluded
        )

    added: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    deleted_hashes = {
        old_files[relative].get("hash"): relative
        for relative in deleted
        if relative in old_files
    }
    for relative in deleted:
        old_files.pop(relative, None)

    for relative, path in sorted(current.items()):
        try:
            digest = hash_file(path)
        except OSError:
            continue
        previous = old_files.get(relative)
        if previous and previous.get("hash") == digest:
            unchanged.append(relative)
            continue
        static = analyze_file(path, repo)
        old_files[relative] = {
            "path": relative,
            "hash": digest,
            **static,
            "dependencies": [],
            "semantic": _pending_semantic(model, analyzer_version),
        }
        _reset_semantic_metadata(relative, old_files[relative])
        renamed_from = deleted_hashes.get(digest)
        if previous is None and renamed_from:
            old_files[relative]["renamed_from"] = renamed_from
            renamed.append((renamed_from, relative))
        (changed if previous else added).append(relative)

    static_changed = bool(added or changed or deleted or not existed)
    if static_changed:
        if existed:
            _emit(progress, IndexProgress("index", "updating"))
            _emit(progress, IndexProgress("scan", "started"))
        _emit(progress, IndexProgress("static", "complete", total=len(old_files)))

    # Make deterministic retrieval durable before the first slow model call.
    rebuild_graph(graph)
    output = save_map(repo, graph)
    if static_changed:
        _emit(progress, IndexProgress(
            "static", "saved", total=len(old_files),
            added=len(added), changed=len(changed), deleted=len(deleted),
        ))

    if not run_semantic or not _analyzer_available(analyzer):
        if not static_changed:
            _emit(progress, IndexProgress("index", "up_to_date", total=len(old_files)))
        return IndexUpdate(
            tuple(added), tuple(changed), tuple(deleted), tuple(unchanged),
            tuple(renamed), output,
        )

    scope = current if paths is not None else {
        relative: repo / relative for relative in old_files
    }
    eligible: list[tuple[str, Path, dict[str, Any]]] = []
    metadata_changed = False
    for relative, path in sorted(scope.items()):
        entry = old_files.get(relative)
        if entry is None or not path.is_file() or not _semantic_eligible(analyzer, path, entry):
            continue
        if not _semantic_is_current(entry, model, analyzer_version):
            entry["semantic"] = _pending_semantic(model, analyzer_version)
            _reset_semantic_metadata(relative, entry)
            eligible.append((relative, path, entry))
            metadata_changed = True

    all_eligible = [
        entry
        for relative, entry in old_files.items()
        if (repo / relative).is_file()
        and _semantic_eligible(analyzer, repo / relative, entry)
    ]
    completed_before = sum(
        _semantic_is_current(entry, model, analyzer_version)
        for entry in all_eligible
    )
    total = len(all_eligible)
    if metadata_changed:
        rebuild_graph(graph)
        output = save_map(repo, graph)

    if not eligible:
        if not static_changed:
            _emit(progress, IndexProgress("index", "up_to_date", total=len(old_files)))
        return IndexUpdate(
            tuple(added), tuple(changed), tuple(deleted), tuple(unchanged),
            tuple(renamed), output, completed_before, total, 0,
        )

    _emit(progress, IndexProgress(
        "semantic", "started", completed_before, total, model=model,
    ))
    run_started = time.monotonic()
    processed = 0
    failed = 0
    for relative, path, entry in eligible:
        try:
            result = _normalize_analysis(analyzer.analyze(path, repo, entry))
        except KeyboardInterrupt:
            completed = completed_before + processed
            _emit(progress, IndexProgress(
                "semantic", "interrupted", completed, total,
                current_file=relative, elapsed_seconds=time.monotonic() - run_started,
                model=model, failed=failed,
            ))
            raise SemanticIndexInterrupted(completed, total, output) from None
        except Exception as exc:
            result = SemanticAnalysis("failed", [], str(exc))

        processed += 1
        if result.status == "complete":
            entry["summary"] = result.summary.strip()[:600] or _static_summary(relative, entry)
            entry["tags"] = list(dict.fromkeys(
                tag.strip()[:80] for tag in result.tags if tag.strip()
            ))[:10] or _static_tags(relative, entry)
            entry["important_symbols"] = list(dict.fromkeys(
                name.strip()[:120] for name in result.important_symbols if name.strip()
            ))[:20]
            entry["semantic"] = {
                "status": "complete",
                "analyzed_hash": entry["hash"],
                "model": model,
                "analyzer_version": analyzer_version,
            }
        else:
            failed += 1
            _reset_semantic_metadata(relative, entry)
            entry["semantic"] = {
                "status": "failed",
                "analyzed_hash": entry["hash"],
                "model": model,
                "analyzer_version": analyzer_version,
                "error": result.error or "semantic analysis failed",
            }

        rebuild_graph(graph)
        output = save_map(repo, graph)
        elapsed = time.monotonic() - run_started
        average = elapsed / processed
        remaining = len(eligible) - processed
        eta = average * remaining if processed >= 2 and remaining else None
        _emit(progress, IndexProgress(
            "semantic", "progress", completed_before + processed, total,
            current_file=relative, elapsed_seconds=elapsed,
            eta_seconds=eta, model=model, failed=failed, processed=processed,
        ))

    completed = completed_before + processed
    _emit(progress, IndexProgress(
        "semantic", "complete", completed, total,
        elapsed_seconds=time.monotonic() - run_started,
        model=model, failed=failed, processed=processed,
    ))
    return IndexUpdate(
        tuple(added), tuple(changed), tuple(deleted), tuple(unchanged),
        tuple(renamed), output, completed, total, failed,
    )
