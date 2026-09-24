"""Retrieval orchestration for normal tasks and failing-test roots."""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from ..config import get_boolean_setting
from ..indexing.index_manager import IndexProgress
from ..retrieval.graph_retriever import retrieve_files as _default_retrieve_files
from ..retrieval.hybrid_retriever import (
    QwenCompletenessChecker,
    QwenTaskHintAnalyzer,
    RetrievalResult,
    expand_candidates,
    implementation_closure,
    resolve_transport_roots,
    resolve_explicit_targets,
    resolve_task_surface_roots,
    resolve_semantic_hints,
    resolve_test_callsite_closure,
)
from .files import (
    _ambiguous_task_file_paths,
    _index_scope_paths,
    _is_context_internal_artifact,
    _is_internal_workspace_path,
    _purge_internal_workspace_from_index,
    _resolve_task_file_references,
    is_source_file,
)
from .materializer import (
    _changed_python_definition_roots,
    _read_current_text_and_hash,
)


def collect_relevant_files(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
    include_target_symbols: bool = False,
    *,
    max_files: int,
    update_project_map_fn: Callable,
    retrieve_files_fn: Callable,
    get_changed_files_fn: Callable[[Path], set[Path]],
    runtime_module_file: str,
) -> list[Path] | tuple[list[Path], dict[Path, list[str]]]:
    # Synchronizing here also catches edits made manually since the previous
    # ChatCode invocation. Indexing is an enhancement, so a damaged/unwritable
    # cache must not prevent the established retrieval path from working.
    graph_files: list[Path] = []
    explicit_targets: list[Path] = []
    explicit_target_reasons: dict[Path, list[str]] = {}
    surface_result = RetrievalResult([])
    semantic_targets: list[Path] = []
    semantic_target_reasons: dict[Path, list[str]] = {}
    semantic_hints: list[str] = []
    semantic_hint_status = "not_run"
    task_hint_analyzer: QwenTaskHintAnalyzer | None = None
    effective_mode = "static"
    try:
        _purge_internal_workspace_from_index(repo)
        index_paths = _index_scope_paths(repo)
        update = update_project_map_fn(
            repo,
            paths=index_paths,
            progress=index_progress,
        )
        effective_mode = update.effective_mode
        explicit_target_result = resolve_explicit_targets(repo, task)
        explicit_targets = explicit_target_result.files
        explicit_target_reasons = explicit_target_result.reasons
        surface_result = resolve_task_surface_roots(repo, task)
        graph_files = [
            path
            for path in retrieve_files_fn(
                repo,
                task,
                max_files=max_files,
                index_mode=effective_mode,
            )
            if (
                not _is_internal_workspace_path(path, repo)
                and not _is_context_internal_artifact(path, repo)
            )
        ]
        if effective_mode == "ai":
            task_hint_analyzer = QwenTaskHintAnalyzer()
            semantic_hints = task_hint_analyzer.hints(repo, task, graph_files)
            semantic_hint_status = task_hint_analyzer.last_status
            semantic_target_result = resolve_semantic_hints(repo, semantic_hints)
            semantic_targets = semantic_target_result.files
            semantic_target_reasons = semantic_target_result.reasons
    except Exception:
        graph_files = []

    changed_files = get_changed_files_fn(repo)
    explicit_files = _resolve_task_file_references(repo, task)
    ambiguous_files = _ambiguous_task_file_paths(repo, task)

    # Graph retrieval is the semantic/Qwen seed in AI mode and a useful
    # deterministic seed in static mode.  In both cases it is deliberately
    # expanded by the same structural ranker rather than treated as authority.
    seed_files = list(dict.fromkeys([
        *explicit_targets, *surface_result.files, *semantic_targets, *graph_files,
    ]))
    # Dirty state is preservation metadata, not relevance evidence. A changed
    # file participates here only when another retrieval signal selected it.
    transport_result = resolve_transport_roots(repo, seed_files)
    seed_files = list(dict.fromkeys([*seed_files, *transport_result.files]))
    hybrid = expand_candidates(repo, task, seed_files, limit=max_files)
    for path, reasons in explicit_target_reasons.items():
        hybrid.reasons[path] = list(dict.fromkeys([
            *reasons,
            *hybrid.reasons.get(path, []),
        ]))
    for path, reasons in surface_result.reasons.items():
        hybrid.reasons[path] = list(dict.fromkeys([
            *reasons, *hybrid.reasons.get(path, []),
        ]))
    for path, reasons in semantic_target_reasons.items():
        hybrid.reasons[path] = list(dict.fromkeys([
            *reasons,
            *hybrid.reasons.get(path, []),
        ]))
    for path, reasons in transport_result.reasons.items():
        hybrid.reasons[path] = list(dict.fromkeys([
            *reasons, *hybrid.reasons.get(path, []),
        ]))
    callsite_closure = resolve_test_callsite_closure(repo, task, [*hybrid.files, *graph_files])
    implementation_roots: dict[Path, list[str]] = {}
    for result in (
        locals().get("explicit_target_result"),
        surface_result,
        locals().get("semantic_target_result"),
        transport_result,
        callsite_closure,
    ):
        if result is None:
            continue
        for path, symbols in result.required_symbols.items():
            implementation_roots.setdefault(path, [])
            implementation_roots[path] = list(dict.fromkeys([
                *implementation_roots[path],
                *symbols,
            ]))
    closure_focus = {
        symbol
        for symbols in implementation_roots.values()
        for symbol in symbols
    }
    closure = (
        implementation_closure(
            repo,
            task,
            [*hybrid.files, *graph_files],
            closure_focus,
            root_symbols=implementation_roots,
        )
        if implementation_roots else RetrievalResult([])
    )
    for path in callsite_closure.files:
        if path not in hybrid.files:
            hybrid.files.append(path)
        hybrid.reasons[path] = list(dict.fromkeys([
            *callsite_closure.reasons.get(path, []), *hybrid.reasons.get(path, []),
        ]))
    for path in closure.files:
        if path not in hybrid.files:
            hybrid.files.append(path)
        hybrid.reasons[path] = list(dict.fromkeys([
            *closure.reasons.get(path, []), *hybrid.reasons.get(path, []),
        ]))
    completeness_files: list[Path] = []
    if effective_mode == "ai":
        completeness = QwenCompletenessChecker().check(repo, task, hybrid)
        for path in completeness.files:
            completeness_files.append(path)
            label = completeness.reasons[path]
            if label not in hybrid.reasons.setdefault(path, []):
                hybrid.reasons[path].append(label)
            if path not in hybrid.files:
                hybrid.files.append(path)

    changed_source_files = sorted(
        path for path in changed_files
        if path.is_file() and is_source_file(path)
        and not _is_context_internal_artifact(path, repo)
    )
    changed_definition_roots = _changed_python_definition_roots(
        repo, changed_source_files
    )
    # Dirty state is normally preservation metadata, not relevance evidence.
    # Keep the legacy overflow safeguard: when the dirty working tree itself
    # exceeds the normal selection cap, retaining every dirty source file
    # avoids silently dropping user changes. Smaller unrelated dirty sets still
    # require an independent task or retrieval signal.
    preserve_dirty_overflow = len(changed_source_files) > max_files
    selected: list[Path] = [
        path
        for path in changed_source_files
        if preserve_dirty_overflow and path not in ambiguous_files
    ]
    for path in explicit_files:
        if path not in selected:
            selected.append(path)
    for path in callsite_closure.files:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in explicit_targets:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in surface_result.files:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in semantic_targets:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in transport_result.files:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in closure.files:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    selection_limit = max(max_files, len(selected))
    # Completeness additions have been specifically confirmed after the first
    # ranking pass, so reserve their place ahead of lower-ranked seed results.
    for path in [*completeness_files, *hybrid.files]:
        if path.is_file() and is_source_file(path) and path not in ambiguous_files and path not in selected:
            selected.append(path)
        if len(selected) >= selection_limit:
            break
    target_symbols: dict[Path, list[str]] = {}
    for source in (
        value for value in (
            surface_result,
            locals().get("explicit_target_result"),
            locals().get("semantic_target_result"),
            transport_result,
            closure,
            callsite_closure,
        ) if value is not None
    ):
        for path, symbols in source.required_symbols.items():
            if path in selected:
                target_symbols.setdefault(path, [])
                target_symbols[path] = list(dict.fromkeys([
                    *target_symbols[path], *symbols,
                ]))
    for path, symbols in changed_definition_roots.items():
        if path not in selected:
            continue
        target_symbols.setdefault(path, [])
        target_symbols[path] = list(dict.fromkeys([
            *target_symbols[path], *symbols,
        ]))
    if get_boolean_setting("CHATCODE_RETRIEVAL_DEBUG"):
        print("Retrieval diagnostics:", file=os.sys.stderr)
        print(
            "Runtime modules: " + runtime_module_file + " | " + QwenTaskHintAnalyzer.__module__,
            file=os.sys.stderr,
        )
        if task_hint_analyzer is not None:
            print(f"Task-hint vocabulary: count {len(task_hint_analyzer.last_vocabulary)}", file=os.sys.stderr)
            print("Vocabulary sample: " + ", ".join(task_hint_analyzer.last_vocabulary[:30]), file=os.sys.stderr)
            print("Raw Qwen task-hint response: " + (task_hint_analyzer.last_raw_response or "<none>"), file=os.sys.stderr)
            print("Parsed symbol hints: " + (", ".join(task_hint_analyzer.last_parsed_hints) or "none"), file=os.sys.stderr)
            print("Normalized symbol hints: " + (", ".join(task_hint_analyzer.last_normalized_hints) or "none"), file=os.sys.stderr)
            if task_hint_analyzer.last_rejections:
                print("Rejected symbol hints: " + "; ".join(task_hint_analyzer.last_rejections), file=os.sys.stderr)
        print(
            "Qwen symbol hints (" + semantic_hint_status + "): "
            + (", ".join(semantic_hints) if semantic_hints else "none"),
            file=os.sys.stderr,
        )
        if semantic_target_reasons:
            print("Resolved semantic symbols:", file=os.sys.stderr)
            for path in sorted(semantic_target_reasons, key=lambda value: str(value).lower()):
                print(f"- {path.relative_to(repo)}: {', '.join(semantic_target_reasons[path])}", file=os.sys.stderr)
        if closure.reasons:
            print("Implementation closure additions:", file=os.sys.stderr)
            for path in sorted(closure.reasons, key=lambda value: str(value).lower()):
                print(f"- {path.relative_to(repo)}: {', '.join(closure.reasons[path])}", file=os.sys.stderr)
        if callsite_closure.reasons:
            print("Deterministic call-site roots and expansion:", file=os.sys.stderr)
            for diagnostic in callsite_closure.diagnostics:
                print(f"- {diagnostic}", file=os.sys.stderr)
            for path in sorted(callsite_closure.reasons, key=lambda value: str(value).lower()):
                print(f"- {path.relative_to(repo)}: {', '.join(callsite_closure.reasons[path])}", file=os.sys.stderr)
        if surface_result.diagnostics:
            print("Task surfaces:", file=os.sys.stderr)
            for diagnostic in surface_result.diagnostics:
                print(f"- {diagnostic}", file=os.sys.stderr)
        for path, symbols in target_symbols.items():
            print(f"Required symbols: {path.relative_to(repo)}::{', '.join(symbols)}", file=os.sys.stderr)
        for path in selected:
            labels = ", ".join(hybrid.reasons.get(path, ["explicit or changed file"]))
            print(f"- {path.relative_to(repo)}: {labels}", file=os.sys.stderr)
    return (selected, target_symbols) if include_target_symbols else selected


def build_source_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
    *,
    max_files: int,
    max_file_chars: int,
    max_total_chars: int,
    collect_relevant_files_fn: Callable,
    redact_sensitive_text_fn: Callable[[str], str],
) -> str:
    files = collect_relevant_files_fn(
        repo,
        task,
        index_progress=index_progress,
    )

    sections: list[str] = []
    total_chars = 0

    for path in files:
        try:
            content, _digest = _read_current_text_and_hash(path)
        except OSError:
            continue

        content = redact_sensitive_text_fn(
            content
        )

        if len(content) > max_file_chars:
            content = (
                content[:max_file_chars]
                + "\n\n"
                + "[File truncated by ChatCode]"
            )

        relative = path.relative_to(repo)

        section = (
            f"\n===== FILE: {relative} =====\n\n"
            f"{content}\n"
        )

        if (
            total_chars + len(section)
            > max_total_chars
        ):
            break

        sections.append(section)
        total_chars += len(section)

    if not sections:
        return (
            "No relevant source files found."
        )

    return "\n".join(sections)


def build_context_from_test_roots(
    repo: Path,
    test_ids: frozenset[str],
    traceback_paths: list[Path] | None = None,
    *,
    update_project_map_fn: Callable,
    build_stable_patch_source_context_fn: Callable,
) -> tuple[str, list[Path]]:
    """Build verified current source from exact failing-test roots.

    This deliberately skips natural-language/semantic discovery.  Test IDs
    seed the same deterministic call-site closure and hardened materializer
    used by normal patch contexts.
    """
    roots: list[Path] = []
    required: dict[Path, list[str]] = {}
    for test_id in sorted(test_ids):
        if "::" in test_id:
            raw_path, *parts = test_id.split("::")
            candidate = repo / raw_path
            symbol = parts[-1] if parts else ""
        else:
            parts = test_id.rsplit(".", 2)
            module = parts[0] if len(parts) == 3 else test_id.split(".", 1)[0]
            candidate = repo / (module.replace(".", "/") + ".py")
            if not candidate.is_file():
                candidate = repo / "tests" / (module.replace(".", "/") + ".py")
            symbol = parts[-1] if len(parts) == 3 else ""
        if candidate.is_file() and candidate not in roots:
            roots.append(candidate)
        if candidate.is_file() and symbol:
            required.setdefault(candidate, []).append(symbol)

    # Keep index-derived ownership current while retaining the known test as
    # the only retrieval root.
    try:
        update_project_map_fn(repo, paths=_index_scope_paths(repo))
    except Exception:
        pass
    # The closure's task matcher works on identifier tokens. Split test IDs
    # (including snake_case names) so an exact test method is never skipped.
    closure_task = " ".join(
        token
        for test_id in sorted(test_ids)
        for token in re.findall(r"[A-Za-z0-9]+", test_id)
    )
    closure = resolve_test_callsite_closure(repo, closure_task, roots)
    files = list(dict.fromkeys([*roots, *closure.files]))
    for path, symbols in closure.required_symbols.items():
        required.setdefault(path, []).extend(symbols)
    for path in traceback_paths or []:
        try:
            resolved = path.resolve()
            resolved.relative_to(repo.resolve())
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved not in files:
            files.append(resolved)
    required = {path: list(dict.fromkeys(symbols)) for path, symbols in required.items()}
    source, _hashes = build_stable_patch_source_context_fn(
        repo, closure_task, files, required
    )
    return source, files
