"""Patch-context assembly and publication orchestration."""
from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable
from pathlib import Path

from ..config import get_boolean_setting
from ..context_state import save_context_state
from ..git_utils import get_branch
from ..indexing.index_manager import IndexProgress
from ..retrieval.hybrid_retriever import resolve_explicit_targets
from ..workspace import atomic_write_text, get_repo_workspace
from .materializer import (
    PATCH_DEPENDENCY_RESERVE_RATIO,
    _context_budget_chars,
    _context_contract_blocks,
    _dependency_paths,
    _effective_context_budget,
    _enforce_materialization_invariant,
    _fresh_symbol_node,
    _materialization_targets,
    _render_context_contract_blocks,
    _render_patch_file_context,
    _render_required_symbols,
)


def build_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path] | None = None,
    target_symbols: dict[Path, list[str]] | None = None,
    critical_paths: list[Path] | None = None,
    *,
    collect_relevant_files_fn: Callable,
    get_changed_files_fn: Callable[[Path], set[Path]],
    resolve_task_file_references_fn: Callable[[Path, str], list[Path]],
    plan_source_coverage_fn: Callable,
    plan_context_contract_fn: Callable,
) -> str:
    if files is None:
        files = collect_relevant_files_fn(repo, task)

    changed_files = get_changed_files_fn(repo)
    explicit_files = set(resolve_task_file_references_fn(repo, task))
    base_target_metadata = (
        target_symbols or resolve_explicit_targets(repo, task).required_symbols
    )
    target_metadata = {
        path: list(dict.fromkeys(symbols))
        for path, symbols in base_target_metadata.items()
    }
    coverage_plan = plan_source_coverage_fn(
        repo,
        task,
        files,
        target_metadata,
    )
    contract_plan = plan_context_contract_fn(
        repo,
        task,
        files,
        target_metadata,
        coverage_plan,
    )
    # Keep every discovered root available for optional source focusing.
    for source in (coverage_plan.symbols, contract_plan.symbols):
        for path, symbols in source.items():
            target_metadata.setdefault(path, [])
            for symbol in symbols:
                if symbol not in target_metadata[path]:
                    target_metadata[path].append(symbol)

    base_materialization_targets = _materialization_targets(
        repo,
        base_target_metadata,
    )
    base_keys = {
        (path, symbol)
        for path, symbol, _priority in base_materialization_targets
    }
    coverage_targets: list[tuple[Path, str, str]] = []
    other_targets: list[tuple[Path, str, str]] = []

    if contract_plan.active:
        mandatory_keys = {
            (path, symbol)
            for path, symbols in contract_plan.mandatory_symbols.items()
            for symbol in symbols
        }
        # Direct build_patch_source_context callers historically use
        # target_symbols as an explicit contract. Production context capture
        # passes critical_paths, so retrieval candidates are not promoted here.
        if critical_paths is None:
            mandatory_keys.update(
                (path, symbol)
                for path, symbols in base_target_metadata.items()
                for symbol in symbols
            )
        # Exact paths named by the user are explicit contract surfaces.
        mandatory_keys.update(
            (path, symbol)
            for path, symbols in base_target_metadata.items()
            if path in explicit_files
            for symbol in symbols
        )

        seen_mandatory: set[tuple[Path, str]] = set()
        contract_owned = {
            (path, symbol)
            for path, symbols in contract_plan.mandatory_symbols.items()
            for symbol in symbols
        }
        for path, symbol, priority in base_materialization_targets:
            key = (path, symbol)
            if key in mandatory_keys:
                coverage_targets.append((
                    path,
                    symbol,
                    "requirement contract" if key in contract_owned else priority,
                ))
                seen_mandatory.add(key)
            else:
                other_targets.append((path, symbol, priority))
                target_metadata.setdefault(path, [])
                if symbol not in target_metadata[path]:
                    target_metadata[path].append(symbol)

        for path, symbols in contract_plan.mandatory_symbols.items():
            for symbol in symbols:
                key = (path, symbol)
                if key in seen_mandatory:
                    continue
                coverage_targets.append((path, symbol, "requirement contract"))
                seen_mandatory.add(key)

        # Only mandatory targets participate in the hard materialization
        # invariant. Support roots remain opportunistic below.
        materialization_targets = list(coverage_targets)
    else:
        coverage_keys = {
            (path, symbol)
            for path, symbols in coverage_plan.symbols.items()
            for symbol in symbols
        }
        for path, symbol, priority in base_materialization_targets:
            if (path, symbol) in coverage_keys:
                coverage_targets.append((path, symbol, "source coverage"))
            else:
                other_targets.append((path, symbol, priority))
        for path, symbols in coverage_plan.symbols.items():
            for symbol in symbols:
                if (path, symbol) not in base_keys and not any(
                    target_path == path and target_symbol == symbol
                    for target_path, target_symbol, _priority in coverage_targets
                ):
                    coverage_targets.append((path, symbol, "source coverage"))
        materialization_targets = [*coverage_targets, *other_targets]

    dependencies = _dependency_paths(repo, files)
    support_target_paths = (
        [path for path, _symbol, _priority in other_targets]
        if contract_plan.active else []
    )
    primary = list(dict.fromkeys([
        *[path for path in files if path in explicit_files],
        *[path for path in files if path in changed_files],
        *files,
        *support_target_paths,
    ]))
    supporting = [path for path in sorted(dependencies) if path not in primary]

    budget = _context_budget_chars()
    contract_blocks: list[tuple[str, list[tuple[Path, str, str]], str]] = []
    contract_unresolved: list[tuple[Path, str, str]] = []
    if contract_plan.active and coverage_targets:
        contract_blocks, contract_unresolved = _context_contract_blocks(
            repo, coverage_targets
        )
        contract_cost = sum(
            len(section) for section, _keys, _state in contract_blocks
        )
        budget = _effective_context_budget(budget, contract_cost)

    if contract_plan.active:
        required_paths = list(dict.fromkeys([
            *[path for path, _symbol, _priority in coverage_targets],
            *[path for path in primary if path in explicit_files],
        ]))
    else:
        required_paths = list(dict.fromkeys([
            *[
                path for path in primary
                if path in changed_files
                and "tests" not in {part.casefold() for part in path.relative_to(repo).parts}
                and "test" not in path.stem.casefold()
                and "spec" not in path.stem.casefold()
            ],
            *(critical_paths if critical_paths is not None else base_target_metadata),
        ]))

    coverage_context = ""
    coverage_states: dict[tuple[Path, str], tuple[str, str]] = {}
    if coverage_targets:
        if contract_plan.active:
            coverage_context, coverage_states = _render_context_contract_blocks(
                repo,
                contract_blocks,
                contract_unresolved,
                budget,
            )
        else:
            coverage_context, coverage_states = _render_required_symbols(
                repo,
                task,
                coverage_targets,
                coverage_plan.paths,
                budget,
                changed_files,
                explicit_files,
            )

        # Validate coverage against the exact fresh definition text. A fallback
        # excerpt that does not contain the complete node is not enough for a
        # patch target. Spend one bounded local repair pass on those gaps before
        # lower-priority selected files receive any context budget.
        missing_coverage: list[tuple[Path, str, str]] = []
        for path, symbol, priority in coverage_targets:
            node = _fresh_symbol_node(path, symbol)
            if node is None or node[0] not in coverage_context:
                missing_coverage.append((path, symbol, priority))

        coverage_rendered_chars = sum(
            len(section)
            for section in coverage_context.splitlines(keepends=True)
            if "REQUIRED SOURCE UNAVAILABLE:" not in section
        )
        repair_budget = max(0, budget - coverage_rendered_chars)
        if missing_coverage and repair_budget > 0:
            repair_context, repair_states = _render_required_symbols(
                repo,
                task,
                missing_coverage,
                list(dict.fromkeys(path for path, _symbol, _priority in missing_coverage)),
                repair_budget,
                changed_files,
                explicit_files,
            )
            if repair_context:
                repaired_keys: set[tuple[Path, str]] = set()
                for path, symbol, _priority in missing_coverage:
                    node = _fresh_symbol_node(path, symbol)
                    if node is not None and node[0] in repair_context:
                        repaired_keys.add((path, symbol))
                if repaired_keys:
                    marker_prefixes = {
                        f"===== REQUIRED SOURCE UNAVAILABLE: "
                        f"{path.relative_to(repo).as_posix()}::{symbol} "
                        for path, symbol in repaired_keys
                    }
                    coverage_context = "".join(
                        line
                        for line in coverage_context.splitlines(keepends=True)
                        if not any(line.startswith(prefix) for prefix in marker_prefixes)
                    )
                coverage_context = "\n".join(
                    part for part in (coverage_context, repair_context) if part
                )
                coverage_states.update(repair_states)

        for path, symbol, _priority in coverage_targets:
            node = _fresh_symbol_node(path, symbol)
            if node is None or node[0] not in coverage_context:
                coverage_states[(path, symbol)] = (
                    "unavailable",
                    "source coverage validation",
                )

    coverage_rendered_chars = sum(
        len(section)
        for section in coverage_context.splitlines(keepends=True)
        if "REQUIRED SOURCE UNAVAILABLE:" not in section
    )
    remaining_required_budget = max(0, budget - coverage_rendered_chars)
    if contract_plan.active:
        coverage_path_set = {
            path for path, _symbol, _priority in coverage_targets
        }
        other_required_paths = [
            path for path in required_paths
            if path not in coverage_path_set
        ]
        required_other_targets: list[tuple[Path, str, str]] = []
    else:
        coverage_path_set = set(coverage_plan.paths)
        other_required_paths = [
            path
            for path in required_paths
            if path not in coverage_path_set or path in changed_files
        ]
        required_other_targets = other_targets

    other_context, other_states = _render_required_symbols(
        repo,
        task,
        required_other_targets,
        other_required_paths,
        remaining_required_budget,
        changed_files,
        explicit_files,
    )
    mandatory_context = "\n".join(
        part for part in (coverage_context, other_context) if part
    )
    materialization_states = {**coverage_states, **other_states}

    mandatory_rendered_chars = sum(
        len(section) for section in mandatory_context.splitlines(keepends=True)
        if "REQUIRED SOURCE UNAVAILABLE:" not in section
    )
    priority_budget = max(0, budget - mandatory_rendered_chars)
    priority_sections: list[str] = []
    priority_used = 0
    if contract_plan.active and priority_budget > 0:
        for path, symbols in contract_plan.priority_symbols.items():
            relative = path.relative_to(repo).as_posix()
            for symbol in symbols:
                node = _fresh_symbol_node(path, symbol)
                if node is None:
                    continue
                section = (
                    f"===== SYMBOL CONTEXT: {relative}::{symbol} "
                    f"[priority support] source lines {node[2]}-{node[3]} =====\n"
                    f"{node[0]}\n"
                )
                if priority_used + len(section) > priority_budget:
                    continue
                priority_sections.append(section)
                priority_used += len(section)

    priority_context = "\n".join(priority_sections)
    required_context = "\n".join(
        part for part in (mandatory_context, priority_context) if part
    )
    required_rendered_chars = mandatory_rendered_chars + priority_used
    remaining_budget = max(0, budget - required_rendered_chars)
    dependency_reserve = int(remaining_budget * PATCH_DEPENDENCY_RESERVE_RATIO)
    primary_budget = max(0, remaining_budget - dependency_reserve)
    sections: list[str] = [required_context] if required_context else []
    used = 0

    # Selected files are possible patch targets. Mandatory paths were already
    # rendered above. Support roots are opportunistic and never veto publication
    # merely because their exact source does not fit.
    required_path_set = set(required_paths)
    for position, path in enumerate(primary):
        if (
            (contract_plan.active and path in required_path_set)
            or (not contract_plan.active and path in target_metadata)
        ):
            continue
        try:
            remaining = len(primary) - position
            allowance = max(1, (primary_budget - used) // remaining)
            section = _render_patch_file_context(
                repo,
                path,
                task,
                changed_files,
                explicit_files,
                max_chars=allowance,
                required_symbols=target_metadata.get(path),
                focus_symbols=target_metadata.get(path),
            )
        except OSError:
            continue
        if not section or used + len(section) > primary_budget:
            continue
        sections.append(section)
        used += len(section)

    for path in supporting:
        if (
            (contract_plan.active and path in required_path_set)
            or (not contract_plan.active and path in target_metadata)
        ):
            continue
        try:
            if contract_plan.active:
                allowance = max(1, remaining_budget - used)
                section = _render_patch_file_context(
                    repo,
                    path,
                    task,
                    changed_files,
                    explicit_files,
                    max_chars=allowance,
                    required_symbols=target_metadata.get(path),
                    focus_symbols=target_metadata.get(path),
                )
            else:
                section = _render_patch_file_context(
                    repo,
                    path,
                    task,
                    changed_files,
                    explicit_files,
                )
        except OSError:
            continue
        if not section or used + len(section) > remaining_budget:
            continue
        sections.append(section)
        used += len(section)

    rendered = "\n".join(sections) or "No relevant source files found."
    rendered = _enforce_materialization_invariant(
        repo, rendered, materialization_targets, materialization_states
    )
    if contract_plan.active:
        incomplete: list[str] = []
        for path, symbol, _priority in coverage_targets:
            state, detail = coverage_states.get(
                (path, symbol), ("unavailable", "missing state")
            )
            node = _fresh_symbol_node(path, symbol)
            exact_present = node is not None and node[0] in rendered
            full_present = (
                detail == "FULL FILE"
                and f"===== FULL FILE: {path.relative_to(repo).as_posix()} ====="
                in rendered
            )
            if state not in {"rendered", "fallback"} or not (
                exact_present or full_present
            ):
                incomplete.append(
                    f"{path.relative_to(repo).as_posix()}::{symbol}"
                )
        if incomplete:
            rendered += (
                "\n===== CONTEXT CONTRACT INCOMPLETE =====\n"
                + "\n".join(f"- {item}" for item in dict.fromkeys(incomplete))
                + "\n"
            )

    if get_boolean_setting("CHATCODE_RETRIEVAL_DEBUG") and contract_plan.active:
        print("Context contract planner:", file=os.sys.stderr)
        for diagnostic in contract_plan.diagnostics:
            print(f"- {diagnostic}", file=os.sys.stderr)
    if get_boolean_setting("CHATCODE_RETRIEVAL_DEBUG") and coverage_plan.symbols:
        print("Source coverage planner:", file=os.sys.stderr)
        for diagnostic in coverage_plan.diagnostics:
            print(f"- {diagnostic}", file=os.sys.stderr)
        for path in coverage_plan.paths:
            symbols = coverage_plan.symbols.get(path, [])
            if symbols:
                print(
                    f"- {path.relative_to(repo)}: {', '.join(symbols)}",
                    file=os.sys.stderr,
                )
    if get_boolean_setting("CHATCODE_RETRIEVAL_DEBUG") and materialization_states:
        print("Final materialization:", file=os.sys.stderr)
        priorities = {(path, symbol): priority for path, symbol, priority in materialization_targets}
        diagnostic_items = list(materialization_states.items())
        for (path, symbol), (state, detail) in diagnostic_items[:40]:
            node = _fresh_symbol_node(path, symbol)
            relative = path.relative_to(repo).as_posix()
            expected_header = (
                f"===== SYMBOL CONTEXT: {relative}::{symbol} "
                f"[{priorities[(path, symbol)]}] source lines {node[2]}-{node[3]} ====="
                if node is not None else ""
            )
            print(f"{path.relative_to(repo)}::{symbol}", file=os.sys.stderr)
            print(f"- priority: {priorities[(path, symbol)]}", file=os.sys.stderr)
            print(f"- fresh AST: {'yes' if node is not None else 'no'}", file=os.sys.stderr)
            if node is not None:
                print(f"- source lines: {node[2]}-{node[3]}", file=os.sys.stderr)
            print(f"- rendered: {'yes' if state == 'rendered' else 'no'}", file=os.sys.stderr)
            if state == "rendered":
                verified = expected_header in rendered and node is not None and node[0] in rendered
                print("- rendered block: SYMBOL CONTEXT", file=os.sys.stderr)
                print(f"- range verified: {'yes' if verified else 'no'}", file=os.sys.stderr)
            if state != "rendered":
                print(f"- reason: {detail}", file=os.sys.stderr)
                print("- status: non-patchable", file=os.sys.stderr)
        if len(diagnostic_items) > 40:
            print(
                f"- {len(diagnostic_items) - 40} lower-priority symbol states omitted",
                file=os.sys.stderr,
            )
    return rendered


def build_patch_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
    *,
    patch_context_header: str,
    upload_instructions: str,
    patch_response_instructions: str,
    build_safe_status_fn: Callable[[Path], str],
    collect_relevant_files_fn: Callable,
    ensure_explicit_task_files_fn: Callable[[Path, str, list[Path]], list[Path]],
    build_stable_patch_source_context_fn: Callable,
    assert_publishable_context_contract_fn: Callable[[Path, str], None],
    format_selected_files_fn: Callable[[list[Path], Path, str], str],
) -> Path:
    output_file = get_repo_workspace(repo) / "UPLOAD_TO_CHATGPT.md"
    repo_display = repo.resolve().as_posix()
    status = build_safe_status_fn(repo).replace("\\", "/")
    retrieved = collect_relevant_files_fn(
        repo,
        task,
        index_progress=index_progress,
        include_target_symbols=True,
    )
    files, target_symbols = retrieved if isinstance(retrieved, tuple) else (retrieved, {})
    files = ensure_explicit_task_files_fn(
        repo,
        task,
        files,
    )
    source_context, source_hashes = build_stable_patch_source_context_fn(
        repo,
        task,
        files,
        target_symbols,
    )
    assert_publishable_context_contract_fn(output_file, source_context)
    parts = [
        "# ChatCode Patch Context",
        "",
        patch_context_header,
        "",
        "## ChatGPT instructions",
        upload_instructions,
        "",
        "## Task (verbatim)",
        task,
        "",
        "## Repository root",
        repo_display,
        "",
        "## Current branch",
        get_branch(repo),
        "",
        "## Git status --short",
        status or "Working tree clean",
        "",
        "## Selected relevant files",
        format_selected_files_fn(files, repo, source_context),
        "",
        "## Exact current working-tree contents",
        source_context,
        "",
        "## Response instructions",
        patch_response_instructions,
        "All patch paths must use forward slashes (`/`).",
        "Patch only against the current contents above; do not reconstruct files from Git history or a separate diff.",
        "",
    ]
    content = "\n".join(parts)
    save_context_state(
        repo,
        task=task,
        source_hashes=source_hashes,
        context_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        generation_id=uuid.uuid4().hex,
    )
    # Publish last: watchers cannot observe this generation before its hashes.
    atomic_write_text(output_file, content, newline="\n")
    return output_file
