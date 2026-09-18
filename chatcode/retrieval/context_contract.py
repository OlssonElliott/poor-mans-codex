"""Requirement-level source contract planning for broad maintenance tasks.

The retrieval and source-coverage stages identify useful project evidence. This
module turns that evidence into a stricter contract: every explicit requirement
gets a bounded chance to claim concrete definitions from the files retrieval
already selected. It does not invent paths and performs no model calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from chatcode.indexing.project_graph import load_map
from chatcode.retrieval.hybrid_retriever import _task_is_complex
from chatcode.retrieval.source_coverage import (
    ACTION_TERMS,
    COMMON_TERMS,
    GENERIC_SUFFIXES,
    ROLE_TRIGGERS,
    DefinitionEvidence,
    SourceCoveragePlan,
    _definitions,
    _domain_terms,
    _kind_score,
    _matching_terms,
    _path_roles,
    _tokens,
)


MAX_CONTRACT_SYMBOLS = 64
MAX_CONTRACT_SYMBOLS_PER_FILE = 16
MAX_REQUIREMENT_UNITS = 8
SUPPORTED_SUFFIXES = {".py"} | GENERIC_SUFFIXES

REQUIREMENT_KIND_LIMITS = {
    "model": 2,
    "domain_flow": 2,
    "schema": 1,
    "persistence_crud": 2,
    "transport": 2,
    "ui": 2,
    "behavior": 2,
    "tests": 2,
}


@dataclass
class ContextContractPlan:
    active: bool = False
    requirements: list[str] = field(default_factory=list)
    # Compatibility union used by diagnostics and optional source focusing.
    symbols: dict[Path, list[str]] = field(default_factory=dict)
    # Only requirement-owned source may veto publication.
    mandatory_symbols: dict[Path, list[str]] = field(default_factory=dict)
    # Retrieval/coverage roots remain useful evidence but never veto publication.
    support_symbols: dict[Path, list[str]] = field(default_factory=dict)
    requirement_symbols: dict[str, dict[Path, list[str]]] = field(default_factory=dict)
    paths: list[Path] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def _requirement_roles(requirement: str) -> set[str]:
    tokens = _tokens(requirement)
    roles = {
        role
        for role, triggers in ROLE_TRIGGERS.items()
        if _matching_terms(set(triggers), tokens)
    }
    if not roles:
        roles.add("domain")
    return roles


def _requirement_kinds(requirement: str) -> list[str]:
    roles = _requirement_roles(requirement)
    kinds: list[str] = []
    if "domain" in roles:
        kinds.extend(("model", "domain_flow"))
    if "persistence" in roles:
        kinds.extend(("schema", "persistence_crud"))
    if "transport" in roles:
        kinds.append("transport")
    if "ui" in roles:
        kinds.append("ui")
    if "behavior" in roles:
        kinds.append("behavior")
    if "tests" in roles:
        kinds.append("tests")
    return list(dict.fromkeys(kinds))


def _requirement_terms(requirement: str, task_terms: set[str]) -> set[str]:
    terms = _domain_terms(requirement)
    if terms:
        return terms
    role_terms = {
        term
        for triggers in ROLE_TRIGGERS.values()
        for term in triggers
    }
    lexical = _tokens(requirement) - set(ACTION_TERMS) - set(COMMON_TERMS) - role_terms
    return lexical or task_terms


def _requirement_is_support_only(requirement: str) -> bool:
    """Return True for constraints/reference clauses that should not own patches."""
    text = " ".join(requirement.casefold().split())
    constraint_fragments = (
        "do not change",
        "do not modify",
        "do not implement",
        "do not convert",
        "do not turn",
        "do not use",
        "don't change",
        "don't modify",
        "don't implement",
        "without changing",
        "separate from",
        "ändra inte",
        "implementera inte",
        "använd inte",
        "gör inte om",
        "separat från",
    )
    if any(fragment in text for fragment in constraint_fragments):
        return True
    if "inspect first" in text or "inspektera först" in text:
        return True
    pattern_terms = ("pattern", "patterns", "style", "stil", "mönster")
    if ("follow" in text or "följ" in text) and any(
        term in text for term in pattern_terms
    ):
        return True
    return False


def _path_tokens(path: Path, repo: Path, index: dict) -> set[str]:
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError:
        relative = path.as_posix()
    entry = index.get(relative, {}) if isinstance(index, dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    semantic = " ".join([
        str(entry.get("summary", "")),
        *(str(tag) for tag in entry.get("tags", []) if isinstance(tag, str)),
    ])
    return _tokens(relative + " " + semantic)


def _evidence(
    repo: Path,
    files: list[Path],
    index: dict,
) -> list[tuple[Path, set[str], set[str], list[DefinitionEvidence]]]:
    rows: list[tuple[Path, set[str], set[str], list[DefinitionEvidence]]] = []
    for path in files:
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        definitions = _definitions(path, source)
        if not definitions:
            continue
        rows.append((
            path,
            _path_tokens(path, repo, index),
            _path_roles(path, repo),
            definitions,
        ))
    return rows


def plan_context_contract(
    repo: Path,
    task: str,
    files: list[Path],
    existing_targets: dict[Path, list[str]],
    coverage_plan: SourceCoveragePlan,
) -> ContextContractPlan:
    """Promote concrete mandatory evidence for each explicit requirement.

    Retrieval roots and broad source-coverage roots remain support evidence.
    They may focus optional context, but only definitions actually claimed by
    an implementation requirement are allowed to veto context publication.
    """
    requirements = list(coverage_plan.requirements)
    if not _task_is_complex(task, requirements):
        return ContextContractPlan(requirements=requirements)

    plan = ContextContractPlan(active=True, requirements=requirements)
    selected_keys: set[tuple[Path, str]] = set()
    union_keys: set[tuple[Path, str]] = set()
    per_file: dict[Path, int] = {}

    def add_union(path: Path, symbol: str) -> None:
        key = (path, symbol)
        if key in union_keys:
            return
        union_keys.add(key)
        if path not in plan.paths:
            plan.paths.append(path)
        plan.symbols.setdefault(path, []).append(symbol)

    def add_support(path: Path, symbol: str) -> None:
        add_union(path, symbol)
        bucket = plan.support_symbols.setdefault(path, [])
        if symbol not in bucket:
            bucket.append(symbol)

    def add_required(path: Path, symbol: str) -> bool:
        key = (path, symbol)
        if key in selected_keys:
            return False
        if len(selected_keys) >= MAX_CONTRACT_SYMBOLS:
            return False
        if per_file.get(path, 0) >= MAX_CONTRACT_SYMBOLS_PER_FILE:
            return False
        selected_keys.add(key)
        per_file[path] = per_file.get(path, 0) + 1
        add_union(path, symbol)
        plan.mandatory_symbols.setdefault(path, []).append(symbol)
        return True

    # Retrieval and broad coverage are candidate evidence, not proof that the
    # task requires those definitions. Requirement ownership below is the only
    # automatic promotion path to mandatory.
    for source in (existing_targets, coverage_plan.symbols):
        for path, symbols in source.items():
            for symbol in symbols:
                add_support(path, symbol)

    try:
        index = load_map(repo).get("files", {})
    except Exception:
        index = {}
    evidence = _evidence(repo, files, index)
    task_terms = _domain_terms(task)

    for requirement in requirements:
        if _requirement_is_support_only(requirement):
            plan.requirement_symbols[requirement] = {}
            plan.diagnostics.append(
                f"requirement support-only: {requirement}"
            )
            continue

        requirement_map: dict[Path, list[str]] = {}
        requirement_terms = _requirement_terms(requirement, task_terms)
        added_for_requirement = 0

        for kind in _requirement_kinds(requirement):
            ranked: list[tuple[int, DefinitionEvidence]] = []
            for path, path_tokens, file_roles, definitions in evidence:
                existing = set(plan.symbols.get(path, []))
                for definition in definitions:
                    score = _kind_score(
                        kind,
                        definition,
                        file_roles,
                        path_tokens,
                        requirement_terms,
                        existing,
                    )
                    if score >= 120:
                        ranked.append((score, definition))
            ranked.sort(
                key=lambda item: (
                    -item[0],
                    item[1].line_count,
                    item[1].path.as_posix().casefold(),
                    item[1].name.casefold(),
                )
            )

            kind_added = 0
            for score, definition in ranked:
                key = (definition.path, definition.name)
                accepted = key in selected_keys or add_required(
                    definition.path, definition.name
                )
                if not accepted:
                    continue
                requirement_map.setdefault(definition.path, [])
                if definition.name not in requirement_map[definition.path]:
                    requirement_map[definition.path].append(definition.name)
                    added_for_requirement += 1
                    kind_added += 1
                plan.diagnostics.append(
                    f"requirement evidence {kind}: {definition.path.name}::"
                    f"{definition.name} (score {score})"
                )
                if (
                    kind_added >= REQUIREMENT_KIND_LIMITS.get(kind, 1)
                    or added_for_requirement >= MAX_REQUIREMENT_UNITS
                ):
                    break
            if added_for_requirement >= MAX_REQUIREMENT_UNITS:
                break

        clean_map = {
            path: list(dict.fromkeys(symbols))
            for path, symbols in requirement_map.items()
            if symbols
        }
        plan.requirement_symbols[requirement] = clean_map
        if not clean_map:
            plan.diagnostics.append(f"requirement evidence unresolved: {requirement}")

    # Coverage already proved these definitions are important architectural
    # evidence. If a requirement has made the file itself mandatory, keep the
    # coverage definitions in the mandatory block too so they are rendered
    # before unrelated selected source. Coverage on support-only files remains
    # non-blocking and cannot veto publication.
    mandatory_paths = set(plan.mandatory_symbols)
    for path, symbols in coverage_plan.symbols.items():
        if path not in mandatory_paths:
            continue
        bucket = plan.mandatory_symbols.setdefault(path, [])
        for symbol in symbols:
            if symbol in bucket:
                continue
            bucket.append(symbol)
            add_union(path, symbol)
            plan.diagnostics.append(
                f"mandatory-path coverage: {path.name}::{symbol}"
            )

    # A requirement-owned definition is mandatory, not support. Keep the two
    # classes disjoint so publication checks cannot accidentally see retrieval
    # noise as a veto again.
    for path in list(plan.support_symbols):
        mandatory = set(plan.mandatory_symbols.get(path, []))
        kept = [
            symbol
            for symbol in plan.support_symbols[path]
            if symbol not in mandatory
        ]
        if kept:
            plan.support_symbols[path] = kept
        else:
            del plan.support_symbols[path]

    return plan
