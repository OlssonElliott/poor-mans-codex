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
    symbols: dict[Path, list[str]] = field(default_factory=dict)
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
    """Promote concrete evidence for each explicit requirement.

    Existing retrieval roots and source-coverage roots are always preserved.
    The extra requirement pass is bounded, deterministic, and limited to files
    retrieval already selected so the stable-source hash invariant stays intact.
    """
    requirements = list(coverage_plan.requirements)
    if not _task_is_complex(task, requirements):
        return ContextContractPlan(requirements=requirements)

    plan = ContextContractPlan(active=True, requirements=requirements)
    selected_keys: set[tuple[Path, str]] = set()
    per_file: dict[Path, int] = {}

    def add(path: Path, symbol: str, *, force: bool = False) -> bool:
        key = (path, symbol)
        if key in selected_keys:
            return False
        if not force:
            if len(selected_keys) >= MAX_CONTRACT_SYMBOLS:
                return False
            if per_file.get(path, 0) >= MAX_CONTRACT_SYMBOLS_PER_FILE:
                return False
        selected_keys.add(key)
        per_file[path] = per_file.get(path, 0) + 1
        if path not in plan.paths:
            plan.paths.append(path)
        plan.symbols.setdefault(path, []).append(symbol)
        return True

    for source in (existing_targets, coverage_plan.symbols):
        for path, symbols in source.items():
            for symbol in symbols:
                add(path, symbol, force=True)

    try:
        index = load_map(repo).get("files", {})
    except Exception:
        index = {}
    evidence = _evidence(repo, files, index)
    task_terms = _domain_terms(task)

    for requirement in requirements:
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
                accepted = key in selected_keys or add(
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

    return plan
