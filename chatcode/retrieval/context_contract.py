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

OWNER_SURFACE_LIMITS = {
    "transport": 4,
    "ui": 2,
    "tests": 3,
}
TRANSPORT_DISPATCH_NAMES = {
    "handle",
    "_handle",
    "dispatch",
    "_dispatch",
    "route",
    "_route",
    "do_get",
    "do_post",
    "do_put",
    "do_patch",
    "do_delete",
    "do_options",
}
TRANSPORT_SERIALIZER_NAMES = {
    "_node_data",
    "node_data",
    "_graph_data",
    "graph_data",
    "serialize",
    "_serialize",
}
UI_OWNER_TERMS = {
    "editor",
    "inspector",
    "page",
    "view",
    "panel",
    "form",
    "dialog",
}
TEST_SETUP_NAMES = {
    "setup",
    "setup_method",
    "setup_class",
    "setUp",
}
TEST_ACTION_TERMS = {
    "create",
    "add",
    "update",
    "edit",
    "delete",
    "remove",
    "get",
    "list",
    "route",
    "api",
    "room",
}


@dataclass
class ContextContractPlan:
    active: bool = False
    requirements: list[str] = field(default_factory=list)
    # Compatibility union used by diagnostics and optional source focusing.
    symbols: dict[Path, list[str]] = field(default_factory=dict)
    # Only requirement-owned source may veto publication.
    mandatory_symbols: dict[Path, list[str]] = field(default_factory=dict)
    # High-priority coverage support renders before ordinary support, but does
    # not participate in the publication veto.
    priority_symbols: dict[Path, list[str]] = field(default_factory=dict)
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


def _owner_surface_score(
    role: str,
    definition: DefinitionEvidence,
    path_tokens: set[str],
    owner_terms: set[str],
) -> int:
    """Score complete patch owners separately from ordinary evidence."""
    lowered = definition.name.casefold()
    name_tokens = _tokens(definition.name)
    body_lower = definition.text.casefold()
    name_hits = _matching_terms(owner_terms, name_tokens)
    path_hits = _matching_terms(owner_terms, path_tokens)
    body_hits = {
        term for term in owner_terms
        if term in body_lower
    }

    if role == "transport":
        if definition.kind not in {"function", "method"}:
            return -1_000
        if lowered in TRANSPORT_DISPATCH_NAMES:
            score = 1_300
        elif lowered in TRANSPORT_SERIALIZER_NAMES:
            score = 1_000
        else:
            return -1_000
        return (
            score
            + 180 * len(name_hits)
            + 45 * min(5, len(body_hits))
            + 60 * len(path_hits)
        )

    if role == "ui":
        if definition.kind not in {"function", "class"}:
            return -1_000
        owner_hits = _matching_terms(UI_OWNER_TERMS, name_tokens)
        path_owner_hits = _matching_terms(name_tokens, path_tokens)
        if not owner_hits:
            return -1_000
        if not (name_hits or body_hits or path_hits or path_owner_hits):
            return -1_000
        score = (
            360 * len(owner_hits)
            + 260 * len(name_hits)
            + 50 * min(6, len(body_hits))
            + 80 * len(path_hits)
            + 180 * len(path_owner_hits)
        )
        if "editor" in name_tokens or "inspector" in name_tokens:
            score += 180
        return score

    if role == "tests":
        if (
            definition.kind not in {"function", "method"}
            or not lowered.startswith("test")
        ):
            return -1_000
        action_hits = _matching_terms(TEST_ACTION_TERMS, name_tokens)
        if not (name_hits or body_hits):
            return -1_000
        return (
            320 * len(name_hits)
            + 55 * min(8, len(body_hits))
            + 90 * len(action_hits)
            + 50 * len(path_hits)
        )

    return -1_000


def _owner_surface_candidates(
    requirement: str,
    requirement_terms: set[str],
    task_terms: set[str],
    preferred_paths: set[Path],
    evidence: list[tuple[Path, set[str], set[str], list[DefinitionEvidence]]],
) -> list[tuple[str, int, DefinitionEvidence]]:
    """Return bounded owners that must be patchable for this requirement."""
    roles = _requirement_roles(requirement)
    owner_terms = requirement_terms | task_terms
    selected: list[tuple[str, int, DefinitionEvidence]] = []

    for role in ("transport", "ui", "tests"):
        if role not in roles:
            continue
        ranked: list[tuple[int, DefinitionEvidence]] = []
        for path, path_tokens, file_roles, definitions in evidence:
            if role not in file_roles:
                continue
            # Do not turn owner expansion into broad retrieval. Transport and
            # UI owners must already have concrete evidence from retrieval or
            # source coverage. Tests may be claimed from already selected test
            # files because test roots are deliberately excluded elsewhere.
            if role != "tests" and path not in preferred_paths:
                continue
            for definition in definitions:
                score = _owner_surface_score(
                    role,
                    definition,
                    path_tokens,
                    owner_terms,
                )
                if score >= 0:
                    ranked.append((score, definition))
        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1].line_count,
                item[1].path.as_posix().casefold(),
                item[1].name.casefold(),
            )
        )
        selected.extend(
            (role, score, definition)
            for score, definition in ranked[:OWNER_SURFACE_LIMITS[role]]
        )

    return selected


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

    def add_priority(path: Path, symbol: str) -> None:
        add_union(path, symbol)
        if symbol in plan.mandatory_symbols.get(path, []):
            return
        bucket = plan.priority_symbols.setdefault(path, [])
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
    # Support-only clauses must not leak nouns into owner scoring for an
    # unrelated implementation requirement.
    implementation_terms: set[str] = set()
    for candidate_requirement in requirements:
        if _requirement_is_support_only(candidate_requirement):
            continue
        implementation_terms.update(_domain_terms(candidate_requirement))
    task_terms = implementation_terms or _domain_terms(task)

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

        # Requirement scoring finds evidence. Owner expansion answers the
        # separate question "which complete definition must be patchable?".
        # Reserve dispatchers, major UI owners, and representative API tests
        # before support context can consume the remaining budget.
        preferred_paths = (
            set(requirement_map)
            | set(existing_targets)
            | set(coverage_plan.symbols)
        )
        owner_candidates = _owner_surface_candidates(
            requirement,
            requirement_terms,
            task_terms,
            preferred_paths,
            evidence,
        )
        selected_test_paths: set[Path] = set()
        for role, score, definition in owner_candidates:
            key = (definition.path, definition.name)
            accepted = key in selected_keys or add_required(
                definition.path, definition.name
            )
            if not accepted:
                continue
            requirement_map.setdefault(definition.path, [])
            if definition.name not in requirement_map[definition.path]:
                requirement_map[definition.path].append(definition.name)
            if role == "tests":
                selected_test_paths.add(definition.path)
            plan.diagnostics.append(
                f"requirement owner {role}: {definition.path.name}::"
                f"{definition.name} (score {score})"
            )
            if role == "ui":
                owner_definitions = next(
                    (
                        definitions
                        for evidence_path, _path_tokens, _roles, definitions in evidence
                        if evidence_path == definition.path
                    ),
                    [],
                )
                bundle_added = 0
                for support in owner_definitions:
                    if support.kind not in {"type", "interface", "enum"}:
                        continue
                    if support.name not in definition.text:
                        continue
                    support_key = (support.path, support.name)
                    support_accepted = (
                        support_key in selected_keys
                        or add_required(support.path, support.name)
                    )
                    if not support_accepted:
                        continue
                    requirement_map.setdefault(support.path, [])
                    if support.name not in requirement_map[support.path]:
                        requirement_map[support.path].append(support.name)
                    plan.diagnostics.append(
                        f"requirement owner type: {support.path.name}::{support.name}"
                    )
                    bundle_added += 1
                    if bundle_added >= 8:
                        break

        # A route test is not useful patch context without its fixture owner.
        # Promote setup only for test files that actually supplied an owner.
        for path in selected_test_paths:
            definitions = next(
                (
                    definitions
                    for evidence_path, _path_tokens, _roles, definitions in evidence
                    if evidence_path == path
                ),
                [],
            )
            for definition in definitions:
                if definition.name not in TEST_SETUP_NAMES:
                    continue
                key = (definition.path, definition.name)
                accepted = key in selected_keys or add_required(
                    definition.path, definition.name
                )
                if not accepted:
                    continue
                requirement_map.setdefault(definition.path, [])
                if definition.name not in requirement_map[definition.path]:
                    requirement_map[definition.path].append(definition.name)
                plan.diagnostics.append(
                    f"requirement owner tests: {definition.path.name}::"
                    f"{definition.name} (setup)"
                )
                break

        clean_map = {
            path: list(dict.fromkeys(symbols))
            for path, symbols in requirement_map.items()
            if symbols
        }
        plan.requirement_symbols[requirement] = clean_map
        if not clean_map:
            plan.diagnostics.append(f"requirement evidence unresolved: {requirement}")

    # Coverage on a requirement-owned file is useful high-priority context,
    # but it is not itself proof that the task must patch that definition.
    # Keep it ahead of ordinary support without inflating the publication
    # contract or stealing budget from the actual owner bundle.
    mandatory_paths = set(plan.mandatory_symbols)
    for path, symbols in coverage_plan.symbols.items():
        if path not in mandatory_paths:
            continue
        for symbol in symbols:
            if symbol in plan.mandatory_symbols.get(path, []):
                continue
            add_priority(path, symbol)
            plan.diagnostics.append(
                f"priority-path coverage: {path.name}::{symbol}"
            )

    # Keep mandatory, priority and ordinary support disjoint. Publication
    # checks consume only mandatory_symbols.
    for path in list(plan.priority_symbols):
        mandatory = set(plan.mandatory_symbols.get(path, []))
        kept = [
            symbol
            for symbol in plan.priority_symbols[path]
            if symbol not in mandatory
        ]
        if kept:
            plan.priority_symbols[path] = kept
        else:
            del plan.priority_symbols[path]

    for path in list(plan.support_symbols):
        reserved = (
            set(plan.mandatory_symbols.get(path, []))
            | set(plan.priority_symbols.get(path, []))
        )
        kept = [
            symbol
            for symbol in plan.support_symbols[path]
            if symbol not in reserved
        ]
        if kept:
            plan.support_symbols[path] = kept
        else:
            del plan.support_symbols[path]

    return plan
