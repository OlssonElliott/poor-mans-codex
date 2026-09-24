"""Requirement-level source contract planning for broad maintenance tasks.

The retrieval and source-coverage stages identify useful project evidence. This
module turns that evidence into a stricter contract: every explicit requirement
gets a bounded chance to claim concrete definitions from the files retrieval
already selected. It does not invent paths and performs no model calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from chatcode.indexing.project_graph import load_map
from chatcode.retrieval.evidence import (
    ACTION_TERMS,
    COMMON_TERMS,
    GENERIC_SUFFIXES,
    ROLE_TRIGGERS,
    DefinitionEvidence,
    definitions as extract_definitions,
    domain_terms as extract_domain_terms,
    kind_score as score_definition,
    matching_terms as match_terms,
    path_roles as infer_path_roles,
    tokens as evidence_tokens,
)
from chatcode.retrieval.source_coverage import SourceCoveragePlan
from chatcode.retrieval.task_semantics import task_is_complex


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

STRUCTURAL_OWNER_SCORE = 10_000
STRUCTURAL_OWNER_ROLES = frozenset({"transport", "ui", "tests"})
EVIDENCE_ACTIVATED_OWNER_ROLES = frozenset({"transport", "tests"})
TRANSPORT_CORE_OWNER_ORDER = (
    "handle",
    "_handle",
    "dispatch",
    "_dispatch",
    "route",
    "_route",
)
TRANSPORT_HTTP_OWNER_ORDER = (
    "do_get",
    "do_post",
    "do_put",
    "do_patch",
    "do_delete",
    "do_options",
)
TRANSPORT_SERIALIZER_ORDER = (
    "_node_data",
    "node_data",
    "_graph_data",
    "graph_data",
    "serialize",
    "_serialize",
)
MAX_STRUCTURAL_UI_OWNERS = 4
MAX_STRUCTURAL_TEST_METHODS = 3
MAX_STRUCTURAL_TRANSPORT_HELPERS = 6


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
    tokens = evidence_tokens(requirement)
    roles = {
        role
        for role, triggers in ROLE_TRIGGERS.items()
        if match_terms(set(triggers), tokens)
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
    terms = extract_domain_terms(requirement)
    if terms:
        return terms
    role_terms = {
        term
        for triggers in ROLE_TRIGGERS.values()
        for term in triggers
    }
    lexical = evidence_tokens(requirement) - set(ACTION_TERMS) - set(COMMON_TERMS) - role_terms
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
    pathtokens: set[str],
    owner_terms: set[str],
) -> int:
    """Score complete patch owners separately from ordinary evidence."""
    lowered = definition.name.casefold()
    nametokens = evidence_tokens(definition.name)
    body_lower = definition.text.casefold()
    name_hits = match_terms(owner_terms, nametokens)
    path_hits = match_terms(owner_terms, pathtokens)
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
        owner_hits = match_terms(UI_OWNER_TERMS, nametokens)
        path_owner_hits = match_terms(nametokens, pathtokens)
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
        if "editor" in nametokens or "inspector" in nametokens:
            score += 180
        return score

    if role == "tests":
        if (
            definition.kind not in {"function", "method"}
            or not lowered.startswith("test")
        ):
            return -1_000
        action_hits = match_terms(TEST_ACTION_TERMS, nametokens)
        if not (name_hits or body_hits):
            return -1_000
        return (
            320 * len(name_hits)
            + 55 * min(8, len(body_hits))
            + 90 * len(action_hits)
            + 50 * len(path_hits)
        )

    return -1_000


def _mentions_symbol(text: str, symbol: str) -> bool:
    if not symbol:
        return False
    return re.search(
        rf"(?<![A-Za-z0-9_$]){re.escape(symbol)}(?![A-Za-z0-9_$])",
        text,
    ) is not None


def _definition_target(definition: DefinitionEvidence) -> str:
    """Return the exact source locator used by the materializer."""
    return definition.materialization_name


def _is_ui_owner(definition: DefinitionEvidence) -> bool:
    if definition.kind not in {"function", "class"}:
        return False
    nametokens = evidence_tokens(definition.name)
    return bool(
        match_terms(UI_OWNER_TERMS, nametokens)
        or definition.name[:1].isupper()
    )


def _effective_owner_roles(
    requirement: str,
    preferred_paths: set[Path],
    anchor_symbols: dict[Path, list[str]],
    evidence: list[tuple[Path, set[str], set[str], list[DefinitionEvidence]]],
) -> set[str]:
    """Combine wording roles with structural roles proven by selected files.

    Retrieval is allowed to discover transport and test surfaces even when the
    natural-language requirement describes only the user-facing UI. Once such
    a file is selected, anchored, and exposes the expected structural shape,
    wording must not veto owner materialization.
    """
    roles = _requirement_roles(requirement)
    missing = EVIDENCE_ACTIVATED_OWNER_ROLES - roles
    if not missing:
        return roles

    dispatch_names = {
        *TRANSPORT_CORE_OWNER_ORDER,
        *TRANSPORT_HTTP_OWNER_ORDER,
    }
    for path, _pathtokens, file_roles, definitions in evidence:
        if path not in preferred_paths or not anchor_symbols.get(path):
            continue

        eligible = missing & file_roles
        if "transport" in eligible and any(
            definition.kind in {"function", "method"}
            and definition.name.casefold() in dispatch_names
            for definition in definitions
        ):
            roles.add("transport")

        if "tests" in eligible and any(
            definition.kind in {"function", "method"}
            and definition.name.casefold().startswith("test")
            for definition in definitions
        ):
            roles.add("tests")

        if missing <= roles:
            break

    return roles


def _ui_owner_text(
    definition: DefinitionEvidence,
    definitions: list[DefinitionEvidence],
) -> str:
    """Rebuild a UI owner's evidence across nested local definitions.

    Generic JS/TS evidence is split at every regex-recognized declaration,
    including local arrow functions inside a component. Join those non-owner
    fragments back onto the component so parent/child references remain
    visible to structural owner resolution.
    """
    try:
        start = definitions.index(definition)
    except ValueError:
        return definition.text

    chunks = [definition.text]
    for trailing in definitions[start + 1:]:
        if _is_ui_owner(trailing):
            break
        chunks.append(trailing.text)
    return "".join(chunks)


def _structural_owner_candidates(
    requirement: str,
    owner_terms: set[str],
    preferred_paths: set[Path],
    anchor_symbols: dict[Path, list[str]],
    evidence: list[tuple[Path, set[str], set[str], list[DefinitionEvidence]]],
) -> tuple[list[tuple[str, int, DefinitionEvidence]], set[str]]:
    """Resolve known patch owners without making them compete on relevance score.

    Retrieval/scoring decides which files and anchors are relevant. Once a
    selected file exposes a known architectural surface, structural rules own
    the final mandatory-source decision.
    """
    roles = _effective_owner_roles(
        requirement,
        preferred_paths,
        anchor_symbols,
        evidence,
    )
    selected: list[tuple[str, int, DefinitionEvidence]] = []
    resolved_roles: set[str] = set()
    seen: set[tuple[str, Path, str]] = set()

    def add(role: str, definition: DefinitionEvidence) -> None:
        key = (role, definition.path, _definition_target(definition))
        if key in seen:
            return
        seen.add(key)
        selected.append((role, STRUCTURAL_OWNER_SCORE, definition))

    if "transport" in roles:
        rows = [
            row
            for row in evidence
            if row[0] in preferred_paths and "transport" in row[2]
        ]
        anchored = [row for row in rows if anchor_symbols.get(row[0])]
        candidate_rows = anchored or rows

        # Prefer the application-level dispatcher over lower HTTP adapters.
        # A selected DashboardAPI.handle/_handle surface owns route behavior;
        # request-handler do_GET/do_POST wrappers should only become structural
        # owners when no core dispatcher exists in the selected evidence.
        core_rows = [
            row
            for row in candidate_rows
            if any(
                definition.kind in {"function", "method"}
                and definition.name.casefold() in TRANSPORT_CORE_OWNER_ORDER
                for definition in row[3]
            )
        ]
        owner_rows = core_rows or candidate_rows

        for path, pathtokens, _file_roles, definitions in owner_rows[:2]:
            anchors = {
                symbol.casefold()
                for symbol in anchor_symbols.get(path, [])
            }
            callables = [
                definition
                for definition in definitions
                if definition.kind in {"function", "method"}
            ]
            owner_groups: dict[str | None, list[DefinitionEvidence]] = {}
            for definition in callables:
                owner_groups.setdefault(definition.owner, []).append(definition)

            ranked_owners: list[
                tuple[int, int, int, bool, str, list[DefinitionEvidence], list[str]]
            ] = []
            for owner, ownerdefinitions in owner_groups.items():
                names = {
                    definition.name.casefold()
                    for definition in ownerdefinitions
                }
                core = [
                    name for name in TRANSPORT_CORE_OWNER_ORDER
                    if name in names
                ]
                http = [
                    name for name in TRANSPORT_HTTP_OWNER_ORDER
                    if name in names
                ]
                dispatch = core or http
                if not dispatch:
                    continue
                anchor_hits = sum(
                    1
                    for symbol in anchors
                    if (
                        symbol == (owner or "").casefold()
                        or (
                            owner is not None
                            and symbol.startswith(owner.casefold() + ".")
                        )
                        or symbol in names
                    )
                )
                path_overlap = len(match_terms(
                    evidence_tokens(owner or ""),
                    pathtokens,
                ))
                ranked_owners.append((
                    anchor_hits,
                    path_overlap,
                    len(core),
                    owner is not None,
                    owner or "",
                    ownerdefinitions,
                    dispatch,
                ))

            if not ranked_owners:
                continue

            ranked_owners.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    -int(item[3]),
                    item[4].casefold(),
                )
            )
            (
                _anchor_hits,
                _path_overlap,
                _core_count,
                _owned,
                owner,
                ownerdefinitions,
                dispatch,
            ) = ranked_owners[0]
            owner_name = owner or None
            by_name = {
                definition.name.casefold(): definition
                for definition in ownerdefinitions
            }
            dispatchdefinitions: list[DefinitionEvidence] = []
            for name in dispatch:
                definition = by_name.get(name)
                if definition is None:
                    continue
                add("transport", definition)
                dispatchdefinitions.append(definition)

            if not dispatchdefinitions:
                continue

            # Known serializers belong to the transport patch surface. Prefer
            # the selected dispatcher owner, but allow an unambiguous top-level
            # serializer in the same file.
            for name in TRANSPORT_SERIALIZER_ORDER:
                candidates = [
                    definition
                    for definition in callables
                    if definition.name.casefold() == name
                ]
                owned = [
                    definition
                    for definition in candidates
                    if definition.owner == owner_name
                ]
                definition = (
                    owned[0] if len(owned) == 1
                    else candidates[0] if len(candidates) == 1
                    else None
                )
                if definition is not None:
                    add("transport", definition)

            # Follow directly referenced same-file helpers only when their
            # definition also carries requirement evidence. This recovers
            # route-specific serializers/helpers without turning _handle into
            # an unbounded closure over every endpoint in the file.
            dispatch_text = "\n".join(
                definition.text for definition in dispatchdefinitions
            )
            helper_count = 0
            serializer_names = set(TRANSPORT_SERIALIZER_ORDER)
            for candidate in callables:
                if candidate in dispatchdefinitions:
                    continue
                if candidate.owner not in {None, owner_name}:
                    continue
                if candidate.name.casefold() in serializer_names:
                    continue
                if not _mentions_symbol(dispatch_text, candidate.name):
                    continue
                name_hits = match_terms(
                    owner_terms,
                    evidence_tokens(candidate.name),
                )
                body_lower = candidate.text.casefold()
                body_hits = {
                    term for term in owner_terms
                    if term in body_lower
                }
                if not (name_hits or body_hits):
                    continue
                add("transport", candidate)
                helper_count += 1
                if helper_count >= MAX_STRUCTURAL_TRANSPORT_HELPERS:
                    break

            resolved_roles.add("transport")

    if "ui" in roles:
        rows = [
            row
            for row in evidence
            if row[0] in preferred_paths and "ui" in row[2]
        ]
        for path, _pathtokens, _file_roles, definitions in rows:
            anchors = {
                symbol.casefold()
                for symbol in anchor_symbols.get(path, [])
            }
            if not anchors:
                continue
            owners = [definition for definition in definitions if _is_ui_owner(definition)]
            owner_text = {
                definition: _ui_owner_text(definition, definitions)
                for definition in owners
            }
            seeds = [
                definition
                for definition in owners
                if definition.name.casefold() in anchors
            ]
            if not seeds:
                continue

            path_selected: list[DefinitionEvidence] = []
            for definition in seeds:
                if definition not in path_selected:
                    path_selected.append(definition)

            # One structural hop in both directions is enough to recover the
            # common parent/child component pair without walking every dialog
            # referenced by a large editor component.
            for candidate in owners:
                if candidate in path_selected:
                    continue
                if any(
                    _mentions_symbol(owner_text[candidate], seed.name)
                    or _mentions_symbol(owner_text[seed], candidate.name)
                    for seed in seeds
                ):
                    path_selected.append(candidate)
                if len(path_selected) >= MAX_STRUCTURAL_UI_OWNERS:
                    break

            for definition in path_selected[:MAX_STRUCTURAL_UI_OWNERS]:
                add("ui", definition)
            resolved_roles.add("ui")

    if "tests" in roles:
        rows = [
            row
            for row in evidence
            if "tests" in row[2]
        ]
        preferred = [row for row in rows if row[0] in preferred_paths]
        for path, pathtokens, _file_roles, definitions in (preferred or rows):
            matching_tests = [
                definition
                for definition in definitions
                if (
                    definition.kind in {"function", "method"}
                    and definition.name.casefold().startswith("test")
                    and (
                        match_terms(owner_terms, evidence_tokens(definition.name))
                        or any(term in definition.text.casefold() for term in owner_terms)
                    )
                )
            ]
            if not matching_tests:
                continue

            suites: dict[str | None, list[DefinitionEvidence]] = {}
            for definition in matching_tests:
                suites.setdefault(definition.owner, []).append(definition)
            ranked_suites = sorted(
                suites.items(),
                key=lambda item: (
                    -len(match_terms(
                        evidence_tokens(item[0] or ""),
                        pathtokens,
                    )),
                    -len(item[1]),
                    -len(match_terms(
                        owner_terms,
                        evidence_tokens(item[0] or ""),
                    )),
                    (item[0] or "").casefold(),
                ),
            )
            suite_owner, suite_tests = ranked_suites[0]
            for definition in suite_tests[:MAX_STRUCTURAL_TEST_METHODS]:
                add("tests", definition)

            # The suite class is structural identity, not required patch text.
            # Materialize its fixture and relevant methods instead of forcing
            # a potentially huge enclosing class into the hard contract.
            for definition in definitions:
                if (
                    definition.kind in {"function", "method"}
                    and definition.name in TEST_SETUP_NAMES
                    and definition.owner == suite_owner
                ):
                    add("tests", definition)
                    break
            resolved_roles.add("tests")

    return selected, resolved_roles


def _owner_surface_candidates(
    requirement: str,
    requirement_terms: set[str],
    task_terms: set[str],
    preferred_paths: set[Path],
    anchor_symbols: dict[Path, list[str]],
    evidence: list[tuple[Path, set[str], set[str], list[DefinitionEvidence]]],
) -> list[tuple[str, int, DefinitionEvidence]]:
    """Return bounded owners that must be patchable for this requirement."""
    owner_terms = requirement_terms | task_terms
    roles = _effective_owner_roles(
        requirement,
        preferred_paths,
        anchor_symbols,
        evidence,
    )
    selected, structurally_resolved = _structural_owner_candidates(
        requirement,
        owner_terms,
        preferred_paths,
        anchor_symbols,
        evidence,
    )

    for role in ("transport", "ui", "tests"):
        if role not in roles or role in structurally_resolved:
            continue
        ranked: list[tuple[int, DefinitionEvidence]] = []
        for path, pathtokens, file_roles, definitions in evidence:
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
                    pathtokens,
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


def _pathtokens(path: Path, repo: Path, index: dict) -> set[str]:
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
    return evidence_tokens(relative + " " + semantic)


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
        definitions = extract_definitions(path, source)
        if not definitions:
            continue
        rows.append((
            path,
            _pathtokens(path, repo, index),
            infer_path_roles(path, repo),
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
    if not task_is_complex(task, requirements):
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
        implementation_terms.update(extract_domain_terms(candidate_requirement))
    task_terms = implementation_terms or extract_domain_terms(task)

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
            for path, pathtokens, file_roles, definitions in evidence:
                existing = set(plan.symbols.get(path, []))
                for definition in definitions:
                    score = score_definition(
                        kind,
                        definition,
                        file_roles,
                        pathtokens,
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
                symbol = _definition_target(definition)
                key = (definition.path, symbol)
                if key in selected_keys:
                    accepted = True
                elif kind in STRUCTURAL_OWNER_ROLES:
                    # Scoring identifies useful evidence for owner-managed
                    # surfaces, but only structural owner resolution below may
                    # hard-claim it as mandatory source.
                    add_priority(definition.path, symbol)
                    accepted = True
                else:
                    accepted = add_required(
                        definition.path, symbol
                    )
                if not accepted:
                    continue
                requirement_map.setdefault(definition.path, [])
                if symbol not in requirement_map[definition.path]:
                    requirement_map[definition.path].append(symbol)
                    added_for_requirement += 1
                    kind_added += 1
                plan.diagnostics.append(
                    f"requirement evidence {kind}: {definition.path.name}::"
                    f"{symbol} (score {score})"
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
        owner_anchors: dict[Path, list[str]] = {}
        for path in preferred_paths:
            symbols = (
                existing_targets.get(path)
                or requirement_map.get(path)
                or coverage_plan.symbols.get(path)
                or []
            )
            if symbols:
                owner_anchors[path] = list(dict.fromkeys(symbols))
        owner_candidates = _owner_surface_candidates(
            requirement,
            requirement_terms,
            task_terms,
            preferred_paths,
            owner_anchors,
            evidence,
        )
        selected_test_owners: dict[Path, set[str | None]] = {}
        for role, score, definition in owner_candidates:
            symbol = _definition_target(definition)
            key = (definition.path, symbol)
            accepted = key in selected_keys or add_required(
                definition.path, symbol
            )
            if not accepted:
                continue
            requirement_map.setdefault(definition.path, [])
            if symbol not in requirement_map[definition.path]:
                requirement_map[definition.path].append(symbol)
            if role == "tests":
                selected_test_owners.setdefault(definition.path, set()).add(
                    definition.owner
                )
            owner_basis = (
                "structural"
                if score == STRUCTURAL_OWNER_SCORE
                else f"score {score}"
            )
            plan.diagnostics.append(
                f"requirement owner {role}: {definition.path.name}::"
                f"{symbol} ({owner_basis})"
            )
            if role == "ui":
                ownerdefinitions = next(
                    (
                        definitions
                        for evidence_path, _pathtokens, _roles, definitions in evidence
                        if evidence_path == definition.path
                    ),
                    [],
                )
                bundle_added = 0
                for support in ownerdefinitions:
                    if support.kind not in {"type", "interface", "enum"}:
                        continue
                    if support.name not in definition.text:
                        continue
                    support_symbol = _definition_target(support)
                    support_key = (support.path, support_symbol)
                    support_accepted = (
                        support_key in selected_keys
                        or add_required(support.path, support_symbol)
                    )
                    if not support_accepted:
                        continue
                    requirement_map.setdefault(support.path, [])
                    if support_symbol not in requirement_map[support.path]:
                        requirement_map[support.path].append(support_symbol)
                    plan.diagnostics.append(
                        f"requirement owner type: {support.path.name}::{support_symbol}"
                    )
                    bundle_added += 1
                    if bundle_added >= 8:
                        break

        # A route test is not useful patch context without the fixture from
        # its own suite. Qualified owners keep multiple setUp methods distinct.
        for path, owners in selected_test_owners.items():
            definitions = next(
                (
                    definitions
                    for evidence_path, _pathtokens, _roles, definitions in evidence
                    if evidence_path == path
                ),
                [],
            )
            for definition in definitions:
                if (
                    definition.name not in TEST_SETUP_NAMES
                    or definition.owner not in owners
                ):
                    continue
                symbol = _definition_target(definition)
                key = (definition.path, symbol)
                accepted = key in selected_keys or add_required(
                    definition.path, symbol
                )
                if not accepted:
                    continue
                requirement_map.setdefault(definition.path, [])
                if symbol not in requirement_map[definition.path]:
                    requirement_map[definition.path].append(symbol)
                plan.diagnostics.append(
                    f"requirement owner tests: {definition.path.name}::"
                    f"{symbol} (setup)"
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
