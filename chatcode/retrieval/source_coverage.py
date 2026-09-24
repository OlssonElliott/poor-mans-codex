"""Plan exact source evidence for broad multi-layer maintenance tasks.

Retrieval answers which files are relevant.  This module answers the narrower
question that matters before publishing patch context: which complete existing
definitions inside those files are needed to understand the requested change.

The planner is deterministic and bounded.  It never invents paths, never adds
files outside the already selected set, and performs no model calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from chatcode.indexing.project_graph import load_map
from chatcode.retrieval.evidence import (
    GENERIC_SUFFIXES,
    ROLE_TRIGGERS,
    DefinitionEvidence,
    definitions as extract_definitions,
    domain_terms as extract_domain_terms,
    file_corroboration,
    kind_score as score_definition,
    matching_terms as match_terms,
    path_roles as infer_path_roles,
    tokens as evidence_tokens,
)
from chatcode.retrieval.task_semantics import (
    task_is_complex,
    task_surfaces,
)


MAX_COVERAGE_FILES = 8
MAX_COVERAGE_SYMBOLS = 28
MAX_COVERAGE_SYMBOLS_PER_FILE = 6
LIFECYCLE_TERMS = frozenset({
    "add", "create", "edit", "update", "delete", "remove", "save", "load", "migrate",
    "lägg", "skapa", "ändra", "spara", "ladda", "migrera",
})

REFERENCE_TERMS = frozenset({
    "existing", "current", "same", "similar", "reuse", "pattern", "inspect",
    "befintlig", "befintliga", "nuvarande", "samma", "återanvänd", "ungefär",
})

@dataclass
class SourceCoveragePlan:
    symbols: dict[Path, list[str]] = field(default_factory=dict)
    reasons: dict[tuple[Path, str], str] = field(default_factory=dict)
    paths: list[Path] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def _task_roles(task: str, files: list[Path], repo: Path, requirements: list[str]) -> set[str]:
    tokens = evidence_tokens(task)
    roles = {
        role
        for role, triggers in ROLE_TRIGGERS.items()
        if match_terms(set(triggers), tokens)
    }
    roles.add("domain")

    path_roles = {role for path in files for role in infer_path_roles(path, repo)}
    lifecycle = bool(tokens & LIFECYCLE_TERMS)
    if lifecycle and "persistence" in path_roles:
        roles.add("persistence")
    if "ui" in roles and lifecycle and "transport" in path_roles:
        roles.add("transport")
    if any(evidence_tokens(requirement) & REFERENCE_TERMS for requirement in requirements):
        roles.add("domain")
    return roles


def _evidence_kinds(roles: set[str]) -> list[tuple[str, int]]:
    kinds: list[tuple[str, int]] = [("model", 4), ("domain_flow", 6)]
    if "persistence" in roles:
        kinds.extend((("schema", 2), ("persistence_crud", 4)))
    if "transport" in roles:
        kinds.append(("transport", 4))
    if "ui" in roles:
        kinds.append(("ui", 5))
    if "behavior" in roles:
        kinds.append(("behavior", 4))
    if "tests" in roles:
        kinds.append(("tests", 3))
    return kinds


def plan_source_coverage(
    repo: Path,
    task: str,
    files: list[Path],
    existing_targets: dict[Path, list[str]] | None = None,
) -> SourceCoveragePlan:
    """Return exact definitions needed to cover a broad task's source layers.

    The planner only promotes definitions from files retrieval already selected.
    That keeps project-wide discovery bounded and leaves path trust with the
    existing retriever while fixing under-materialization inside large files.
    """
    requirements = task_surfaces(task)
    if not task_is_complex(task, requirements):
        return SourceCoveragePlan(requirements=requirements)

    existing_targets = existing_targets or {}
    domain_terms = extract_domain_terms(task)
    roles = _task_roles(task, files, repo, requirements)
    min_corroboration = 2 if len(domain_terms) >= 4 else 1

    try:
        index = load_map(repo).get("files", {})
    except Exception:
        index = {}

    evidence: list[
        tuple[Path, str, set[str], set[str], list[DefinitionEvidence]]
    ] = []
    for path in files:
        if not path.is_file() or path.suffix.casefold() not in ({".py"} | GENERIC_SUFFIXES):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        definitions = extract_definitions(path, source)
        if not definitions:
            continue
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
        pathtokens = evidence_tokens(relative + " " + semantic)
        file_roles = infer_path_roles(path, repo)
        corroboration = file_corroboration(
            path, repo, source, definitions, domain_terms
        )
        has_existing = bool(existing_targets.get(path))
        if (
            len(corroboration) < min_corroboration
            and not (has_existing and corroboration)
            and not ("tests" in file_roles and "tests" in roles and corroboration)
        ):
            continue
        evidence.append((path, source, pathtokens, file_roles, definitions))

    plan = SourceCoveragePlan(requirements=requirements)
    per_file: dict[Path, int] = {}
    selected_keys: set[tuple[Path, str]] = set()

    def add(definition: DefinitionEvidence, reason: str) -> bool:
        symbol = definition.materialization_name
        key = (definition.path, symbol)
        if key in selected_keys:
            return False
        if len(selected_keys) >= MAX_COVERAGE_SYMBOLS:
            return False
        if per_file.get(definition.path, 0) >= MAX_COVERAGE_SYMBOLS_PER_FILE:
            return False
        if definition.path not in plan.paths and len(plan.paths) >= MAX_COVERAGE_FILES:
            return False
        selected_keys.add(key)
        per_file[definition.path] = per_file.get(definition.path, 0) + 1
        if definition.path not in plan.paths:
            plan.paths.append(definition.path)
        plan.symbols.setdefault(definition.path, []).append(symbol)
        plan.reasons[key] = reason
        return True

    for kind, limit in _evidence_kinds(roles):
        ranked: list[tuple[int, DefinitionEvidence]] = []
        for path, _source, pathtokens, file_roles, definitions in evidence:
            existing = set(existing_targets.get(path, []))
            for definition in definitions:
                score = score_definition(
                    kind,
                    definition,
                    file_roles,
                    pathtokens,
                    domain_terms,
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
        added = 0
        for score, definition in ranked:
            if add(definition, f"source coverage {kind} (score {score})"):
                added += 1
            if added >= limit:
                break
        plan.diagnostics.append(f"coverage {kind}: {added}/{limit}")

    return plan
