"""Shared deterministic source evidence for retrieval planners.

This module owns source-definition extraction, lexical matching, path-role
classification, and evidence scoring shared by source coverage and context
contract planning. It performs no repository-wide discovery and no model calls.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace
from pathlib import Path

from chatcode.indexing.generic_structure import (
    GENERIC_SYMBOL,
    GENERIC_SYMBOL_SUFFIXES,
    generic_symbol_span,
)
from chatcode.retrieval.task_semantics import surface_tokens


GENERIC_SUFFIXES = set(GENERIC_SYMBOL_SUFFIXES)

ROLE_TRIGGERS = {
    "persistence": frozenset({
        "database", "db", "persist", "persistence", "schema", "storage",
        "store", "save", "saved", "load", "loaded", "table", "migration",
        "sqlite", "repository", "lagra", "lagring", "spara", "sparas",
        "ladda", "tabell", "migrera",
    }),
    "transport": frozenset({
        "api", "endpoint", "route", "request", "response", "http", "handler",
        "controller", "server", "transport",
    }),
    "ui": frozenset({
        "ui", "frontend", "dashboard", "dialog", "modal", "editor", "button",
        "form", "component", "render", "display", "view", "panel", "library",
        "gränssnitt", "knapp", "visa",
    }),
    "behavior": frozenset({
        "behavior", "behaviour", "gameplay", "mechanic", "mechanics", "lock",
        "lockpick", "door", "command", "validation", "validate", "interaction",
        "beteende", "mekanik", "lås", "dörr", "validera",
    }),
    "tests": frozenset({
        "test", "tests", "testing", "spec", "regression", "tester", "testa",
        "regressionstest",
    }),
    "domain": frozenset({
        "model", "entity", "state", "type", "instance", "template", "world",
        "room", "inventory", "item", "container", "object", "connection",
        "modell", "entitet", "tillstånd", "typ", "instans", "mall", "rum",
    }),
}

PATH_ROLE_TERMS = {
    "persistence": frozenset({"database", "db", "repository", "storage", "store", "persistence"}),
    "transport": frozenset({"api", "route", "routes", "controller", "server", "transport"}),
    "ui": frozenset({"dashboard", "frontend", "editor", "ui", "component", "components", "web"}),
    "behavior": frozenset({"command", "commands", "mechanic", "mechanics", "interaction", "world"}),
    "tests": frozenset({"test", "tests", "spec", "specs"}),
    "domain": frozenset({"world", "model", "models", "service", "services", "entity", "entities", "inventory"}),
}

ROLE_SYMBOL_TERMS = {
    "persistence": frozenset({
        "init", "initialize", "schema", "migrate", "migration", "setup", "ensure",
        "create", "insert", "save", "store", "persist", "load", "read", "get",
        "list", "update", "delete", "remove", "execute",
    }),
    "transport": frozenset({
        "handle", "dispatch", "route", "request", "response", "serialize",
        "deserialize", "parse", "node", "data", "get", "post", "put", "patch",
        "delete", "options",
    }),
    "ui": frozenset({
        "editor", "modal", "dialog", "form", "panel", "inspector", "library",
        "contents", "content", "add", "create", "render", "room", "item",
        "container", "button", "submit", "open", "close",
    }),
    "behavior": frozenset({
        "lock", "lockpick", "validate", "validation", "command", "action", "door",
        "trap", "interaction", "can", "check",
    }),
    "tests": frozenset({"test", "spec", "fixture", "assert"}),
    "domain": frozenset({
        "entity", "kind", "model", "room", "world", "inventory", "holder", "item",
        "container", "template", "connection", "lock", "state", "type", "instance",
        "create", "update", "delete", "place", "move", "transfer",
    }),
}

ACTION_TERMS = frozenset({
    "add", "allow", "change", "create", "delete", "edit", "implement", "make",
    "migrate", "new", "remove", "replace", "reuse", "save", "support", "update",
    "use", "when", "with", "without", "should", "same", "existing", "current",
    "inspect", "roughly", "similar", "general", "generic", "future", "later",
    "lägg", "ändra", "skapa", "ta", "stöd", "använd", "ska", "samma", "befintlig",
    "befintliga", "nuvarande", "återanvänd", "återanvända", "ungefär", "senare",
})


COMMON_TERMS = frozenset({
    "and", "are", "but", "can", "for", "from", "have", "into", "not", "that",
    "the", "their", "them", "then", "this", "through", "with", "without", "your",
    "att", "det", "den", "och", "för", "från", "har", "inte", "med", "ska", "som",
    "till", "utan", "vara", "blir", "även", "samt",
})

SQL_MARKERS = (
    "create table", "alter table", "pragma ", "sqlite_master", "create index",
    "drop table", "foreign key",
)


@dataclass(frozen=True)
class DefinitionEvidence:
    path: Path
    name: str
    kind: str
    text: str
    line_count: int
    owner: str | None = None
    qualified_name: str | None = None
    ambiguous: bool = False

    @property
    def materialization_name(self) -> str:
        """Return a source locator that stays safe when method names collide."""
        if self.ambiguous and self.qualified_name:
            return self.qualified_name
        return self.name



def raw_tokens(text: str) -> set[str]:
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text.replace("_", " "))
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", split)
        if len(token) >= 3
    }


def tokens(text: str) -> set[str]:
    return surface_tokens(
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text.replace("_", " "))
    )


def term_matches(term: str, token: str) -> bool:
    if term == token:
        return True
    if min(len(term), len(token)) < 4:
        return False
    return token.startswith(term) or term.startswith(token)


def matching_terms(terms: set[str], tokens: set[str]) -> set[str]:
    return {
        term
        for term in terms
        if any(term_matches(term, token) for token in tokens)
    }


def text_matching_terms(terms: set[str], text: str) -> set[str]:
    lowered = text.casefold()
    return {
        term
        for term in terms
        if term in lowered
    }


def path_roles(path: Path, repo: Path) -> set[str]:
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError:
        relative = path.as_posix()
    path_tokens = tokens(relative)
    roles = {
        role
        for role, role_terms in PATH_ROLE_TERMS.items()
        if matching_terms(set(role_terms), path_tokens)
    }
    suffix = path.suffix.casefold()
    if suffix in {".tsx", ".jsx"}:
        roles.add("ui")
    if "transport" in roles and suffix == ".py":
        explicit_ui_terms = {"frontend", "editor", "ui", "component", "components", "web"}
        if not matching_terms(explicit_ui_terms, path_tokens):
            roles.discard("ui")
    if (
        path.name.casefold().startswith("test_")
        or path.name.casefold().endswith(("_test.py", ".test.ts", ".spec.ts", ".test.js", ".spec.js"))
    ):
        roles.add("tests")
    return roles


def definition_text(lines: list[str], node: ast.AST) -> tuple[str, int]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start = min(start, *(item.lineno for item in decorators))
    return "".join(lines[start - 1:end]), max(1, end - start + 1)


def python_definitions(path: Path, source: str) -> list[DefinitionEvidence]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines(keepends=True)
    raw: list[DefinitionEvidence] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        text, line_count = definition_text(lines, node)
        raw.append(DefinitionEvidence(
            path,
            node.name,
            "class" if isinstance(node, ast.ClassDef) else "function",
            text,
            line_count,
            qualified_name=node.name,
        ))
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            member_text, member_lines = definition_text(lines, member)
            raw.append(DefinitionEvidence(
                path,
                member.name,
                "class" if isinstance(member, ast.ClassDef) else "method",
                member_text,
                member_lines,
                owner=node.name,
                qualified_name=f"{node.name}.{member.name}",
            ))
    counts: dict[str, int] = {}
    for definition in raw:
        counts[definition.name] = counts.get(definition.name, 0) + 1
    # Preserve ambiguous methods instead of deleting their evidence. Their
    # qualified locator is used only when an unqualified name would be unsafe.
    return [
        replace(definition, ambiguous=counts[definition.name] > 1)
        for definition in raw
    ]


def generic_definitions(path: Path, source: str) -> list[DefinitionEvidence]:
    lines = source.splitlines(keepends=True)
    definitions: list[DefinitionEvidence] = []
    for match in GENERIC_SYMBOL.finditer(source):
        name = (
            match.group("named")
            or match.group("type_name")
            or match.group("binding")
        )
        if not name:
            continue
        span = generic_symbol_span(source, name)
        if span is None:
            continue
        kind = match.group("kind") or ("type" if match.group("type_name") else "function")
        start, end = span
        text = "".join(lines[start - 1:end])
        definitions.append(
            DefinitionEvidence(path, name, kind, text, max(1, end - start + 1))
        )
    return definitions


def definitions(path: Path, source: str) -> list[DefinitionEvidence]:
    suffix = path.suffix.casefold()
    if suffix == ".py":
        return python_definitions(path, source)
    if suffix in GENERIC_SUFFIXES:
        return generic_definitions(path, source)
    return []



def domain_terms(task: str) -> set[str]:
    role_terms = {
        term
        for role in ("persistence", "transport", "ui", "tests")
        for term in ROLE_TRIGGERS[role]
    }
    terms = raw_tokens(task)
    return {
        term
        for term in terms
        if term not in ACTION_TERMS
        and term not in COMMON_TERMS
        and term not in role_terms
        and len(term) >= 4
    }


def file_corroboration(
    path: Path,
    repo: Path,
    source: str,
    definitions: list[DefinitionEvidence],
    domain_terms: set[str],
) -> set[str]:
    try:
        relative = path.relative_to(repo).as_posix()
    except ValueError:
        relative = path.as_posix()
    names = " ".join(definition.name for definition in definitions)
    evidence = f"{relative}\n{names}\n{source[:180_000]}"
    return text_matching_terms(domain_terms, evidence)


def kind_score(
    kind: str,
    definition: DefinitionEvidence,
    file_roles: set[str],
    path_tokens: set[str],
    domain_terms: set[str],
    existing: set[str],
) -> int:
    name_tokens = tokens(definition.name)
    body_lower = definition.text.casefold()
    name_hits = matching_terms(domain_terms, name_tokens)
    body_hits = text_matching_terms(domain_terms, body_lower)
    path_hits = matching_terms(domain_terms, path_tokens)

    # Evidence kinds represent architectural roles, not generic lexical hits.
    # Reject obviously wrong surfaces before scoring so names such as
    # ChartContainer cannot satisfy a domain/persistence requirement merely
    # because they contain one task noun.
    if kind == "model":
        if definition.kind not in {"class", "interface", "type", "enum"}:
            return -1_000
        core_model_terms = {
            "entity", "kind", "room", "world", "inventory", "holder", "item",
            "container", "state", "type", "model", "connection",
        }
        if not name_hits and not matching_terms(core_model_terms, name_tokens):
            return -1_000
        if name_tokens & {"api", "commands", "command", "database", "service"}:
            return -1_000
    elif kind == "schema":
        if "persistence" not in file_roles or definition.kind not in {"function", "method"}:
            return -1_000
        strong_schema_name = matching_terms(
            {"init", "initialize", "schema", "migrate", "migration", "setup"},
            name_tokens,
        )
        if not strong_schema_name and not any(marker in body_lower for marker in SQL_MARKERS):
            return -1_000
    elif kind == "persistence_crud":
        if "persistence" not in file_roles or definition.kind not in {"function", "method"}:
            return -1_000
        crud_name = matching_terms(
            {"create", "insert", "save", "store", "persist", "load", "read", "get",
             "list", "update", "delete", "remove"},
            name_tokens,
        )
        if not crud_name or not (name_hits or body_hits):
            return -1_000
    elif kind == "transport":
        if "transport" not in file_roles or definition.kind not in {"function", "method"}:
            return -1_000
    elif kind == "ui":
        if "ui" not in file_roles:
            return -1_000
    elif kind == "domain_flow":
        if "domain" not in file_roles or definition.kind not in {"function", "method"}:
            return -1_000
        if not (name_hits or body_hits):
            return -1_000
    elif kind == "tests":
        if "tests" not in file_roles or definition.kind not in {"function", "method"}:
            return -1_000
        if not (name_hits or body_hits):
            return -1_000
    elif kind == "behavior":
        if definition.kind not in {"function", "method"}:
            return -1_000
        behavior_name = matching_terms(set(ROLE_SYMBOL_TERMS["behavior"]), name_tokens)
        if not behavior_name and not (
            any(term in body_lower for term in ("lock", "door", "validate", "lockpick"))
            and (name_hits or body_hits)
        ):
            return -1_000

    score = 150 * len(name_hits) + 24 * min(5, len(body_hits)) + 35 * len(path_hits)
    if definition.name in existing:
        score += 90

    if kind == "schema":
        if any(marker in body_lower for marker in SQL_MARKERS):
            score += 280
        score += 180 * len(matching_terms(
            {"init", "initialize", "schema", "migrate", "migration", "setup"},
            name_tokens,
        ))
    elif kind == "persistence_crud":
        score += 55 * len(matching_terms(set(ROLE_SYMBOL_TERMS["persistence"]), name_tokens))
    elif kind == "transport":
        score += 55 * len(matching_terms(set(ROLE_SYMBOL_TERMS["transport"]), name_tokens))
        if definition.name.casefold() in {"handle", "dispatch", "_node_data", "node_data"}:
            score += 180
    elif kind == "ui":
        score += 45 * len(matching_terms(set(ROLE_SYMBOL_TERMS["ui"]), name_tokens))
        if not (name_hits or body_hits):
            score -= 130
    elif kind == "behavior":
        score += 65 * len(matching_terms(set(ROLE_SYMBOL_TERMS["behavior"]), name_tokens))
    elif kind == "model":
        score += 120
        score += 45 * len(matching_terms(set(ROLE_SYMBOL_TERMS["domain"]), name_tokens))
    elif kind == "domain_flow":
        score += 60
        score += 35 * len(matching_terms(
            set(ROLE_SYMBOL_TERMS["domain"] | ROLE_SYMBOL_TERMS["persistence"]),
            name_tokens,
        ))
    elif kind == "tests":
        score += 45 * len(matching_terms(set(ROLE_SYMBOL_TERMS["tests"]), name_tokens))

    required_role = {
        "schema": "persistence",
        "persistence_crud": "persistence",
        "transport": "transport",
        "ui": "ui",
        "model": "domain",
        "domain_flow": "domain",
        "tests": "tests",
    }.get(kind)
    if required_role is not None and required_role in file_roles:
        score += 100

    # Huge enclosing definitions can consume the whole context budget. Prefer
    # precise methods and small models when the evidence is otherwise similar.
    if definition.line_count > 500:
        score -= 220
    elif definition.line_count > 250:
        score -= 90
    if definition.kind == "class" and definition.line_count > 350 and not name_hits:
        score -= 180
    return score


