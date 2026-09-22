"""Conservative deterministic expansion around semantic retrieval seeds.

The project map is intentionally a lightweight index.  This module uses it as
evidence, not as a compiler: direct dependencies, reverse dependencies,
symbol definitions and tests are useful signals, but never automatic closure.
"""
from __future__ import annotations

import json
import ast
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from chatcode.config import get_setting
from chatcode.indexing.generic_structure import (
    GENERIC_SYMBOL,
    GENERIC_SYMBOL_SUFFIXES,
    generic_symbol_span,
)
from chatcode.indexing.project_graph import load_map


MAX_CANDIDATES = 36
MAX_COMPLETENESS_RESOLVED = 6
MAX_EXPLICIT_TARGETS = 8
MAX_SEMANTIC_HINTS = 20
MAX_CLOSURE_FILES = 8
MAX_ROOTED_CLOSURE_DEPTH = 3
MAX_ROOTED_CLOSURE_FILES = 12
MAX_ROOTED_CLOSURE_SYMBOLS = 24
MAX_TASK_SURFACES = 12
# A surface can have a small set of equally strong concrete definitions (for
# example renderer, view, presenter, and autocomplete). Keep this bounded,
# while allowing each independently promoted root to reach materialization.
MAX_SURFACE_ROOTS = 4
MAX_TRANSPORT_ROOTS = 2
MAX_SURFACE_SUPPORT_SYMBOLS = 4
GENERIC_PATH_PARTS = {"utils", "util", "common", "config", "settings", "constants", "helpers"}

REQUIREMENT_ROLE_TERMS = {
    "persistence": frozenset({
        "database", "db", "persist", "persistence", "schema", "storage", "store",
        "save", "saved", "load", "loaded", "table", "repository",
        "databas", "lagra", "lagring", "spara", "sparas", "ladda", "tabell",
    }),
    "transport": frozenset({
        "api", "endpoint", "route", "request", "response", "http", "handler",
        "controller", "server",
    }),
    "ui": frozenset({
        "ui", "frontend", "dashboard", "dialog", "modal", "editor", "button",
        "form", "component", "render", "display", "view", "panel",
        "gränssnitt", "knapp", "visa",
    }),
    "tests": frozenset({
        "test", "tests", "testing", "spec", "regression",
        "tester", "testa", "regressionstest",
    }),
    "domain": frozenset({
        "model", "entity", "service", "state", "flow", "rule", "behavior",
        "behaviour", "type", "instance", "template",
        "modell", "entitet", "tjänst", "flöde", "regel", "instans", "mall",
    }),
}

REQUIREMENT_ROLE_SYMBOL_TERMS = {
    "persistence": frozenset({
        "init", "initialize", "schema", "migrate", "migration", "create", "insert",
        "save", "store", "persist", "load", "read", "get", "list", "update",
        "delete", "remove",
    }),
    "transport": frozenset({
        "api", "route", "handler", "request", "response", "get", "post", "put",
        "patch", "delete", "options", "serialize", "deserialize",
    }),
    "ui": frozenset({
        "dialog", "modal", "editor", "form", "component", "render", "view",
        "panel", "button", "submit", "handle", "open", "close",
    }),
    "tests": frozenset({"test", "spec", "fixture", "assert"}),
    "domain": frozenset({
        "model", "entity", "service", "state", "flow", "rule", "create", "update",
        "delete", "place", "move", "transfer",
    }),
}

GENERIC_REQUIREMENT_TERMS = frozenset({
    "add", "allow", "change", "create", "edit", "make", "new", "replace",
    "support", "update", "use", "when", "with", "without", "should", "same",
    "lägg", "ändra", "skapa", "stöd", "använd", "ska", "samma",
})


@dataclass
class RetrievalResult:
    files: list[Path]
    reasons: dict[Path, list[str]] = field(default_factory=dict)
    candidates: list[Path] = field(default_factory=list)
    required_symbols: dict[Path, list[str]] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class CompletenessResult:
    files: list[Path] = field(default_factory=list)
    reasons: dict[Path, str] = field(default_factory=dict)


def _surface_tokens(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+",
            text.replace("_", " "),
        )
        if len(token) >= 3
    }
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 5:
            expanded.add(token[:-3] + "y")
        if token.endswith("ing") and len(token) > 5:
            expanded.add(token[:-3])
        if token.endswith("ed") and len(token) > 4:
            expanded.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            expanded.add(token[:-1])
    return expanded


def _surface_roles(tokens: set[str]) -> set[str]:
    return {
        role
        for role, trigger_terms in REQUIREMENT_ROLE_TERMS.items()
        if tokens & trigger_terms
    }


def _task_is_complex(task: str, surfaces: list[str] | None = None) -> bool:
    surfaces = surfaces if surfaces is not None else _task_surfaces(task)
    roles = {
        role
        for surface in surfaces
        for role in _surface_roles(_surface_tokens(surface))
    }
    return len(surfaces) >= 4 or len(task) >= 480 or len(roles) >= 3


def _related_role_symbols(
    entry: dict,
    task_tokens: set[str],
    roles: set[str],
    primary: str,
) -> list[str]:
    if not roles:
        return []
    domain_tokens = task_tokens - GENERIC_REQUIREMENT_TERMS
    role_symbol_terms = {
        term
        for role in roles
        for term in REQUIREMENT_ROLE_SYMBOL_TERMS.get(role, ())
    }
    important = {
        str(value).casefold()
        for value in entry.get("important_symbols", [])
        if isinstance(value, str)
    }
    ranked: list[tuple[int, str, str]] = []
    for symbol in entry.get("symbols", []):
        if not isinstance(symbol, dict) or not symbol.get("name"):
            continue
        name = str(symbol["name"])
        definition = str(symbol.get("definition_name") or name)
        if definition == primary:
            continue
        split_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
        symbol_tokens = _surface_tokens(split_name)
        score = 100 * len(domain_tokens & symbol_tokens)
        score += 45 * len(role_symbol_terms & symbol_tokens)
        if name.casefold() in important:
            score += 20
        if (
            "persistence" in roles
            and symbol_tokens
            & {"init", "initialize", "schema", "migrate", "migration", "setup"}
        ):
            score += 40
        if score >= 45:
            ranked.append((score, name, definition))
    ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
    return list(
        dict.fromkeys(item[2] for item in ranked)
    )[:MAX_SURFACE_SUPPORT_SYMBOLS]


def _task_surfaces(task: str) -> list[str]:
    """Split broad maintenance tasks into bounded explicit requirements."""
    normalized = re.sub(
        r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",
        "",
        task,
    )
    action_words = (
        r"add|allow|create|delete|display|edit|load|make|persist|remove|render|"
        r"replace|save|show|support|update|use|when|lägg|skapa|ta|visa|ändra|"
        r"spara|ladda|stöd"
    )
    clauses = re.split(
        rf"\s*(?:;|\n+|[.!?]+\s+|,\s+(?=(?:{action_words})\b)|"
        rf"\b(?:and|och)\s+(?=(?:{action_words})\b))\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    return list(
        dict.fromkeys(clause.strip() for clause in clauses if clause.strip())
    )[:MAX_TASK_SURFACES]


def _python_definition_literal_terms(path: Path) -> dict[str, dict[str, int]]:
    """Return user-visible literal words owned by each Python definition."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return {}
    result: dict[str, dict[str, int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        terms: dict[str, int] = {}
        docstring = ast.get_docstring(node, clean=False)
        for child in ast.walk(node):
            if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                continue
            if child.value == docstring:
                continue
            for word in re.findall(r"[A-Za-z0-9]+", child.value):
                if len(word) >= 3:
                    terms.setdefault(word.casefold(), 1)
        # Static fragments inside an f-string/template are stronger evidence
        # of displayed output than prose used only in an exception branch.
        for template in (child for child in ast.walk(node) if isinstance(child, ast.JoinedStr)):
            for part in template.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    for word in re.findall(r"[A-Za-z0-9]+", part.value):
                        if len(word) >= 3:
                            terms[word.casefold()] = 2
        if terms:
            result[node.name] = terms
    return result


_GENERIC_STRING_LITERAL = re.compile(
    r"""(?s)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)"""
)
_GENERIC_JSX_TEXT = re.compile(r">([^<>{}]+)<")


def _generic_definition_literal_terms(path: Path) -> dict[str, dict[str, int]]:
    """Return literal/JSX words owned by complete JS/TS definitions."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    lines = source.splitlines(keepends=True)
    result: dict[str, dict[str, int]] = {}
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
        start, end = span
        text = "".join(lines[start - 1:end])
        terms: dict[str, int] = {}
        for literal in _GENERIC_STRING_LITERAL.finditer(text):
            value = literal.group(0)[1:-1]
            for word in re.findall(r"[A-Za-z0-9]+", value):
                if len(word) >= 3:
                    terms.setdefault(word.casefold(), 1)
        for jsx_text in _GENERIC_JSX_TEXT.finditer(text):
            for word in re.findall(r"[A-Za-z0-9]+", jsx_text.group(1)):
                if len(word) >= 3:
                    terms[word.casefold()] = 2
        if terms:
            owned = result.setdefault(name, {})
            for word, weight in terms.items():
                owned[word] = max(weight, owned.get(word, 0))
    return result


def _definition_literal_terms(path: Path) -> dict[str, dict[str, int]]:
    if path.suffix.casefold() == ".py":
        return _python_definition_literal_terms(path)
    if path.suffix.casefold() in GENERIC_SYMBOL_SUFFIXES:
        return _generic_definition_literal_terms(path)
    return {}


def resolve_task_surface_roots(repo: Path, task: str) -> RetrievalResult:
    """Find concrete indexed roots for each explicit requirement in the task."""
    surfaces = _task_surfaces(task)
    complex_task = _task_is_complex(task, surfaces)
    # A single surface remains too broad for name-only promotion. Concrete
    # runtime text owned by a relevant definition is narrow enough to use.
    runtime_evidence_only = len(surfaces) < 2
    index = load_map(repo).get("files", {})
    paths: list[Path] = []
    reasons: dict[Path, list[str]] = defaultdict(list)
    required: dict[Path, list[str]] = defaultdict(list)
    diagnostics: list[str] = []
    literal_terms_by_file: dict[str, dict[str, dict[str, int]]] = {}
    presentation_terms = {"display", "displayed", "show", "render", "rendering", "view", "embed", "ui"}
    for surface in surfaces:
        tokens = _surface_tokens(surface)
        roles = _surface_roles(tokens) if complex_task else set()
        role_symbol_terms = {
            term
            for role in roles
            for term in REQUIREMENT_ROLE_SYMBOL_TERMS.get(role, ())
        }
        candidates: list[tuple[int, str, str]] = []
        for relative, entry in index.items():
            if not (repo / relative).is_file():
                continue
            # Surface roots are patchable implementation definitions. Tests
            # remain available through the dedicated test/call-site closure,
            # but cannot consume this bounded implementation-root set.
            if _is_test(relative):
                continue
            important = {
                str(value).casefold()
                for value in entry.get("important_symbols", [])
                if isinstance(value, str)
            }
            file_tokens = _surface_tokens(relative)
            semantic_tokens = _surface_tokens(
                " ".join([
                    str(entry.get("summary", "")),
                    *[
                        str(tag)
                        for tag in entry.get("tags", [])
                        if isinstance(tag, str)
                    ],
                ])
            )
            if relative not in literal_terms_by_file:
                literal_terms_by_file[relative] = _definition_literal_terms(
                    repo / relative
                )
            for symbol in entry.get("symbols", []):
                if not isinstance(symbol, dict) or not symbol.get("name"):
                    continue
                name = str(symbol["name"])
                symbol_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
                symbol_tokens = _surface_tokens(symbol_text)
                overlap = tokens & (symbol_tokens | file_tokens | semantic_tokens)
                role_symbol_overlap = role_symbol_terms & symbol_tokens
                definition_name = str(symbol.get("definition_name") or name)
                # Runtime examples in task/feedback are direct evidence for
                # the definition that owns their literal/template fragments.
                owned_literal_terms = literal_terms_by_file[relative].get(
                    definition_name, {}
                )
                literal_overlap = tokens & owned_literal_terms.keys()
                if not overlap and not role_symbol_overlap and not literal_overlap:
                    continue
                if runtime_evidence_only and not literal_overlap:
                    continue
                score = (
                    70 * len(tokens & symbol_tokens)
                    + 25 * len(tokens & file_tokens)
                    + 15 * len(tokens & semantic_tokens)
                )
                score += 240 * sum(owned_literal_terms[word] for word in literal_overlap)
                if name.casefold() in tokens:
                    score += 120
                if tokens & presentation_terms and symbol_tokens & presentation_terms:
                    score += 180
                if complex_task:
                    score += 45 * len(role_symbol_overlap)
                    file_role_terms = {
                        term
                        for role in roles
                        for term in REQUIREMENT_ROLE_TERMS.get(role, ())
                    }
                    score += 12 * len(file_role_terms & (file_tokens | semantic_tokens))
                if name.casefold() in important:
                    score += 80
                if score >= 70:
                    candidates.append((score, relative, definition_name))
        selected = sorted(
            candidates,
            key=lambda item: (-item[0], item[1].lower(), item[2].lower()),
        )[:MAX_SURFACE_ROOTS]
        if not selected:
            diagnostics.append(f"surface unresolved: {surface}")
            continue
        diagnostics.append(f"surface definition roots resolved: {surface}")
        for _score, relative, symbol in selected:
            path = repo / relative
            if path not in paths:
                paths.append(path)
            label = f"explicit task surface: {surface}"
            if label not in reasons[path]:
                reasons[path].append(label)
            if symbol not in required[path]:
                required[path].append(symbol)
            if not complex_task:
                continue
            for support_symbol in _related_role_symbols(
                index.get(relative, {}),
                tokens,
                roles,
                symbol,
            ):
                if support_symbol not in required[path] and len(required[path]) < 8:
                    required[path].append(support_symbol)
                    support_label = (
                        f"requirement support for {surface}: {support_symbol}"
                    )
                    if support_label not in reasons[path]:
                        reasons[path].append(support_label)
    return RetrievalResult(paths, dict(reasons), paths.copy(), dict(required), diagnostics)


def resolve_alternative_callback_roots(
    repo: Path, task: str, attempted_symbols: set[str], limit: int = 3,
) -> RetrievalResult:
    """Find bounded callback alternatives to an attempted implementation path."""
    if not attempted_symbols:
        return RetrievalResult([])
    index = load_map(repo).get("files", {})
    bindings: list[tuple[str, str, str, str]] = []
    for relative in index:
        path = repo / relative
        if not relative.casefold().endswith(".py") or not path.is_file() or _is_test(relative):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                decorator_name = (
                    decorator.func.attr if isinstance(decorator.func, ast.Attribute)
                    else decorator.func.id if isinstance(decorator.func, ast.Name) else ""
                )
                if not decorator_name.casefold().endswith("autocomplete"):
                    continue
                for keyword in decorator.keywords:
                    if keyword.arg and isinstance(keyword.value, ast.Name):
                        bindings.append((relative, node.name, keyword.arg, keyword.value.id))

    attempted_parameters = {
        parameter
        for _relative, _command, parameter, callback in bindings
        if callback in attempted_symbols
    }
    if not attempted_parameters:
        return RetrievalResult([])
    task_tokens = {
        token.casefold() for token in re.findall(r"[A-Za-z0-9]+", task)
        if len(token) >= 3
    }
    ranked: list[tuple[int, str, str, str]] = []
    for relative, command, parameter, callback in bindings:
        if callback in attempted_symbols or parameter not in attempted_parameters:
            continue
        callback_tokens = set(re.findall(r"[a-z0-9]+", callback.casefold().replace("_", " ")))
        command_tokens = set(re.findall(r"[a-z0-9]+", command.casefold().replace("_", " ")))
        literal_terms = _python_definition_literal_terms(repo / relative).get(callback, {})
        score = 300
        score += 70 * len(task_tokens & callback_tokens)
        score += 50 * len(task_tokens & command_tokens)
        score += 120 * sum(literal_terms[word] for word in task_tokens & literal_terms.keys())
        ranked.append((score, relative, callback, command))
    selected = sorted(ranked, key=lambda item: (-item[0], item[1].lower(), item[2].lower()))[:limit]
    paths: list[Path] = []
    reasons: dict[Path, list[str]] = defaultdict(list)
    required: dict[Path, list[str]] = defaultdict(list)
    for _score, relative, callback, command in selected:
        path = repo / relative
        if path not in paths:
            paths.append(path)
        reasons[path].append(f"alternative callback for unresolved follow-up: {command} -> {callback}")
        if callback not in required[path]:
            required[path].append(callback)
    return RetrievalResult(paths, dict(reasons), paths.copy(), dict(required))


def resolve_explicit_targets(repo: Path, task: str) -> RetrievalResult:
    """Resolve unambiguous code-shaped task references before semantic ranking."""
    index = load_map(repo).get("files", {})
    commands = {value.lower() for value in re.findall(r"/(?P<name>[A-Za-z][A-Za-z0-9_-]*)", task)}
    classes = set(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", task))
    functions = set(re.findall(r"\b([a-z_][A-Za-z0-9_]*)\s*\(\s*\)", task))
    scores: dict[str, int] = defaultdict(int)
    reasons: dict[str, list[str]] = defaultdict(list)
    required: dict[str, list[str]] = defaultdict(list)
    symbol_owners: dict[str, set[str]] = defaultdict(set)

    def add(relative: str, score: int, reason: str) -> None:
        if relative in index and (repo / relative).is_file():
            scores[relative] += score
            if reason not in reasons[relative]:
                reasons[relative].append(reason)

    def require(relative: str, symbol: str) -> None:
        if symbol and symbol not in required[relative]:
            required[relative].append(symbol)

    for relative, entry in index.items():
        for symbol in entry.get("symbols", []):
            if not isinstance(symbol, dict):
                continue
            name = str(symbol.get("name", ""))
            qualified = str(symbol.get("qualified_name", name))
            if name:
                symbol_owners[name.lower()].add(relative)
            if name in classes or qualified in classes:
                add(relative, 200, f"explicit class {name}")
                require(relative, name)
            if name in functions or qualified in functions:
                add(relative, 180, f"explicit function {name}()")
                require(relative, name)

        if not commands:
            continue
        try:
            source = (repo / relative).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for command in sorted(commands):
            # Covers discord.py app_commands, classic command decorators, and
            # equivalent decorators with an explicit name argument.
            pattern = rf"@[^\n]*command\s*\([^)]*name\s*=\s*['\"]{re.escape(command)}['\"]"
            inferred_name = rf"@[^\n]*command\s*\([^)]*\)\s*(?:async\s+)?def\s+{re.escape(command)}\s*\("
            match = re.search(pattern + r"[\s\S]{0,500}?(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source, re.IGNORECASE)
            inferred = re.search(inferred_name, source, re.IGNORECASE)
            if match or inferred:
                add(relative, 240, f"explicit command /{command}")
                require(relative, (match.group(1) if match else command))

    # Follow only method calls textually present in resolved entry points. The
    # owner lookup is index-backed and bounded; it is not a transitive walk.
    call_sources = [name for name in scores if any("explicit command" in reason or "explicit function" in reason for reason in reasons[name])]
    inspected: set[str] = set()
    for depth in range(2):
        next_sources: list[str] = []
        for relative in call_sources:
            if relative in inspected:
                continue
            inspected.add(relative)
            # Follow the resolved entry-point definitions, not every call in
            # their containing module. Scanning the whole file makes one
            # explicit command (for example /drop) promote unrelated commands
            # and their dependency trees ahead of other task surfaces.
            source = "\n".join(
                _fresh_python_symbol_text(repo / relative, symbol)
                for symbol in required[relative]
            )
            for method in re.findall(r"\.([a-z_][A-Za-z0-9_]*)\s*\(", source):
                for owner in symbol_owners.get(method.lower(), set()):
                    if owner != relative:
                        add(owner, 90 - depth * 20, f"implementation call {method}()")
                        require(owner, method)
                        next_sources.append(owner)
        call_sources = list(dict.fromkeys(next_sources))

    ordered = sorted(scores, key=lambda name: (-scores[name], name.lower()))[:MAX_EXPLICIT_TARGETS]
    paths = [repo / name for name in ordered]
    return RetrievalResult(
        paths, {repo / name: reasons[name] for name in ordered}, paths.copy(),
        {repo / name: required[name] for name in ordered if required[name]},
    )


class QwenTaskHintAnalyzer:
    """The first, compact task-level Qwen stage; paths are never trusted."""
    def __init__(self, model: str | None = None, timeout: float = 30.0) -> None:
        self.model = model or get_setting("CHATCODE_QWEN_MODEL")
        self.timeout = timeout
        self.last_status = "not_run"
        self.last_vocabulary: list[str] = []
        self.last_raw_response = ""
        self.last_parsed_hints: list[str] = []
        self.last_normalized_hints: list[str] = []
        self.last_rejections: list[str] = []
        self.last_candidate_ids: dict[str, str] = {}

    def is_available(self) -> bool:
        return bool(self.model and shutil.which("ollama"))

    def hints(self, repo: Path, task: str, preferred_paths: list[Path] | None = None) -> list[str]:
        self.last_raw_response = ""
        self.last_parsed_hints = []
        self.last_normalized_hints = []
        self.last_rejections = []
        if not self.is_available():
            self.last_status = "unavailable"
            return []
        index = load_map(repo).get("files", {})
        preferred_names = {
            path.relative_to(repo).as_posix() for path in (preferred_paths or []) if path.is_file()
        }
        terms = {word.lower() for word in re.findall(r"[A-Za-z_]+", task) if len(word) >= 3}
        terms |= {word[:-2] for word in terms if word.endswith("ed") and len(word) > 4}
        candidates: list[tuple[int, str, dict]] = []
        for relative, entry in index.items():
            for symbol in entry.get("symbols", []):
                if not isinstance(symbol, dict) or not symbol.get("name"):
                    continue
                name = str(symbol["name"])
                kind = str(symbol.get("kind", "symbol"))
                haystack = f"{relative} {name} {symbol.get('qualified_name', '')}".lower()
                score = 100 if relative in preferred_names else 1
                score += 25 if kind in {"function", "method"} else 5 if kind == "class" else 0
                score += sum(45 for term in terms if term in haystack or haystack.find(term) >= 0)
                if score:
                    candidates.append((score, relative, {"id": f"{relative}::{name}", "name": name, "kind": kind, "path": relative}))
        candidates.sort(key=lambda item: (-item[0], item[1].lower(), item[2]["name"].lower()))
        # Broad tasks need more vocabulary coverage, but still use one bounded
        # model call and only real indexed IDs.
        surfaces = _task_surfaces(task)
        complex_task = _task_is_complex(task, surfaces)
        entry_limit = 240 if complex_task else 180
        hint_limit = MAX_SEMANTIC_HINTS if complex_task else 12
        entries = [entry for _, _, entry in candidates[:entry_limit]]
        vocabulary = list(dict.fromkeys(entry["name"] for entry in entries))
        self.last_candidate_ids = {entry["id"]: entry["name"] for entry in entries}
        self.last_vocabulary = vocabulary
        prompt = (
            "Map this maintenance task to likely existing repository symbol names by meaning, not literal word overlap. "
            "Natural-language actions may use different verbs than code; choose the semantically closest vocabulary names. "
            f"Return ONLY JSON {{\"candidate_ids\":[\"id\"]}}. Use only IDs from Candidates; at most {hint_limit}. "
            "Cover each explicit requirement when the task spans multiple layers. "
            "Prefer behavior-changing functions, methods, handlers, and commands over passive domain types when appropriate. "
            f"Task: {task}\nRequirements: {json.dumps(surfaces, ensure_ascii=False)}"
            f"\nCandidates: {json.dumps(entries, ensure_ascii=False)}"
        )
        try:
            run = subprocess.run(["ollama", "run", self.model, "--format", "json"], input=prompt,
                text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=self.timeout)
            self.last_raw_response = run.stdout.strip()[:2000]
            if run.returncode != 0:
                self.last_status = "failed"
                return []
        except (OSError, subprocess.TimeoutExpired):
            self.last_status = "failed"
            return []
        payload = _parse_task_hint_payload(self.last_raw_response)
        if payload is None:
            self.last_status = "failed"
            return []
        raw_ids = payload.get("candidate_ids") if isinstance(payload.get("candidate_ids"), list) else None
        raw = raw_ids if raw_ids is not None else payload.get("symbol_hints", payload.get("hints", payload.get("symbols", [])))
        self.last_parsed_hints = _bounded_strings(raw, MAX_SEMANTIC_HINTS)
        lookup = {name.casefold(): name for name in vocabulary}
        hints = []
        for hint in self.last_parsed_hints:
            if raw_ids is not None:
                accepted = self.last_candidate_ids.get(hint)
                if accepted:
                    if accepted not in hints:
                        hints.append(accepted)
                    continue
                self.last_rejections.append(f"{hint}: unknown candidate ID")
                continue
            normalized = hint.strip().split("::")[-1].removesuffix("()").strip()
            self.last_normalized_hints.append(normalized)
            accepted = lookup.get(normalized.casefold())
            if accepted:
                if accepted not in hints:
                    hints.append(accepted)
            else:
                self.last_rejections.append(f"{hint}: not in supplied vocabulary")
        self.last_status = "complete" if hints else "empty"
        return hints


def _parse_task_hint_payload(text: str) -> dict | None:
    """Tolerate a fenced/surrounded JSON object from the same model stage."""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def resolve_semantic_hints(repo: Path, hints: list[str]) -> RetrievalResult:
    """Resolve exact accepted symbols; closure is a separate downstream stage."""
    if not hints:
        return RetrievalResult([])
    wanted = {hint.casefold() for hint in hints}
    index = load_map(repo).get("files", {})
    paths: list[Path] = []
    reasons: dict[Path, list[str]] = {}
    required: dict[Path, list[str]] = {}
    for relative, entry in sorted(index.items()):
        matches = [
            str(symbol.get("name")) for symbol in entry.get("symbols", [])
            if isinstance(symbol, dict) and str(symbol.get("name", "")).casefold() in wanted
        ]
        if matches and (repo / relative).is_file():
            path = repo / relative
            paths.append(path)
            required[path] = list(dict.fromkeys(matches))
            reasons[path] = [f"semantic symbol hint {symbol}" for symbol in required[path]]
    return RetrievalResult(paths, reasons, paths.copy(), required)


def _indexed_definition_owners(
    index: dict,
) -> dict[str, list[tuple[str, str]]]:
    """Map callable/class names to concrete indexed definition owners."""
    owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for relative, entry in index.items():
        if not isinstance(entry, dict):
            continue
        for symbol in entry.get("symbols", []):
            if not isinstance(symbol, dict) or not symbol.get("name"):
                continue
            name = str(symbol["name"])
            definition = str(symbol.get("definition_name") or name)
            owner = (relative, definition)
            for lookup in {name.casefold(), definition.casefold()}:
                if owner not in owners[lookup]:
                    owners[lookup].append(owner)
    return owners


def _fresh_python_definitions(
    path: Path,
) -> list[tuple[str, str | None]]:
    """Return current top-level functions/classes and direct class methods."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    definitions: list[tuple[str, str | None]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append((node.name, None))
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.append((member.name, node.name))
    return definitions


def _fresh_python_call_edges(
    path: Path,
    symbol: str,
) -> tuple[str | None, list[tuple[str, str]]] | None:
    """Return direct call names for one unambiguous current Python definition."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None
    matches: list[tuple[ast.AST, str | None]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            matches.append((node, None))
        if not isinstance(node, ast.ClassDef):
            continue
        for member in node.body:
            if (
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name == symbol
            ):
                matches.append((member, node.name))
    if len(matches) != 1:
        return None
    node, class_name = matches[0]
    calls: list[tuple[str, str]] = []
    for call in sorted(
        (item for item in ast.walk(node) if isinstance(item, ast.Call)),
        key=lambda item: (item.lineno, item.col_offset),
    ):
        if isinstance(call.func, ast.Attribute):
            receiver = call.func.value
            root = receiver
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(receiver, ast.Name) and receiver.id in {"self", "cls"}:
                receiver_kind = "self"
            elif isinstance(root, ast.Name) and root.id in {"self", "cls"}:
                receiver_kind = "member"
            else:
                receiver_kind = "qualified"
            calls.append((call.func.attr, receiver_kind))
        elif isinstance(call.func, ast.Name):
            calls.append((call.func.id, "bare"))
    for decorator in getattr(node, "decorator_list", []):
        for call in (
            item for item in ast.walk(decorator) if isinstance(item, ast.Call)
        ):
            calls.extend(
                (keyword.value.id, "bare")
                for keyword in call.keywords
                if isinstance(keyword.value, ast.Name)
            )
    return class_name, list(dict.fromkeys(calls))


def _root_definition_name(
    index: dict,
    relative: str,
    symbol: str,
) -> str:
    """Translate aliases such as command names to their real definition name."""
    entry = index.get(relative, {})
    for candidate in entry.get("symbols", []) if isinstance(entry, dict) else []:
        if not isinstance(candidate, dict):
            continue
        names = {
            str(candidate.get("name", "")).casefold(),
            str(candidate.get("qualified_name", "")).casefold(),
        }
        if symbol.casefold() in names:
            return str(candidate.get("definition_name") or candidate.get("name") or symbol)
    return symbol


def _rooted_implementation_closure(
    repo: Path,
    root_symbols: dict[Path, list[str]],
) -> RetrievalResult:
    """Follow concrete project-local calls from already accepted task roots.

    Generic names such as get/save/load/update are allowed when their owner can
    be resolved structurally. Ambiguous calls are deliberately not expanded.
    """
    index = load_map(repo).get("files", {})
    owners = _indexed_definition_owners(index)
    frontier: list[tuple[Path, str, int]] = []
    root_keys: set[tuple[Path, str]] = set()
    for path, symbols in root_symbols.items():
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(repo).as_posix()
        except ValueError:
            continue
        for symbol in symbols:
            definition = _root_definition_name(index, relative, symbol)
            key = (path, definition)
            if key in root_keys:
                continue
            root_keys.add(key)
            frontier.append((path, definition, 0))

    files: list[Path] = []
    reasons: dict[Path, list[str]] = defaultdict(list)
    required: dict[Path, list[str]] = defaultdict(list)
    expanded: set[tuple[Path, str]] = set()
    discovered: set[tuple[Path, str]] = set()

    def resolve_owner(
        source_path: Path,
        class_name: str | None,
        called: str,
        receiver_kind: str,
    ) -> tuple[Path, str] | None:
        same_file = [
            (source_path, name)
            for name, owner_class in _fresh_python_definitions(source_path)
            if name == called
            and (
                receiver_kind == "bare"
                or (
                    receiver_kind == "self"
                    and class_name is not None
                    and owner_class == class_name
                )
            )
        ]
        if len(same_file) == 1:
            return same_file[0]
        if receiver_kind == "self":
            return None
        try:
            relative = source_path.relative_to(repo).as_posix()
        except ValueError:
            return None
        dependencies = {
            str(value)
            for value in index.get(relative, {}).get("dependencies", [])
            if isinstance(value, str)
        }
        indexed = list(dict.fromkeys(owners.get(called.casefold(), [])))
        direct = [owner for owner in indexed if owner[0] in dependencies]
        if len(direct) == 1:
            return repo / direct[0][0], direct[0][1]
        if receiver_kind in {"bare", "member"} and len(indexed) == 1:
            return repo / indexed[0][0], indexed[0][1]
        return None

    while frontier and len(discovered) < MAX_ROOTED_CLOSURE_SYMBOLS:
        path, symbol, depth = frontier.pop(0)
        key = (path, symbol)
        if key in expanded or depth >= MAX_ROOTED_CLOSURE_DEPTH:
            continue
        expanded.add(key)
        call_data = _fresh_python_call_edges(path, symbol)
        if call_data is None:
            continue
        class_name, calls = call_data
        for called, receiver_kind in calls:
            owner = resolve_owner(path, class_name, called, receiver_kind)
            if owner is None or owner in root_keys or owner in discovered:
                continue
            owner_path, definition = owner
            if not owner_path.is_file():
                continue
            if owner_path not in files and len(files) >= MAX_ROOTED_CLOSURE_FILES:
                continue
            discovered.add(owner)
            if owner_path not in files:
                files.append(owner_path)
            if definition not in required[owner_path]:
                required[owner_path].append(definition)
            label = (
                f"structural implementation from "
                f"{path.relative_to(repo).as_posix()}::{symbol}: {called}()"
            )
            if label not in reasons[owner_path]:
                reasons[owner_path].append(label)
            frontier.append((owner_path, definition, depth + 1))
            if len(discovered) >= MAX_ROOTED_CLOSURE_SYMBOLS:
                break

    return RetrievalResult(
        files,
        dict(reasons),
        files.copy(),
        dict(required),
    )


def implementation_closure(
    repo: Path,
    task: str,
    seeds: list[Path],
    focus_symbols: set[str] | None = None,
    root_symbols: dict[Path, list[str]] | None = None,
) -> RetrievalResult:
    """Follow implementation calls using exact roots when they are available."""
    if root_symbols:
        return _rooted_implementation_closure(repo, root_symbols)

    # Compatibility path for callers that only have coarse seed files.
    index = load_map(repo).get("files", {})
    owners: dict[str, set[str]] = defaultdict(set)
    for relative, entry in index.items():
        for symbol in entry.get("symbols", []):
            if isinstance(symbol, dict) and symbol.get("name"):
                owners[str(symbol["name"]).lower()].add(relative)
    terms = {word.lower() for word in re.findall(r"[A-Za-z_]+", task) if len(word) >= 3}
    terms |= {word[:-2] for word in terms if word.endswith("ed") and len(word) > 4}
    terms |= {word[:-3] for word in terms if word.endswith("ing") and len(word) > 5}
    terms |= {word[:-1] for word in terms if len(word) > 3 and word[-1:] == word[-2:-1]}
    focus = {value.lower() for value in (focus_symbols or set())}
    generic_calls = {"get", "set", "create", "delete", "update", "room", "log", "save", "load"}
    scores: dict[str, int] = defaultdict(int)
    reasons: dict[str, list[str]] = defaultdict(list)
    required: dict[str, list[str]] = defaultdict(list)
    sources = [path.relative_to(repo).as_posix() for path in seeds if path.is_file()]
    seen: set[str] = set()
    for depth in range(2):
        next_sources: list[str] = []
        for relative in sorted(set(sources)):
            if relative in seen:
                continue
            seen.add(relative)
            try:
                text = (repo / relative).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for name in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*(?:\.callback)?\s*\(", text):
                lowered = name.lower()
                relevant = lowered in focus or any(term in lowered or lowered in term for term in terms)
                if depth == 0 and focus:
                    relevant = lowered in focus or any(lowered.startswith(value + "_") for value in focus)
                if not relevant or (lowered in generic_calls and lowered not in focus):
                    continue
                for owner in owners.get(lowered, set()):
                    if owner == relative:
                        continue
                    scores[owner] += 150 - depth * 30
                    label = f"implementation closure {name}()"
                    if label not in reasons[owner]:
                        reasons[owner].append(label)
                    if name not in required[owner]:
                        required[owner].append(name)
                    next_sources.append(owner)
        sources = next_sources
    ordered = sorted(scores, key=lambda name: (-scores[name], name.lower()))[:MAX_CLOSURE_FILES]
    paths = [repo / name for name in ordered]
    return RetrievalResult(paths, {repo / name: reasons[name] for name in ordered}, paths.copy(), {repo / name: required[name] for name in ordered})


def resolve_transport_roots(repo: Path, seeds: list[Path]) -> RetrievalResult:
    """Resolve a local HTTP-handler owner when selected code names a method.

    The method that needs implementing may deliberately be absent.  In that
    case a class owning several sibling HTTP handlers is stronger evidence
    than an exact-symbol lookup, provided its indexed module is structurally
    connected to one of the already selected application surfaces.
    """
    index = load_map(repo).get("files", {})
    seed_names = {
        path.relative_to(repo).as_posix()
        for path in seeds
        if path.is_file()
    }
    requested: set[str] = set()
    method_pattern = re.compile(
        r"\bmethod\s*[:=]\s*[^\r\n]{0,80}?['\"]"
        r"(GET|POST|PUT|PATCH|DELETE|OPTIONS)['\"]",
        re.IGNORECASE,
    )
    for path in seeds:
        if not path.is_file():
            continue
        try:
            requested.update(
                value.upper() for value in method_pattern.findall(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            )
        except OSError:
            continue
    if not requested:
        return RetrievalResult([])

    candidates: list[tuple[int, str, str, list[str]]] = []
    for relative, entry in index.items():
        path = repo / relative
        if relative in seed_names or path.suffix.casefold() != ".py" or not path.is_file():
            continue
        dependencies = {str(value) for value in entry.get("dependencies", [])}
        # This relation anchors the transport owner to a selected application
        # surface and prevents unrelated web servers from entering the result.
        if not dependencies & seed_names:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        indexed_names = {
            str(symbol.get("name", ""))
            for symbol in entry.get("symbols", [])
            if isinstance(symbol, dict)
        }
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name not in indexed_names:
                continue
            handlers = sorted({
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and re.fullmatch(r"do_(GET|POST|PUT|PATCH|DELETE|OPTIONS)", child.name)
            })
            methods = {name.removeprefix("do_") for name in handlers}
            if len(methods) < 3:
                continue
            score = len(methods) * 20 + 120
            score += 50 if "OPTIONS" in methods else 0
            score += 80 if requested - methods else 40
            candidates.append((score, relative, node.name, handlers))

    chosen = sorted(
        candidates, key=lambda value: (-value[0], value[1].casefold(), value[2].casefold())
    )[:MAX_TRANSPORT_ROOTS]
    paths = [repo / relative for _score, relative, _owner, _handlers in chosen]
    reasons = {
        repo / relative: [
            "HTTP transport owner for selected request method(s) "
            + ", ".join(sorted(requested)),
            "sibling handlers: " + ", ".join(handlers),
        ]
        for _score, relative, _owner, handlers in chosen
    }
    required = {
        repo / relative: [owner]
        for _score, relative, owner, _handlers in chosen
    }
    return RetrievalResult(paths, reasons, paths.copy(), required)


def resolve_test_callsite_closure(
    repo: Path,
    task: str,
    seeds: list[Path],
) -> RetrievalResult:
    """Resolve calls from relevant test bodies, then walk exact definitions twice."""
    index = load_map(repo).get("files", {})
    owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for relative, entry in index.items():
        for symbol in entry.get("symbols", []):
            if isinstance(symbol, dict) and symbol.get("name"):
                owners[str(symbol["name"]).casefold()].append((
                    relative, str(symbol.get("definition_name") or symbol["name"]),
                ))
    terms = {word.lower() for word in re.findall(r"[A-Za-z_]+", task) if len(word) >= 4}
    terms |= {word[:-1] for word in terms if word.endswith("s") and len(word) > 4}
    terms |= {word[:-2] for word in terms if word.endswith("ed") and len(word) > 5}
    roots: list[tuple[str, str]] = []
    callback_roots: list[tuple[str, str]] = []
    diagnostics: list[str] = []
    parsed_tests: list[tuple[Path, str, ast.AST, list[str]]] = []
    for path in sorted(set(seeds), key=lambda value: str(value).lower()):
        if not path.is_file() or "test" not in path.stem.casefold() or path.suffix.lower() != ".py":
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        parsed_tests.append((path, source, tree, source.splitlines(keepends=True)))

    # A task-relevant service test can reveal the action used by a separately
    # named command test (for example, take_loose_item() -> take.callback()).
    # Derive those action names only from already-selected function bodies; this
    # keeps the second pass bounded to the same candidate test files.
    discovered_actions: set[str] = set()
    for _path, _source, tree, lines in parsed_tests:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            lowered = node.name.casefold()
            name_tokens = set(lowered.split("_"))
            if terms and not (terms & name_tokens):
                continue
            body = "".join(lines[node.lineno - 1:node.end_lineno])
            for name in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", body):
                discovered_actions.add(name.casefold().split("_", 1)[0])

    for path, _source, tree, lines in parsed_tests:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            lowered = node.name.casefold()
            body = "".join(lines[node.lineno - 1:node.end_lineno])
            callback_nodes = sorted((
                call for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "callback"
                and isinstance(call.func.value, ast.Attribute)
            ), key=lambda call: (call.lineno, call.col_offset))
            callback_calls = [call.func.value.attr for call in callback_nodes]
            body_lower = body.casefold()
            name_tokens = set(lowered.split("_"))
            body_tokens = {
                part
                for identifier in re.findall(r"[a-z_]+", body_lower)
                for part in identifier.split("_")
                if part
            }
            body_matches = terms & body_tokens
            name_relevant = not terms or bool(terms & name_tokens)
            action_relevant = any(name.casefold() in discovered_actions for name in callback_calls)
            if not name_relevant and not action_relevant:
                continue
            diagnostics.append(
                f"relevant test {path.relative_to(repo).as_posix()}::{node.name} "
                f"(name_match={name_relevant}, discovered_action={action_relevant}, "
                f"body_terms={','.join(sorted(body_matches)) or 'none'})"
            )
            for call, name in zip(callback_nodes, callback_calls):
                matches = sorted(owners.get(name.casefold(), []))
                diagnostics.extend((
                    f"callback call {ast.unparse(call)}",
                    f"  extracted logical target: {name}",
                    f"  command alias lookup: {'matched' if matches else 'no match'}",
                    "  indexed definition: " + (
                        f"{matches[0][0]}::{matches[0][1]}" if matches else "none"
                    ),
                    f"  promoted required symbol: {'yes' if matches else 'no'}",
                    *( [] if matches else ["  reason if no: no indexed symbol or command alias"] ),
                ))
            calls = list(callback_calls)
            calls += [
                name for name in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", body)
                if any(term in name.casefold() or name.casefold() in term for term in terms)
            ]
            for name in dict.fromkeys(calls):
                diagnostics.append(f"extracted call {path.relative_to(repo).as_posix()}::{node.name} -> {name}")
                for owner, definition_name in sorted(owners.get(name.casefold(), []))[:2]:
                    target = (owner, definition_name)
                    (callback_roots if name in callback_calls else roots).append(target)
                    diagnostics.append(
                        f"resolved call {name} -> {owner}::{definition_name}"
                    )

    scores: dict[str, int] = defaultdict(int)
    reasons: dict[str, list[str]] = defaultdict(list)
    required: dict[str, list[str]] = defaultdict(list)
    frontier = list(dict.fromkeys([*callback_roots, *roots]))
    for owner, name in frontier:
        scores[owner] += 220
        reasons[owner].append(f"test call-site resolution {name}()")
        required[owner].append(name)
    generic = {"get", "set", "log", "save", "load", "send", "respond"}
    for depth in range(2):
        next_frontier: list[tuple[str, str]] = []
        for relative, symbol in frontier:
            body = _fresh_python_symbol_text(repo / relative, symbol)
            for called in dict.fromkeys(re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", body)):
                if called.casefold() in generic:
                    continue
                for owner, definition_name in sorted(owners.get(called.casefold(), []))[:2]:
                    if (owner, definition_name) == (relative, symbol):
                        continue
                    scores[owner] += 150 - depth * 30
                    label = f"implementation closure from {symbol}(): {called}()"
                    if label not in reasons[owner]:
                        reasons[owner].append(label)
                    if definition_name not in required[owner]:
                        required[owner].append(definition_name)
                    next_frontier.append((owner, definition_name))
        frontier = list(dict.fromkeys(next_frontier))[:MAX_CLOSURE_FILES]
    ordered = sorted(scores, key=lambda name: (-scores[name], name.lower()))[:MAX_CLOSURE_FILES]
    paths = [repo / name for name in ordered]
    return RetrievalResult(
        paths,
        {repo / name: reasons[name] for name in ordered},
        paths.copy(),
        {repo / name: required[name] for name in ordered},
        diagnostics,
    )


def _fresh_python_symbol_text(path: Path, symbol: str) -> str:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return ""
    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            return "".join(lines[node.lineno - 1:node.end_lineno])
    return ""


def expand_candidates(repo: Path, task: str, seeds: list[Path], limit: int = 12) -> RetrievalResult:
    """Rank direct, local evidence around seeds without transitive expansion."""
    index = load_map(repo).get("files", {})
    seed_names = {path.relative_to(repo).as_posix() for path in seeds if path.is_file()}
    words = {word.lower() for word in re.findall(r"[A-Za-z0-9_]+", task) if len(word) >= 3}
    reverse: dict[str, set[str]] = defaultdict(set)
    symbol_owners: dict[str, set[str]] = defaultdict(set)
    for relative, entry in index.items():
        for dependency in entry.get("dependencies", []):
            reverse[str(dependency)].add(relative)
        for symbol in entry.get("symbols", []):
            if isinstance(symbol, dict):
                for key in ("name", "qualified_name"):
                    name = str(symbol.get(key, "")).lower()
                    if name:
                        symbol_owners[name].add(relative)

    scores: dict[str, int] = defaultdict(int)
    reasons: dict[str, list[str]] = defaultdict(list)
    def add(relative: str, score: int, reason: str) -> None:
        if relative not in index or not (repo / relative).is_file():
            return
        scores[relative] += score
        if reason not in reasons[relative]:
            reasons[relative].append(reason)

    for seed in sorted(seed_names):
        # Keep a valid seed even when a caller supplied it before the cache was
        # populated (for example during a first-run fallback).
        scores[seed] += 100
        reasons[seed].append("Qwen seed")
        entry = index.get(seed, {})
        for dependency in entry.get("dependencies", []):
            # Direct imports are evidence, but generic plumbing is deliberately weak.
            weight = 8 if any(part in GENERIC_PATH_PARTS for part in Path(dependency).parts) else 24
            add(str(dependency), weight, "direct dependency of seed")
        for importer in reverse.get(seed, set()):
            add(importer, 18, "imports seed")
        for symbol in entry.get("symbols", []):
            if not isinstance(symbol, dict):
                continue
            name = str(symbol.get("name", "")).lower()
            if not name or len(name) < 4:
                continue
            for owner in symbol_owners.get(name, set()):
                if owner != seed:
                    add(owner, 14, "defines seed symbol")

    for relative, entry in index.items():
        lower = relative.lower()
        for word in words:
            if word in lower:
                add(relative, 20, "lexical path match")
            for symbol in entry.get("symbols", []):
                if isinstance(symbol, dict) and word in str(symbol.get("qualified_name") or symbol.get("name", "")).lower():
                    add(relative, 28, "symbol match")
        # Tests named after a seed module are a strong, bounded relation.
        if _is_test(relative):
            stem = Path(relative).stem.lower().removeprefix("test_")
            if any(stem and stem in Path(seed).stem.lower() for seed in seed_names):
                add(relative, 22, "related test")

    # rg is much cheaper and more accurate than reopening every indexed file
    # in Python.  It is optional: the map-based signals remain fully usable on
    # installations without ripgrep.
    if shutil.which("rg"):
        for word in sorted(words)[:12]:
            try:
                search = subprocess.run(
                    ["rg", "-l", "-i", "--no-messages", "--glob", "!*.lock", word, str(repo)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3,
                )
            except (OSError, subprocess.TimeoutExpired):
                break
            for raw_path in search.stdout.splitlines()[:80]:
                try:
                    relative = Path(raw_path).resolve().relative_to(repo.resolve()).as_posix()
                except ValueError:
                    continue
                add(relative, 12, "lexical text match")

    ordered = sorted(scores, key=lambda value: (-scores[value], value.lower()))
    selected_names = ordered[:max(limit, len(seed_names))]
    paths = [repo / name for name in selected_names]
    candidate_limit = MAX_CANDIDATES if _task_is_complex(task) else 24
    return RetrievalResult(
        paths,
        {repo / name: reasons[name] for name in selected_names},
        [repo / name for name in ordered[:candidate_limit]],
    )


def _is_test(relative: str) -> bool:
    path = Path(relative)
    return path.name.startswith("test_") or path.name.endswith(("_test.py", ".test.ts", ".spec.ts", ".test.js", ".spec.js"))


class QwenCompletenessChecker:
    """One compact check; symbol/concept requests are resolved locally once."""
    def __init__(self, model: str | None = None, timeout: float = 30.0) -> None:
        self.model = model or get_setting("CHATCODE_QWEN_MODEL")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.model and shutil.which("ollama"))

    def check(self, repo: Path, task: str, result: RetrievalResult) -> CompletenessResult:
        if not self.is_available():
            return CompletenessResult()
        surfaces = _task_surfaces(task)
        complex_task = _task_is_complex(task, surfaces)
        symbol_limit = 14 if complex_task else 8
        dependency_limit = 10 if complex_task else 6
        resolved_limit = MAX_COMPLETENESS_RESOLVED if complex_task else 3
        selected = {path.relative_to(repo).as_posix() for path in result.files}
        candidate_rows = []
        index = load_map(repo).get("files", {})
        selected_rows = []
        for relative in sorted(selected):
            entry = index.get(relative, {})
            selected_rows.append({
                "path": relative,
                "symbols": [
                    str(item.get("qualified_name") or item.get("name"))
                    for item in entry.get("symbols", [])[:symbol_limit]
                    if isinstance(item, dict)
                ],
                "dependencies": entry.get("dependencies", [])[:dependency_limit],
            })
        for path in result.candidates:
            relative = path.relative_to(repo).as_posix()
            if relative in selected:
                continue
            entry = index.get(relative, {})
            symbols = [
                str(item.get("qualified_name") or item.get("name"))
                for item in entry.get("symbols", [])[:symbol_limit]
                if isinstance(item, dict)
            ]
            candidate_rows.append({
                "path": relative,
                "symbols": symbols,
                "dependencies": entry.get("dependencies", [])[:dependency_limit],
            })
        prompt = (
            "You check whether code-retrieval context is complete. Return ONLY JSON: "
            '{"missing_files":["candidate/path"],"missing_symbols":["ClassOrFunction"],'
            '"missing_concepts":["short implementation concept"]}. missing_files may ONLY be paths in Candidates. '
            "Check every explicit requirement below, including persistence, domain, transport, UI, and tests when present. "
            "For a required file absent from Candidates, name its project symbol or a short concept; do not invent paths.\n"
            f"Task: {task}\nRequirements: {json.dumps(surfaces, ensure_ascii=False)}"
            f"\nSelected: {json.dumps(selected_rows, ensure_ascii=False)}"
            f"\nCandidates: {json.dumps(candidate_rows, ensure_ascii=False)}"
        )
        try:
            run = subprocess.run(["ollama", "run", self.model, "--format", "json"], input=prompt,
                text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=self.timeout)
            payload = json.loads(run.stdout) if run.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return CompletenessResult()
        requested = payload.get("missing_files", []) if isinstance(payload, dict) else []
        available = {path.relative_to(repo).as_posix(): path for path in result.candidates}
        output = CompletenessResult()
        for name in requested:
            if not isinstance(name, str) or name not in available or name in selected:
                continue
            path = available[name]
            if path not in output.files:
                output.files.append(path)
                output.reasons[path] = "Qwen completeness known-path addition"
        symbols = _bounded_strings(
            payload.get("missing_symbols", []) if isinstance(payload, dict) else [],
            12 if complex_task else 8,
        )
        concepts = _bounded_strings(
            payload.get("missing_concepts", []) if isinstance(payload, dict) else [],
            8 if complex_task else 4,
        )
        # This remains one bounded repair pass. Complex tasks may resolve a few
        # more concrete files, but never trigger an open-ended model loop.
        for path, reason in _resolve_missing_requests(
            repo,
            index,
            selected | {p.relative_to(repo).as_posix() for p in output.files},
            symbols,
            concepts,
        ):
            if len(output.files) >= resolved_limit:
                break
            if path not in output.files:
                output.files.append(path)
                output.reasons[path] = reason
        return output

    def missing_files(self, repo: Path, task: str, result: RetrievalResult) -> list[Path]:
        """Compatibility helper for callers that only need the additions."""
        return self.check(repo, task, result).files


def _bounded_strings(value: object, limit: int) -> list[str]:
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    output: list[str] = []
    for item in values:
        cleaned = re.sub(r"\s+", " ", item).strip()[:120] if isinstance(item, str) else ""
        if cleaned and cleaned.casefold() not in {item.casefold() for item in output}:
            output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _resolve_missing_requests(repo: Path, index: dict, selected: set[str], symbols: list[str], concepts: list[str]) -> list[tuple[Path, str]]:
    """Resolve Qwen's non-path requests solely through real index/search hits."""
    scores: dict[str, int] = defaultdict(int)
    reasons: dict[str, str] = {}
    normalized_symbols = {re.sub(r"[^a-z0-9]", "", value.lower()) for value in symbols}
    concept_words = [word.lower() for concept in concepts for word in re.findall(r"[A-Za-z0-9_]+", concept) if len(word) >= 3]
    for relative, entry in index.items():
        if relative in selected or not (repo / relative).is_file():
            continue
        names = [str(symbol.get("qualified_name") or symbol.get("name", "")) for symbol in entry.get("symbols", []) if isinstance(symbol, dict)]
        for name in names:
            if re.sub(r"[^a-z0-9]", "", name.lower()) in normalized_symbols:
                scores[relative] += 100
                reasons[relative] = "Qwen completeness symbol resolution"
        haystack = " ".join([relative, str(entry.get("summary", "")), *entry.get("tags", []), *names]).lower()
        matched = sum(word in haystack for word in concept_words)
        if matched >= 2:
            scores[relative] += matched * 12
            reasons.setdefault(relative, "Qwen completeness concept resolution")
    # A concept can name behavior rather than a declared symbol. Search only
    # its significant terms, and require corroboration from index evidence.
    if concept_words and shutil.which("rg"):
        for word in sorted(set(concept_words))[:8]:
            try:
                run = subprocess.run(["rg", "-l", "-i", "--no-messages", word, str(repo)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                break
            for raw in run.stdout.splitlines()[:50]:
                try:
                    relative = Path(raw).resolve().relative_to(repo.resolve()).as_posix()
                except ValueError:
                    continue
                if relative in index and relative not in selected:
                    scores[relative] += 6
                    reasons.setdefault(relative, "Qwen completeness concept resolution")
    ranked = sorted((name for name, score in scores.items() if score >= 24), key=lambda name: (-scores[name], name.lower()))
    return [(repo / name, reasons[name]) for name in ranked[:MAX_COMPLETENESS_RESOLVED]]
