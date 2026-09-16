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
from chatcode.indexing.project_graph import load_map


MAX_CANDIDATES = 24
MAX_COMPLETENESS_RESOLVED = 3
MAX_EXPLICIT_TARGETS = 8
MAX_SEMANTIC_HINTS = 12
MAX_CLOSURE_FILES = 8
MAX_TASK_SURFACES = 4
# A surface can have a small set of equally strong concrete definitions (for
# example renderer, view, presenter, and autocomplete). Keep this bounded,
# while allowing each independently promoted root to reach materialization.
MAX_SURFACE_ROOTS = 4
GENERIC_PATH_PARTS = {"utils", "util", "common", "config", "settings", "constants", "helpers"}


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


def _task_surfaces(task: str) -> list[str]:
    """Split only explicit functional clauses, never arbitrary nouns."""
    clauses = re.split(
        r"\s*(?:;|\n+|[.!?]\s+(?=(?:when|make|show|render|display|use|update|save|delete|drop|take|/))|"
        r"\band\s+(?=(?:make|show|render|display|use|update|save|delete|drop|take|/)))\s*",
        task,
        flags=re.IGNORECASE,
    )
    return list(dict.fromkeys(clause.strip() for clause in clauses if clause.strip()))[:MAX_TASK_SURFACES]


def resolve_task_surface_roots(repo: Path, task: str) -> RetrievalResult:
    """Find a few real indexed roots for every explicit task surface."""
    surfaces = _task_surfaces(task)
    # Preserve established single-surface ranking/materialization exactly.
    if len(surfaces) < 2:
        return RetrievalResult([])
    index = load_map(repo).get("files", {})
    paths: list[Path] = []
    reasons: dict[Path, list[str]] = defaultdict(list)
    required: dict[Path, list[str]] = defaultdict(list)
    diagnostics: list[str] = []
    presentation_terms = {"display", "displayed", "show", "render", "rendering", "view", "embed", "ui"}
    for surface in surfaces:
        tokens = {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", surface.replace("_", " "))
            if len(token) >= 3
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
            file_tokens = set(re.findall(r"[a-z0-9]+", relative.casefold().replace("_", " ")))
            for symbol in entry.get("symbols", []):
                if not isinstance(symbol, dict) or not symbol.get("name"):
                    continue
                name = str(symbol["name"])
                symbol_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
                symbol_tokens = set(re.findall(r"[a-z0-9]+", symbol_text.casefold().replace("_", " ")))
                overlap = tokens & (symbol_tokens | file_tokens)
                if not overlap:
                    continue
                score = 70 * len(tokens & symbol_tokens) + 25 * len(tokens & file_tokens)
                if name.casefold() in tokens:
                    score += 120
                # Normalize the already-supported presentation vocabulary so
                # inflected task wording ("displayed") can resolve concrete
                # show/render/view/embed definitions for the same noun surface.
                if tokens & presentation_terms and symbol_tokens & presentation_terms:
                    score += 180
                # The index already records a bounded set of central symbols.
                # Use that existing definition metadata only to break close
                # matches; it does not add files or candidates to the search.
                if name.casefold() in important:
                    score += 80
                # A surface must have meaningful evidence, not a lone generic
                # filename coincidence.
                if score >= 70:
                    candidates.append((score, relative, str(symbol.get("definition_name") or name)))
        selected = sorted(candidates, key=lambda item: (-item[0], item[1].lower(), item[2].lower()))[:MAX_SURFACE_ROOTS]
        if not selected:
            diagnostics.append(f"surface unresolved: {surface}")
            continue
        # This means concrete definition roots were resolved. Rendering is
        # verified later by the shared required-source materializer; references
        # or imports alone never produce this state.
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
    return RetrievalResult(paths, dict(reasons), paths.copy(), dict(required), diagnostics)


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
        # A compact high-quality list is more useful than a flat alphabetical
        # dump; all entries remain real indexed definitions.
        entries = [entry for _, _, entry in candidates[:180]]
        vocabulary = list(dict.fromkeys(entry["name"] for entry in entries))
        self.last_candidate_ids = {entry["id"]: entry["name"] for entry in entries}
        self.last_vocabulary = vocabulary
        prompt = (
            "Map this maintenance task to likely existing repository symbol names by meaning, not literal word overlap. "
            "Natural-language actions may use different verbs than code; choose the semantically closest vocabulary names. "
            "Return ONLY JSON {\"candidate_ids\":[\"id\"]}. Use only IDs from Candidates; at most 12. "
            "Prefer behavior-changing functions, methods, handlers, and commands over passive domain types when appropriate. "
            f"Task: {task}\nCandidates: {json.dumps(entries, ensure_ascii=False)}"
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


def implementation_closure(
    repo: Path, task: str, seeds: list[Path], focus_symbols: set[str] | None = None,
) -> RetrievalResult:
    """Follow task-related project calls from selected tests/call sites, twice."""
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


def test_callsite_closure(repo: Path, task: str, seeds: list[Path]) -> RetrievalResult:
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
    return RetrievalResult(paths, {repo / name: reasons[name] for name in selected_names}, [repo / name for name in ordered[:MAX_CANDIDATES]])


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
        selected = {path.relative_to(repo).as_posix() for path in result.files}
        candidate_rows = []
        index = load_map(repo).get("files", {})
        selected_rows = []
        for relative in sorted(selected):
            entry = index.get(relative, {})
            selected_rows.append({"path": relative, "symbols": [str(item.get("qualified_name") or item.get("name")) for item in entry.get("symbols", [])[:8] if isinstance(item, dict)], "dependencies": entry.get("dependencies", [])[:6]})
        for path in result.candidates:
            relative = path.relative_to(repo).as_posix()
            if relative in selected:
                continue
            entry = index.get(relative, {})
            symbols = [str(item.get("qualified_name") or item.get("name")) for item in entry.get("symbols", [])[:8] if isinstance(item, dict)]
            candidate_rows.append({"path": relative, "symbols": symbols, "dependencies": entry.get("dependencies", [])[:6]})
        prompt = (
            "You check whether code-retrieval context is complete. Return ONLY JSON: "
            '{"missing_files":["candidate/path"],"missing_symbols":["ClassOrFunction"],'
            '"missing_concepts":["short implementation concept"]}. missing_files may ONLY be paths in Candidates. '
            "For a required file absent from Candidates, name its project symbol or a short concept; do not invent paths.\n"
            f"Task: {task}\nSelected: {json.dumps(selected_rows, ensure_ascii=False)}\nCandidates: {json.dumps(candidate_rows, ensure_ascii=False)}"
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
        symbols = _bounded_strings(payload.get("missing_symbols", []) if isinstance(payload, dict) else [], 8)
        concepts = _bounded_strings(payload.get("missing_concepts", []) if isinstance(payload, dict) else [], 4)
        # The initial pool is intentionally broad but bounded (task rg hits plus
        # one-hop graph evidence). These requests cover the rare case where it
        # contains no route to a required implementation, without trusting an
        # invented filename or asking Qwen a third question.
        for path, reason in _resolve_missing_requests(repo, index, selected | {p.relative_to(repo).as_posix() for p in output.files}, symbols, concepts):
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
