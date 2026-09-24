"""Current-source materialization for ChatCode patch context.

This module owns authoritative working-tree reads, exact symbol location,
bounded source rendering, context-contract rendering, and dependency
materialization. Retrieval and high-level context orchestration stay outside
this module.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path

from ..git_utils import GitError, run_git
from ..indexing.project_graph import load_map
from ..retrieval.task_semantics import get_task_words


PATCH_FULL_FILE_CHARS = 20_000
PATCH_DIRTY_FULL_FILE_CHARS = 48_000
PATCH_EXCERPT_RADIUS = 32
PATCH_CONTEXT_BUDGET_CHARS = 90_000
PATCH_CONTEXT_HARD_BUDGET_CHARS = 180_000
PATCH_CONTEXT_ELASTIC_RESERVE_CHARS = 12_000
PATCH_CONTRACT_FULL_FILE_MAX_CHARS = 120_000
PATCH_CONTRACT_FULL_FILE_RATIO = 1.20
PATCH_DEPENDENCY_RESERVE_RATIO = 0.25


def _changed_python_definition_roots(
    repo: Path,
    paths: list[Path],
) -> dict[Path, list[str]]:
    """Resolve dirty definitions when an oversized file cannot render in full."""
    roots: dict[Path, list[str]] = {}
    hunk_pattern = re.compile(
        r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@",
        re.MULTILINE,
    )
    for path in paths:
        if path.suffix.casefold() != ".py" or not path.is_file():
            continue
        relative = path.relative_to(repo).as_posix()
        try:
            diff = run_git(
                "diff", "HEAD", "--unified=0", "--no-renames", "--", relative,
                cwd=repo,
            )
            source = path.read_text(encoding="utf-8", errors="replace")
            # Small files already render as authoritative FULL FILE source.
            # Larger dirty files can lose the expanded dirty-file allowance
            # when the primary budget is shared, so their changed definitions
            # still need deterministic symbol materialization.
            if len(source) <= PATCH_FULL_FILE_CHARS:
                continue
            tree = ast.parse(source)
        except (GitError, OSError, SyntaxError):
            continue
        changed_ranges = []
        for match in hunk_pattern.finditer(diff):
            start = int(match.group("start"))
            count = int(match.group("count") or "1")
            changed_ranges.append((max(1, start), max(1, start + count - 1)))
        if not changed_ranges:
            continue

        # Patchable identities are module definitions and direct class
        # members. Nested helpers remain part of their enclosing definition,
        # avoiding ambiguous unqualified identities such as nested on_submit.
        definitions: list[tuple[int, int, str]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min(
                    [node.lineno, *(item.lineno for item in node.decorator_list)]
                    if node.decorator_list else [node.lineno]
                )
                definitions.append((start, node.end_lineno, node.name))
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    start = min(
                        [member.lineno, *(item.lineno for item in member.decorator_list)]
                        if member.decorator_list else [member.lineno]
                    )
                    definitions.append((start, member.end_lineno, member.name))

        for changed_start, changed_end in changed_ranges:
            overlaps = [
                item for item in definitions
                if item[0] <= changed_end and changed_start <= item[1]
            ]
            if not overlaps:
                continue
            _start, _end, name = min(
                overlaps, key=lambda item: (item[1] - item[0], item[0], item[2])
            )
            if name not in roots.setdefault(path, []):
                roots[path].append(name)
    return roots


def _read_current_text_and_hash(
    path: Path,
) -> tuple[str, str]:
    """Return one authoritative working-tree snapshot of ``path``.

    Context rendering must never consult Git blobs or the project-map cache for
    source text.  Keeping the decode and digest tied to the same byte read also
    prevents a hash from describing a different revision than the text below it.
    """
    data = path.read_bytes()
    return (
        data.decode("utf-8", errors="replace"),
        hashlib.sha256(data).hexdigest(),
    )


def _merge_line_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []

    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )
        else:
            merged.append([start, end])

    return [
        (start, end)
        for start, end in merged
    ]


def _configured_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _context_budget_chars() -> int:
    return _configured_positive_int(
        "CHATCODE_CONTEXT_BUDGET_CHARS",
        PATCH_CONTEXT_BUDGET_CHARS,
    )


def _context_hard_budget_chars() -> int:
    return max(
        _context_budget_chars(),
        _configured_positive_int(
            "CHATCODE_CONTEXT_HARD_BUDGET_CHARS",
            PATCH_CONTEXT_HARD_BUDGET_CHARS,
        ),
    )


def _dirty_current_line_ranges(repo: Path, path: Path) -> list[tuple[int, int]]:
    """Return new-side dirty hunk coordinates without treating diff as source."""
    try:
        relative = path.relative_to(repo).as_posix()
        diff = run_git(
            "diff", "HEAD", "--unified=0", "--no-renames", "--", relative,
            cwd=repo,
        )
    except (GitError, ValueError):
        return []
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(
        r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@",
        diff,
        re.MULTILINE,
    ):
        start = max(1, int(match.group("start")))
        count = int(match.group("count") or "1")
        ranges.append((start, start if count == 0 else start + count - 1))
    return ranges


def _symbol_aware_ranges(
    content: str,
    task: str,
    *,
    test_file: bool = False,
    focus_symbols: list[str] | None = None,
    preferred_ranges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    lines = content.splitlines()
    count = len(lines)
    task_words = get_task_words(task) - {
        "able", "and", "are", "like", "not", "properly", "should", "the",
        "themselves", "there", "they", "way", "you",
    }
    matches: set[int] = set()
    match_scores: dict[int, int] = {}
    definition = re.compile(
        r"^\s*(?:async\s+)?(?:def|class|function|interface|type|enum|struct|trait)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)"
    )
    symbols: set[str] = set()

    term_frequency = {
        word: sum(1 for line in lines if word in line.lower())
        for word in task_words
    }
    for index, line in enumerate(lines, start=1):
        lowered = line.lower()
        symbol_hits = sum(
            1
            for symbol in (focus_symbols or [])
            if re.search(rf"\b{re.escape(symbol)}\b", line, re.IGNORECASE)
        )
        word_hits = [word for word in task_words if word in lowered]
        if symbol_hits or word_hits:
            matches.add(index)
            match_scores[index] = symbol_hits * 100 + sum(
                min(
                    5000,
                    max(
                        1,
                        (len(word) ** 3 * 100)
                        // max(1, int(term_frequency[word] ** 0.5)),
                    ),
                )
                * max(1, lowered.count(word))
                for word in word_hits
            )
            if any(len(word) >= 6 for word in word_hits) and any(
                start <= index <= end for start, end in (preferred_ranges or [])
            ):
                match_scores[index] += 10_000
            found = definition.match(line)
            if found:
                symbols.add(found.group(1))
                if found.group(1).casefold() in task_words:
                    match_scores[index] += 20_000

    if test_file and task_words:
        for index, line in enumerate(lines, start=1):
            lowered = line.lower()
            if ("test" in lowered or "spec" in lowered) and any(
                word in lowered for word in task_words
            ):
                matches.add(index)
                match_scores.setdefault(index, 1)

    if symbols:
        for index, line in enumerate(lines, start=1):
            if any(
                re.search(rf"\b{re.escape(symbol)}\b", line)
                for symbol in symbols
            ):
                matches.add(index)

    ranges: list[tuple[int, int]] = []
    import_end = 0
    for index, line in enumerate(lines[:200], start=1):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "#include", "using ")):
            import_end = index
    if import_end:
        ranges.append((1, min(count, import_end + 8)))

    if not matches:
        fallback_end = min(count, PATCH_EXCERPT_RADIUS * 2 + 1)
        return _merge_line_ranges(
            [*ranges, (1, fallback_end)] if count else ranges
        )

    strongest_matches = set(
        sorted(
            matches,
            key=lambda line_number: (-match_scores.get(line_number, 1), line_number),
        )[:64]
    )
    for line_number in sorted(strongest_matches):
        ranges.append((
            max(1, line_number - PATCH_EXCERPT_RADIUS),
            min(count, line_number + PATCH_EXCERPT_RADIUS),
        ))

    merged = _merge_line_ranges(ranges)
    focused_ranges: list[tuple[int, int, int]] = []
    for start, end in merged:
        anchors = [
            line_number
            for line_number in strongest_matches
            if start <= line_number <= end
        ]
        if not anchors:
            focused_ranges.append((start, end, 0))
            continue
        anchor = min(
            anchors,
            key=lambda line_number: (-match_scores.get(line_number, 1), line_number),
        )
        structural_start = max(start, anchor - PATCH_EXCERPT_RADIUS)
        found_structure = False
        for line_number in range(anchor, structural_start - 1, -1):
            if definition.match(lines[line_number - 1]):
                structural_start = line_number
                found_structure = True
                break
        if not found_structure:
            structural_start = max(start, anchor - 10)
        focused_ranges.append((
            structural_start,
            min(end, anchor + PATCH_EXCERPT_RADIUS),
            match_scores.get(anchor, 1),
        ))
        secondary_candidates = [
            line_number
            for line_number in anchors
            if abs(line_number - anchor) >= 30
        ]
        secondary = max(secondary_candidates, default=None)
        if secondary is not None:
            focused_ranges.append((
                max(start, secondary - 10),
                min(end, secondary + PATCH_EXCERPT_RADIUS),
                match_scores.get(anchor, 1),
            ))
    # Under a constrained required-file allowance, patch-relevant regions
    # must be emitted before supplemental imports or file headers.
    return [
        (start, end)
        for start, end, _score in sorted(
            focused_ranges, key=lambda item: (-item[2], item[0])
        )
    ]


def _render_patch_file_context(
    repo: Path,
    path: Path,
    task: str,
    changed_files: set[Path],
    explicit_files: set[Path],
    max_chars: int | None = None,
    required_symbols: list[str] | None = None,
    focus_symbols: list[str] | None = None,
) -> str:
    content, digest = _read_current_text_and_hash(path)
    relative = path.relative_to(repo).as_posix()
    line_count = max(1, len(content.splitlines()))
    include_full = (
        len(content) <= PATCH_FULL_FILE_CHARS
        or (
            path in explicit_files
            and len(content) <= PATCH_DIRTY_FULL_FILE_CHARS
        )
    )
    if include_full:
        diagnostic = ""
        if os.getenv("CHATCODE_DEBUG_CONTEXT", "").strip():
            dirty = "yes" if path in changed_files else "no"
            diagnostic = f"Source: working-tree; dirty={dirty}; sha256={digest}\n"
        full_section = (
            f"===== FULL FILE: {relative} =====\n"
            f"SHA-256: {digest}\n"
            f"{diagnostic}"
            f"Source line range: 1-{line_count}\n\n"
            f"{content}\n"
        )
        if max_chars is None or len(full_section) <= max_chars:
            return full_section

    lines = content.splitlines(keepends=True)
    excerpts = []
    test_file = "test" in path.stem.casefold() or "spec" in path.stem.casefold()
    preferred_ranges = (
        _dirty_current_line_ranges(repo, path) if path in changed_files else []
    )
    target_ranges = _fresh_symbol_ranges(content, path, required_symbols or [])
    ranges = target_ranges or _symbol_aware_ranges(
        content,
        task,
        test_file=test_file,
        focus_symbols=focus_symbols or required_symbols,
        preferred_ranges=preferred_ranges,
    )
    for start, end in ranges:
        excerpts.append(
            f"----- EXCERPT {relative} source lines {start}-{end} -----\n"
            + "".join(lines[start - 1:end])
        )
    section = (
        f"===== SYMBOL CONTEXT: {relative} =====\n"
        f"SHA-256 (complete source file): {digest}\n"
        + "\n".join(excerpts)
        + "\n"
    )
    if max_chars is None or len(section) <= max_chars:
        return section
    if target_ranges:
        # Keep each requested node atomic, but do not make every requested
        # handler an all-or-nothing bundle. A large sibling handler must not
        # prevent smaller, independently resolved targets from materializing.
        header = f"===== SELECTED SOURCE: {relative} =====\n"
        kept: list[str] = []
        used = len(header)
        for start, end in target_ranges:
            excerpt = (
                f"----- EXCERPT {relative} source lines {start}-{end} -----\n"
                + "".join(lines[start - 1:end])
                + "\n"
            )
            if used + len(excerpt) <= max_chars:
                kept.append(excerpt)
                used += len(excerpt)
        return header + "".join(kept) if kept else ""

    # Every primary selection gets a fresh, exact (though possibly shorter)
    # excerpt before supporting dependencies consume the budget. Never splice
    # arbitrary characters: preserve complete current working-tree lines.
    header = (
        f"===== EXCERPT: {relative} =====\n"
        f"SHA-256 (complete source file): {digest}\n"
    )
    available = max_chars - len(header) - 1
    if available <= 0:
        return ""
    excerpt_lines: list[str] = []
    used = 0
    fallback_ranges = _symbol_aware_ranges(
        content,
        task,
        test_file=test_file,
        focus_symbols=focus_symbols or required_symbols,
        preferred_ranges=preferred_ranges,
    )[:4]
    for position, (start, end) in enumerate(fallback_ranges):
        remaining_ranges = len(fallback_ranges) - position
        range_allowance = max(1, (available - used) // remaining_ranges)
        marker = f"----- source lines "
        if len(marker) > range_allowance:
            break
        captured: list[str] = []
        actual_end = start - 1
        for line_number, line in enumerate(lines[start - 1:end], start=start):
            # Reserve the final, truthful range marker before accepting a line.
            possible_marker = f"----- source lines {start}-{line_number} -----\n"
            if len(possible_marker) + sum(map(len, captured)) + len(line) > range_allowance:
                break
            captured.append(line)
            actual_end = line_number
        if actual_end < start:
            continue
        final_marker = f"----- source lines {start}-{actual_end} -----\n"
        excerpt_lines.append(final_marker)
        excerpt_lines.extend(captured)
        used += len(final_marker) + sum(map(len, captured))
        if used >= available:
            break
    return header + "".join(excerpt_lines) + "\n" if excerpt_lines else ""


GENERIC_SYMBOL_SUFFIXES = {".js", ".jsx", ".ts", ".tsx"}


def _balanced_source_end(
    content: str,
    start: int,
    opener: str,
    closer: str,
) -> int | None:
    """Return the index just after a balanced JS/TS delimiter."""
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = start
    while index < len(content):
        char = content[index]
        next_char = content[index + 1] if index + 1 < len(content) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
                continue
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in {'"', "'", "`"}:
            quote = char
            index += 1
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _generic_symbol_span(
    content: str,
    symbol: str,
) -> tuple[int, int] | None:
    """Locate one named JS/TS declaration without requiring a parser runtime."""
    escaped_symbol = re.escape(symbol)
    declaration = re.compile(
        rf"(?m)^[ \t]*(?:export\s+)?(?:default\s+)?"
        rf"(?:(?P<kind>async\s+function|function|class|interface|enum)\s+{escaped_symbol}\b"
        rf"|type\s+{escaped_symbol}\s*="
        rf"|(?P<binding>const|let|var)\s+{escaped_symbol}(?:\s*:\s*[^=\n]+)?\s*=)"
    )
    match = declaration.search(content)
    if match is None:
        return None

    search_from = match.end()
    end_index: int | None = None
    if match.group("binding"):
        arrow = content.find("=>", search_from)
        semicolon = content.find(";", search_from)
        if arrow >= 0 and (semicolon < 0 or arrow < semicolon):
            cursor = arrow + 2
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
            if cursor < len(content) and content[cursor] in "{(":
                opener = content[cursor]
                closer = "}" if opener == "{" else ")"
                end_index = _balanced_source_end(content, cursor, opener, closer)
        if end_index is None:
            brace = content.find("{", search_from)
            if brace >= 0 and (semicolon < 0 or brace < semicolon):
                end_index = _balanced_source_end(content, brace, "{", "}")
    elif match.group("kind"):
        body_search_from = search_from
        kind = match.group("kind") or ""
        if "function" in kind:
            # A typed/destructured parameter can contain braces before the
            # actual function body, e.g. function Panel({ value }: { ... }) {.
            # Skip the complete parameter list before looking for the body.
            params = content.find("(", search_from)
            if params >= 0:
                params_end = _balanced_source_end(content, params, "(", ")")
                if params_end is not None:
                    body_search_from = params_end
        brace = content.find("{", body_search_from)
        if brace >= 0:
            end_index = _balanced_source_end(content, brace, "{", "}")
    else:
        equals = content.find("=", match.start(), search_from + 1)
        cursor = equals + 1 if equals >= 0 else search_from
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor < len(content) and content[cursor] in "{(":
            opener = content[cursor]
            closer = "}" if opener == "{" else ")"
            end_index = _balanced_source_end(content, cursor, opener, closer)

    if end_index is None:
        semicolon = content.find(";", search_from)
        newline = content.find("\n", search_from)
        candidates = [
            value
            for value in (
                semicolon + 1 if semicolon >= 0 else -1,
                newline,
            )
            if value >= 0
        ]
        end_index = min(candidates) if candidates else len(content)

    cursor = end_index
    while cursor < len(content) and content[cursor] in " \t":
        cursor += 1
    if cursor < len(content) and content[cursor] == ";":
        end_index = cursor + 1

    start_line = content.count("\n", 0, match.start()) + 1
    end_line = content.count("\n", 0, max(match.start(), end_index - 1)) + 1
    return start_line, end_line


def _python_symbol_node(tree: ast.Module, symbol: str) -> ast.AST | None:
    """Resolve one current Python definition, including Class.method locators."""
    parts = symbol.split(".")
    if len(parts) == 1:
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        ]
        return matches[0] if len(matches) == 1 else None

    body: list[ast.stmt] = list(tree.body)
    current: ast.AST | None = None
    for position, part in enumerate(parts):
        matches = [
            node
            for node in body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == part
        ]
        if len(matches) != 1:
            return None
        current = matches[0]
        if position < len(parts) - 1:
            if not isinstance(current, ast.ClassDef):
                return None
            body = list(current.body)
    return current


def _fresh_symbol_ranges(content: str, path: Path, symbols: list[str]) -> list[tuple[int, int]]:
    """Locate requested definitions in fresh Python or JS/TS source."""
    if not symbols:
        return []
    suffix = path.suffix.lower()
    if suffix in GENERIC_SYMBOL_SUFFIXES:
        spans = {
            symbol: _generic_symbol_span(content, symbol)
            for symbol in symbols
        }
        return [
            span
            for symbol in symbols
            if (span := spans.get(symbol)) is not None
        ]
    if suffix != ".py":
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    ranges: list[tuple[int, int, bool, str]] = []
    for symbol in symbols:
        node = _python_symbol_node(tree, symbol)
        if (
            node is None
            or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            or not hasattr(node, "end_lineno")
        ):
            continue
        start = min(
            [node.lineno, *(item.lineno for item in node.decorator_list)]
            if node.decorator_list else [node.lineno]
        )
        ranges.append((
            start,
            node.end_lineno,
            isinstance(node, ast.ClassDef),
            symbol,
        ))
    # When an explicit class and concrete methods inside it are both targets,
    # the methods are the precise patch surface. Emitting the enclosing class
    # would turn distant methods into one enormous range and waste the budget.
    filtered = [
        (start, end, symbol)
        for start, end, is_class, symbol in ranges
        if not is_class or not any(
            not other_is_class and start <= other_start and other_end <= end
            for other_start, other_end, other_is_class, _other_symbol in ranges
        )
    ]
    order = {symbol: index for index, symbol in enumerate(symbols)}
    return [
        (start, end)
        for start, end, symbol in sorted(
            filtered, key=lambda item: (order.get(item[2], len(order)), item[0])
        )
    ]


def _fresh_symbol_node(
    path: Path, symbol: str,
) -> tuple[str, list[str], int, int] | None:
    """Return one complete current definition, calls, and exact source range."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = content.splitlines(keepends=True)
    if path.suffix.lower() in GENERIC_SYMBOL_SUFFIXES:
        span = _generic_symbol_span(content, symbol)
        if span is None:
            return None
        start, end = span
        return (
            "".join(lines[start - 1:end]),
            [],
            start,
            end,
        )

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    node = _python_symbol_node(tree, symbol)
    if (
        node is None
        or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        or not hasattr(node, "end_lineno")
    ):
        return None
    start = min(
        [node.lineno, *(item.lineno for item in node.decorator_list)]
        if node.decorator_list else [node.lineno]
    )
    calls = [
        call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
        for call in sorted(
            (item for item in ast.walk(node) if isinstance(item, ast.Call)),
            key=lambda item: (item.lineno, item.col_offset),
        )
        if isinstance(call.func, (ast.Attribute, ast.Name))
    ]
    # Decorator callbacks are implementation dependencies even though the
    # callback is passed by name rather than invoked in the function body
    # (for example ``@autocomplete(item=choice_provider)``).
    for decorator in node.decorator_list:
        for call in (
            item for item in ast.walk(decorator) if isinstance(item, ast.Call)
        ):
            calls.extend(
                keyword.value.id
                for keyword in call.keywords
                if isinstance(keyword.value, ast.Name)
            )
    return (
        "".join(lines[start - 1:node.end_lineno]),
        list(dict.fromkeys(calls)),
        start,
        node.end_lineno,
    )


def _full_current_file_section(repo: Path, path: Path) -> str:
    content, digest = _read_current_text_and_hash(path)
    relative = path.relative_to(repo).as_posix()
    line_count = max(1, len(content.splitlines()))
    return (
        f"===== FULL FILE: {relative} =====\n"
        f"SHA-256: {digest}\n"
        f"Source line range: 1-{line_count}\n\n"
        f"{content}\n"
    )


def _context_contract_blocks(
    repo: Path,
    targets: list[tuple[Path, str, str]],
) -> tuple[
    list[tuple[str, list[tuple[Path, str, str]], str]],
    list[tuple[Path, str, str]],
]:
    """Build exact mandatory source blocks before any budget allocation."""
    grouped: dict[Path, list[tuple[str, str]]] = {}
    seen: set[tuple[Path, str]] = set()
    for path, symbol, priority in targets:
        key = (path, symbol)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(path, []).append((symbol, priority))

    blocks: list[tuple[str, list[tuple[Path, str, str]], str]] = []
    unresolved: list[tuple[Path, str, str]] = []
    for path, symbols in grouped.items():
        exact: list[tuple[str, tuple[Path, str, str]]] = []
        missing: list[tuple[Path, str, str]] = []
        relative = path.relative_to(repo).as_posix()
        for symbol, priority in symbols:
            node = _fresh_symbol_node(path, symbol)
            key = (path, symbol, priority)
            if node is None:
                missing.append(key)
                continue
            section = (
                f"===== SYMBOL CONTEXT: {relative}::{symbol} [{priority}] "
                f"source lines {node[2]}-{node[3]} =====\n"
                f"{node[0]}\n"
            )
            exact.append((section, key))

        full_section = ""
        try:
            candidate = _full_current_file_section(repo, path)
            if len(candidate) <= PATCH_CONTRACT_FULL_FILE_MAX_CHARS:
                full_section = candidate
        except OSError:
            pass

        exact_chars = sum(len(section) for section, _key in exact)
        promote_full = bool(full_section) and (
            bool(missing)
            or (
                len(exact) >= 3
                and len(full_section)
                <= int(max(1, exact_chars) * PATCH_CONTRACT_FULL_FILE_RATIO)
            )
        )
        if promote_full:
            keys = [
                (path, symbol, priority)
                for symbol, priority in symbols
            ]
            blocks.append((full_section, keys, "fallback"))
            continue

        blocks.extend(
            (section, [key], "rendered")
            for section, key in exact
        )
        unresolved.extend(missing)

    return blocks, unresolved


def _render_context_contract_blocks(
    repo: Path,
    blocks: list[tuple[str, list[tuple[Path, str, str]], str]],
    unresolved: list[tuple[Path, str, str]],
    budget: int,
) -> tuple[str, dict[tuple[Path, str], tuple[str, str]]]:
    sections: list[str] = []
    states: dict[tuple[Path, str], tuple[str, str]] = {}
    missing = list(unresolved)
    used = 0

    for section, keys, state in blocks:
        if used + len(section) > budget:
            missing.extend(keys)
            continue
        sections.append(section)
        used += len(section)
        for path, symbol, priority in keys:
            states[(path, symbol)] = (
                ("fallback", "FULL FILE")
                if state == "fallback"
                else ("rendered", priority)
            )

    for path, symbol, priority in dict.fromkeys(missing):
        if (path, symbol) in states:
            continue
        relative = path.relative_to(repo).as_posix()
        states[(path, symbol)] = ("unavailable", "context contract")
        sections.append(
            f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
            "[reason: context contract; do not patch] =====\n"
        )

    return "\n".join(sections), states


def _effective_context_budget(base_budget: int, required_chars: int) -> int:
    """Grow only broad contract contexts, never past the configured hard cap."""
    if required_chars <= base_budget:
        return base_budget
    hard_budget = _context_hard_budget_chars()
    desired = required_chars + PATCH_CONTEXT_ELASTIC_RESERVE_CHARS
    return min(hard_budget, max(base_budget, desired))


def _materialization_targets(
    repo: Path, target_symbols: dict[Path, list[str]], max_inherited: int = 64,
) -> list[tuple[Path, str, str]]:
    """Order patch targets first and inherit priority through two direct calls."""
    index = load_map(repo).get("files", {})
    owners: dict[str, list[tuple[Path, str]]] = {}
    for relative, entry in index.items():
        for symbol in entry.get("symbols", []):
            if not isinstance(symbol, dict) or not symbol.get("name"):
                continue
            owners.setdefault(str(symbol["name"]).casefold(), []).append((
                repo / relative,
                str(symbol.get("definition_name") or symbol["name"]),
            ))
    # Large or dirty files can contain fresh methods not represented by the
    # cached symbol list. Required files are already in scope, so supplement
    # owner lookup from their current AST without broadening retrieval.
    for path in target_symbols:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            owner = (path, node.name)
            bucket = owners.setdefault(node.name.casefold(), [])
            if owner not in bucket:
                bucket.append(owner)

    original = [(path, symbol) for path, symbols in target_symbols.items() for symbol in symbols]
    command_roots = [
        item for item in original
        if "commands" in {part.casefold() for part in item[0].relative_to(repo).parts}
    ]
    ordered: list[tuple[Path, str, str]] = []
    seen: set[tuple[Path, str]] = set()

    def add(path: Path, symbol: str, priority: str) -> bool:
        key = (path, symbol)
        if key in seen:
            return False
        seen.add(key)
        ordered.append((path, symbol, priority))
        return True

    # Every command/task-surface root is patch-critical. Register all of them
    # before walking dependencies so a broad first command cannot exhaust the
    # budget before a disconnected later surface reaches materialization.
    for path, symbol in command_roots:
        add(path, symbol, "patch target")

    # Complete same-file implementation chains first. They are part of the
    # concrete patch surface, not optional project-wide dependency expansion;
    # otherwise a broad sibling class can consume the external-dependency cap
    # before a small renderer chain in the same module is materialized.
    def add_same_file_dependencies(path: Path, symbol: str, depth: int) -> None:
        if depth >= 2:
            return
        node = _fresh_symbol_node(path, symbol)
        if node is None:
            return
        for called in node[1]:
            for owner, definition in owners.get(called.casefold(), []):
                if owner != path:
                    continue
                if add(owner, definition, "direct implementation dependency"):
                    add_same_file_dependencies(owner, definition, depth + 1)

    for path, symbol in command_roots:
        add_same_file_dependencies(path, symbol, 0)

    frontier = [(path, symbol, 0) for path, symbol in command_roots]
    inherited = 0
    while frontier and inherited < max_inherited:
        path, symbol, depth = frontier.pop(0)
        if depth >= 2:
            continue
        node = _fresh_symbol_node(path, symbol)
        if node is None:
            continue
        for called in node[1]:
            for owner, definition in sorted(
                owners.get(called.casefold(), []), key=lambda item: str(item[0]).lower()
            ):
                if inherited >= max_inherited:
                    break
                if add(owner, definition, "direct implementation dependency"):
                    inherited += 1
                    frontier.append((owner, definition, depth + 1))

    for path, symbol in original:
        add(path, symbol, "required implementation")
    return ordered


def _render_required_symbols(
    repo: Path,
    task: str,
    targets: list[tuple[Path, str, str]],
    required_paths: list[Path],
    budget: int,
    changed_files: set[Path],
    explicit_files: set[Path],
) -> tuple[str, dict[tuple[Path, str], tuple[str, str]]]:
    """Render required paths first, with current-source fallback per file."""
    sections: list[str] = []
    states: dict[tuple[Path, str], tuple[str, str]] = {}
    used = 0
    grouped: dict[Path, list[tuple[str, str]]] = {}
    for path in required_paths:
        grouped.setdefault(path, [])
    for path, symbol, priority in targets:
        grouped.setdefault(path, []).append((symbol, priority))

    grouped_items = list(grouped.items())
    required_path_set = set(required_paths)
    coverage_demands: dict[Path, int] = {}
    for path, symbols in grouped_items:
        coverage_count = sum(
            priority == "source coverage"
            for _symbol, priority in symbols
        )
        if coverage_count:
            # Demand is intentionally structural rather than byte-perfect. It
            # gives files with several required regions a larger share without
            # reparsing large source files just to estimate their size.
            coverage_demands[path] = 1_600 + coverage_count * 2_800

    for position, (path, symbols) in enumerate(grouped_items):
        relative = path.relative_to(repo).as_posix()
        coverage_path = path in coverage_demands
        if coverage_path:
            remaining_demand = sum(
                coverage_demands.get(candidate, 0)
                for candidate, _symbols in grouped_items[position:]
            )
            allowance = max(
                1,
                max(0, budget - used)
                * coverage_demands[path]
                // max(1, remaining_demand),
            )
        else:
            remaining_paths = (
                sum(
                    candidate in required_path_set
                    for candidate, _symbols in grouped_items[position:]
                )
                if path in required_path_set
                else len(grouped_items) - position
            )
            allowance = max(1, (budget - used) // remaining_paths)
        if path in changed_files and path.suffix.casefold() == ".py":
            dirty_symbols = set(
                _changed_python_definition_roots(repo, [path]).get(path, [])
            )
            task_identifiers = get_task_words(task)
            symbols = sorted(
                enumerate(symbols),
                key=lambda item: (
                    -sum(
                        len(word)
                        for word in task_identifiers
                        if word in item[1][0].casefold()
                    ),
                    item[1][0] not in dirty_symbols,
                    item[0],
                ),
            )
            symbols = [item for _index, item in symbols]
        resolved = [
            (symbol, priority, _fresh_symbol_node(path, symbol))
            for symbol, priority in symbols
        ]
        fallback_symbols = [symbol for symbol, _priority, node in resolved if node is None]
        path_fallback_required = not symbols or path in required_paths
        exact_size = sum(
            len(
                f"===== SYMBOL CONTEXT: {relative}::{symbol} [{priority}] "
                f"source lines {node[2]}-{node[3]} =====\n{node[0]}\n"
            )
            for symbol, priority, node in resolved
            if node is not None
        )
        needs_fallback = path_fallback_required or bool(fallback_symbols) or exact_size > allowance
        if coverage_path and needs_fallback:
            fallback_reserve = min(2_200, max(600, allowance // 10))
            exact_allowance = max(1, allowance - fallback_reserve)
        else:
            exact_allowance = allowance // 2 if needs_fallback else allowance
        local_used = 0

        for symbol, priority, node in resolved:
            if node is None:
                continue
            section = (
                f"===== SYMBOL CONTEXT: {relative}::{symbol} [{priority}] "
                f"source lines {node[2]}-{node[3]} =====\n"
                f"{node[0]}\n"
            )
            if local_used + len(section) <= exact_allowance:
                sections.append(section)
                local_used += len(section)
                states[(path, symbol)] = ("rendered", priority)
            else:
                fallback_symbols.append(symbol)

        fallback_section = ""
        fallback_allowance = allowance - local_used
        if (path_fallback_required or fallback_symbols) and fallback_allowance > 0:
            try:
                unique_fallback_symbols = list(dict.fromkeys(fallback_symbols))
                fallback_section = _render_patch_file_context(
                    repo,
                    path,
                    task,
                    changed_files,
                    explicit_files,
                    max_chars=fallback_allowance,
                    required_symbols=unique_fallback_symbols or None,
                    focus_symbols=unique_fallback_symbols,
                )
            except OSError:
                fallback_section = ""
        if fallback_section:
            sections.append(fallback_section)
            local_used += len(fallback_section)
            fallback_type = (
                "FULL FILE" if fallback_section.startswith("===== FULL FILE:")
                else "EXCERPT"
            )
            for symbol in fallback_symbols:
                states[(path, symbol)] = ("fallback", fallback_type)

        used += local_used
        for symbol, _priority in symbols:
            if (path, symbol) in states:
                continue
            reason = (
                "source could not be read or bounded safely"
                if not path.is_file() else "budget"
            )
            states[(path, symbol)] = ("unavailable", reason)
            sections.append(
                f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
                f"[reason: {reason}; do not patch] =====\n"
            )
    return "\n".join(sections), states


def _enforce_materialization_invariant(
    repo: Path,
    rendered: str,
    targets: list[tuple[Path, str, str]],
    states: dict[tuple[Path, str], tuple[str, str]],
) -> str:
    """Ensure every required symbol is complete or explicitly non-patchable."""
    repairs: list[str] = []
    for path, symbol, _priority in targets:
        state = states.get((path, symbol), ("unavailable", "missing state"))[0]
        if state == "fallback":
            continue
        relative = path.relative_to(repo).as_posix()
        unavailable = f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
        if unavailable in rendered:
            continue
        node = _fresh_symbol_node(path, symbol)
        header = (
            f"===== SYMBOL CONTEXT: {relative}::{symbol} [{_priority}] "
            f"source lines {node[2]}-{node[3]} ====="
            if node is not None else ""
        )
        if node is not None and header in rendered and node[0] in rendered:
            continue
        repairs.append(
            f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
            "[reason: post-render validation; do not patch] =====\n"
        )
    return rendered + ("\n" + "\n".join(repairs) if repairs else "")


def _dependency_paths(repo: Path, files: list[Path]) -> set[Path]:
    try:
        graph = load_map(repo)
    except Exception:
        return set()
    entries = graph.get("files", {})
    if not isinstance(entries, dict):
        return set()
    selected = {
        path.relative_to(repo).as_posix()
        for path in files
        if path.is_file()
    }
    dependencies: set[Path] = set()
    for relative in selected:
        entry = entries.get(relative, {})
        if not isinstance(entry, dict):
            continue
        for raw_dependency in entry.get("dependencies", []):
            if not isinstance(raw_dependency, str):
                continue
            candidate = (repo / raw_dependency).resolve()
            if candidate.is_file():
                dependencies.add(candidate)
    return dependencies


