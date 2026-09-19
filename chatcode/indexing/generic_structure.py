from __future__ import annotations

import re


GENERIC_SYMBOL_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx"})

GENERIC_SYMBOL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:(?P<kind>class|interface|function|enum)\s+"
    r"(?P<named>[A-Za-z_$][\w$]*)|"
    r"type\s+(?P<type_name>[A-Za-z_$][\w$]*)\s*=|"
    r"(?:const|let|var)\s+(?P<binding>[A-Za-z_$][\w$]*)"
    r"(?:\s*:\s*[^=\n]+)?\s*=\s*(?:async\s*)?"
    r"(?:<[^>\n]+>\s*)?\([^)]*\)(?:\s*:\s*[^=\n]+)?\s*=>)",
    re.MULTILINE,
)


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


def generic_symbol_span(
    content: str,
    symbol: str,
) -> tuple[int, int] | None:
    """Locate one complete named JS/TS declaration in current source."""
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
