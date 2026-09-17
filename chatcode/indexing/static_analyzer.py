from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


MAX_SYMBOLS = 100
MAX_IMPORTS = 100


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[dict[str, str]] = []
        self.imports: list[str] = []
        self.class_name: str | None = None
        self.function_depth = 0

    def _symbol(self, name: str, qualified: str, kind: str) -> None:
        if len(self.symbols) < MAX_SYMBOLS:
            self.symbols.append({
                "name": name,
                "qualified_name": qualified,
                "kind": kind,
            })

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.function_depth == 0:
            qualified = f"{self.class_name}.{node.name}" if self.class_name else node.name
            self._symbol(node.name, qualified, "method" if self.class_name else "function")
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                decorator_name = (
                    decorator.func.attr if isinstance(decorator.func, ast.Attribute)
                    else decorator.func.id if isinstance(decorator.func, ast.Name)
                    else ""
                )
                if decorator_name != "command":
                    continue
                command_name = next((
                    keyword.value.value
                    for keyword in decorator.keywords
                    if keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ), node.name)
                if command_name == node.name:
                    # The command name is still meaningful metadata even when
                    # it matches the Python handler name.
                    for symbol in reversed(self.symbols):
                        if symbol.get("qualified_name") == qualified:
                            symbol["kind"] = "command"
                            symbol["definition_name"] = node.name
                            break
                elif len(self.symbols) < MAX_SYMBOLS:
                    self.symbols.append({
                        "name": command_name,
                        "qualified_name": qualified,
                        "kind": "command",
                        "definition_name": node.name,
                    })
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.function_depth:
            return
        previous = self.class_name
        qualified = f"{previous}.{node.name}" if previous else node.name
        self._symbol(node.name, qualified, "class")
        self.class_name = qualified
        self.generic_visit(node)
        self.class_name = previous

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = "." * node.level + (node.module or "")
        if base:
            self.imports.append(base)
        for alias in node.names:
            if alias.name != "*":
                separator = "" if base.endswith(".") else "."
                self.imports.append(f"{base}{separator}{alias.name}")


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
GENERIC_IMPORT = re.compile(
    r"(?:import\s+(?:[^;]*?\s+from\s+)?|require\s*\(\s*)['\"]([^'\"]+)['\"]"
)


def _analyze_generic(text: str) -> dict[str, Any]:
    symbols: list[dict[str, str]] = []
    for match in GENERIC_SYMBOL.finditer(text):
        kind = match.group("kind") or ("type" if match.group("type_name") else "function")
        name = match.group("named") or match.group("type_name") or match.group("binding")
        symbols.append({"name": name, "qualified_name": name, "kind": kind})
        if len(symbols) >= MAX_SYMBOLS:
            break
    return {
        "language": "javascript_typescript",
        "symbols": symbols,
        "imports": list(dict.fromkeys(GENERIC_IMPORT.findall(text)))[:MAX_IMPORTS],
    }


def analyze_file(
    path: Path,
    repo: Path,
    source: str | None = None,
) -> dict[str, Any]:
    """Analyze a caller-supplied source snapshot when one is available."""
    text = source if source is not None else path.read_text(
        encoding="utf-8", errors="replace"
    )
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(text, filename=path.relative_to(repo).as_posix())
        except SyntaxError as exc:
            return {
                "language": "python",
                "symbols": [],
                "imports": [],
                "error": str(exc),
            }
        visitor = _PythonVisitor()
        visitor.visit(tree)
        return {
            "language": "python",
            "symbols": visitor.symbols[:MAX_SYMBOLS],
            "imports": list(dict.fromkeys(visitor.imports))[:MAX_IMPORTS],
        }
    return _analyze_generic(text)
