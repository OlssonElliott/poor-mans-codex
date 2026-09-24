"""Follow-up workflow state and attempted-symbol history."""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path, PurePosixPath

from ..models import ApplyResult, TestValidation
from ...workspace import atomic_write_text, get_followup_state_file


MAX_FOLLOWUP_ROUNDS = 5


def load_followup_state(repo: Path) -> dict | None:
    try:
        state = json.loads(get_followup_state_file(repo).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) and state.get("version") == 1 else None


def save_followup_state(repo: Path, state: dict) -> None:
    atomic_write_text(
        get_followup_state_file(repo),
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        newline="\n",
    )


def mark_followup_resolved(repo: Path) -> None:
    state = load_followup_state(repo)
    if state is None or not state.get("unresolved"):
        return
    state["unresolved"] = False
    state["updated_at"] = time.time()
    save_followup_state(repo, state)


def followup_attempt(result: ApplyResult, validation: TestValidation, patch_summary: str) -> dict:
    return {
        "history_entry": str(result.history_entry.resolve()),
        "paths": sorted(result.paths),
        "patch_summary": patch_summary,
        "validation_status": validation.status,
    }


def result_from_followup_attempt(attempt: dict) -> ApplyResult | None:
    history_entry = attempt.get("history_entry")
    paths = attempt.get("paths")
    if not isinstance(history_entry, str) or not isinstance(paths, list):
        return None
    return ApplyResult(
        {path for path in paths if isinstance(path, str)}, Path(history_entry)
    )


def attempted_python_symbols_by_path(
    result: ApplyResult,
) -> dict[str, set[str]]:
    """Recover changed Python definitions while preserving their file owner."""
    attempted: dict[str, set[str]] = {}
    for relative in result.paths:
        if not relative.casefold().endswith(".py"):
            continue
        parts = PurePosixPath(relative).parts
        snapshots: list[dict[str, str]] = []
        for state in ("before", "after"):
            path = result.history_entry.joinpath(state, *parts)
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (OSError, SyntaxError):
                snapshots.append({})
                continue
            snapshots.append({
                node.name: ast.get_source_segment(source, node) or ""
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            })
        before, after = snapshots
        changed = {
            name for name in before.keys() | after.keys()
            if before.get(name) != after.get(name)
        }
        if changed:
            attempted[relative] = changed
    return attempted


def attempted_python_symbols(result: ApplyResult) -> set[str]:
    """Compatibility helper for follow-up alternative discovery."""
    return {
        symbol
        for symbols in attempted_python_symbols_by_path(result).values()
        for symbol in symbols
    }


