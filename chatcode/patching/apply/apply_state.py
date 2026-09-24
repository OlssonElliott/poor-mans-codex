"""Cleanup of single-use patch workflow artifacts."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def clear_repair_context(
    repo: Path,
    *,
    get_repair_context_file_fn: Callable[[Path], Path],
    repair_state_file_fn: Callable[[Path], Path],
) -> None:
    for path in (get_repair_context_file_fn(repo), repair_state_file_fn(repo)):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def clear_incoming_patch(
    repo: Path,
    *,
    get_default_patch_file_fn: Callable[[Path], Path],
) -> None:
    incoming = get_default_patch_file_fn(repo)
    try:
        if incoming.exists():
            incoming.write_text(
                "",
                encoding="utf-8",
                newline="\n",
            )
    except OSError:
        pass
