"""Public patch API orchestration."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .models import ApplyResult, UndoResult


def apply_patch(
    repo: Path,
    patch_file: Path,
    *,
    new_task: bool = False,
    cli_apply_invocation: bool,
    cli_apply_yes: bool,
    run_apply_flow_fn: Callable,
    apply_patch_core_fn: Callable,
) -> ApplyResult:
    if cli_apply_invocation:
        run_apply_flow_fn(
            repo,
            patch_file,
            yes=cli_apply_yes,
            new_task=new_task,
        )
        raise SystemExit(0)

    result = apply_patch_core_fn(
        repo,
        patch_file,
        new_task=new_task,
    )
    assert isinstance(result, ApplyResult)
    return result


def undo_last_patch(
    repo: Path,
    *,
    undo_last_patch_fn: Callable,
    validate_patch_paths_fn: Callable[[str], set[str]],
) -> UndoResult:
    return undo_last_patch_fn(
        repo,
        validate_patch_paths_fn=validate_patch_paths_fn,
    )
