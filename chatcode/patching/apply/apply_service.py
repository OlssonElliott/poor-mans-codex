"""Apply-core and apply-state wiring with explicit compatibility hooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from ..models import ApplyResult, PatchPreview
from . import apply_core, apply_state


@dataclass(frozen=True)
class ApplyHooks:
    get_default_patch_file_fn: Callable[[Path], Path]
    patch_supersedes_active_repair_fn: Callable
    supersede_active_repair_fn: Callable[[Path], None]
    get_stale_context_reason_fn: Callable
    build_syntax_repair_context_fn: Callable
    build_stale_repair_context_fn: Callable
    build_patch_repair_context_fn: Callable
    get_context_kind_fn: Callable[[Path], str | None]
    get_context_task_fn: Callable[[Path], str | None]
    save_context_state_fn: Callable
    get_repair_context_file_fn: Callable[[Path], Path]
    repair_state_file_fn: Callable[[Path], Path]


def apply_patch_core(
    hooks: ApplyHooks,
    repo: Path,
    patch_file: Path,
    *,
    dry_run: bool = False,
    new_task: bool = False,
) -> ApplyResult | PatchPreview:
    return apply_core.apply_patch_core(
        repo,
        patch_file,
        dry_run=dry_run,
        new_task=new_task,
        get_default_patch_file_fn=hooks.get_default_patch_file_fn,
        patch_supersedes_active_repair_fn=hooks.patch_supersedes_active_repair_fn,
        supersede_active_repair_fn=hooks.supersede_active_repair_fn,
        get_stale_context_reason_fn=hooks.get_stale_context_reason_fn,
        build_syntax_repair_context_fn=hooks.build_syntax_repair_context_fn,
        build_stale_repair_context_fn=hooks.build_stale_repair_context_fn,
        build_patch_repair_context_fn=hooks.build_patch_repair_context_fn,
        get_context_kind_fn=hooks.get_context_kind_fn,
        get_context_task_fn=hooks.get_context_task_fn,
        save_context_state_fn=hooks.save_context_state_fn,
        clear_repair_context_fn=lambda current_repo: clear_repair_context(
            hooks, current_repo
        ),
        clear_incoming_patch_fn=lambda current_repo: clear_incoming_patch(
            hooks, current_repo
        ),
    )


def clear_repair_context(
    hooks: ApplyHooks,
    repo: Path,
) -> None:
    return apply_state.clear_repair_context(
        repo,
        get_repair_context_file_fn=hooks.get_repair_context_file_fn,
        repair_state_file_fn=hooks.repair_state_file_fn,
    )


def clear_incoming_patch(
    hooks: ApplyHooks,
    repo: Path,
) -> None:
    return apply_state.clear_incoming_patch(
        repo,
        get_default_patch_file_fn=hooks.get_default_patch_file_fn,
    )
