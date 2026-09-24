"""Patch repair workflow wiring with explicit compatibility hooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from ..models import ApplyResult, TestValidation
from . import repair_state
from . import repair_context


@dataclass(frozen=True)
class RepairHooks:
    get_context_task_fn: Callable[[Path], str | None]
    save_context_state_fn: Callable
    clear_incoming_fn: Callable[[Path], None]
    clear_repair_fn: Callable[[Path], None]
    get_context_kind_fn: Callable[[Path], str | None]
    safe_candidate_paths_fn: Callable[[Path, str], list[str]]
    working_tree_section_fn: Callable
    patch_hunks_fn: Callable


def write_repair_context(
    hooks: RepairHooks,
    repo: Path,
    content: str,
    paths: set[str] | list[str],
    repair_targets: set[str] | frozenset[str] = frozenset(),
) -> Path:
    return repair_state.write_repair_context(
        repo,
        content,
        paths,
        repair_targets,
        get_context_task_fn=hooks.get_context_task_fn,
        save_context_state_fn=hooks.save_context_state_fn,
        clear_incoming_fn=hooks.clear_incoming_fn,
    )


def patch_supersedes_active_repair(
    hooks: RepairHooks,
    repo: Path,
    patch_paths: set[str],
    *,
    force: bool = False,
) -> bool:
    return repair_state.patch_supersedes_active_repair(
        repo,
        patch_paths,
        force=force,
        get_context_kind_fn=hooks.get_context_kind_fn,
    )


def supersede_active_repair(
    hooks: RepairHooks,
    repo: Path,
) -> None:
    return repair_state.supersede_active_repair(
        repo,
        save_context_state_fn=hooks.save_context_state_fn,
        clear_repair_fn=hooks.clear_repair_fn,
    )


def build_syntax_repair_context(
    hooks: RepairHooks,
    write_repair_context_fn: Callable,
    repo: Path,
    original_patch: str,
    error: str,
    malformed_hunk: str | None = None,
    failure_type: str = "invalid_patch_syntax",
) -> Path:
    return repair_context.build_syntax_repair_context(
        repo,
        original_patch,
        error,
        malformed_hunk,
        failure_type,
        safe_candidate_paths_fn=hooks.safe_candidate_paths_fn,
        working_tree_section_fn=hooks.working_tree_section_fn,
        write_repair_context_fn=write_repair_context_fn,
        get_context_task_fn=hooks.get_context_task_fn,
    )


def build_stale_repair_context(
    hooks: RepairHooks,
    write_repair_context_fn: Callable,
    repo: Path,
    patch_text: str,
    stale_reason: str,
    paths: set[str],
) -> Path:
    return repair_context.build_stale_repair_context(
        repo,
        patch_text,
        stale_reason,
        paths,
        working_tree_section_fn=hooks.working_tree_section_fn,
        write_repair_context_fn=write_repair_context_fn,
        get_context_task_fn=hooks.get_context_task_fn,
    )


def build_patch_repair_context(
    hooks: RepairHooks,
    write_repair_context_fn: Callable,
    repo: Path,
    patch_text: str,
    apply_error: str,
) -> Path:
    return repair_context.build_patch_repair_context(
        repo,
        patch_text,
        apply_error,
        patch_hunks_fn=hooks.patch_hunks_fn,
        safe_candidate_paths_fn=hooks.safe_candidate_paths_fn,
        working_tree_section_fn=hooks.working_tree_section_fn,
        write_repair_context_fn=write_repair_context_fn,
        get_context_task_fn=hooks.get_context_task_fn,
    )


def build_test_failure_repair_context(
    hooks: RepairHooks,
    write_repair_context_fn: Callable,
    repo: Path,
    result: ApplyResult,
    validation: TestValidation,
) -> Path:
    return repair_context.build_test_failure_repair_context(
        repo,
        result,
        validation,
        working_tree_section_fn=hooks.working_tree_section_fn,
        write_repair_context_fn=write_repair_context_fn,
        get_context_task_fn=hooks.get_context_task_fn,
    )
