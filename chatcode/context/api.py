"""Canonical context-building API for internal subsystem callers."""
from __future__ import annotations

from pathlib import Path

from ..indexing.index_manager import update_project_map
from ..retrieval.graph_retriever import retrieve_files
from ..retrieval.source_coverage import plan_source_coverage
from ..retrieval.context_contract import plan_context_contract
from . import builder, capture, retrieval_pipeline
from .files import (
    _resolve_task_file_references,
    get_changed_files,
)


MAX_FILES = 12


def collect_relevant_files(
    repo: Path,
    task: str,
    include_target_symbols: bool = False,
):
    return retrieval_pipeline.collect_relevant_files(
        repo,
        task,
        include_target_symbols=include_target_symbols,
        max_files=MAX_FILES,
        update_project_map_fn=update_project_map,
        retrieve_files_fn=retrieve_files,
        get_changed_files_fn=get_changed_files,
        runtime_module_file=__file__,
    )


def build_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path] | None = None,
    target_symbols: dict[Path, list[str]] | None = None,
    critical_paths: list[Path] | None = None,
) -> str:
    return builder.build_patch_source_context(
        repo,
        task,
        files,
        target_symbols,
        critical_paths,
        collect_relevant_files_fn=collect_relevant_files,
        get_changed_files_fn=get_changed_files,
        resolve_task_file_references_fn=_resolve_task_file_references,
        plan_source_coverage_fn=plan_source_coverage,
        plan_context_contract_fn=plan_context_contract,
    )


def build_stable_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path],
    target_symbols: dict[Path, list[str]] | None = None,
):
    return capture.build_stable_patch_source_context(
        repo,
        task,
        files,
        target_symbols,
        build_patch_source_context_fn=build_patch_source_context,
    )


def build_context_from_test_roots(
    repo: Path,
    test_ids: frozenset[str],
    traceback_paths: list[Path] | None = None,
) -> tuple[str, list[Path]]:
    return retrieval_pipeline.build_context_from_test_roots(
        repo,
        test_ids,
        traceback_paths,
        update_project_map_fn=update_project_map,
        build_stable_patch_source_context_fn=build_stable_patch_source_context,
    )
