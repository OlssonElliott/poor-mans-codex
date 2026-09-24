from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

# Compatibility patch points used by tests and external callers.
from .file_filter import iter_repository_files
from .indexing.index_manager import IndexProgress, update_project_map
from .retrieval.graph_retriever import retrieve_files
from .retrieval.hybrid_retriever import QwenCompletenessChecker
from .retrieval.source_coverage import plan_source_coverage
from .retrieval.context_contract import plan_context_contract

from .context.errors import ContextBuildError
from .context import capture as _capture
from .context import retrieval_pipeline as _retrieval_pipeline
from .context import builder as _context_builder
from .context import safety as _safety
from .context import renderer as _renderer
from .context import prompts as _prompts
from .context.files import (
    _resolve_task_file_references,
    _ensure_explicit_task_files,
    build_tree,
    get_changed_files,
)

MAX_FILES = 12
MAX_FILE_CHARS = 20_000
MAX_TOTAL_CHARS = 120_000

CONTEXT_PURPOSE = _prompts.CONTEXT_PURPOSE
PATCH_RESPONSE_INSTRUCTIONS = _prompts.PATCH_RESPONSE_INSTRUCTIONS
PATCH_CONTEXT_HEADER = _prompts.PATCH_CONTEXT_HEADER
UPLOAD_INSTRUCTIONS = _prompts.UPLOAD_INSTRUCTIONS

def build_safe_status(repo: Path) -> str:
    return _safety.build_safe_status(repo)


def redact_sensitive_text(text: str) -> str:
    return _safety.redact_sensitive_text(text)

def collect_relevant_files(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
    include_target_symbols: bool = False,
) -> list[Path] | tuple[list[Path], dict[Path, list[str]]]:
    return _retrieval_pipeline.collect_relevant_files(
        repo,
        task,
        index_progress=index_progress,
        include_target_symbols=include_target_symbols,
        max_files=MAX_FILES,
        update_project_map_fn=update_project_map,
        retrieve_files_fn=retrieve_files,
        get_changed_files_fn=get_changed_files,
        runtime_module_file=__file__,
    )


def build_source_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> str:
    return _retrieval_pipeline.build_source_context(
        repo,
        task,
        index_progress=index_progress,
        max_files=MAX_FILES,
        max_file_chars=MAX_FILE_CHARS,
        max_total_chars=MAX_TOTAL_CHARS,
        collect_relevant_files_fn=collect_relevant_files,
        redact_sensitive_text_fn=redact_sensitive_text,
    )

def build_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path] | None = None,
    target_symbols: dict[Path, list[str]] | None = None,
    critical_paths: list[Path] | None = None,
) -> str:
    return _context_builder.build_patch_source_context(
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

def build_context_from_test_roots(
    repo: Path,
    test_ids: frozenset[str],
    traceback_paths: list[Path] | None = None,
) -> tuple[str, list[Path]]:
    return _retrieval_pipeline.build_context_from_test_roots(
        repo,
        test_ids,
        traceback_paths,
        update_project_map_fn=update_project_map,
        build_stable_patch_source_context_fn=_build_stable_patch_source_context,
    )


from .context.capture import (
    _format_selected_files,
    _assert_publishable_context_contract,
)


def _build_stable_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path],
    target_symbols: dict[Path, list[str]] | None = None,
) -> tuple[str, dict[str, str]]:
    return _capture.build_stable_patch_source_context(
        repo,
        task,
        files,
        target_symbols,
        build_patch_source_context_fn=build_patch_source_context,
    )

def build_patch_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> Path:
    return _context_builder.build_patch_context(
        repo,
        task,
        index_progress=index_progress,
        patch_context_header=PATCH_CONTEXT_HEADER,
        upload_instructions=UPLOAD_INSTRUCTIONS,
        patch_response_instructions=PATCH_RESPONSE_INSTRUCTIONS,
        build_safe_status_fn=build_safe_status,
        collect_relevant_files_fn=collect_relevant_files,
        ensure_explicit_task_files_fn=_ensure_explicit_task_files,
        build_stable_patch_source_context_fn=_build_stable_patch_source_context,
        assert_publishable_context_contract_fn=_assert_publishable_context_contract,
        format_selected_files_fn=_format_selected_files,
    )

def build_safe_diff(
    repo: Path,
    staged: bool,
) -> str:
    return _safety.build_safe_diff(repo, staged)


def build_failed_test_context(repo: Path) -> str:
    return _safety.build_failed_test_context(repo)

def build_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> Path:
    return _renderer.build_context(
        repo,
        task,
        index_progress=index_progress,
        upload_instructions=UPLOAD_INSTRUCTIONS,
        patch_response_instructions=PATCH_RESPONSE_INSTRUCTIONS,
        build_safe_status_fn=build_safe_status,
        build_safe_diff_fn=build_safe_diff,
        build_tree_fn=build_tree,
        collect_relevant_files_fn=collect_relevant_files,
        ensure_explicit_task_files_fn=_ensure_explicit_task_files,
        build_failed_test_context_fn=build_failed_test_context,
        build_stable_patch_source_context_fn=_build_stable_patch_source_context,
        assert_publishable_context_contract_fn=_assert_publishable_context_contract,
        format_selected_files_fn=_format_selected_files,
    )

