"""Top-level full context document assembly."""
from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from pathlib import Path

from ..context_state import save_context_state
from ..git_utils import get_branch
from ..indexing.index_manager import IndexProgress
from ..workspace import atomic_write_text, get_repo_workspace


def build_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
    *,
    upload_instructions: str,
    patch_response_instructions: str,
    build_safe_status_fn: Callable[[Path], str],
    build_safe_diff_fn: Callable,
    build_tree_fn: Callable,
    collect_relevant_files_fn: Callable,
    ensure_explicit_task_files_fn: Callable,
    build_failed_test_context_fn: Callable[[Path], str],
    build_stable_patch_source_context_fn: Callable,
    assert_publishable_context_contract_fn: Callable[[Path, str], None],
    format_selected_files_fn: Callable[[list[Path], Path, str], str],
) -> Path:
    output_dir = get_repo_workspace(repo)

    output_file = (
        output_dir / "UPLOAD_TO_CHATGPT.md"
    )

    branch = get_branch(repo)
    status = build_safe_status_fn(repo)

    unstaged = build_safe_diff_fn(
        repo,
        staged=False,
    )

    staged = build_safe_diff_fn(
        repo,
        staged=True,
    )

    tree = build_tree_fn(repo)

    retrieved = collect_relevant_files_fn(
        repo,
        task,
        index_progress=index_progress,
        include_target_symbols=True,
    )
    files, target_symbols = retrieved if isinstance(retrieved, tuple) else (retrieved, {})
    files = ensure_explicit_task_files_fn(
        repo,
        task,
        files,
    )
    failed_tests = (
        build_failed_test_context_fn(repo)
    )

    # Capture source last: it is the only material sent as an authoritative
    # patch target, and this keeps its final hash recheck immediately before
    # the generated context is finalized.
    source_context, source_hashes = build_stable_patch_source_context_fn(
        repo,
        task,
        files,
        target_symbols,
    )
    assert_publishable_context_contract_fn(output_file, source_context)

    parts = [
        "# ChatCode Context",
        "",
        "## ChatGPT instructions",
        upload_instructions,
        "",
        "## Task",
        task,
        "",
        "## Response instructions",
        patch_response_instructions,
        "",
        "## Repository",
        str(repo),
        "",
        "## Current branch",
        branch,
        "",
        "## Git status",
        status or "Working tree clean",
        "",
        "## Project structure",
        tree,
        "",
        "## Selected relevant files",
        format_selected_files_fn(files, repo, source_context),
        "",
        "## Relevant source files",
        source_context,
        "",
        "## Unstaged changes",
        unstaged or "No unstaged changes.",
        "",
        "## Staged changes",
        staged or "No staged changes.",
        "",
    ]

    if failed_tests:
        parts.extend([
            "## Latest failed test run",
            "",
            failed_tests,
            "",
        ])

    content = "\n".join(parts)

    save_context_state(
        repo,
        task=task,
        source_hashes=source_hashes,
        context_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        generation_id=uuid.uuid4().hex,
    )
    # Publish last: watchers cannot observe this generation before its hashes.
    atomic_write_text(output_file, content, newline="\n")

    return output_file
