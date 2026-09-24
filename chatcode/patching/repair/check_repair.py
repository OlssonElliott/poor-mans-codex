"""Focused repair context for standalone health-check failures."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from ..errors import PatchError
from .existing_failure import failure_output_excerpt
from ...git_utils import GitError, get_status, run_git
from ...workspace import atomic_write_text, get_check_repair_context_file


CHECK_REPAIR_COMPANION_PROMPT = (
    "Use the attached CHECK_REPAIR_CONTEXT.md as the current repository context "
    "and source of truth. Fix the listed failing test(s) without weakening tests "
    "or reverting unrelated working-tree changes. Trace the failure through the "
    "supplied current implementation and repair the underlying behavior. Return "
    "exactly one complete unified diff with at least three unchanged context lines "
    "around each hunk and no explanation outside the diff."
)


def build_check_repair_context(
    repo: Path,
    test_result,
    selected_failures: frozenset[str],
    *,
    build_context_from_test_roots_fn: Callable | None = None,
) -> Path:
    """Materialize a focused new task from a standalone health check."""
    selected = frozenset(selected_failures) & getattr(test_result, "failed_tests", frozenset())
    if not selected:
        raise PatchError("Choose at least one failing test from this health check.")
    try:
        output = test_result.output_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = "[Saved test output is unavailable.]"
    traceback_paths: list[Path] = []
    repo_root = repo.resolve()
    for raw_path in re.findall(r'^\s*File "([^"]+)", line \d+', output, re.MULTILINE):
        try:
            path = Path(raw_path).resolve()
            path.relative_to(repo_root)
        except (OSError, ValueError):
            continue
        traceback_paths.append(path)
    if build_context_from_test_roots_fn is None:
        from ...context.api import build_context_from_test_roots
        build_context_from_test_roots_fn = build_context_from_test_roots
    source_context, selected_paths = build_context_from_test_roots_fn(
        repo, selected, traceback_paths
    )
    try:
        dirty = get_status(repo) or "[Working tree clean.]"
    except GitError:
        dirty = "[Working-tree status unavailable.]"
    try:
        diff_paths = [path.relative_to(repo).as_posix() for path in selected_paths]
        current_diff = run_git("diff", "--no-renames", "--", *diff_paths, cwd=repo) if diff_paths else ""
    except GitError:
        current_diff = ""
    diagnostics = []
    for failure_id in sorted(selected):
        diagnostics.extend([f"### {failure_id}", "```text", failure_output_excerpt(output, failure_id), "```", ""])
    content = "\n".join([
        "# ChatCode Check Repair Context", "",
        "Generated from `chatcode check` against the current working tree.",
        f"Repository: `{repo.resolve()}`", "",
        "## Repair targets", *(f"- {failure}" for failure in sorted(selected)), "",
        "## Relevant failure output", *diagnostics,
        "## Dirty working-tree state", "```text", dirty, "```", "",
        "## Directly relevant current diff", "```diff", current_diff or "[No unstaged diff for selected source files.]", "```", "",
        "## Exact current source and bounded dependencies",
        source_context,
        "## Repair success criterion",
        "The listed failing tests are the repair targets. The repair is successful when those tests pass without introducing new regressions.",
        "Do not weaken or remove tests. Do not revert unrelated working-tree changes.", "",
    ])
    destination = get_check_repair_context_file(repo)
    atomic_write_text(destination, content, newline="\n")
    return destination
