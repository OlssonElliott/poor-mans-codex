"""Focused contexts for failures proven to predate the current patch."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from ..errors import PatchError
from ..models import ApplyResult, TestValidation
from ...git_utils import GitError, get_status
from ...history import get_history_patch_file
from ...indexing.project_graph import load_map
from ...workspace import atomic_write_text, get_existing_failure_context_file


def failure_output_excerpt(output: str, failure_id: str) -> str:
    """Keep the selected failure's diagnostic, not unrelated suite noise."""
    lines = output.splitlines()
    index = next((i for i, line in enumerate(lines) if failure_id in line), None)
    if index is None:
        return output[-12_000:] or "[Selected failure details unavailable.]"
    end = next(
        (i for i in range(index + 1, len(lines))
         if lines[i].startswith(("FAIL: ", "ERROR: "))),
        min(len(lines), index + 180),
    )
    return "\n".join(lines[max(0, index - 2):end]).strip()


def build_existing_failure_context(
    repo: Path,
    result: ApplyResult,
    validation: TestValidation,
    selected_failures: frozenset[str],
    *,
    working_tree_section_fn: Callable,
    fallback_patch_summary_fn: Callable[[str, set[str]], str],
    get_context_task_fn: Callable[[Path], str | None],
) -> Path:
    """Create a new-task context for failures proven to predate this patch."""
    selected_failures = frozenset(selected_failures) & validation.existing_failures
    if not selected_failures:
        raise PatchError("Choose at least one currently reported pre-existing failure.")

    def report_text(test_result) -> str:
        if test_result is None:
            return "[Not run.]"
        try:
            return test_result.output_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "[Saved test output is unavailable.]"

    baseline_output = report_text(validation.baseline)
    post_output = report_text(validation.full)
    relevant_paths: list[str] = []

    def add_path(raw_path: str) -> None:
        normalized = PurePosixPath(raw_path.replace("\\", "/")).as_posix()
        path = PurePosixPath(normalized)
        if (path.is_absolute() or ".." in path.parts or ".git" in path.parts
                or normalized in relevant_paths):
            return
        if repo.joinpath(*path.parts).is_file():
            relevant_paths.append(normalized)

    test_paths: list[str] = []
    for failure_id in sorted(selected_failures):
        module = failure_id.split(".", 1)[0]
        candidate = f"tests/{module.replace('.', '/')}" + ("" if module.endswith(".py") else ".py")
        if (repo / candidate).is_file():
            test_paths.append(candidate)
            add_path(candidate)

    # A small deterministic fallback keeps a useful implementation file in
    # the context even when the project graph has not been built yet.
    for test_path in test_paths:
        try:
            test_source = (repo / test_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        modules = re.findall(r"^\s*from\s+([\w.]+)\s+import\s+", test_source, re.MULTILINE)
        modules += re.findall(r"^\s*import\s+([\w.]+)", test_source, re.MULTILINE)
        for module in modules[:12]:
            add_path(module.replace(".", "/") + ".py")
            add_path(module.replace(".", "/") + "/__init__.py")

    # The graph supplies bounded direct implementation dependencies from the
    # high-confidence failing-test seed. Content is always read from disk now.
    try:
        indexed_files = load_map(repo).get("files", {})
        for test_path in test_paths:
            for dependency in indexed_files.get(test_path, {}).get("dependencies", [])[:8]:
                if isinstance(dependency, str):
                    add_path(dependency)
    except (OSError, AttributeError, TypeError):
        pass
    for path in sorted(result.paths):
        add_path(path)

    sections: list[str] = []
    budget = 140_000
    for path in relevant_paths:
        section = working_tree_section_fn(repo, path, full_limit=35_000)
        size = len("\n".join(section))
        if sections and size > budget:
            continue
        sections.extend(section)
        budget -= size

    try:
        patch_text = get_history_patch_file(result.history_entry).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        patch_text = "[Recently applied patch is unavailable.]"
    try:
        dirty_state = get_status(repo) or "[Working tree clean.]"
    except GitError:
        dirty_state = "[Working-tree status unavailable.]"

    failure_sections: list[str] = []
    for failure_id in sorted(selected_failures):
        failure_sections.extend([
            f"### {failure_id}",
            "This test failed before the previous patch and still fails now.",
            "#### Baseline failure output",
            "```text", failure_output_excerpt(baseline_output, failure_id), "```",
            "#### Current post-patch failure output",
            "```text", failure_output_excerpt(post_output, failure_id), "```", "",
        ])

    content = "\n".join([
        "# ChatCode Existing Failure Context",
        "",
        "## New-task provenance",
        "Created from a pre-existing failure discovered during post-apply validation.",
        f"Repository: `{repo.resolve()}`",
        "The previous patch passed its targeted tests and introduced zero new regressions.",
        "This is a separate bug-fix task, not a repair of the previous patch.",
        "",
        "## Original task for the previous patch",
        get_context_task_fn(repo),
        "",
        "## Selected pre-existing failure(s)",
        *failure_sections,
        "## Recently applied patch summary",
        fallback_patch_summary_fn(patch_text, result.paths),
        "",
        "## Dirty working-tree state",
        "```text", dirty_state, "```",
        "",
        "## Exact current source and bounded dependencies",
        *(sections or ["[No selected test source could be materialized.]", ""]),
        "## Required response",
        "Fix only the selected pre-existing failure(s). Preserve unrelated working behavior and do not revert the previous successful patch merely to restore an old state.",
        "Return one complete unified diff against the current files, with no explanation outside the diff.",
        "",
    ])
    output_file = get_existing_failure_context_file(repo)
    atomic_write_text(output_file, content, newline="\n")
    return output_file
