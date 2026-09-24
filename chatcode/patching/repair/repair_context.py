"""Repair-context builders for patch syntax and target failures."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from ..models import ApplyResult, TestValidation
from ...history import get_history_patch_file
from ...indexing.project_graph import load_map


def build_syntax_repair_context(
    repo: Path,
    original_patch: str,
    error: str,
    malformed_hunk: str | None = None,
    failure_type: str = "invalid_patch_syntax",
    *,
    safe_candidate_paths_fn: Callable[[Path, str], list[str]],
    working_tree_section_fn: Callable,
    write_repair_context_fn: Callable,
    get_context_task_fn: Callable[[Path], str | None],
) -> Path:
    sections: list[str] = []
    for raw_path in safe_candidate_paths_fn(repo, original_patch):
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        if path.is_file():
            sections.extend(
                working_tree_section_fn(repo, raw_path, original_patch)
            )
        else:
            sections.extend([
                f"Current working-tree file: {raw_path}",
                "```text",
                "[File does not currently exist.]",
                "```",
                "",
            ])

    content_parts = [
        "# ChatCode Patch Repair Context",
        "",
        f"Failure type: `{failure_type}`",
        "",
        "## Original task",
        get_context_task_fn(repo) or "",
        "",
        "## Git/parser error",
        "```text",
        error,
        "```",
    ]
    if malformed_hunk:
        content_parts.extend([
            "",
            "## Malformed hunk",
            "```diff",
            malformed_hunk,
            "```",
        ])
    content_parts.extend([
        "",
        "## Complete generated patch",
        "```diff",
        original_patch.rstrip(),
        "```",
        "",
        "## Exact current working-tree content",
        *(sections or ["No safe affected working-tree file could be identified.", ""]),
        "## Required response",
        "Return only one COMPLETE corrected unified diff. Do not return a fragment or explanation.",
        "Use repository-relative POSIX paths and preserve all uncommitted changes.",
        "Include unchanged current source lines around every hunk whenever possible; do not rely on a guessed line number.",
        "",
    ])
    return write_repair_context_fn(
        repo,
        "\n".join(content_parts),
        safe_candidate_paths_fn(repo, original_patch),
    )


def build_stale_repair_context(
    repo: Path,
    patch_text: str,
    stale_reason: str,
    paths: set[str],
    *,
    working_tree_section_fn: Callable,
    write_repair_context_fn: Callable,
    get_context_task_fn: Callable[[Path], str | None],
) -> Path:
    sections: list[str] = []
    for raw_path in sorted(paths):
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        if path.is_file():
            sections.extend(
                working_tree_section_fn(repo, raw_path, patch_text)
            )

    content = "\n".join([
        "# ChatCode Patch Repair Context",
        "",
        "Failure type: `stale_context`",
        "",
        "## Original task",
        get_context_task_fn(repo) or "",
        "",
        "## Stale-context error",
        stale_reason,
        "",
        "## Complete generated patch",
        "```diff",
        patch_text.rstrip(),
        "```",
        "",
        "## Exact CURRENT working-tree content",
        *sections,
        "## Required response",
        "Return only one COMPLETE corrected unified diff against the current files above.",
        "",
    ])
    return write_repair_context_fn(repo, content, paths)


def build_patch_repair_context(
    repo: Path,
    patch_text: str,
    apply_error: str,
    *,
    patch_hunks_fn: Callable[[str], list[tuple[str, int, str]]],
    safe_candidate_paths_fn: Callable[[Path, str], list[str]],
    working_tree_section_fn: Callable,
    write_repair_context_fn: Callable,
    get_context_task_fn: Callable[[Path], str | None],
) -> Path:
    error_locations: dict[str, set[int]] = {}
    for raw_path, raw_line in re.findall(
        r"patch failed: (.*?):(\d+)",
        apply_error,
    ):
        path = PurePosixPath(raw_path.replace("\\", "/")).as_posix()
        error_locations.setdefault(path, set()).add(int(raw_line))

    hunks = patch_hunks_fn(patch_text)
    if error_locations:
        failed_hunks = [
            hunk for hunk in hunks
            if hunk[0] in error_locations
            and hunk[1] in error_locations[hunk[0]]
        ]
        if not failed_hunks:
            failed_hunks = [
                hunk for hunk in hunks
                if hunk[0] in error_locations
            ]
    else:
        failed_hunks = hunks

    sections: list[str] = []
    for raw_path, line_number, hunk in failed_hunks:
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        if path.is_file():
            source_section = working_tree_section_fn(
                repo,
                raw_path,
                hunk,
                line_number,
            )
        else:
            source_section = [
                f"Current working-tree file: {raw_path}",
                "```text",
                "[File does not currently exist.]",
                "```",
                "",
            ]
        sections.extend([
            f"### {raw_path}",
            *source_section,
            "Failed hunk:",
            "```diff",
            hunk,
            "```",
            "",
        ])

    if not sections:
        sections = [
            "No individual hunk could be identified; inspect the complete patch below.",
            "```diff",
            patch_text.rstrip(),
            "```",
            "",
        ]

    content = "\n".join([
        "# ChatCode Patch Repair Context",
        "",
        "Failure type: `patch_target_mismatch`",
        "",
        "## Original task",
        get_context_task_fn(repo) or "",
        "",
        "The patch failed `git apply --check`. Return only a corrected unified diff against the exact CURRENT working-tree excerpts below.",
        "Preserve all existing uncommitted changes. Use repository-relative paths with forward slashes.",
        "",
        "## git apply error",
        "```text",
        apply_error,
        "```",
        "",
        "## Failed hunks and current contents",
        *sections,
        "## Complete generated patch",
        "```diff",
        patch_text.rstrip(),
        "```",
        "",
    ])
    return write_repair_context_fn(
        repo,
        content,
        safe_candidate_paths_fn(repo, patch_text),
    )


def build_test_failure_repair_context(
    repo: Path,
    result: ApplyResult,
    validation: TestValidation,
    *,
    working_tree_section_fn: Callable,
    write_repair_context_fn: Callable,
    get_context_task_fn: Callable[[Path], str | None],
) -> Path:
    """Create a repair prompt from the applied patch and *current* source."""
    patch_file = get_history_patch_file(result.history_entry)
    patch_text = patch_file.read_text(encoding="utf-8", errors="replace")

    raw_reports: dict[str, str] = {}
    for label, test_result in (
        ("Baseline", validation.baseline),
        ("Targeted tests", validation.targeted),
        ("Full suite", validation.full),
    ):
        if test_result is None:
            raw_reports[label] = ""
            continue
        try:
            raw_reports[label] = test_result.output_file.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            raw_reports[label] = "[Saved test output is unavailable.]"

    # A repair prompt must focus on failures this patch is responsible for.
    # Unchanged baseline failures are diagnostic information, never automatic
    # repair targets.
    repair_failure_ids = (
        validation.new_failures
        | validation.remaining_repair_failures
        | (
            getattr(validation.targeted, "failed_tests", frozenset())
            if validation.status == "targeted_failed" else frozenset()
        )
        # This function may also be called explicitly to investigate an
        # existing failure.  It is never reached automatically for that state.
        | (validation.existing_failures if validation.status == "existing" else frozenset())
        | (
            getattr(validation.full, "failed_tests", frozenset())
            if validation.status == "unclassified_failure" else frozenset()
        )
    )
    failure_ids = sorted(repair_failure_ids)
    anchors: dict[str, int] = {}
    relevant_paths: list[str] = []

    def add_relevant(raw_path: str) -> None:
        normalized = PurePosixPath(raw_path.replace("\\", "/")).as_posix()
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or ".." in path.parts
            or ".git" in path.parts
            or normalized in relevant_paths
        ):
            return
        if repo.joinpath(*path.parts).is_file():
            relevant_paths.append(normalized)

    repo_resolved = repo.resolve()
    for output in raw_reports.values():
        for raw_file, raw_line in re.findall(
            r'^\s*File "([^"]+)", line (\d+)', output, flags=re.MULTILINE
        ):
            try:
                relative = Path(raw_file).resolve().relative_to(repo_resolved).as_posix()
            except (OSError, ValueError):
                continue
            add_relevant(relative)
            anchors.setdefault(relative, int(raw_line))

    failed_test_paths: list[str] = []
    for failure_id in failure_ids:
        module = failure_id.split(".", 1)[0]
        candidate = f"tests/{module.replace('.', '/')}" + (
            "" if module.endswith(".py") else ".py"
        )
        if (repo / candidate).is_file():
            failed_test_paths.append(candidate)
            add_relevant(candidate)

    # Existing index relationships provide direct imports for the failing test
    # files. They only select paths; every byte included below is read fresh.
    try:
        indexed_files = load_map(repo).get("files", {})
        for test_path in failed_test_paths:
            metadata = indexed_files.get(test_path, {})
            for dependency in metadata.get("dependencies", [])[:8]:
                if isinstance(dependency, str):
                    add_relevant(dependency)
    except (OSError, AttributeError, TypeError):
        pass

    for raw_path in sorted(result.paths):
        add_relevant(raw_path)

    current_sections: list[str] = []
    source_budget = 140_000
    included_paths: list[str] = []
    for raw_path in relevant_paths:
        section = working_tree_section_fn(
            repo,
            raw_path,
            line_number=anchors.get(raw_path, 1),
            full_limit=35_000,
        )
        section_size = len("\n".join(section))
        if current_sections and section_size > source_budget:
            continue
        current_sections.extend(section)
        included_paths.append(raw_path)
        source_budget -= section_size

    reports: list[str] = []
    for label, test_result in (("Baseline", validation.baseline), ("Targeted tests", validation.targeted), ("Full suite", validation.full)):
        reports.extend([f"### {label}"])
        if test_result is None:
            reports.extend(["Not run.", ""])
            continue
        output = raw_reports[label]
        lines = output.splitlines()
        metadata = [
            line for line in lines
            if line.startswith(("Status:", "Exit code:", "Command:", "Duration:"))
        ]
        if test_result.returncode != 0:
            first_failure = next(
                (index for index, line in enumerate(lines) if line.strip().startswith(("ERROR: ", "FAIL: "))),
                None,
            )
            detail = lines[max(0, first_failure - 1):] if first_failure is not None else lines[-200:]
        else:
            detail = [line for line in lines if line.strip().startswith(("Ran ", "OK"))]
        compact = "\n".join([*metadata, "", *detail]).strip()
        reports.extend(["```text", compact or "[No relevant output.]", "```", ""])

    if validation.status == "repair_failed":
        classifications = [
            *(f"- unresolved repair target: {name}" for name in sorted(
                validation.remaining_repair_failures
            )),
            *(f"- new regression: {name}" for name in sorted(
                validation.new_failures - validation.remaining_repair_failures
            )),
        ]
    elif validation.status == "targeted_failed":
        classifications = [
            *(f"- failed targeted test: {name}" for name in sorted(
                getattr(validation.targeted, "failed_tests", frozenset())
            )),
            *(f"- regression: {name}" for name in sorted(validation.new_failures)),
        ]
    elif validation.status == "unclassified_failure":
        classifications = [
            f"- observed after patch; baseline unavailable: {name}"
            for name in sorted(repair_failure_ids)
        ]
    else:
        classifications = [
            *(f"- regression: {name}" for name in sorted(validation.new_failures)),
            *(f"- pre-existing (explicit investigation): {name}" for name in sorted(
                validation.existing_failures
            )),
        ]
    classifications = classifications or [
        "- unclear: test runner did not expose stable failure identifiers"
    ]
    content = "\n".join([
        "# ChatCode Test Failure Repair Context",
        "",
        "## Original task",
        get_context_task_fn(repo),
        "",
        "## Classification",
        *classifications,
        "",
        "## Unsuccessful patch attempt",
        "```diff",
        patch_text.rstrip(),
        "```",
        "",
        "## Changed files",
        *(f"- {path}" for path in sorted(result.paths)),
        "",
        "## Test results",
        *reports,
        "## Exact CURRENT working-tree contents",
        *current_sections,
        "## Required response",
        "Make every unresolved repair target above pass with one complete unified diff against the current files above.",
        "The unsuccessful patch attempt is evidence of what did not work; do not repeat it.",
        "Do not restore or overwrite unrelated user changes.",
        "",
    ])
    return write_repair_context_fn(
        repo,
        content,
        included_paths,
        repair_failure_ids,
    )
