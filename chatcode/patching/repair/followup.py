"""Follow-up context generation after a technically successful patch."""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path, PurePosixPath

from ..errors import PatchError
from ..models import ApplyResult, TestValidation
from .followup_state import (
    MAX_FOLLOWUP_ROUNDS,
    attempted_python_symbols,
    attempted_python_symbols_by_path,
    followup_attempt,
    load_followup_state,
    result_from_followup_attempt,
    save_followup_state,
)
from ...context_state import get_context_task, save_context_state
from ...git_utils import GitError, get_status
from ...workspace import atomic_write_text, get_followup_context_file
from ...context.api import (
    build_stable_patch_source_context,
    collect_relevant_files,
)
from ...context.files import _ensure_explicit_task_files, is_source_file


def build_followup_context(
    repo: Path,
    result: ApplyResult,
    validation: TestValidation,
    feedback: str,
    patch_summary: str,
    *,
    original_task: str | None = None,
    prior_attempts: list[ApplyResult] | None = None,
    feedback_history: list[str] | None = None,
    persist_state: bool = True,
) -> Path:
    """Create a fresh, post-patch task context from explicit user feedback."""
    feedback = feedback.strip()
    if not feedback:
        raise PatchError("Describe what is still not working before creating a follow-up context.")
    original_task = original_task or get_context_task(repo)
    existing_state = load_followup_state(repo) if persist_state else None
    prior_feedback = list(feedback_history or [])
    prior_results = list(prior_attempts or [])
    if existing_state is not None and existing_state.get("unresolved"):
        prior_feedback = [
            item for item in existing_state.get("feedback", [])
            if isinstance(item, str) and item
        ]
        prior_results = [
            parsed for parsed in (
                result_from_followup_attempt(item)
                for item in existing_state.get("attempts", [])
                if isinstance(item, dict)
            )
            if parsed is not None
        ]
    all_feedback = [*prior_feedback, feedback]
    all_feedback = list(dict.fromkeys(all_feedback))[-MAX_FOLLOWUP_ROUNDS:]
    retrieval_feedback = "\n".join(
        f"- {item}" for item in all_feedback
    )
    retrieval_task = "\n\n".join((
        original_task,
        "Previous patch applied successfully; automated validation passed.",
        "User-provided unresolved runtime feedback:\n" + retrieval_feedback,
    ))
    # This is the normal current-working-tree retrieval/materialization path,
    # deliberately rerun with feedback rather than reusing an old upload.
    from ...retrieval.hybrid_retriever import resolve_alternative_callback_roots
    retrieved = collect_relevant_files(repo, retrieval_task, include_target_symbols=True)
    files, target_symbols = retrieved if isinstance(retrieved, tuple) else (retrieved, {})
    attempted_results = [*prior_results, result][-MAX_FOLLOWUP_ROUNDS:]
    attempted_symbols = {
        symbol for attempted in attempted_results
        for symbol in attempted_python_symbols(attempted)
    }
    attempted_paths: list[Path] = []
    attempted_target_symbols: dict[Path, list[str]] = {}
    repo_root = repo.resolve()
    for attempted in attempted_results:
        symbols_by_path = attempted_python_symbols_by_path(attempted)
        for raw_path in sorted(attempted.paths):
            relative = PurePosixPath(raw_path.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
                continue
            candidate = repo.joinpath(*relative.parts)
            try:
                candidate.resolve().relative_to(repo_root)
            except (OSError, ValueError):
                continue
            if (
                candidate.is_file()
                and is_source_file(candidate)
                and candidate not in attempted_paths
            ):
                attempted_paths.append(candidate)
            symbols = symbols_by_path.get(raw_path, set())
            if candidate.is_file() and symbols:
                attempted_target_symbols.setdefault(candidate, [])
                attempted_target_symbols[candidate] = list(dict.fromkeys([
                    *attempted_target_symbols[candidate],
                    *sorted(symbols),
                ]))
    alternatives = resolve_alternative_callback_roots(
        repo, retrieval_task, attempted_symbols
    )
    # Follow-up-only priority: keep the attempted dirty path in normal
    # retrieval, but put structurally supported alternatives first so the
    # failed hypothesis cannot consume all patchable-source allowance.
    files = list(dict.fromkeys([*alternatives.files, *attempted_paths, *files]))
    merged_targets: dict[Path, list[str]] = {}
    for source in (
        alternatives.required_symbols,
        target_symbols,
        attempted_target_symbols,
    ):
        for path, symbols in source.items():
            merged_targets.setdefault(path, [])
            merged_targets[path] = list(dict.fromkeys([
                *merged_targets[path], *symbols,
            ]))
    target_symbols = merged_targets
    files = _ensure_explicit_task_files(repo, retrieval_task, files)
    source_context, source_hashes = build_stable_patch_source_context(
        repo, retrieval_task, files, target_symbols
    )
    try:
        dirty_state = get_status(repo) or "[Working tree clean.]"
    except GitError:
        dirty_state = "[Working-tree status unavailable.]"
    content = "\n".join((
        "# ChatCode Follow-up Context", "",
        "## Original task", original_task, "",
        "## Previous patch result",
        "- The previous patch applied successfully.",
        "- Automated validation passed with no new regressions.",
        f"- Validation classification: `{validation.status}`.",
        "", "### Previous patch summary", patch_summary, "",
        "## User feedback", "The following is user-provided runtime feedback:",
        "```text", feedback, "```", "",
        "## Bounded unresolved feedback history",
        "\n".join(f"- {item}" for item in all_feedback), "",
        "## Follow-up status",
        "The previous patch applied successfully and automated validation passed, "
        "but the user reports that the real problem remains unresolved.", "",
        "## Current repository context", "## Dirty working-tree state",
        "```text", dirty_state, "```", "",
        "## Exact current source and bounded dependencies", source_context, "",
        "## Required response",
        "Investigate the unresolved behavior using the user feedback and current "
        "source. Preserve unrelated working-tree changes and the previous patch.",
        "Return one complete unified diff against the current files, with no explanation outside the diff.",
        "",
    ))
    output = get_followup_context_file(repo)
    save_context_state(
        repo,
        task=original_task,
        source_hashes=source_hashes,
        context_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        generation_id=uuid.uuid4().hex,
        context_filename=output.name,
        context_kind="followup",
    )
    atomic_write_text(output, content, newline="\n")
    if persist_state:
        attempts = []
        if existing_state is not None and existing_state.get("unresolved"):
            attempts = [
                item for item in existing_state.get("attempts", [])
                if isinstance(item, dict)
            ]
        attempts.append(followup_attempt(result, validation, patch_summary))
        save_followup_state(repo, {
            "version": 1,
            "id": (
                existing_state.get("id")
                if existing_state is not None and existing_state.get("unresolved")
                else uuid.uuid4().hex
            ),
            "unresolved": True,
            "original_task": original_task,
            "feedback": all_feedback,
            "attempts": attempts[-MAX_FOLLOWUP_ROUNDS:],
            "updated_at": time.time(),
        })
    return output


def regenerate_followup_context(repo: Path) -> Path | None:
    """Regenerate the active follow-up from structured evidence and current source."""
    state = load_followup_state(repo)
    if state is None or not state.get("unresolved"):
        return None
    attempts = [item for item in state.get("attempts", []) if isinstance(item, dict)]
    feedback = [item for item in state.get("feedback", []) if isinstance(item, str) and item]
    if not attempts or not feedback or not isinstance(state.get("original_task"), str):
        return None
    latest = attempts[-1]
    result = result_from_followup_attempt(latest)
    if result is None:
        return None
    prior_results = [
        parsed for parsed in (result_from_followup_attempt(item) for item in attempts[:-1])
        if parsed is not None
    ]
    validation = TestValidation(
        None, None, None, str(latest.get("validation_status") or "passed")
    )
    return build_followup_context(
        repo,
        result,
        validation,
        feedback[-1],
        str(latest.get("patch_summary") or "[Previous patch summary unavailable.]"),
        original_task=state["original_task"],
        prior_attempts=prior_results,
        feedback_history=feedback[:-1],
        persist_state=False,
    )
