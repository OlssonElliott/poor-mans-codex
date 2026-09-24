"""Safe patch application engine independent of CLI orchestration."""
from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..errors import PatchAlreadyApplied, PatchError
from ..models import ApplyResult, PatchPreview
from .patch_validation import (
    _expand_thin_hunk_context,
    _validate_hunk_context,
    normalize_patch_file,
    patch_is_already_applied,
    validate_patch_paths,
)
from ...git_utils import GitError, run_git
from ...history import (
    HistoryError,
    begin_history_entry,
    discard_pending_entry,
    finalize_history_entry,
    get_history_patch_file,
)
from ...unified_diff import UnifiedDiffError, canonicalize_unified_diff


def apply_patch_core(
    repo: Path,
    patch_file: Path,
    *,
    dry_run: bool = False,
    new_task: bool = False,
    get_default_patch_file_fn: Callable[[Path], Path],
    patch_supersedes_active_repair_fn: Callable,
    supersede_active_repair_fn: Callable[[Path], None],
    get_stale_context_reason_fn: Callable,
    build_syntax_repair_context_fn: Callable,
    build_stale_repair_context_fn: Callable,
    build_patch_repair_context_fn: Callable,
    get_context_kind_fn: Callable[[Path], str | None],
    get_context_task_fn: Callable[[Path], str | None],
    save_context_state_fn: Callable,
    clear_repair_context_fn: Callable[[Path], None],
    clear_incoming_patch_fn: Callable[[Path], None],
) -> ApplyResult | PatchPreview:
    patch_file = patch_file.resolve()

    if not patch_file.exists():
        raise PatchError(
            "Patchfilen finns inte:\n"
            f"{patch_file}"
        )

    if not patch_file.is_file():
        raise PatchError(
            "Detta är inte en fil: "
            f"{patch_file}"
        )

    original_patch = normalize_patch_file(patch_file, write_back=not dry_run)

    try:
        patch_text, _normalized_counts = canonicalize_unified_diff(
            original_patch
        )
    except UnifiedDiffError as exc:
        repair_context = build_syntax_repair_context_fn(
            repo,
            original_patch,
            str(exc),
            malformed_hunk=exc.hunk,
        )
        raise PatchError(
            "Generated patch is not a valid unified diff. "
            "No changes were made.\n"
            f"Reason: {exc}\n"
            f"Repair context: {repair_context}",
            failure_type="invalid_patch_syntax",
            repair_context=repair_context,
        ) from exc

    paths = validate_patch_paths(
        patch_text
    )
    patch_text = _expand_thin_hunk_context(repo, patch_text)
    try:
        _validate_hunk_context(repo, patch_text)
    except UnifiedDiffError as exc:
        repair_context = build_syntax_repair_context_fn(
            repo,
            original_patch,
            str(exc),
            malformed_hunk=exc.hunk,
            failure_type="insufficient_patch_context",
        )
        raise PatchError(
            "Generated patch is too fragile to apply safely. "
            "No changes were made.\n"
            f"Reason: {exc}\n"
            f"Repair context: {repair_context}",
            failure_type="insufficient_patch_context",
            repair_context=repair_context,
        ) from exc

    if not dry_run:
        patch_file.write_text(
            patch_text,
            encoding="utf-8",
            newline="\n",
        )

    validation_patch_file = patch_file
    temporary_validation_file: Path | None = None
    if dry_run:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".diff",
            delete=False,
        ) as temporary:
            temporary.write(patch_text)
            temporary_validation_file = Path(temporary.name)
        validation_patch_file = temporary_validation_file

    try:
        run_git(
            "apply",
            "--numstat",
            str(validation_patch_file),
            cwd=repo,
        )
    except GitError as exc:
        if temporary_validation_file is not None:
            temporary_validation_file.unlink(missing_ok=True)
        repair_context = build_syntax_repair_context_fn(
            repo,
            original_patch,
            str(exc),
        )
        raise PatchError(
            "Generated patch is not a valid unified diff. "
            "No changes were made.\n"
            f"Reason: {exc}\n"
            f"Repair context: {repair_context}",
            failure_type="invalid_patch_syntax",
            repair_context=repair_context,
        ) from exc

    default_patch_file = (
        get_default_patch_file_fn(
            repo
        ).resolve()
    )

    supersedes_repair = (
        patch_file == default_patch_file
        and patch_supersedes_active_repair_fn(
            repo,
            paths,
            force=new_task,
        )
    )
    if supersedes_repair and not dry_run:
        supersede_active_repair_fn(repo)

    if patch_file == default_patch_file and not supersedes_repair:
        stale_reason = (
            get_stale_context_reason_fn(
                repo,
                paths,
            )
        )

        if stale_reason is not None:
            if temporary_validation_file is not None:
                temporary_validation_file.unlink(missing_ok=True)
            repair_context = build_stale_repair_context_fn(
                repo,
                patch_text,
                stale_reason,
                paths,
            )
            raise PatchError(
                f"{stale_reason}\n\nRepair context: {repair_context}",
                failure_type="stale_context",
                repair_context=repair_context,
            )

    try:
        run_git(
            "apply",
            "--check",
            str(validation_patch_file),
            cwd=repo,
        )

    except GitError as strict_error:
        already_applied = patch_is_already_applied(
            repo,
            validation_patch_file,
        )
        if temporary_validation_file is not None:
            temporary_validation_file.unlink(missing_ok=True)
        if already_applied:
            raise PatchAlreadyApplied(
                "Patchen verkar redan vara "
                "applicerad."
            )

        repair_context = build_patch_repair_context_fn(
            repo,
            patch_text,
            str(strict_error),
        )
        raise PatchError(
            "Patch did not pass git apply --check. No changes were made.\n"
            f"{strict_error}\n\n"
            f"Repair context: {repair_context}",
            failure_type="patch_target_mismatch",
            repair_context=repair_context,
        ) from strict_error

    if dry_run:
        if temporary_validation_file is not None:
            temporary_validation_file.unlink(missing_ok=True)
        return PatchPreview(
            paths=paths,
            patch_text=patch_text,
        )

    try:
        pending_entry = (
            begin_history_entry(
                repo,
                patch_text,
                paths,
            )
        )
    except HistoryError as exc:
        raise PatchError(
            f"Kunde inte skapa historik: {exc}"
        ) from exc

    history_patch = (
        get_history_patch_file(
            pending_entry
        )
    )

    apply_args = [
        "apply",
        str(history_patch),
    ]

    try:
        run_git(
            *apply_args,
            cwd=repo,
        )

    except GitError as exc:
        discard_pending_entry(
            pending_entry
        )

        raise PatchError(
            "Git kunde inte applicera "
            "patchen:\n"
            f"{exc}"
        ) from exc

    try:
        history_entry = (
            finalize_history_entry(
                repo,
                pending_entry,
            )
        )
    except HistoryError as exc:
        raise PatchError(
            "Patchen applicerades, men "
            "ChatCode kunde inte slutföra "
            f"historiken:\n{exc}"
        ) from exc

    # Refresh a valid (or absent) derived map after an apply, but never
    # overwrite a malformed workspace artifact as a side effect of consuming
    # the canonical incoming patch. Context generation can rebuild it later.
    try:
        from ...indexing.index_manager import update_project_map
        from ...indexing.project_graph import SCHEMA_VERSION, map_path

        project_map = map_path(repo)
        valid_map = not project_map.exists()
        if project_map.exists():
            raw_map = json.loads(project_map.read_text(encoding="utf-8"))
            valid_map = (
                isinstance(raw_map, dict)
                and raw_map.get("version") == SCHEMA_VERSION
                and isinstance(raw_map.get("files"), dict)
            )
        if valid_map:
            update_project_map(repo, paths=paths, run_semantic=False)
    except Exception:
        pass

    # Normal and repair contexts are single-use. Once their patch has been
    # applied, keeping that generation active makes a later unrelated patch
    # appear stale. Repair contexts are especially vulnerable because their
    # Markdown file is removed immediately below while context-state.json would
    # otherwise continue pointing at it.
    if get_context_kind_fn(repo) in {"normal", "repair"}:
        save_context_state_fn(
            repo,
            task=get_context_task_fn(repo),
            context_kind="consumed",
        )

    clear_repair_context_fn(repo)
    clear_incoming_patch_fn(repo)

    return ApplyResult(
        paths=paths,
        history_entry=history_entry,
    )
