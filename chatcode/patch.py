from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .context_state import (
    get_context_task,
    get_stale_context_reason,
)
from .git_utils import GitError, run_git
from .history import (
    HistoryError,
    begin_history_entry,
    discard_pending_entry,
    finalize_history_entry,
    get_history_patch_file,
    get_latest_applied_entry,
    move_entry_to_undone,
)
from .workspace import (
    get_default_patch_file,
    get_repair_context_file,
)
from .unified_diff import (
    UnifiedDiffError,
    canonicalize_unified_diff,
)


class PatchError(RuntimeError):
    def __init__(
        self,
        message: str,
        failure_type: str | None = None,
        repair_context: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.repair_context = repair_context


class PatchAlreadyApplied(RuntimeError):
    pass


class PatchUndoError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApplyResult:
    paths: set[str]
    history_entry: Path


@dataclass(frozen=True)
class UndoResult:
    paths: set[str]
    history_entry: Path


def extract_patch_paths(
    patch_text: str,
) -> set[str]:
    paths: set[str] = set()

    for line in patch_text.splitlines():
        if not line.startswith(
            ("--- ", "+++ ")
        ):
            continue

        raw_path = (
            line[4:]
            .strip()
            .split("\t", 1)[0]
        )

        if raw_path == "/dev/null":
            continue

        if raw_path.startswith(
            ("a/", "b/")
        ):
            raw_path = raw_path[2:]

        paths.add(raw_path)

    return paths


def validate_patch_paths(
    patch_text: str,
) -> set[str]:
    paths = extract_patch_paths(
        patch_text
    )

    if not paths:
        raise PatchError(
            "Patchen innehåller inga filer."
        )

    for raw_path in paths:
        path = PurePosixPath(
            raw_path
        )

        if path.is_absolute():
            raise PatchError(
                "Absolut sökväg är inte "
                f"tillåten: {raw_path}"
            )

        if ".." in path.parts:
            raise PatchError(
                "Sökväg utanför repot är "
                f"inte tillåten: {raw_path}"
            )

        if ".git" in path.parts:
            raise PatchError(
                "Patchen får inte ändra "
                f".git: {raw_path}"
            )

    return paths


def strip_markdown_fence(
    patch_text: str,
) -> str:
    lines = patch_text.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) >= 2:
        opening = lines[0].strip().lower()
        closing = lines[-1].strip()

        if (
            opening in {
                "```diff",
                "```patch",
                "```",
            }
            and closing == "```"
        ):
            lines = lines[1:-1]

    return "\n".join(lines)


def normalize_patch_file(
    patch_file: Path,
) -> str:
    try:
        patch_text = patch_file.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise PatchError(
            "Patchfilen måste vara UTF-8."
        ) from exc

    patch_text = strip_markdown_fence(
        patch_text
    )

    if not patch_text.endswith("\n"):
        patch_text += "\n"

    patch_file.write_text(
        patch_text,
        encoding="utf-8",
        newline="\n",
    )

    return patch_text


def patch_is_already_applied(
    repo: Path,
    patch_file: Path,
) -> bool:
    try:
        run_git(
            "apply",
            "--reverse",
            "--check",
            "--recount",
            str(patch_file),
            cwd=repo,
        )

        return True

    except GitError:
        pass

    try:
        run_git(
            "apply",
            "--reverse",
            "--check",
            "--recount",
            "--ignore-space-change",
            str(patch_file),
            cwd=repo,
        )

        return True

    except GitError:
        return False


def _patch_hunks(
    patch_text: str,
) -> list[tuple[str, int, str]]:
    lines = patch_text.splitlines()
    hunks: list[tuple[str, int, str]] = []
    current_path = ""
    old_path = ""
    index = 0
    hunk_header = re.compile(
        r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@"
    )

    while index < len(lines):
        line = lines[index]
        if line.startswith("--- "):
            old_path = line[4:].split("\t", 1)[0].strip()
            if old_path.startswith("a/"):
                old_path = old_path[2:]

        if line.startswith("+++ "):
            raw_path = line[4:].split("\t", 1)[0].strip()
            if raw_path.startswith("b/"):
                raw_path = raw_path[2:]
            current_path = (
                old_path
                if raw_path == "/dev/null"
                else raw_path
            )

        match = hunk_header.match(line)
        if not match:
            index += 1
            continue

        hunk_lines = [line]
        index += 1
        while index < len(lines) and not lines[index].startswith(
            ("@@ ", "diff --git ", "--- ", "+++ ")
        ):
            hunk_lines.append(lines[index])
            index += 1
        hunks.append((
            current_path,
            int(match.group(1)),
            "\n".join(hunk_lines),
        ))

    return hunks


def _safe_candidate_paths(
    repo: Path,
    patch_text: str,
) -> list[str]:
    candidates: set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw_path = line[4:].split("\t", 1)[0].strip().replace("\\", "/")
        if raw_path == "/dev/null":
            continue
        if raw_path.startswith(("a/", "b/")):
            raw_path = raw_path[2:]
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            continue
        candidates.add(path.as_posix())
    return sorted(candidates)


def _hunk_anchor(
    current_lines: list[str],
    hunk: str,
    fallback_line: int,
) -> int:
    candidates = []
    for line in hunk.splitlines():
        if line.startswith((" ", "-")) and len(line) > 1:
            text = line[1:]
            if text.strip():
                candidates.append(text)
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            return current_lines.index(candidate) + 1
        except ValueError:
            continue
    if not current_lines:
        return 0
    return min(max(1, fallback_line), len(current_lines))


def _working_tree_section(
    repo: Path,
    raw_path: str,
    hunk: str = "",
    line_number: int = 1,
    full_limit: int = 80_000,
) -> list[str]:
    path = repo.joinpath(*PurePosixPath(raw_path).parts)
    data = path.read_bytes()
    content = data.decode("utf-8", errors="replace")
    if len(content) <= full_limit:
        label = f"Complete current working-tree file: {raw_path}"
        source = content.rstrip("\r\n") or "[File is empty.]"
    else:
        current_lines = content.splitlines()
        anchor = _hunk_anchor(current_lines, hunk, line_number)
        start = max(1, anchor - 45)
        end = min(len(current_lines), anchor + 65)
        if start > end:
            start = end = max(1, min(len(current_lines), anchor))
        label = f"Current working-tree lines {start}-{end}: {raw_path}"
        source = "\n".join(current_lines[start - 1:end])
        if not source:
            source = "[No non-empty lines in this range.]"
    return [label, "```text", source, "```", ""]


def build_syntax_repair_context(
    repo: Path,
    original_patch: str,
    error: str,
    malformed_hunk: str | None = None,
) -> Path:
    output_file = get_repair_context_file(repo)
    sections: list[str] = []
    for raw_path in _safe_candidate_paths(repo, original_patch):
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        if path.is_file():
            sections.extend(_working_tree_section(repo, raw_path, original_patch))
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
        "Failure type: `invalid_patch_syntax`",
        "",
        "## Original task",
        get_context_task(repo),
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
        "",
    ])
    output_file.write_text("\n".join(content_parts), encoding="utf-8", newline="\n")
    return output_file


def build_stale_repair_context(
    repo: Path,
    patch_text: str,
    stale_reason: str,
    paths: set[str],
) -> Path:
    output_file = get_repair_context_file(repo)
    sections: list[str] = []
    for raw_path in sorted(paths):
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        if path.is_file():
            sections.extend(_working_tree_section(repo, raw_path, patch_text))
    output_file.write_text(
        "\n".join([
            "# ChatCode Patch Repair Context",
            "",
            "Failure type: `stale_context`",
            "",
            "## Original task",
            get_context_task(repo),
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
        ]),
        encoding="utf-8",
        newline="\n",
    )
    return output_file


def build_patch_repair_context(
    repo: Path,
    patch_text: str,
    apply_error: str,
) -> Path:
    output_file = get_repair_context_file(repo)
    error_locations: dict[str, set[int]] = {}
    for raw_path, raw_line in re.findall(
        r"patch failed: (.*?):(\d+)",
        apply_error,
    ):
        path = PurePosixPath(raw_path).as_posix()
        error_locations.setdefault(path, set()).add(int(raw_line))
    hunks = _patch_hunks(patch_text)
    if error_locations:
        failed_hunks = [
            hunk for hunk in hunks
            if (
                hunk[0] in error_locations
                and hunk[1] in error_locations[hunk[0]]
            )
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
            source_section = _working_tree_section(
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
        get_context_task(repo),
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
    output_file.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )
    return output_file


def apply_patch(
    repo: Path,
    patch_file: Path,
) -> ApplyResult:
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

    original_patch = normalize_patch_file(
        patch_file
    )

    try:
        patch_text, _normalized_counts = canonicalize_unified_diff(
            original_patch
        )
    except UnifiedDiffError as exc:
        repair_context = build_syntax_repair_context(
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

    patch_file.write_text(
        patch_text,
        encoding="utf-8",
        newline="\n",
    )

    try:
        run_git(
            "apply",
            "--numstat",
            str(patch_file),
            cwd=repo,
        )
    except GitError as exc:
        repair_context = build_syntax_repair_context(
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
        get_default_patch_file(
            repo
        ).resolve()
    )

    if patch_file == default_patch_file:
        stale_reason = (
            get_stale_context_reason(
                repo,
                paths,
            )
        )

        if stale_reason is not None:
            repair_context = build_stale_repair_context(
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
            str(patch_file),
            cwd=repo,
        )

    except GitError as strict_error:
        if patch_is_already_applied(
            repo,
            patch_file,
        ):
            raise PatchAlreadyApplied(
                "Patchen verkar redan vara "
                "applicerad."
            )

        repair_context = build_patch_repair_context(
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

    # Keep the persistent map current immediately after an apply. Failure to
    # update this derived cache must never turn a successful patch into a
    # reported patch failure; the next context command will repair it by hash.
    try:
        from .indexing.index_manager import update_project_map

        update_project_map(repo, paths=paths, run_semantic=False)
    except Exception:
        pass

    return ApplyResult(
        paths=paths,
        history_entry=history_entry,
    )


def undo_last_patch(
    repo: Path,
) -> UndoResult:
    try:
        history_entry = (
            get_latest_applied_entry(
                repo
            )
        )
    except HistoryError as exc:
        raise PatchUndoError(
            str(exc)
        ) from exc

    patch_file = (
        get_history_patch_file(
            history_entry
        )
    )

    try:
        patch_text = patch_file.read_text(
            encoding="utf-8",
            errors="strict",
        )
    except (
        OSError,
        UnicodeDecodeError,
    ) as exc:
        raise PatchUndoError(
            "Historikpatchen kunde inte "
            "läsas."
        ) from exc

    try:
        paths = validate_patch_paths(
            patch_text
        )
    except PatchError as exc:
        raise PatchUndoError(
            str(exc)
        ) from exc

    ignore_space = False

    try:
        run_git(
            "apply",
            "--reverse",
            "--check",
            "--recount",
            str(patch_file),
            cwd=repo,
        )

    except GitError:
        try:
            run_git(
                "apply",
                "--reverse",
                "--check",
                "--recount",
                "--ignore-space-change",
                str(patch_file),
                cwd=repo,
            )

            ignore_space = True

        except GitError as exc:
            raise PatchUndoError(
                "Den senaste ChatCode patchen "
                "kan inte backas säkert.\n"
                "Filerna kan ha ändrats efter "
                "att patchen applicerades.\n\n"
                f"{exc}"
            ) from exc

    undo_args = [
        "apply",
        "--reverse",
        "--recount",
    ]

    if ignore_space:
        undo_args.append(
            "--ignore-space-change"
        )

    undo_args.append(
        str(patch_file)
    )

    try:
        run_git(
            *undo_args,
            cwd=repo,
        )

    except GitError as exc:
        raise PatchUndoError(
            "Git kunde inte backa "
            "patchen:\n"
            f"{exc}"
        ) from exc

    try:
        undone_entry = (
            move_entry_to_undone(
                repo,
                history_entry,
            )
        )
    except HistoryError as exc:
        raise PatchUndoError(
            str(exc)
        ) from exc

    return UndoResult(
        paths=paths,
        history_entry=undone_entry,
    )
