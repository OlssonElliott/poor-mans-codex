"""Patch path validation, hunk normalization, and current-source excerpts."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path, PurePosixPath

from ..errors import PatchError
from ...git_utils import GitError, run_git
from ...unified_diff import (
    DiffLine,
    Hunk,
    UnifiedDiffError,
    parse_unified_diff,
    serialize_unified_diff,
)


def extract_patch_paths(
    patch_text: str,
) -> set[str]:
    paths: set[str] = set()

    try:
        parsed = parse_unified_diff(
            patch_text
        )
    except UnifiedDiffError:
        candidates = []
        for line in patch_text.splitlines():
            if not line.startswith(
                ("--- ", "+++ ")
            ):
                continue
            candidates.append(
                line[4:]
                .strip()
                .split("\t", 1)[0]
            )
    else:
        candidates = [
            raw_path
            for file_patch in parsed.files
            for raw_path in (
                file_patch.old_path,
                file_patch.new_path,
            )
        ]

    for raw_path in candidates:
        if raw_path == "/dev/null":
            continue

        if raw_path.startswith(
            ("a/", "b/")
        ):
            raw_path = raw_path[2:]

        raw_path = raw_path.replace("\\", "/")
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
        raw_path = raw_path.replace("\\", "/")
        path = PurePosixPath(
            raw_path
        )

        if path.is_absolute() or re.match(r"^[A-Za-z]:", raw_path):
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


MIN_HUNK_CONTEXT_LINES = 3


def _required_hunk_context(source_lines: list[str], hunk: Hunk) -> int:
    removed_lines = sum(line.kind == "-" for line in hunk.lines)
    return min(
        MIN_HUNK_CONTEXT_LINES,
        max(0, len(source_lines) - removed_lines),
    )


def _old_hunk_lines(hunk: Hunk) -> list[str]:
    return [line.text for line in hunk.lines if line.kind in {" ", "-"}]


def _unique_sequence_start(
    source_lines: list[str],
    needle: list[str],
) -> int | None:
    """Return a zero-based start only when the exact old-side text is unique."""
    if not needle or len(needle) > len(source_lines):
        return None
    match: int | None = None
    final_start = len(source_lines) - len(needle)
    for start in range(final_start + 1):
        if source_lines[start:start + len(needle)] != needle:
            continue
        if match is not None:
            return None
        match = start
    return match


def _expand_hunk_from_current_source(
    source_lines: list[str],
    hunk: Hunk,
) -> Hunk:
    context_lines = sum(line.kind == " " for line in hunk.lines)
    required_context = _required_hunk_context(source_lines, hunk)
    if context_lines >= required_context:
        return hunk

    old_lines = _old_hunk_lines(hunk)
    match_start = _unique_sequence_start(source_lines, old_lines)
    if match_start is None:
        return hunk

    match_end = match_start + len(old_lines)
    missing_context = required_context - context_lines
    before_capacity = min(MIN_HUNK_CONTEXT_LINES, match_start)
    after_capacity = min(
        MIN_HUNK_CONTEXT_LINES,
        len(source_lines) - match_end,
    )

    before_count = min(before_capacity, (missing_context + 1) // 2)
    after_count = min(after_capacity, missing_context - before_count)
    remaining = missing_context - before_count - after_count
    if remaining:
        extra_before = min(before_capacity - before_count, remaining)
        before_count += extra_before
        remaining -= extra_before
    if remaining:
        extra_after = min(after_capacity - after_count, remaining)
        after_count += extra_after
        remaining -= extra_after
    if remaining:
        return hunk

    expanded_old_start = match_start - before_count + 1
    expanded_new_start = expanded_old_start + (hunk.new_start - hunk.old_start)
    if expanded_new_start < 1:
        return hunk

    before = tuple(
        DiffLine(" ", line)
        for line in source_lines[match_start - before_count:match_start]
    )
    after = tuple(
        DiffLine(" ", line)
        for line in source_lines[match_end:match_end + after_count]
    )
    return replace(
        hunk,
        old_start=expanded_old_start,
        new_start=expanded_new_start,
        lines=before + hunk.lines + after,
    )


def _nonblank_sequence_spans(
    source_lines: list[str],
    needle: list[str],
) -> list[tuple[int, int]]:
    """Return source spans whose nonblank lines exactly match the needle."""
    wanted = [line for line in needle if line.strip()]
    if not wanted:
        return []

    indexed_source = [
        (index, line)
        for index, line in enumerate(source_lines)
        if line.strip()
    ]
    if len(wanted) > len(indexed_source):
        return []

    spans: list[tuple[int, int]] = []
    final_start = len(indexed_source) - len(wanted)
    for start in range(final_start + 1):
        candidate = indexed_source[start:start + len(wanted)]
        if [line for _, line in candidate] != wanted:
            continue
        spans.append((candidate[0][0], candidate[-1][0] + 1))
    return spans


def _unique_blank_gap(
    source_lines: list[str],
    before_anchor: list[str],
    after_anchor: list[str],
) -> tuple[int, int] | None:
    """Find one blank-only gap uniquely identified by its surrounding anchors."""
    before_spans = _nonblank_sequence_spans(source_lines, before_anchor)
    after_spans = _nonblank_sequence_spans(source_lines, after_anchor)
    matches: list[tuple[int, int]] = []

    for _before_start, before_end in before_spans:
        for after_start, _after_end in after_spans:
            if before_end > after_start:
                continue
            if any(line.strip() for line in source_lines[before_end:after_start]):
                continue
            matches.append((before_end, after_start))
            if len(matches) > 1:
                return None

    return matches[0] if matches else None


def _rebuild_hunk_context(
    source_lines: list[str],
    hunk: Hunk,
    body: tuple[DiffLine, ...],
    core_start: int,
    core_end: int,
) -> Hunk:
    """Rebuild surrounding context from exact current working-tree source."""
    internal_context = sum(line.kind == " " for line in body)
    required_context = _required_hunk_context(source_lines, hunk)
    missing_context = max(0, required_context - internal_context)

    before_capacity = min(MIN_HUNK_CONTEXT_LINES, core_start)
    after_capacity = min(
        MIN_HUNK_CONTEXT_LINES,
        len(source_lines) - core_end,
    )

    before_count = min(
        before_capacity,
        (missing_context + 1) // 2,
    )
    after_count = min(
        after_capacity,
        missing_context - before_count,
    )
    remaining = missing_context - before_count - after_count

    if remaining:
        extra_before = min(
            before_capacity - before_count,
            remaining,
        )
        before_count += extra_before
        remaining -= extra_before

    if remaining:
        extra_after = min(
            after_capacity - after_count,
            remaining,
        )
        after_count += extra_after
        remaining -= extra_after

    if remaining:
        return hunk

    old_start = core_start - before_count + 1
    shift = old_start - hunk.old_start
    new_start = hunk.new_start + shift
    if new_start < 1:
        return hunk

    before = tuple(
        DiffLine(" ", line)
        for line in source_lines[
            core_start - before_count:core_start
        ]
    )
    after = tuple(
        DiffLine(" ", line)
        for line in source_lines[
            core_end:core_end + after_count
        ]
    )

    return replace(
        hunk,
        old_start=old_start,
        new_start=new_start,
        lines=before + body + after,
    )


def _reanchor_hunk_from_current_source(
    source_lines: list[str],
    hunk: Hunk,
) -> Hunk:
    """Safely re-anchor a context-rich mismatched hunk against current source."""
    context_lines = sum(line.kind == " " for line in hunk.lines)
    required_context = _required_hunk_context(source_lines, hunk)
    if context_lines < required_context:
        return hunk

    if any(line.kind == "\\" for line in hunk.lines):
        return hunk

    old_lines = _old_hunk_lines(hunk)
    declared_start = hunk.old_start - 1
    if (
        declared_start >= 0
        and source_lines[
            declared_start:declared_start + len(old_lines)
        ] == old_lines
    ):
        return hunk

    exact_start = _unique_sequence_start(source_lines, old_lines)
    if exact_start is not None:
        shift = exact_start + 1 - hunk.old_start
        new_start = hunk.new_start + shift
        if new_start < 1:
            return hunk
        return replace(
            hunk,
            old_start=exact_start + 1,
            new_start=new_start,
        )

    changed_indexes = [
        index
        for index, line in enumerate(hunk.lines)
        if line.kind in {"+", "-"}
    ]
    if not changed_indexes:
        return hunk

    first_change = changed_indexes[0]
    last_change = changed_indexes[-1]
    leading = hunk.lines[:first_change]
    body = hunk.lines[first_change:last_change + 1]
    trailing = hunk.lines[last_change + 1:]

    # Replacements/deletions are only repaired when their exact old change
    # body occurs once in the current working tree.
    if any(line.kind == "-" for line in body):
        old_body = [
            line.text
            for line in body
            if line.kind in {" ", "-"}
        ]
        core_start = _unique_sequence_start(source_lines, old_body)
        if core_start is None:
            return hunk
        return _rebuild_hunk_context(
            source_lines,
            hunk,
            body,
            core_start,
            core_start + len(old_body),
        )

    # For pure insertions, only tolerate blank-line drift between surrounding
    # anchors. Never skip or replace nonblank current source.
    if any(line.kind != "+" for line in body):
        return hunk

    before_anchor = [
        line.text
        for line in leading
        if line.kind == " " and line.text.strip()
    ][-MIN_HUNK_CONTEXT_LINES:]
    after_anchor = [
        line.text
        for line in trailing
        if line.kind == " " and line.text.strip()
    ][:MIN_HUNK_CONTEXT_LINES]
    if not before_anchor or not after_anchor:
        return hunk

    gap = _unique_blank_gap(
        source_lines,
        before_anchor,
        after_anchor,
    )
    if gap is None:
        return hunk

    _gap_start, insertion_index = gap
    return _rebuild_hunk_context(
        source_lines,
        hunk,
        body,
        insertion_index,
        insertion_index,
    )


def _hunks_are_ordered_and_disjoint(
    hunks: tuple[Hunk, ...],
) -> bool:
    previous_end = 0
    for hunk in hunks:
        old_count = max(1, hunk.actual_old_count)
        current_end = hunk.old_start + old_count - 1
        if hunk.old_start <= previous_end:
            return False
        previous_end = current_end
    return True


def _expand_thin_hunk_context(repo: Path, patch_text: str) -> str:
    """Normalize patch hunks against exact current working-tree source."""
    parsed = parse_unified_diff(patch_text)
    files = []
    changed = False
    for file_patch in parsed.files:
        if file_patch.old_path == "/dev/null":
            files.append(file_patch)
            continue
        raw_path = file_patch.old_path.removeprefix("a/")
        source = repo.joinpath(*PurePosixPath(raw_path).parts)
        try:
            source_lines = source.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            files.append(file_patch)
            continue

        expanded_hunks = tuple(
            _expand_hunk_from_current_source(source_lines, hunk)
            for hunk in file_patch.hunks
        )
        normalized_hunks = tuple(
            _reanchor_hunk_from_current_source(source_lines, hunk)
            for hunk in expanded_hunks
        )
        if (
            normalized_hunks != expanded_hunks
            and not _hunks_are_ordered_and_disjoint(normalized_hunks)
        ):
            normalized_hunks = expanded_hunks

        if normalized_hunks != file_patch.hunks:
            changed = True
            file_patch = replace(file_patch, hunks=normalized_hunks)
        files.append(file_patch)
    if not changed:
        return patch_text
    return serialize_unified_diff(replace(parsed, files=tuple(files)))


def _validate_hunk_context(
    repo: Path,
    patch_text: str,
) -> None:
    """Reject fragile, line-number-only edits to non-trivial existing files."""
    parsed = parse_unified_diff(patch_text)
    for file_patch in parsed.files:
        if file_patch.old_path == "/dev/null":
            continue
        raw_path = file_patch.old_path.removeprefix("a/")
        source = repo.joinpath(*PurePosixPath(raw_path).parts)
        try:
            source_lines = source.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        for hunk in file_patch.hunks:
            context_lines = sum(line.kind == " " for line in hunk.lines)
            required_context = _required_hunk_context(source_lines, hunk)
            # Replacing/deleting the complete file has no possible context.
            if context_lines >= required_context:
                continue
            raise UnifiedDiffError(
                f"hunk has only {context_lines} unchanged context line(s); "
                f"include at least {required_context} surrounding current "
                "source line(s) instead of relying on a line number",
                hunk.source_line,
                "\n".join([
                    hunk.original_header,
                    *(line.kind + line.text for line in hunk.lines),
                ]),
            )


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
    *,
    write_back: bool = True,
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

    if write_back:
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
    digest = hashlib.sha256(data).hexdigest()
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
    return [
        label,
        f"SHA-256 (current file): `{digest}`",
        "```text",
        source,
        "```",
        "",
    ]
