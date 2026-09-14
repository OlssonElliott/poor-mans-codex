from __future__ import annotations

import re
from dataclasses import dataclass


class UnifiedDiffError(ValueError):
    def __init__(
        self,
        message: str,
        line_number: int | None = None,
        hunk: str | None = None,
    ) -> None:
        self.line_number = line_number
        self.hunk = hunk
        location = (
            f" near line {line_number}"
            if line_number is not None
            else ""
        )
        super().__init__(f"{message}{location}")


@dataclass(frozen=True)
class DiffLine:
    kind: str
    text: str


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: tuple[DiffLine, ...]
    source_line: int
    original_header: str

    @property
    def actual_old_count(self) -> int:
        return sum(
            line.kind in {" ", "-"}
            for line in self.lines
        )

    @property
    def actual_new_count(self) -> int:
        return sum(
            line.kind in {" ", "+"}
            for line in self.lines
        )


@dataclass(frozen=True)
class FilePatch:
    old_path: str
    new_path: str
    hunks: tuple[Hunk, ...]


@dataclass(frozen=True)
class UnifiedDiff:
    files: tuple[FilePatch, ...]


HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)


def _header_path(line: str, prefix: str) -> str:
    value = line[len(prefix):].split("\t", 1)[0].strip()
    if not value:
        raise UnifiedDiffError("empty file header")
    return value.replace("\\", "/")


def _is_file_header(lines: list[str], index: int) -> bool:
    return (
        lines[index].startswith("--- ")
        and index + 1 < len(lines)
        and lines[index + 1].startswith("+++ ")
    )


def _is_file_boundary(lines: list[str], index: int) -> bool:
    return (
        _is_file_header(lines, index)
        and index + 2 < len(lines)
        and lines[index + 2].startswith("@@ ")
    )


def _nearby_hunk(lines: list[str], index: int) -> str:
    end = min(len(lines), index + 40)
    for candidate in range(index + 1, end):
        if lines[candidate].startswith("diff --git "):
            end = candidate
            break
        if _is_file_header(lines, candidate):
            end = candidate
            break
    return "\n".join(lines[index:end])


def parse_unified_diff(text: str) -> UnifiedDiff:
    lines = text.splitlines()
    files: list[FilePatch] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith((
            "diff --git ",
            "index ",
            "new file mode ",
            "deleted file mode ",
            "old mode ",
            "new mode ",
            "similarity index ",
            "rename from ",
            "rename to ",
        )):
            index += 1
            continue
        if not _is_file_header(lines, index):
            raise UnifiedDiffError(
                "expected valid ---/+++ file headers",
                index + 1,
            )

        old_path = _header_path(lines[index], "--- ")
        new_path = _header_path(lines[index + 1], "+++ ")
        if old_path == "/dev/null" and new_path == "/dev/null":
            raise UnifiedDiffError(
                "both file headers refer to /dev/null",
                index + 1,
            )
        index += 2
        hunks: list[Hunk] = []

        while index < len(lines):
            if _is_file_boundary(lines, index) or lines[index].startswith(
                "diff --git "
            ):
                break
            if not lines[index]:
                raise UnifiedDiffError(
                    "unexpected empty line outside a hunk",
                    index + 1,
                )

            header = lines[index]
            match = HUNK_HEADER.match(header)
            if not match:
                raise UnifiedDiffError(
                    "missing or malformed hunk header",
                    index + 1,
                    _nearby_hunk(lines, index),
                )

            source_line = index + 1
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            section = match.group(5)
            index += 1
            body: list[DiffLine] = []

            while index < len(lines):
                if lines[index].startswith("@@ "):
                    break
                if lines[index].startswith("diff --git "):
                    break
                if _is_file_boundary(lines, index):
                    break
                if _is_file_header(lines, index):
                    raise UnifiedDiffError(
                        "ambiguous or incomplete file boundary",
                        index + 1,
                        _nearby_hunk(lines, index),
                    )

                body_line = lines[index]
                if body_line.startswith((" ", "+", "-")):
                    body.append(DiffLine(body_line[0], body_line[1:]))
                elif body_line == r"\ No newline at end of file":
                    if not body or body[-1].kind == "\\":
                        raise UnifiedDiffError(
                            "misplaced no-newline metadata",
                            index + 1,
                            "\n".join([header, *lines[source_line:index + 1]]),
                        )
                    body.append(DiffLine("\\", body_line[2:]))
                else:
                    raise UnifiedDiffError(
                        "invalid line in hunk body",
                        index + 1,
                        "\n".join([header, *lines[source_line:index + 1]]),
                    )
                index += 1

            if not body or not any(line.kind != "\\" for line in body):
                raise UnifiedDiffError(
                    "empty or incomplete hunk",
                    source_line,
                    header,
                )

            hunks.append(Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                section=section,
                lines=tuple(body),
                source_line=source_line,
                original_header=header,
            ))

        if not hunks:
            raise UnifiedDiffError(
                "file patch has no hunks",
                index + 1,
            )
        files.append(FilePatch(old_path, new_path, tuple(hunks)))

    if not files:
        raise UnifiedDiffError("patch contains no file patches")
    return UnifiedDiff(tuple(files))


def count_mismatches(diff: UnifiedDiff) -> list[str]:
    mismatches: list[str] = []
    for file_patch in diff.files:
        path = (
            file_patch.new_path
            if file_patch.new_path != "/dev/null"
            else file_patch.old_path
        )
        for hunk in file_patch.hunks:
            if (
                hunk.old_count != hunk.actual_old_count
                or hunk.new_count != hunk.actual_new_count
            ):
                mismatches.append(
                    f"{path}:{hunk.source_line}: header declares "
                    f"-{hunk.old_count}/+{hunk.new_count}, body is "
                    f"-{hunk.actual_old_count}/+{hunk.actual_new_count}"
                )
    return mismatches


def _canonical_path(path: str, side: str) -> str:
    if path == "/dev/null":
        return path
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return f"{side}/{path}"


def serialize_unified_diff(diff: UnifiedDiff) -> str:
    output: list[str] = []
    for file_patch in diff.files:
        output.append(f"--- {_canonical_path(file_patch.old_path, 'a')}")
        output.append(f"+++ {_canonical_path(file_patch.new_path, 'b')}")
        for hunk in file_patch.hunks:
            old_count = hunk.actual_old_count
            new_count = hunk.actual_new_count
            output.append(
                f"@@ -{hunk.old_start},{old_count} "
                f"+{hunk.new_start},{new_count} @@{hunk.section}"
            )
            for line in hunk.lines:
                if line.kind == "\\":
                    output.append(r"\ No newline at end of file")
                else:
                    output.append(line.kind + line.text)
    return "\n".join(output) + "\n"


def canonicalize_unified_diff(text: str) -> tuple[str, list[str]]:
    parsed = parse_unified_diff(text)
    return serialize_unified_diff(parsed), count_mismatches(parsed)
