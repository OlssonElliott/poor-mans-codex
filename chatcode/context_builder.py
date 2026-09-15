from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from .file_filter import (
    is_ignored,
    iter_repository_files,
)
from .git_utils import (
    get_branch,
    get_staged_changed_paths,
    get_staged_diff,
    get_status,
    get_untracked_paths,
    get_unstaged_changed_paths,
    get_unstaged_diff,
)
from .indexing.index_manager import IndexProgress, update_project_map
from .indexing.project_graph import load_map, save_map
from .retrieval.graph_retriever import retrieve_files
from .workspace import (
    get_repo_workspace,
    get_test_results_file,
)


SOURCE_SUFFIXES = {
    ".py",
    ".java",
    ".kt",
    ".php",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".properties",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".md",
}


IMPORTANT_FILES = {
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "package.json",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
    "pyproject.toml",
}


TASK_SYNONYMS = {
    "bild": {
        "image",
        "images",
        "upload",
        "attachment",
        "attachments",
    },
    "bilder": {
        "image",
        "images",
        "upload",
        "attachment",
        "attachments",
    },
    "mail": {
        "mail",
        "email",
        "smtp",
        "contact",
    },
    "mejl": {
        "mail",
        "email",
        "smtp",
        "contact",
    },
    "skicka": {
        "send",
        "submit",
        "post",
        "request",
    },
    "formulär": {
        "form",
        "contact",
        "submit",
    },
}


STOP_WORDS = {
    "att",
    "det",
    "den",
    "som",
    "och",
    "för",
    "med",
    "hur",
    "var",
    "vad",
    "hitta",
    "koden",
    "hanterar",
}


MAX_FILES = 12
MAX_FILE_CHARS = 20_000
MAX_TOTAL_CHARS = 120_000
MAX_TEST_RESULT_CHARS = 60_000
PATCH_FULL_FILE_CHARS = 80_000
PATCH_EXCERPT_RADIUS = 45
CONTEXT_CAPTURE_RETRIES = 3

CONTEXT_PURPOSE = (
    "This request concerns maintenance of the users own local software repository."
)

PATCH_RESPONSE_INSTRUCTIONS = """\
When this task requires code changes:

1. Return a valid unified diff that can be applied with `git apply`.
2. All file paths must be relative to the repository root.
3. Do not use absolute paths.
4. Put the entire patch inside a single Markdown code block labeled `diff`.
5. Do not include any explanations before or after the code block.
6. Include all required changes in a single patch.
7. Preserve unrelated existing changes.
8. Do not modify `.git` or sensitive files such as `.env`, credentials, private keys or secrets.
9. For new files, use `/dev/null` as the old file.
10. For deleted files, use `/dev/null` as the new file.
11. Build `@@` hunk line numbers from the source-file line ranges shown in the
    FILE/EXCERPT blocks, never from Markdown/document line numbers.

The contents of the code block must be directly saveable as `incoming.diff`
and applicable with:

chatcode apply
"""

PATCH_CONTEXT_HEADER = """\
The files below are the exact CURRENT working-tree contents.
Generate the patch against these contents.
Uncommitted changes are intentional and must be preserved."""

UPLOAD_INSTRUCTIONS = """\
This request concerns maintenance of the users own local software repository.

This file is an execution request.

When this file is uploaded to ChatGPT, immediately perform the task described
under "## Task".

Do not ask the user what they want you to do.
Do not ask for confirmation.
Treat the upload of this file itself as the user's request to perform the task.

Complete the task entirely in the current ChatGPT conversation.

Do not switch to, suggest, invoke, or require ChatGPT Work, Codex, Computer Use,
Canvas, or any other execution mode or external coding environment.

Do not ask the user to continue the task in another mode.

The user is intentionally using ChatCode as the local execution layer.
ChatGPT should analyze the supplied repository context and return the requested
code changes in chat. ChatCode will handle applying, testing, reviewing and
undoing those changes locally.

Do not attempt to directly edit the user's local repository or filesystem.

Use the repository context, source files, Git changes and test results contained
in this file as the basis for the work.

If the task requires code changes, follow the instructions under
"## Response instructions" exactly.
"""


PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?"
    r"-----END [^-]*PRIVATE KEY-----",
    re.DOTALL,
)


KNOWN_SECRET_PATTERNS = [
    re.compile(
        r"\bsk-[A-Za-z0-9_-]{20,}\b"
    ),
    re.compile(
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"
    ),
    re.compile(
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"
    ),
    re.compile(
        r"\bAKIA[0-9A-Z]{16}\b"
    ),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}"
        r"\.[A-Za-z0-9_-]{10,}"
        r"\.[A-Za-z0-9_-]{10,}\b"
    ),
]


QUOTED_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (
        \b
        (?:
            password
            | passwd
            | api[_-]?key
            | secret
            | token
            | access[_-]?token
            | refresh[_-]?token
            | client[_-]?secret
        )
        \b
        \s*
        [:=]
        \s*
        ["']
    )
    [^"'\r\n]{6,}
    (["'])
    """
)


UNQUOTED_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (
        \b
        (?:
            password
            | passwd
            | api[_-]?key
            | secret
            | token
            | access[_-]?token
            | refresh[_-]?token
            | client[_-]?secret
        )
        \b
        \s*
        =
        \s*
    )
    [^\s,;\#\r\n]{8,}
    """
)


BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)(Authorization\s*:\s*Bearer\s+)"
    r"[A-Za-z0-9._~+/=-]{8,}"
)


def build_safe_status(
    repo: Path,
) -> str:
    status = get_status(repo)

    if not status:
        return ""

    safe_lines: list[str] = []
    omitted_count = 0

    for line in status.splitlines():
        if len(line) < 4:
            continue

        raw_path = line[3:].strip()

        paths = [raw_path]

        if " -> " in raw_path:
            paths = raw_path.split(" -> ", 1)

        should_omit = False

        for status_path in paths:
            path = repo / status_path.strip()

            if (
                is_ignored(path, repo)
                or _is_context_internal_artifact(path, repo)
            ):
                should_omit = True
                break

        if should_omit:
            omitted_count += 1
            continue

        safe_lines.append(line)

    if omitted_count:
        safe_lines.append(
            "[ChatCode omitted "
            f"{omitted_count} sensitive or ignored "
            "file(s) from Git status.]"
        )

    return "\n".join(safe_lines)


def redact_sensitive_text(text: str) -> str:
    text = PRIVATE_KEY_PATTERN.sub(
        "[REDACTED PRIVATE KEY]",
        text,
    )

    for pattern in KNOWN_SECRET_PATTERNS:
        text = pattern.sub(
            "[REDACTED SECRET]",
            text,
        )

    text = QUOTED_SECRET_ASSIGNMENT.sub(
        r"\1[REDACTED]\2",
        text,
    )

    text = UNQUOTED_SECRET_ASSIGNMENT.sub(
        r"\1[REDACTED]",
        text,
    )

    text = BEARER_TOKEN_PATTERN.sub(
        r"\1[REDACTED]",
        text,
    )

    return text


def is_source_file(path: Path) -> bool:
    return (
        path.name.lower() in IMPORTANT_FILES
        or path.suffix.lower() in SOURCE_SUFFIXES
    )


def _internal_workspace_root(repo: Path) -> Path | None:
    """Return ChatCode state root when that state lives inside the target repo."""
    repo_root = repo.resolve()
    try:
        repo_workspace = get_repo_workspace(repo).resolve()
    except OSError:
        return None

    for candidate in (repo_workspace, *repo_workspace.parents):
        if candidate.parent == repo_root:
            return candidate
        if candidate == repo_root:
            break

    return None


def _is_internal_workspace_path(
    path: Path,
    repo: Path,
    workspace_root: Path | None = None,
) -> bool:
    workspace_root = workspace_root or _internal_workspace_root(repo)
    if workspace_root is None:
        return False

    try:
        path.resolve().relative_to(workspace_root)
        return True
    except (OSError, ValueError):
        return False


def _is_context_internal_artifact(
    path: Path,
    repo: Path,
) -> bool:
    """Identify ChatCode-owned state without hiding ordinary user files."""
    if _is_internal_workspace_path(path, repo):
        return True

    try:
        relative = path.resolve().relative_to(repo.resolve())
    except (OSError, ValueError):
        return False

    parts = tuple(part.lower() for part in relative.parts)
    if not parts:
        return False

    if parts[0] == ".chatcode":
        return True

    is_chatcode_source_repo = (
        (repo / "chatcode" / "context_builder.py").is_file()
        and (repo / "chatcode" / "workspace.py").is_file()
    )
    if not is_chatcode_source_repo:
        return False

    if len(parts) == 1 and parts[0] in {
        "upload_to_chatgpt.md",
        "patch_repair_context.md",
        "context.md",
        "context-state.json",
        "project-map.json",
    }:
        return True

    return parts[0] in {
        "history",
        "patches",
        "test-results",
    }


def _iter_project_files(repo: Path):
    workspace_root = _internal_workspace_root(repo)
    for path in iter_repository_files(repo):
        if _is_internal_workspace_path(path, repo, workspace_root):
            continue
        yield path


def _iter_context_files(repo: Path):
    for path in _iter_project_files(repo):
        if _is_context_internal_artifact(path, repo):
            continue
        yield path


def _task_file_tokens(task: str) -> list[str]:
    tokens = re.findall(
        r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+",
        task,
    )
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip(".,:;!?()[]{}'\"`")
        if token:
            cleaned.append(token)
    return cleaned


def _is_selectable_task_file(
    path: Path,
    repo: Path,
) -> bool:
    try:
        path.resolve().relative_to(repo.resolve())
    except (OSError, ValueError):
        return False

    return (
        path.is_file()
        and is_source_file(path)
        and not is_ignored(path, repo)
        and not _is_context_internal_artifact(path, repo)
        and not _is_internal_workspace_path(path, repo)
    )


def _resolve_task_file_references(
    repo: Path,
    task: str,
) -> list[Path]:
    """Resolve deterministic file references from task text against the repo."""
    repo_root = repo.resolve()
    resolved: list[Path] = []
    context_files: list[Path] | None = None

    def all_context_files() -> list[Path]:
        nonlocal context_files
        if context_files is None:
            context_files = [
                path.resolve()
                for path in _iter_context_files(repo)
                if _is_selectable_task_file(path, repo)
            ]
        return context_files

    for raw_token in _task_file_tokens(task):
        normalized = raw_token.replace("\\", "/")
        posix = PurePosixPath(normalized)
        parts = posix.parts

        if (
            not parts
            or posix.is_absolute()
            or ".." in parts
            or re.match(r"^[A-Za-z]:", normalized)
        ):
            continue

        candidate = repo_root.joinpath(*parts)
        if _is_selectable_task_file(candidate, repo):
            candidate = candidate.resolve()
            if candidate not in resolved:
                resolved.append(candidate)
            continue

        lowered = normalized.lower()
        matches: list[Path] = []

        if "/" in normalized:
            matches = [
                path
                for path in all_context_files()
                if path.relative_to(repo_root).as_posix().lower() == lowered
            ]
        elif "." in normalized:
            matches = [
                path
                for path in all_context_files()
                if path.name.lower() == lowered
            ]
        else:
            try:
                matches = [
                    path.resolve()
                    for path in repo_root.iterdir()
                    if _is_selectable_task_file(path, repo)
                    and (
                        path.name.lower() == lowered
                        or path.stem.lower() == lowered
                    )
                ]
            except OSError:
                matches = []

        if len(matches) == 1 and matches[0] not in resolved:
            resolved.append(matches[0])

    return resolved


def _ensure_explicit_task_files(
    repo: Path,
    task: str,
    files: list[Path],
) -> list[Path]:
    selected = list(files)
    for path in _resolve_task_file_references(repo, task):
        if path not in selected:
            selected.append(path)
    return selected


def _purge_internal_workspace_from_index(repo: Path) -> None:
    workspace_root = _internal_workspace_root(repo)
    if workspace_root is None:
        return

    try:
        graph = load_map(repo)
    except Exception:
        return

    files = graph.get("files")
    if not isinstance(files, dict):
        return

    removed = [
        raw_path
        for raw_path in files
        if _is_internal_workspace_path(
            repo.joinpath(*PurePosixPath(raw_path).parts),
            repo,
            workspace_root,
        )
    ]
    if not removed:
        return

    for raw_path in removed:
        files.pop(raw_path, None)

    try:
        save_map(repo, graph)
    except Exception:
        pass


def _index_scope_paths(repo: Path) -> list[str] | None:
    """Prevent a repo-local ChatCode workspace from entering the project index."""
    workspace_root = _internal_workspace_root(repo)
    if workspace_root is None:
        return None

    paths = {
        path.relative_to(repo).as_posix()
        for path in _iter_project_files(repo)
        if is_source_file(path)
    }

    changed_paths = set(get_unstaged_changed_paths(repo))
    changed_paths.update(get_staged_changed_paths(repo))
    changed_paths.update(get_untracked_paths(repo))

    for raw_path in changed_paths:
        normalized = raw_path.replace("\\", "/")
        candidate = (repo / normalized).resolve()
        if _is_internal_workspace_path(candidate, repo, workspace_root):
            continue
        if is_source_file(Path(normalized)):
            paths.add(normalized)

    return sorted(paths)


def build_tree(
    repo: Path,
    max_files: int = 500,
) -> str:
    lines: list[str] = []
    file_count = 0

    for path in _iter_context_files(repo):

        file_count += 1

        if file_count > max_files:
            lines.append(
                f"... tree truncated after "
                f"{max_files} files"
            )
            break

        relative = path.relative_to(repo)

        lines.append(
            str(relative)
        )

    return "\n".join(lines)


def get_changed_files(
    repo: Path,
) -> set[Path]:
    changed: set[Path] = set()

    paths = set(
        get_unstaged_changed_paths(repo)
    )

    paths.update(
        get_staged_changed_paths(repo)
    )

    paths.update(
        get_untracked_paths(repo)
    )

    for raw_path in paths:
        path = (repo / raw_path).resolve()

        if not path.exists():
            continue

        if not path.is_file():
            continue

        if _is_context_internal_artifact(path, repo):
            continue

        if _is_internal_workspace_path(path, repo):
            continue

        if is_ignored(path, repo):
            continue

        changed.add(path)

    return changed


def get_task_words(task: str) -> set[str]:
    words = {
        word.lower()
        for word in re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+",
            task,
        )
        if len(word) >= 3
    }

    words -= STOP_WORDS

    expanded = set(words)

    for word in words:
        expanded.update(
            TASK_SYNONYMS.get(
                word,
                set(),
            )
        )

    return expanded


def score_file(
    path: Path,
    repo: Path,
    task_words: set[str],
    changed_files: set[Path],
) -> int:
    score = 0

    relative = str(
        path.relative_to(repo)
    ).lower()

    filename = path.name.lower()

    if path in changed_files:
        score += 1000

    if filename in IMPORTANT_FILES:
        score += 50

    for word in task_words:
        if word in filename:
            score += 100

        elif word in relative:
            score += 50

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).lower()

        content = content[:1_000_000]

        for word in task_words:
            occurrences = content.count(word)

            if occurrences:
                score += (
                    min(occurrences, 10) * 5
                )

    except OSError:
        pass

    return score


def collect_relevant_files(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> list[Path]:
    # Synchronizing here also catches edits made manually since the previous
    # ChatCode invocation. Indexing is an enhancement, so a damaged/unwritable
    # cache must not prevent the established retrieval path from working.
    graph_files: list[Path] = []
    effective_mode = "static"
    try:
        _purge_internal_workspace_from_index(repo)
        index_paths = _index_scope_paths(repo)
        update = update_project_map(
            repo,
            paths=index_paths,
            progress=index_progress,
        )
        effective_mode = update.effective_mode
        graph_files = [
            path
            for path in retrieve_files(
                repo,
                task,
                max_files=MAX_FILES,
                index_mode=effective_mode,
            )
            if (
                not _is_internal_workspace_path(path, repo)
                and not _is_context_internal_artifact(path, repo)
            )
        ]
    except Exception:
        graph_files = []

    task_words = get_task_words(task)
    changed_files = get_changed_files(repo)
    explicit_files = _resolve_task_file_references(repo, task)

    if effective_mode == "ai":
        # AI retrieval is the primary selector in AI mode. Do not also scan
        # every source file's contents through the legacy static ranker.
        changed_source_files = sorted(
            path
            for path in changed_files
            if path.is_file()
            and is_source_file(path)
            and not _is_context_internal_artifact(path, repo)
        )
        selected: list[Path] = list(changed_source_files)
        for path in explicit_files:
            if path not in selected:
                selected.append(path)
        selection_limit = max(
            MAX_FILES,
            len(selected),
        )
        for path in graph_files:
            if path.is_file() and is_source_file(path) and path not in selected:
                selected.append(path)
            if len(selected) >= selection_limit:
                break
        return selected

    candidate_scores: dict[Path, int] = {}

    for path in _iter_context_files(repo):
        if not is_source_file(path):
            continue

        score = score_file(
            path,
            repo,
            task_words,
            changed_files,
        )

        if score > 0:
            candidate_scores[path] = score

    # Graph hits complement keyword/content scoring. Earlier graph results get
    # a larger boost while dirty files retain their existing highest priority.
    for rank, path in enumerate(graph_files):
        candidate_scores[path] = candidate_scores.get(path, 0) + max(
            100,
            300 - rank * 15,
        )

    candidates = sorted(
        candidate_scores.items(),
        key=lambda item: (
            -item[1],
            str(item[0]).lower(),
        )
    )

    changed_source_files = sorted(
        path
        for path in changed_files
        if path.is_file()
        and is_source_file(path)
        and not _is_context_internal_artifact(path, repo)
    )
    selected = list(changed_source_files)
    for path in explicit_files:
        if path not in selected:
            selected.append(path)
    selection_limit = max(
        MAX_FILES,
        len(selected),
    )

    for path, _ in candidates:
        if path not in selected:
            selected.append(path)
        if len(selected) >= selection_limit:
            break

    return selected


def build_source_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> str:
    files = collect_relevant_files(
        repo,
        task,
        index_progress=index_progress,
    )

    sections: list[str] = []
    total_chars = 0

    for path in files:
        try:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        content = redact_sensitive_text(
            content
        )

        if len(content) > MAX_FILE_CHARS:
            content = (
                content[:MAX_FILE_CHARS]
                + "\n\n"
                + "[File truncated by ChatCode]"
            )

        relative = path.relative_to(repo)

        section = (
            f"\n===== FILE: {relative} =====\n\n"
            f"{content}\n"
        )

        if (
            total_chars + len(section)
            > MAX_TOTAL_CHARS
        ):
            break

        sections.append(section)
        total_chars += len(section)

    if not sections:
        return (
            "No relevant source files found."
        )

    return "\n".join(sections)


def _read_current_text_and_hash(
    path: Path,
) -> tuple[str, str]:
    data = path.read_bytes()
    return (
        data.decode("utf-8", errors="replace"),
        hashlib.sha256(data).hexdigest(),
    )


def _merge_line_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []

    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )
        else:
            merged.append([start, end])

    return [
        (start, end)
        for start, end in merged
    ]


def _symbol_aware_ranges(
    content: str,
    task: str,
) -> list[tuple[int, int]]:
    lines = content.splitlines()
    count = len(lines)
    task_words = get_task_words(task)
    matches: set[int] = set()
    definition = re.compile(
        r"^\s*(?:async\s+)?(?:def|class|function|interface|type|enum|struct|trait)\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    symbols: set[str] = set()

    for index, line in enumerate(lines, start=1):
        lowered = line.lower()
        if any(word in lowered for word in task_words):
            matches.add(index)
            found = definition.match(line)
            if found:
                symbols.add(found.group(1))

    # References to matched definitions are useful caller/callee clues.
    if symbols:
        for index, line in enumerate(lines, start=1):
            if any(
                re.search(rf"\b{re.escape(symbol)}\b", line)
                for symbol in symbols
            ):
                matches.add(index)

    ranges: list[tuple[int, int]] = []
    import_end = 0
    for index, line in enumerate(lines[:200], start=1):
        stripped = line.lstrip()
        if stripped.startswith(
            ("import ", "from ", "#include", "using ")
        ):
            import_end = index
    if import_end:
        ranges.append((1, min(count, import_end + 8)))

    if not matches:
        matches.add(1)

    for line_number in matches:
        ranges.append((
            max(1, line_number - PATCH_EXCERPT_RADIUS),
            min(count, line_number + PATCH_EXCERPT_RADIUS),
        ))

    return _merge_line_ranges(ranges)


def build_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path] | None = None,
) -> str:
    if files is None:
        files = collect_relevant_files(repo, task)

    changed_files = get_changed_files(repo)
    explicit_files = set(
        _resolve_task_file_references(repo, task)
    )
    sections: list[str] = []

    for path in files:
        try:
            content, digest = (
                _read_current_text_and_hash(path)
            )
        except OSError:
            continue

        relative = path.relative_to(repo).as_posix()
        include_full = (
            len(content) <= PATCH_FULL_FILE_CHARS
            or path in changed_files
            or path in explicit_files
        )

        if include_full:
            section = (
                f"===== FULL FILE: {relative} =====\n"
                f"SHA-256: {digest}\n"
                f"Source line range: 1-{max(1, len(content.splitlines()))}\n\n"
                f"{content}\n"
            )
        else:
            lines = content.splitlines(keepends=True)
            excerpts: list[str] = []
            for start, end in _symbol_aware_ranges(content, task):
                excerpts.append(
                    f"----- EXCERPT {relative} source lines {start}-{end} -----\n"
                    + "".join(lines[start - 1:end])
                )
            section = (
                f"===== EXCERPTS FROM CURRENT FILE: {relative} =====\n"
                f"SHA-256 (complete source file): {digest}\n"
                + "\n".join(excerpts)
                + "\n"
            )

        sections.append(section)

    return "\n".join(sections) or "No relevant source files found."


def _current_file_hashes(
    files: list[Path],
) -> dict[Path, str | None]:
    hashes: dict[Path, str | None] = {}
    for path in files:
        try:
            hashes[path] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        except OSError:
            hashes[path] = None
    return hashes


def _build_stable_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path],
) -> str:
    """Read source from disk and retry if it changes during context capture."""
    for _attempt in range(CONTEXT_CAPTURE_RETRIES):
        before = _current_file_hashes(files)
        source_context = build_patch_source_context(
            repo,
            task,
            files=files,
        )
        after = _current_file_hashes(files)
        if before == after:
            return source_context

    raise RuntimeError(
        "Source files changed while ChatCode was building context. "
        "Run chatcode context again."
    )


def build_patch_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> Path:
    output_file = get_repo_workspace(repo) / "UPLOAD_TO_CHATGPT.md"
    repo_display = repo.resolve().as_posix()
    status = build_safe_status(repo).replace("\\", "/")
    files = collect_relevant_files(
        repo,
        task,
        index_progress=index_progress,
    )
    files = _ensure_explicit_task_files(
        repo,
        task,
        files,
    )
    source_context = _build_stable_patch_source_context(
        repo,
        task,
        files,
    )
    selected = [
        path.relative_to(repo).as_posix()
        for path in files
    ]

    parts = [
        "# ChatCode Patch Context",
        "",
        PATCH_CONTEXT_HEADER,
        "",
        "## ChatGPT instructions",
        UPLOAD_INSTRUCTIONS,
        "",
        "## Task (verbatim)",
        task,
        "",
        "## Repository root",
        repo_display,
        "",
        "## Current branch",
        get_branch(repo),
        "",
        "## Git status --short",
        status or "Working tree clean",
        "",
        "## Selected relevant files",
        "\n".join(f"- {path}" for path in selected) or "None",
        "",
        "## Exact current working-tree contents",
        source_context,
        "",
        "## Response instructions",
        PATCH_RESPONSE_INSTRUCTIONS,
        "All patch paths must use forward slashes (`/`).",
        "Patch only against the current contents above; do not reconstruct files from Git history or a separate diff.",
        "",
    ]
    output_file.write_text(
        "\n".join(parts),
        encoding="utf-8",
        newline="\n",
    )
    return output_file


def build_safe_diff(
    repo: Path,
    staged: bool,
) -> str:
    if staged:
        changed_paths = (
            get_staged_changed_paths(repo)
        )
    else:
        changed_paths = (
            get_unstaged_changed_paths(repo)
        )

    safe_paths: list[str] = []
    omitted_count = 0

    for raw_path in changed_paths:
        path = repo / raw_path

        if (
            is_ignored(path, repo)
            or _is_context_internal_artifact(path, repo)
        ):
            omitted_count += 1
            continue

        safe_paths.append(raw_path)

    if staged:
        diff = get_staged_diff(
            repo,
            safe_paths,
        )
    else:
        diff = get_unstaged_diff(
            repo,
            safe_paths,
        )

    if diff:
        diff = redact_sensitive_text(
            diff
        )

    if omitted_count:
        note = (
            "[ChatCode omitted "
            f"{omitted_count} sensitive or "
            "ignored file(s) from this diff.]"
        )

        if diff:
            diff = (
                f"{diff}\n\n{note}"
            )
        else:
            diff = note

    return diff


def build_failed_test_context(
    repo: Path,
) -> str:
    test_results_file = (
        get_test_results_file(repo)
    )

    if not test_results_file.exists():
        return ""

    try:
        content = test_results_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return ""

    if "Status: FAILED" not in content:
        return ""

    content = redact_sensitive_text(
        content
    )

    if len(content) > MAX_TEST_RESULT_CHARS:
        content = (
            "[Earlier test output truncated "
            "by ChatCode]\n\n"
            + content[-MAX_TEST_RESULT_CHARS:]
        )

    return content


def build_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> Path:
    output_dir = get_repo_workspace(repo)

    output_file = (
        output_dir / "UPLOAD_TO_CHATGPT.md"
    )

    branch = get_branch(repo)
    status = build_safe_status(repo)

    unstaged = build_safe_diff(
        repo,
        staged=False,
    )

    staged = build_safe_diff(
        repo,
        staged=True,
    )

    tree = build_tree(repo)

    files = collect_relevant_files(
        repo,
        task,
        index_progress=index_progress,
    )
    files = _ensure_explicit_task_files(
        repo,
        task,
        files,
    )
    source_context = _build_stable_patch_source_context(
        repo,
        task,
        files,
    )

    failed_tests = (
        build_failed_test_context(repo)
    )

    parts = [
        "# ChatCode Context",
        "",
        "## ChatGPT instructions",
        UPLOAD_INSTRUCTIONS,
        "",
        "## Task",
        task,
        "",
        "## Response instructions",
        PATCH_RESPONSE_INSTRUCTIONS,
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
        "\n".join(
            f"- {path.relative_to(repo).as_posix()}"
            for path in files
        ) or "None",
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

    output_file.write_text(
        content,
        encoding="utf-8",
    )

    return output_file
