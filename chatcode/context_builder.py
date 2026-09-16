from __future__ import annotations

import ast
import hashlib
import os
import re
import uuid
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
from .context_state import save_context_state
from .config import get_boolean_setting
from .indexing.index_manager import IndexProgress, update_project_map
from .indexing.project_graph import load_map, save_map
from .retrieval.graph_retriever import retrieve_files
from .retrieval.hybrid_retriever import (
    QwenCompletenessChecker,
    QwenTaskHintAnalyzer,
    RetrievalResult,
    expand_candidates,
    implementation_closure,
    resolve_explicit_targets,
    resolve_semantic_hints,
    test_callsite_closure,
)
from .workspace import (
    atomic_write_text,
    get_repo_workspace,
    get_test_results_file,
)
from . import workspace


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
PATCH_FULL_FILE_CHARS = 20_000
PATCH_DIRTY_FULL_FILE_CHARS = 48_000
PATCH_EXCERPT_RADIUS = 32
PATCH_CONTEXT_BUDGET_CHARS = 90_000
PATCH_DEPENDENCY_RESERVE_RATIO = 0.25
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
12. For a change inside an existing file, include at least three unchanged
    context lines when those lines exist. Never return a one-line hunk based
    only on a guessed line number.

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

Do not modify an existing file unless the current source for the affected area
appears in a `FULL FILE`, `SYMBOL CONTEXT`, or `EXCERPT` block in this context. A file listed as
selected but marked source-unavailable is reference-only: do not guess its
contents or generate a patch for it.

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
        # Resolve through the module so the workspace provider remains the
        # single authority (and can be replaced by an embedding application).
        repo_workspace = workspace.get_repo_workspace(repo).resolve()
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


def _ambiguous_task_file_paths(repo: Path, task: str) -> set[Path]:
    """Find bare filename references that cannot be resolved safely."""
    bare_filenames = [
        token.replace("\\", "/")
        for token in _task_file_tokens(task)
        if "/" not in token.replace("\\", "/") and "." in token
    ]
    if not bare_filenames:
        return set()
    candidates = [path.resolve() for path in _iter_context_files(repo)]
    ambiguous: set[Path] = set()
    for normalized in bare_filenames:
        matches = [
            path for path in candidates
            if path.name.casefold() == normalized.casefold()
        ]
        if len(matches) > 1:
            ambiguous.update(matches)
    return ambiguous


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
    include_target_symbols: bool = False,
) -> list[Path] | tuple[list[Path], dict[Path, list[str]]]:
    # Synchronizing here also catches edits made manually since the previous
    # ChatCode invocation. Indexing is an enhancement, so a damaged/unwritable
    # cache must not prevent the established retrieval path from working.
    graph_files: list[Path] = []
    explicit_targets: list[Path] = []
    explicit_target_reasons: dict[Path, list[str]] = {}
    semantic_targets: list[Path] = []
    semantic_target_reasons: dict[Path, list[str]] = {}
    semantic_hints: list[str] = []
    semantic_hint_status = "not_run"
    task_hint_analyzer: QwenTaskHintAnalyzer | None = None
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
        explicit_target_result = resolve_explicit_targets(repo, task)
        explicit_targets = explicit_target_result.files
        explicit_target_reasons = explicit_target_result.reasons
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
        if effective_mode == "ai":
            task_hint_analyzer = QwenTaskHintAnalyzer()
            semantic_hints = task_hint_analyzer.hints(repo, task, graph_files)
            semantic_hint_status = task_hint_analyzer.last_status
            semantic_target_result = resolve_semantic_hints(repo, semantic_hints)
            semantic_targets = semantic_target_result.files
            semantic_target_reasons = semantic_target_result.reasons
    except Exception:
        graph_files = []

    changed_files = get_changed_files(repo)
    explicit_files = _resolve_task_file_references(repo, task)
    ambiguous_files = _ambiguous_task_file_paths(repo, task)

    # Graph retrieval is the semantic/Qwen seed in AI mode and a useful
    # deterministic seed in static mode.  In both cases it is deliberately
    # expanded by the same structural ranker rather than treated as authority.
    seed_files = list(dict.fromkeys([*explicit_targets, *semantic_targets, *graph_files]))
    hybrid = expand_candidates(repo, task, seed_files, limit=MAX_FILES)
    for path, reasons in explicit_target_reasons.items():
        hybrid.reasons[path] = list(dict.fromkeys([
            *reasons,
            *hybrid.reasons.get(path, []),
        ]))
    for path, reasons in semantic_target_reasons.items():
        hybrid.reasons[path] = list(dict.fromkeys([
            *reasons,
            *hybrid.reasons.get(path, []),
        ]))
    closure_focus = {
        symbol
        for result in (locals().get("explicit_target_result"), locals().get("semantic_target_result"))
        if result is not None
        for symbols in result.required_symbols.values()
        for symbol in symbols
    }
    closure = (
        implementation_closure(repo, task, [*hybrid.files, *graph_files], closure_focus)
        if closure_focus else RetrievalResult([])
    )
    callsite_closure = test_callsite_closure(repo, task, [*hybrid.files, *graph_files])
    for path in callsite_closure.files:
        if path not in hybrid.files:
            hybrid.files.append(path)
        hybrid.reasons[path] = list(dict.fromkeys([
            *callsite_closure.reasons.get(path, []), *hybrid.reasons.get(path, []),
        ]))
    for path in closure.files:
        if path not in hybrid.files:
            hybrid.files.append(path)
        hybrid.reasons[path] = list(dict.fromkeys([
            *closure.reasons.get(path, []), *hybrid.reasons.get(path, []),
        ]))
    completeness_files: list[Path] = []
    if effective_mode == "ai":
        completeness = QwenCompletenessChecker().check(repo, task, hybrid)
        for path in completeness.files:
            completeness_files.append(path)
            label = completeness.reasons[path]
            if label not in hybrid.reasons.setdefault(path, []):
                hybrid.reasons[path].append(label)
            if path not in hybrid.files:
                hybrid.files.append(path)

    changed_source_files = sorted(
        path for path in changed_files
        if path.is_file() and is_source_file(path)
        and not _is_context_internal_artifact(path, repo)
    )
    selected: list[Path] = [path for path in changed_source_files if path not in ambiguous_files]
    for path in explicit_files:
        if path not in selected:
            selected.append(path)
    for path in callsite_closure.files:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in explicit_targets:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in semantic_targets:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    for path in closure.files:
        if path not in selected and path not in ambiguous_files:
            selected.append(path)
    selection_limit = max(MAX_FILES, len(selected))
    # Completeness additions have been specifically confirmed after the first
    # ranking pass, so reserve their place ahead of lower-ranked seed results.
    for path in [*completeness_files, *hybrid.files]:
        if path.is_file() and is_source_file(path) and path not in ambiguous_files and path not in selected:
            selected.append(path)
        if len(selected) >= selection_limit:
            break
    target_symbols: dict[Path, list[str]] = {}
    for source in (
        value for value in (
            locals().get("explicit_target_result"),
            locals().get("semantic_target_result"),
            closure,
            callsite_closure,
        ) if value is not None
    ):
        for path, symbols in source.required_symbols.items():
            if path in selected:
                target_symbols[path] = list(dict.fromkeys(symbols))
    if get_boolean_setting("CHATCODE_RETRIEVAL_DEBUG"):
        print("Retrieval diagnostics:", file=os.sys.stderr)
        print(
            "Runtime modules: " + __file__ + " | " + QwenTaskHintAnalyzer.__module__,
            file=os.sys.stderr,
        )
        if task_hint_analyzer is not None:
            print(f"Task-hint vocabulary: count {len(task_hint_analyzer.last_vocabulary)}", file=os.sys.stderr)
            print("Vocabulary sample: " + ", ".join(task_hint_analyzer.last_vocabulary[:30]), file=os.sys.stderr)
            print("Raw Qwen task-hint response: " + (task_hint_analyzer.last_raw_response or "<none>"), file=os.sys.stderr)
            print("Parsed symbol hints: " + (", ".join(task_hint_analyzer.last_parsed_hints) or "none"), file=os.sys.stderr)
            print("Normalized symbol hints: " + (", ".join(task_hint_analyzer.last_normalized_hints) or "none"), file=os.sys.stderr)
            if task_hint_analyzer.last_rejections:
                print("Rejected symbol hints: " + "; ".join(task_hint_analyzer.last_rejections), file=os.sys.stderr)
        print(
            "Qwen symbol hints (" + semantic_hint_status + "): "
            + (", ".join(semantic_hints) if semantic_hints else "none"),
            file=os.sys.stderr,
        )
        if semantic_target_reasons:
            print("Resolved semantic symbols:", file=os.sys.stderr)
            for path in sorted(semantic_target_reasons, key=lambda value: str(value).lower()):
                print(f"- {path.relative_to(repo)}: {', '.join(semantic_target_reasons[path])}", file=os.sys.stderr)
        if closure.reasons:
            print("Implementation closure additions:", file=os.sys.stderr)
            for path in sorted(closure.reasons, key=lambda value: str(value).lower()):
                print(f"- {path.relative_to(repo)}: {', '.join(closure.reasons[path])}", file=os.sys.stderr)
        if callsite_closure.reasons:
            print("Deterministic call-site roots and expansion:", file=os.sys.stderr)
            for diagnostic in callsite_closure.diagnostics:
                print(f"- {diagnostic}", file=os.sys.stderr)
            for path in sorted(callsite_closure.reasons, key=lambda value: str(value).lower()):
                print(f"- {path.relative_to(repo)}: {', '.join(callsite_closure.reasons[path])}", file=os.sys.stderr)
        for path, symbols in target_symbols.items():
            print(f"Required symbols: {path.relative_to(repo)}::{', '.join(symbols)}", file=os.sys.stderr)
        for path in selected:
            labels = ", ".join(hybrid.reasons.get(path, ["explicit or changed file"]))
            print(f"- {path.relative_to(repo)}: {labels}", file=os.sys.stderr)
    return (selected, target_symbols) if include_target_symbols else selected


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
            content, _digest = _read_current_text_and_hash(path)
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
    """Return one authoritative working-tree snapshot of ``path``.

    Context rendering must never consult Git blobs or the project-map cache for
    source text.  Keeping the decode and digest tied to the same byte read also
    prevents a hash from describing a different revision than the text below it.
    """
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


def _configured_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _context_budget_chars() -> int:
    return _configured_positive_int(
        "CHATCODE_CONTEXT_BUDGET_CHARS",
        PATCH_CONTEXT_BUDGET_CHARS,
    )


def _symbol_aware_ranges(
    content: str,
    task: str,
    *,
    test_file: bool = False,
) -> list[tuple[int, int]]:
    lines = content.splitlines()
    count = len(lines)
    task_words = get_task_words(task)
    matches: set[int] = set()
    definition = re.compile(
        r"^\s*(?:async\s+)?(?:def|class|function|interface|type|enum|struct|trait)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)"
    )
    symbols: set[str] = set()

    for index, line in enumerate(lines, start=1):
        lowered = line.lower()
        if any(word in lowered for word in task_words):
            matches.add(index)
            found = definition.match(line)
            if found:
                symbols.add(found.group(1))

    if test_file and task_words:
        for index, line in enumerate(lines, start=1):
            lowered = line.lower()
            if ("test" in lowered or "spec" in lowered) and any(
                word in lowered for word in task_words
            ):
                matches.add(index)

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
        if stripped.startswith(("import ", "from ", "#include", "using ")):
            import_end = index
    if import_end:
        ranges.append((1, min(count, import_end + 8)))

    if not matches:
        fallback_end = min(count, PATCH_EXCERPT_RADIUS * 2 + 1)
        return _merge_line_ranges(
            [*ranges, (1, fallback_end)] if count else ranges
        )

    for line_number in sorted(matches):
        ranges.append((
            max(1, line_number - PATCH_EXCERPT_RADIUS),
            min(count, line_number + PATCH_EXCERPT_RADIUS),
        ))

    return _merge_line_ranges(ranges)


def _render_patch_file_context(
    repo: Path,
    path: Path,
    task: str,
    changed_files: set[Path],
    explicit_files: set[Path],
    max_chars: int | None = None,
    required_symbols: list[str] | None = None,
) -> str:
    content, digest = _read_current_text_and_hash(path)
    relative = path.relative_to(repo).as_posix()
    line_count = max(1, len(content.splitlines()))
    include_full = (
        len(content) <= PATCH_FULL_FILE_CHARS
        or (
            path in explicit_files
            and len(content) <= PATCH_DIRTY_FULL_FILE_CHARS
        )
    )
    if include_full:
        diagnostic = ""
        if os.getenv("CHATCODE_DEBUG_CONTEXT", "").strip():
            dirty = "yes" if path in changed_files else "no"
            diagnostic = f"Source: working-tree; dirty={dirty}; sha256={digest}\n"
        full_section = (
            f"===== FULL FILE: {relative} =====\n"
            f"SHA-256: {digest}\n"
            f"{diagnostic}"
            f"Source line range: 1-{line_count}\n\n"
            f"{content}\n"
        )
        if max_chars is None or len(full_section) <= max_chars:
            return full_section

    lines = content.splitlines(keepends=True)
    excerpts = []
    test_file = "test" in path.stem.casefold() or "spec" in path.stem.casefold()
    target_ranges = _fresh_symbol_ranges(content, path, required_symbols or [])
    ranges = target_ranges or _symbol_aware_ranges(content, task, test_file=test_file)
    for start, end in ranges:
        excerpts.append(
            f"----- EXCERPT {relative} source lines {start}-{end} -----\n"
            + "".join(lines[start - 1:end])
        )
    section = (
        f"===== SYMBOL CONTEXT: {relative} =====\n"
        f"SHA-256 (complete source file): {digest}\n"
        + "\n".join(excerpts)
        + "\n"
    )
    if max_chars is None or len(section) <= max_chars:
        return section
    if target_ranges:
        # Keep each requested node atomic, but do not make every requested
        # handler an all-or-nothing bundle. A large sibling handler must not
        # prevent smaller, independently resolved targets from materializing.
        header = f"===== SELECTED SOURCE: {relative} =====\n"
        kept: list[str] = []
        used = len(header)
        for start, end in target_ranges:
            excerpt = (
                f"----- EXCERPT {relative} source lines {start}-{end} -----\n"
                + "".join(lines[start - 1:end])
                + "\n"
            )
            if used + len(excerpt) <= max_chars:
                kept.append(excerpt)
                used += len(excerpt)
        return header + "".join(kept) if kept else ""

    # Every primary selection gets a fresh, exact (though possibly shorter)
    # excerpt before supporting dependencies consume the budget. Never splice
    # arbitrary characters: preserve complete current working-tree lines.
    header = (
        f"===== EXCERPT: {relative} =====\n"
        f"SHA-256 (complete source file): {digest}\n"
    )
    available = max_chars - len(header) - 1
    if available <= 0:
        return ""
    excerpt_lines: list[str] = []
    used = 0
    for start, end in _symbol_aware_ranges(content, task, test_file=test_file):
        marker = f"----- source lines "
        if used + len(marker) > available:
            break
        captured: list[str] = []
        actual_end = start - 1
        for line_number, line in enumerate(lines[start - 1:end], start=start):
            # Reserve the final, truthful range marker before accepting a line.
            possible_marker = f"----- source lines {start}-{line_number} -----\n"
            if used + len(possible_marker) + sum(map(len, captured)) + len(line) > available:
                break
            captured.append(line)
            actual_end = line_number
        if actual_end < start:
            break
        final_marker = f"----- source lines {start}-{actual_end} -----\n"
        excerpt_lines.append(final_marker)
        excerpt_lines.extend(captured)
        used += len(final_marker) + sum(map(len, captured))
        if used >= available or actual_end < end:
            break
    return header + "".join(excerpt_lines) + "\n" if excerpt_lines else ""


def _fresh_symbol_ranges(content: str, path: Path, symbols: list[str]) -> list[tuple[int, int]]:
    """Locate requested Python definitions in fresh source, never cached text."""
    if path.suffix.lower() != ".py" or not symbols:
        return []
    wanted = set(symbols)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    ranges: list[tuple[int, int, bool, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in wanted or not hasattr(node, "end_lineno"):
            continue
        start = min([node.lineno, *(item.lineno for item in node.decorator_list)] if node.decorator_list else [node.lineno])
        ranges.append((start, node.end_lineno, isinstance(node, ast.ClassDef), node.name))
    # When an explicit class and concrete methods inside it are both targets,
    # the methods are the precise patch surface. Emitting the enclosing class
    # would turn distant methods into one enormous range and waste the budget.
    filtered = [
        (start, end, name)
        for start, end, is_class, name in ranges
        if not is_class or not any(
            not other_is_class and start <= other_start and other_end <= end
            for other_start, other_end, other_is_class, _other_name in ranges
        )
    ]
    order = {name: index for index, name in enumerate(symbols)}
    return [
        (start, end)
        for start, end, _name in sorted(
            filtered, key=lambda item: (order.get(item[2], len(order)), item[0])
        )
    ]


def _fresh_symbol_node(path: Path, symbol: str) -> tuple[str, list[str]] | None:
    """Return one complete current Python definition and its method calls."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content)
    except (OSError, SyntaxError):
        return None
    lines = content.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != symbol or not hasattr(node, "end_lineno"):
            continue
        start = min(
            [node.lineno, *(item.lineno for item in node.decorator_list)]
            if node.decorator_list else [node.lineno]
        )
        calls = [
            call.func.attr
            for call in sorted(
                (item for item in ast.walk(node) if isinstance(item, ast.Call)),
                key=lambda item: (item.lineno, item.col_offset),
            )
            if isinstance(call.func, ast.Attribute)
        ]
        return "".join(lines[start - 1:node.end_lineno]), list(dict.fromkeys(calls))
    return None


def _materialization_targets(
    repo: Path, target_symbols: dict[Path, list[str]], max_inherited: int = 64,
) -> list[tuple[Path, str, str]]:
    """Order patch targets first and inherit priority through two direct calls."""
    index = load_map(repo).get("files", {})
    owners: dict[str, list[tuple[Path, str]]] = {}
    for relative, entry in index.items():
        for symbol in entry.get("symbols", []):
            if not isinstance(symbol, dict) or not symbol.get("name"):
                continue
            owners.setdefault(str(symbol["name"]).casefold(), []).append((
                repo / relative,
                str(symbol.get("definition_name") or symbol["name"]),
            ))
    # Large or dirty files can contain fresh methods not represented by the
    # cached symbol list. Required files are already in scope, so supplement
    # owner lookup from their current AST without broadening retrieval.
    for path in target_symbols:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            owner = (path, node.name)
            bucket = owners.setdefault(node.name.casefold(), [])
            if owner not in bucket:
                bucket.append(owner)

    original = [(path, symbol) for path, symbols in target_symbols.items() for symbol in symbols]
    command_roots = [
        item for item in original
        if "commands" in {part.casefold() for part in item[0].relative_to(repo).parts}
    ]
    ordered: list[tuple[Path, str, str]] = []
    seen: set[tuple[Path, str]] = set()
    inherited = 0

    def visit(path: Path, symbol: str, depth: int, priority: str) -> None:
        nonlocal inherited
        key = (path, symbol)
        if key in seen:
            return
        seen.add(key)
        ordered.append((path, symbol, priority))
        if depth >= 2 or inherited >= max_inherited:
            return
        node = _fresh_symbol_node(path, symbol)
        if node is None:
            return
        for called in node[1]:
            for owner, definition in sorted(owners.get(called.casefold(), []), key=lambda item: str(item[0]).lower()):
                if (owner, definition) == key or inherited >= max_inherited:
                    continue
                if (owner, definition) in seen:
                    continue
                inherited += 1
                visit(owner, definition, depth + 1, "direct implementation dependency")

    for path, symbol in command_roots:
        inherited = 0
        visit(path, symbol, 0, "patch target")
    for path, symbol in original:
        visit(path, symbol, 0, "required implementation")
    return ordered


def _render_required_symbols(
    repo: Path,
    targets: list[tuple[Path, str, str]],
    budget: int,
) -> tuple[str, dict[tuple[Path, str], tuple[str, str]]]:
    sections: list[str] = []
    states: dict[tuple[Path, str], tuple[str, str]] = {}
    used = 0
    for path, symbol, priority in targets:
        relative = path.relative_to(repo).as_posix()
        node = _fresh_symbol_node(path, symbol)
        if node is None:
            reason = "definition not found in current working tree"
            states[(path, symbol)] = ("unavailable", reason)
            sections.append(
                f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
                f"[reason: {reason}; do not patch] =====\n"
            )
            continue
        section = (
            f"===== REQUIRED SOURCE: {relative}::{symbol} [{priority}] =====\n"
            f"{node[0]}\n"
        )
        if used + len(section) <= budget:
            sections.append(section)
            used += len(section)
            states[(path, symbol)] = ("rendered", priority)
        else:
            states[(path, symbol)] = ("unavailable", "budget")
            sections.append(
                f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
                "[reason: budget; do not patch] =====\n"
            )
    return "\n".join(sections), states


def _enforce_materialization_invariant(
    repo: Path,
    rendered: str,
    targets: list[tuple[Path, str, str]],
) -> str:
    """Ensure every required symbol is complete or explicitly non-patchable."""
    repairs: list[str] = []
    for path, symbol, _priority in targets:
        relative = path.relative_to(repo).as_posix()
        unavailable = f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
        if unavailable in rendered:
            continue
        node = _fresh_symbol_node(path, symbol)
        header = f"===== REQUIRED SOURCE: {relative}::{symbol} "
        if node is not None and header in rendered and node[0] in rendered:
            continue
        repairs.append(
            f"===== REQUIRED SOURCE UNAVAILABLE: {relative}::{symbol} "
            "[reason: post-render validation; do not patch] =====\n"
        )
    return rendered + ("\n" + "\n".join(repairs) if repairs else "")


def _dependency_paths(repo: Path, files: list[Path]) -> set[Path]:
    try:
        graph = load_map(repo)
    except Exception:
        return set()
    entries = graph.get("files", {})
    if not isinstance(entries, dict):
        return set()
    selected = {
        path.relative_to(repo).as_posix()
        for path in files
        if path.is_file()
    }
    dependencies: set[Path] = set()
    for relative in selected:
        entry = entries.get(relative, {})
        if not isinstance(entry, dict):
            continue
        for raw_dependency in entry.get("dependencies", []):
            if not isinstance(raw_dependency, str):
                continue
            candidate = (repo / raw_dependency).resolve()
            if candidate.is_file():
                dependencies.add(candidate)
    return dependencies


def build_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path] | None = None,
    target_symbols: dict[Path, list[str]] | None = None,
) -> str:
    if files is None:
        files = collect_relevant_files(repo, task)

    changed_files = get_changed_files(repo)
    explicit_files = set(_resolve_task_file_references(repo, task))
    target_metadata = target_symbols or resolve_explicit_targets(repo, task).required_symbols
    materialization_targets = _materialization_targets(repo, target_metadata)
    dependencies = _dependency_paths(repo, files)
    primary = list(dict.fromkeys([
        *[path for path in files if path in explicit_files],
        *[path for path in files if path in changed_files],
        *files,
    ]))
    supporting = [path for path in sorted(dependencies) if path not in primary]

    budget = _context_budget_chars()
    required_context, materialization_states = _render_required_symbols(
        repo, materialization_targets, budget
    )
    required_rendered_chars = sum(
        len(section) for section in required_context.splitlines(keepends=True)
        if "REQUIRED SOURCE UNAVAILABLE:" not in section
    )
    remaining_budget = max(0, budget - required_rendered_chars)
    dependency_reserve = int(remaining_budget * PATCH_DEPENDENCY_RESERVE_RATIO)
    primary_budget = max(0, remaining_budget - dependency_reserve)
    sections: list[str] = [required_context] if required_context else []
    used = 0

    # Selected files are possible patch targets. Allocate their share first so
    # a broad dependency, README, or an early oversized selection cannot make
    # a later selected test silently vanish from the authoritative source.
    for position, path in enumerate(primary):
        if path in target_metadata:
            continue
        try:
            remaining = len(primary) - position
            allowance = max(1, (primary_budget - used) // remaining)
            section = _render_patch_file_context(
                repo,
                path,
                task,
                changed_files,
                explicit_files,
                max_chars=allowance,
                required_symbols=target_metadata.get(path),
            )
        except OSError:
            continue
        if not section or used + len(section) > primary_budget:
            continue
        sections.append(section)
        used += len(section)

    for path in supporting:
        if path in target_metadata:
            continue
        try:
            section = _render_patch_file_context(repo, path, task, changed_files, explicit_files)
        except OSError:
            continue
        if used + len(section) > budget:
            continue
        sections.append(section)
        used += len(section)

    rendered = "\n".join(sections) or "No relevant source files found."
    rendered = _enforce_materialization_invariant(repo, rendered, materialization_targets)
    if get_boolean_setting("CHATCODE_RETRIEVAL_DEBUG") and materialization_states:
        print("Final materialization:", file=os.sys.stderr)
        priorities = {(path, symbol): priority for path, symbol, priority in materialization_targets}
        diagnostic_items = list(materialization_states.items())
        for (path, symbol), (state, detail) in diagnostic_items[:40]:
            print(f"{path.relative_to(repo)}::{symbol}", file=os.sys.stderr)
            print(f"- priority: {priorities[(path, symbol)]}", file=os.sys.stderr)
            print(f"- rendered: {'yes' if state == 'rendered' else 'no'}", file=os.sys.stderr)
            if state != "rendered":
                print(f"- reason: {detail}", file=os.sys.stderr)
                print("- status: non-patchable", file=os.sys.stderr)
        if len(diagnostic_items) > 40:
            print(
                f"- {len(diagnostic_items) - 40} lower-priority symbol states omitted",
                file=os.sys.stderr,
            )
    return rendered


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


def _captured_source_paths(repo: Path, files: list[Path]) -> list[Path]:
    """Return every working-tree file that may be emitted in source context.

    Dependencies are selected from cached graph metadata, but their contents
    are loaded from disk.  They must therefore participate in the same
    before/after integrity check as primary files.
    """
    return list(dict.fromkeys([
        *files,
        *sorted(_dependency_paths(repo, files)),
    ]))


def _build_stable_patch_source_context(
    repo: Path,
    task: str,
    files: list[Path],
    target_symbols: dict[Path, list[str]] | None = None,
) -> tuple[str, dict[str, str]]:
    """Read source from disk and retry if it changes during context capture."""
    for _attempt in range(CONTEXT_CAPTURE_RETRIES):
        expanded_targets = _materialization_targets(repo, target_symbols or {})
        captured_paths = list(dict.fromkeys([
            *_captured_source_paths(repo, files),
            *(path for path, _symbol, _priority in expanded_targets),
        ]))
        before = _current_file_hashes(captured_paths)
        expanded_metadata: dict[Path, list[str]] = {
            path: list(symbols) for path, symbols in (target_symbols or {}).items()
        }
        for owner, symbol, _priority in expanded_targets:
            expanded_metadata.setdefault(owner, [])
            if symbol not in expanded_metadata[owner]:
                expanded_metadata[owner].append(symbol)
        source_context = build_patch_source_context(
            repo,
            task,
            files=files,
            target_symbols=expanded_metadata,
        )
        # Re-resolve dependencies in case graph metadata was refreshed while
        # rendering, then verify every path that could have been emitted.
        captured_paths = list(dict.fromkeys([
            *captured_paths,
            *_captured_source_paths(repo, files),
        ]))
        after = _current_file_hashes(captured_paths)
        if before == after:
            return source_context, {
                path.relative_to(repo).as_posix(): digest
                for path, digest in after.items()
                if digest is not None
            }

    raise RuntimeError(
        "Source files changed while ChatCode was building context. "
        "Run chatcode context again."
    )


def _format_selected_files(files: list[Path], repo: Path, source_context: str) -> str:
    """Make exceptional non-materialization explicit and non-patchable."""
    materialized = set(re.findall(
        r"^=====(?: FULL FILE| SYMBOL CONTEXT| EXCERPT):? ([^=\n]+?) =====$",
        source_context,
        re.MULTILINE,
    ))
    materialized.update(re.findall(
        r"^===== REQUIRED SOURCE: ([^:\n]+?)::[^\n]+ =====$",
        source_context,
        re.MULTILINE,
    ))
    lines = []
    for path in files:
        relative = path.relative_to(repo).as_posix()
        suffix = "" if relative in materialized else " [source unavailable; do not patch]"
        lines.append(f"- {relative}{suffix}")
    return "\n".join(lines) or "None"


def build_patch_context(
    repo: Path,
    task: str,
    index_progress: Callable[[IndexProgress], None] | None = None,
) -> Path:
    output_file = get_repo_workspace(repo) / "UPLOAD_TO_CHATGPT.md"
    repo_display = repo.resolve().as_posix()
    status = build_safe_status(repo).replace("\\", "/")
    retrieved = collect_relevant_files(
        repo,
        task,
        index_progress=index_progress,
        include_target_symbols=True,
    )
    files, target_symbols = retrieved if isinstance(retrieved, tuple) else (retrieved, {})
    files = _ensure_explicit_task_files(
        repo,
        task,
        files,
    )
    source_context, source_hashes = _build_stable_patch_source_context(
        repo,
        task,
        files,
        target_symbols,
    )
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
        _format_selected_files(files, repo, source_context),
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

    retrieved = collect_relevant_files(
        repo,
        task,
        index_progress=index_progress,
        include_target_symbols=True,
    )
    files, target_symbols = retrieved if isinstance(retrieved, tuple) else (retrieved, {})
    files = _ensure_explicit_task_files(
        repo,
        task,
        files,
    )
    failed_tests = (
        build_failed_test_context(repo)
    )

    # Capture source last: it is the only material sent as an authoritative
    # patch target, and this keeps its final hash recheck immediately before
    # the generated context is finalized.
    source_context, source_hashes = _build_stable_patch_source_context(
        repo,
        task,
        files,
        target_symbols,
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
        _format_selected_files(files, repo, source_context),
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
