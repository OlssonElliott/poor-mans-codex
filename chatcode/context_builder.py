from __future__ import annotations

import re
from pathlib import Path

from .git_utils import (
    get_branch,
    get_staged_changed_paths,
    get_staged_diff,
    get_status,
    get_unstaged_changed_paths,
    get_unstaged_diff,
)
from .workspace import (
    get_repo_workspace,
    get_test_results_file,
)


IGNORED_DIRS = {
    ".git",
    ".chatcode",
    ".vite",
    "node_modules",
    "vendor",
    "target",
    "dist",
    "build",
    "tmp",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    "venv",
    ".aws",
    ".ssh",
}


SECRET_FILENAMES = {
    "credentials.json",
    "secrets.json",
    "service-account.json",
    "serviceaccount.json",
    ".npmrc",
    ".pypirc",
}


SECRET_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
}


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

PATCH_RESPONSE_INSTRUCTIONS = """\
When this task requires code changes:

1. Return a valid unified diff that can be applied with `git apply`.
2. All file paths must be relative to the repository root.
3. Do not use absolute paths.
4. Do not include Markdown code fences.
5. Do not include explanations before or after the patch.
6. Include all required changes in a single patch.
7. Preserve unrelated existing changes.
8. Do not modify `.git` or sensitive files such as `.env`, credentials, private keys or secrets.
9. For new files, use `/dev/null` as the old file.
10. For deleted files, use `/dev/null` as the new file.

The response should be directly saveable as `incoming.diff` and applicable with:

chatcode apply
"""

UPLOAD_INSTRUCTIONS = """\
This file is an execution request.

When this file is uploaded to ChatGPT, immediately perform the task described
under "## Task".

Do not ask the user what they want you to do.
Do not ask for confirmation.
Treat the upload of this file itself as the user's request to perform the task.

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


def is_secret(path: Path) -> bool:
    name = path.name.lower()

    if name == ".env":
        return True

    if name.startswith(".env."):
        return True

    if name in SECRET_FILENAMES:
        return True

    if path.suffix.lower() in SECRET_SUFFIXES:
        return True

    return False


def is_ignored(
    path: Path,
    repo: Path,
) -> bool:
    try:
        relative = path.relative_to(repo)
    except ValueError:
        return True

    for part in relative.parts:
        if part.lower() in IGNORED_DIRS:
            return True

    return is_secret(path)


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

            if is_ignored(path, repo):
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


def build_tree(
    repo: Path,
    max_files: int = 500,
) -> str:
    lines: list[str] = []
    file_count = 0

    for path in sorted(repo.rglob("*")):
        if is_ignored(path, repo):
            continue

        if path.is_dir():
            continue

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

    for raw_path in paths:
        path = (repo / raw_path).resolve()

        if not path.exists():
            continue

        if not path.is_file():
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

        content = content[:100_000]

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
) -> list[Path]:
    task_words = get_task_words(task)
    changed_files = get_changed_files(repo)

    candidates: list[
        tuple[int, Path]
    ] = []

    for path in repo.rglob("*"):
        if not path.is_file():
            continue

        if is_ignored(path, repo):
            continue

        if not is_source_file(path):
            continue

        score = score_file(
            path,
            repo,
            task_words,
            changed_files,
        )

        if score > 0:
            candidates.append(
                (score, path)
            )

    candidates.sort(
        key=lambda item: (
            -item[0],
            str(item[1]).lower(),
        )
    )

    return [
        path
        for _, path
        in candidates[:MAX_FILES]
    ]


def build_source_context(
    repo: Path,
    task: str,
) -> str:
    files = collect_relevant_files(
        repo,
        task,
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

        if is_ignored(path, repo):
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

    source_context = (
        build_source_context(
            repo,
            task,
        )
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