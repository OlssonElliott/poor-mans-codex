from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .context_state import (
    _collect_file_hashes,
    get_active_repair_targets,
    get_context_task,
    get_context_kind,
    get_stale_context_reason,
    save_context_state,
)
from .git_utils import GitError, get_branch, get_status, run_git
from .history import (
    HistoryError,
    begin_history_entry,
    discard_pending_entry,
    finalize_history_entry,
    get_history_patch_file,
    get_latest_applied_entry,
    move_entry_to_undone,
    open_code_diff,
)
from .workspace import (
    atomic_write_text,
    get_default_patch_file,
    get_check_repair_context_file,
    get_existing_failure_context_file,
    get_followup_context_file,
    get_followup_state_file,
    get_repair_context_file,
    get_repo_workspace,
    get_test_results_dir,
    get_verified_baseline_cache_file,
)

VERIFIED_BASELINE_CACHE_VERSION = 1
VERIFIED_BASELINE_MAX_AGE_SECONDS = 30 * 60
from .unified_diff import (
    UnifiedDiffError,
    canonicalize_unified_diff,
    parse_unified_diff,
)
from .test_runner import TestResult


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


def _consume_cli_apply_flags() -> tuple[bool, bool]:
    """Consume --yes before the existing CLI parser sees the apply command."""
    is_apply = len(sys.argv) > 1 and sys.argv[1] == "apply"
    auto_yes = False
    if is_apply and "--yes" in sys.argv[2:]:
        sys.argv.remove("--yes")
        auto_yes = True
    return is_apply, auto_yes


_CLI_APPLY_INVOCATION, _CLI_APPLY_YES = _consume_cli_apply_flags()


@dataclass(frozen=True)
class ApplyResult:
    paths: set[str]
    history_entry: Path


@dataclass(frozen=True)
class PatchPreview:
    paths: set[str]
    patch_text: str


@dataclass(frozen=True)
class UndoResult:
    paths: set[str]
    history_entry: Path


@dataclass(frozen=True)
class TestValidation:
    baseline: object | None
    targeted: object | None
    full: object | None
    status: str
    new_failures: frozenset[str] = frozenset()
    existing_failures: frozenset[str] = frozenset()
    fixed_failures: frozenset[str] = frozenset()
    repair_targets: frozenset[str] = frozenset()
    remaining_repair_failures: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RepositorySnapshot:
    """The full working state that a speculative baseline represents."""
    branch: str
    file_hashes: tuple[tuple[str, str], ...]
    status: str
    staged_diff_sha256: str


def _capture_repository_snapshot(repo: Path) -> RepositorySnapshot:
    staged = run_git("diff", "--cached", "--binary", "--no-ext-diff", cwd=repo)
    return RepositorySnapshot(
        branch=get_branch(repo),
        file_hashes=tuple(sorted(_collect_file_hashes(repo).items())),
        status=get_status(repo),
        staged_diff_sha256=hashlib.sha256(staged.encode("utf-8")).hexdigest(),
    )


def _test_config_identity(repo: Path) -> str | None:
    try:
        from .test_runner import detect_test_command
        return detect_test_command(repo).display
    except Exception:
        return None


def _snapshot_payload(snapshot: RepositorySnapshot) -> dict:
    return {
        "branch": snapshot.branch, "file_hashes": [list(item) for item in snapshot.file_hashes],
        "status": snapshot.status, "staged_diff_sha256": snapshot.staged_diff_sha256,
    }


def _load_verified_baseline(repo: Path, snapshot: RepositorySnapshot):
    try:
        payload = json.loads(get_verified_baseline_cache_file(repo).read_text(encoding="utf-8"))
        if payload.get("version") != VERIFIED_BASELINE_CACHE_VERSION:
            return None
        if time.time() - float(payload["created_at"]) > VERIFIED_BASELINE_MAX_AGE_SECONDS:
            return None
        if payload.get("repository") != str(repo.resolve()):
            return None
        if payload.get("snapshot") != _snapshot_payload(snapshot):
            return None
        if payload.get("test_config") != _test_config_identity(repo):
            return None
        report = Path(payload["report"])
        if not report.is_file():
            return None
        result = payload["result"]
        return TestResult(
            str(result["command"]), int(result["returncode"]), float(result["duration_seconds"]),
            report, frozenset(result["failed_tests"]),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def save_verified_baseline(repo: Path, snapshot: RepositorySnapshot, result) -> None:
    """Persist only a completed full-suite result for this exact state."""
    if _test_config_identity(repo) is None:
        return
    report = get_repo_workspace(repo) / "verified-baseline.md"
    try:
        shutil.copyfile(result.output_file, report)
    except OSError:
        return
    payload = {
        "version": VERIFIED_BASELINE_CACHE_VERSION,
        "created_at": time.time(), "repository": str(repo.resolve()),
        "snapshot": _snapshot_payload(snapshot), "test_config": _test_config_identity(repo),
        "report": str(report),
        "result": {"command": result.command, "returncode": result.returncode,
                   "duration_seconds": result.duration_seconds,
                   "failed_tests": sorted(result.failed_tests)},
    }
    atomic_write_text(get_verified_baseline_cache_file(repo), json.dumps(payload, indent=2) + "\n")


_ANSI = {
    "green": "\x1b[32m", "red": "\x1b[31m", "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}

REPAIR_COMPANION_PROMPT = (
    "Use the attached PATCH_REPAIR_CONTEXT.md as repository context and the "
    "current source of truth. The listed repair targets are the sole success "
    "criterion: each of those tests must pass after the patch. Trace the failing "
    "assertion through the supplied current implementation and do not repeat the "
    "unsuccessful attempted patch. Do not change unrelated behavior from the "
    "original task. Investigate and repair the classified failures "
    "without weakening tests or reverting unrelated working-tree changes. "
    "Return exactly one complete unified diff with at least three unchanged "
    "context lines around each hunk, and no explanation outside the diff."
)


def _supports_color() -> bool:
    return "NO_COLOR" not in os.environ and sys.stdout.isatty()


def _status(text: str, color: str | None = None) -> str:
    if color and _supports_color():
        return f"{_ANSI[color]}{text}\x1b[0m"
    return text


def show_chatgpt_upload_artifact(context: Path) -> None:
    """Make a generated ChatGPT attachment unambiguous in terminal output."""
    print(_status("[UPLOAD THIS FILE]", "cyan"))
    print(_status(str(context), "cyan"))


def show_repair_send_instructions(repair_context: Path) -> None:
    print(f"Repair context ready to send to ChatGPT: {repair_context}")
    show_chatgpt_upload_artifact(repair_context)
    print("Attach that file and send this message with it:")
    print(REPAIR_COMPANION_PROMPT)


def _classify_test_validation(
    baseline,
    targeted,
    full,
    repair_targets: frozenset[str] = frozenset(),
) -> TestValidation:
    """Classify post-patch failures without claiming more than the output shows."""
    if targeted is None and full is None:
        return TestValidation(
            baseline, targeted, full, "unavailable",
            repair_targets=repair_targets,
        )
    # Targeted tests are the explicit success criterion for this patch.  They
    # remain authoritative even when their failing test was already red in the
    # baseline; the baseline only determines full-suite regressions.
    if targeted is not None and targeted.returncode != 0:
        targeted_failures = getattr(targeted, "failed_tests", frozenset())
        before = getattr(baseline, "failed_tests", frozenset())
        full_failures = getattr(full, "failed_tests", frozenset()) if full else frozenset()
        remaining_targets = targeted_failures & repair_targets
        return TestValidation(
            baseline, targeted, full,
            "repair_failed" if remaining_targets else "targeted_failed",
            new_failures=full_failures - before,
            existing_failures=full_failures & before,
            fixed_failures=before - full_failures if full is not None else frozenset(),
            repair_targets=repair_targets,
            remaining_repair_failures=remaining_targets,
        )

    if full is not None and full.returncode == 0:
        before = getattr(baseline, "failed_tests", frozenset())
        return TestValidation(
            baseline, targeted, full,
            "repair_passed" if repair_targets else "passed",
            fixed_failures=before,
            repair_targets=repair_targets,
        )
    if baseline is None:
        return TestValidation(baseline, targeted, full, "unclear")

    before = getattr(baseline, "failed_tests", frozenset())
    # Regressions are deliberately the full-suite delta, B - A.  A targeted
    # failure is handled above as a separate, patch-specific criterion.
    after = getattr(full, "failed_tests", frozenset()) if full is not None else frozenset()
    remaining_targets = after & repair_targets
    if remaining_targets:
        return TestValidation(
            baseline, targeted, full, "repair_failed",
            new_failures=after - before,
            existing_failures=after & before,
            fixed_failures=before - after,
            repair_targets=repair_targets,
            remaining_repair_failures=remaining_targets,
        )
    if baseline.returncode != 0 and not before:
        # We know the baseline failed, but not *which* test failed.  No
        # post-patch identifier can safely be called new in that situation.
        return TestValidation(baseline, targeted, full, "unclear")
    if full is None or not before and not after:
        return TestValidation(baseline, targeted, full, "unclear")
    new = after - before
    existing = after & before
    if new:
        return TestValidation(
            baseline, targeted, full, "regressions", new, existing,
            fixed_failures=before - after,
        )
    if existing:
        return TestValidation(
            baseline, targeted, full, "existing", frozenset(), existing,
            fixed_failures=before - after,
        )
    return TestValidation(baseline, targeted, full, "unclear")


def _show_test_validation(validation: TestValidation) -> None:
    functional_status = {
        "passed": ("PASSED", "green"),
        "existing": ("PASS WITH PRE-EXISTING FAILURES", "yellow"),
        "repair_passed": ("REPAIR SUCCESSFUL", "green"),
        "regressions": ("FAILED", "red"),
        "repair_failed": ("REPAIR UNSUCCESSFUL", "red"),
        "targeted_failed": ("FAILED", "red"),
    }.get(validation.status, ("REVIEW REQUIRED", "yellow"))
    print("\nFunctional validation: " + _status(*functional_status))
    for label, result in (
        ("Baseline", validation.baseline),
        ("Relevant tests", validation.targeted),
        ("Full suite", validation.full),
    ):
        if result is None:
            print(_status(f"[WARN] {label}: not run", "yellow"))
        else:
            expected_repair_baseline = (
                label == "Baseline"
                and validation.status == "repair_passed"
                and result.returncode != 0
                and bool(
                    getattr(result, "failed_tests", frozenset())
                    & validation.repair_targets
                )
            )
            if expected_repair_baseline:
                print(_status(
                    f"[EXPECTED] {label}: failed before repair; target reproduced ({result.command})",
                    "yellow",
                ))
                continue
            if label == "Full suite" and validation.status == "existing":
                count = len(validation.existing_failures)
                plural = "failure remains" if count == 1 else "failures remain"
                print(_status(
                    f"[WARN] {label}: same {count} pre-existing {plural} ({result.command})",
                    "yellow",
                ))
                continue
            if label == "Baseline" and validation.status == "existing":
                count = len(validation.existing_failures)
                print(_status(f"[FAIL] {label}: {count} existing failure(s) ({result.command})", "red"))
                continue
            status = "passed" if result.returncode == 0 else "failed"
            color = "green" if result.returncode == 0 else "red"
            symbol = "[OK]" if result.returncode == 0 else "[FAIL]"
            print(_status(f"{symbol} {label}: {status} ({result.command})", color))

    if validation.status == "repair_passed":
        print(_status(
            "[OK] Assessment: all repair targets now pass and no regressions were detected.",
            "green",
        ))
    elif validation.status == "passed":
        print(_status("[OK] Assessment: no new regressions detected.", "green"))
    elif validation.status == "repair_failed":
        print(_status("[FAIL] Repair target failures remain: " + ", ".join(
            sorted(validation.remaining_repair_failures)
        ), "red"))
        print("Recommended action: review the diff/test output, then undo this unsuccessful repair and create a fresh repair context.")
    elif validation.status == "regressions":
        print(_status("REGRESSION DETECTED", "red"))
        print(_status("Failed: " + ", ".join(sorted(validation.new_failures)), "red"))
        print("Recommended action: run `chatcode repair` to create repair context, then send it to ChatGPT.")
    elif validation.status == "existing":
        print(_status(
            f"[WARN] {len(validation.existing_failures)} pre-existing failure(s) remain: "
            + ", ".join(sorted(validation.existing_failures)),
            "yellow",
        ))
        print("[OK] Assessment: no new regressions detected; existing failures predate this patch.")
    elif validation.status == "targeted_failed":
        print(_status("[FAIL] Relevant tests failed; targeted validation remains required for this patch.", "red"))
        print("Recommended action: repair the targeted failures before keeping this patch.")
    else:
        print(_status("[WARN] Assessment: failures are unclear and require review.", "yellow"))
        print("Recommended action: run `chatcode repair` and send the context to ChatGPT for review.")

    targeted = "PASS" if validation.targeted and validation.targeted.returncode == 0 else "FAIL" if validation.targeted else "N/A"
    full = "PASS" if validation.full and validation.full.returncode == 0 else "FAIL" if validation.full else "N/A"
    regression_count = len(validation.new_failures)
    targeted_color = "green" if targeted == "PASS" else "red" if targeted == "FAIL" else "yellow"
    full_color = "green" if full == "PASS" else "red" if full == "FAIL" else "yellow"
    regression_color = "green" if regression_count == 0 and validation.status in {"passed", "repair_passed", "existing"} else "red" if regression_count else "yellow"
    print(" | ".join([
        _status("APPLY: PASS", "green"),
        _status(f"TARGETED: {targeted}", targeted_color),
        _status(f"FULL SUITE: {full}", full_color),
        _status(f"REGRESSIONS: {regression_count}", regression_color),
        *(
            [_status(f"PRE-EXISTING: {len(validation.existing_failures)}", "yellow")]
            if validation.existing_failures else []
        ),
        *(
            [_status("REPAIR: PASS | TARGET FAILURES REMAIN: 0", "green")]
            if validation.status == "repair_passed" else []
        ),
        *(
            [_status(
                f"REPAIR: FAIL | TARGET FAILURES REMAIN: {len(validation.remaining_repair_failures)}",
                "red",
            )]
            if validation.status == "repair_failed" else []
        ),
    ]))


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
            removed_lines = sum(line.kind == "-" for line in hunk.lines)
            required_context = min(3, max(0, len(source_lines) - removed_lines))
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


def _repair_state_file(repo: Path) -> Path:
    return get_repair_context_file(repo).with_name("patch-repair-state.json")


def _write_repair_context(
    repo: Path,
    content: str,
    paths: set[str] | list[str],
    repair_targets: set[str] | frozenset[str] = frozenset(),
) -> Path:
    output_file = get_repair_context_file(repo)
    hashes: dict[str, str | None] = {}
    for raw_path in sorted(set(paths)):
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        try:
            hashes[raw_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            hashes[raw_path] = None
    state = {
        "version": 1,
        "context_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "files": hashes,
        "repair_targets": sorted(repair_targets),
    }
    # Publish state first and the watched Markdown file last. A mismatched pair
    # is rejected by ``chatcode repair`` rather than presented as current.
    atomic_write_text(
        _repair_state_file(repo),
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        newline="\n",
    )
    # A repair context is the source of truth for the next canonical incoming
    # patch as soon as it is offered to the user. Persist its post-patch
    # working-tree baseline instead of leaving the original task generation
    # active until a separate ``chatcode repair`` invocation.
    original_task = get_context_task(repo)
    save_context_state(
        repo,
        task=original_task,
        context_sha256=state["context_sha256"],
        context_filename=output_file.name,
        context_kind="repair",
        repair_targets=sorted(repair_targets),
    )
    atomic_write_text(output_file, content, newline="\n")
    return output_file


def get_repair_context_targets(repo: Path) -> frozenset[str]:
    try:
        state = json.loads(_repair_state_file(repo).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    targets = state.get("repair_targets")
    if not isinstance(targets, list):
        return frozenset()
    return frozenset(item for item in targets if isinstance(item, str) and item)


def get_repair_context_stale_reason(repo: Path) -> str | None:
    output_file = get_repair_context_file(repo)
    state_file = _repair_state_file(repo)
    try:
        content = output_file.read_bytes()
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Repair context metadata is missing or unreadable."
    if hashlib.sha256(content).hexdigest() != state.get("context_sha256"):
        return "Repair context and its metadata belong to different generations."
    files = state.get("files")
    if not isinstance(files, dict):
        return "Repair context metadata is invalid."
    changed: list[str] = []
    for raw_path, expected in files.items():
        path = repo.joinpath(*PurePosixPath(raw_path).parts)
        try:
            current = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            current = None
        if current != expected:
            changed.append(raw_path)
    if changed:
        return "Working-tree files changed after repair context creation: " + ", ".join(sorted(changed))
    return None


def _test_result_from_saved_report(path: Path):
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fields = dict(re.findall(
        r"^(Status|Exit code|Command|Duration):\s*(.+)$",
        content,
        flags=re.MULTILINE,
    ))
    try:
        returncode = int(fields["Exit code"])
        duration = float(fields.get("Duration", "0").split()[0])
        command = fields["Command"]
    except (KeyError, ValueError):
        return None
    from .test_runner import TestResult, _failed_test_ids

    return TestResult(
        command=command,
        returncode=returncode,
        duration_seconds=duration,
        output_file=path,
        failed_tests=_failed_test_ids(content, ""),
    )


def refresh_test_failure_repair_context(repo: Path) -> Path | None:
    """Rebuild a saved test-failure context from current files on demand."""
    repair_context = get_repair_context_file(repo)
    try:
        if not repair_context.read_text(
            encoding="utf-8", errors="replace"
        ).startswith("# ChatCode Test Failure Repair Context"):
            return None
        history_entry = get_latest_applied_entry(repo)
        patch_text = get_history_patch_file(history_entry).read_text(
            encoding="utf-8", errors="replace"
        )
    except (OSError, HistoryError):
        return None

    reports_dir = get_test_results_dir(repo)
    baseline = _test_result_from_saved_report(reports_dir / "baseline.md")
    targeted = _test_result_from_saved_report(reports_dir / "targeted.md")
    full = _test_result_from_saved_report(reports_dir / "full-suite.md")
    if baseline is None or full is None:
        return None
    validation = _classify_test_validation(baseline, targeted, full)
    result = ApplyResult(
        paths=extract_patch_paths(patch_text),
        history_entry=history_entry,
    )
    return build_test_failure_repair_context(repo, result, validation)


def build_syntax_repair_context(
    repo: Path,
    original_patch: str,
    error: str,
    malformed_hunk: str | None = None,
    failure_type: str = "invalid_patch_syntax",
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
        f"Failure type: `{failure_type}`",
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
        "Include unchanged current source lines around every hunk whenever possible; do not rely on a guessed line number.",
        "",
    ])
    return _write_repair_context(
        repo,
        "\n".join(content_parts),
        _safe_candidate_paths(repo, original_patch),
    )


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
    content = "\n".join([
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
        ])
    return _write_repair_context(repo, content, paths)


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
        path = PurePosixPath(raw_path.replace("\\", "/")).as_posix()
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
    return _write_repair_context(
        repo,
        content,
        _safe_candidate_paths(repo, patch_text),
    )


def build_test_failure_repair_context(
    repo: Path,
    result: ApplyResult,
    validation: TestValidation,
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
        from .indexing.project_graph import load_map

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
        section = _working_tree_section(
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
        get_context_task(repo),
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
    return _write_repair_context(
        repo,
        content,
        included_paths,
        validation.new_failures | validation.existing_failures,
    )


def _apply_patch_core(
    repo: Path,
    patch_file: Path,
    *,
    dry_run: bool = False,
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
    try:
        _validate_hunk_context(repo, patch_text)
    except UnifiedDiffError as exc:
        repair_context = build_syntax_repair_context(
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
            if temporary_validation_file is not None:
                temporary_validation_file.unlink(missing_ok=True)
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
        from .indexing.index_manager import update_project_map
        from .indexing.project_graph import SCHEMA_VERSION, map_path

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

    _clear_repair_context(repo)
    _clear_incoming_patch(repo)

    return ApplyResult(
        paths=paths,
        history_entry=history_entry,
    )


def _clear_repair_context(
    repo: Path,
) -> None:
    for path in (get_repair_context_file(repo), _repair_state_file(repo)):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _clear_incoming_patch(
    repo: Path,
) -> None:
    incoming = get_default_patch_file(repo)
    try:
        if incoming.exists():
            incoming.write_text(
                "",
                encoding="utf-8",
                newline="\n",
            )
    except OSError:
        pass


def _fallback_patch_summary(
    patch_text: str,
    paths: set[str],
) -> str:
    additions = 0
    deletions = 0
    for line in patch_text.splitlines():
        if line.startswith(("+++ ", "--- ")):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1

    lines = ["Changed files:"]
    lines.extend(
        f"  - {path}"
        for path in sorted(paths)
    )
    lines.append(
        f"Diff statistics: +{additions} / -{deletions}"
    )
    return "\n".join(lines)


def _qwen_patch_summary(
    repo: Path,
    patch_text: str,
) -> str | None:
    if shutil.which("ollama") is None:
        return None

    model = os.getenv(
        "CHATCODE_QWEN_MODEL",
        "qwen2.5-coder:1.5b",
    ).strip()
    if not model:
        return None

    prompt = "\n".join([
        "Summarize this code patch for the developer who is about to apply it.",
        "Describe practical behavior changes, not implementation trivia.",
        "Return strict JSON only: {\"bullets\":[\"...\"]}.",
        "Return 3 to 6 short bullets. Do not suggest or modify code.",
        "",
        "Task:",
        get_context_task(repo),
        "",
        "Patch:",
        patch_text[:60_000],
    ])

    try:
        process = subprocess.run(
            ["ollama", "run", model, "--format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if process.returncode != 0 or not process.stdout.strip():
        return None

    try:
        parsed = json.loads(process.stdout)
    except json.JSONDecodeError:
        return None

    bullets = parsed.get("bullets")
    if not isinstance(bullets, list):
        return None

    cleaned = [
        item.strip()[:300]
        for item in bullets
        if isinstance(item, str) and item.strip()
    ][:6]
    if len(cleaned) < 1:
        return None

    return "\n".join(
        f"  - {item}"
        for item in cleaned
    )


def _build_patch_summary(
    repo: Path,
    preview: PatchPreview,
) -> str:
    return (
        _qwen_patch_summary(
            repo,
            preview.patch_text,
        )
        or _fallback_patch_summary(
            preview.patch_text,
            preview.paths,
        )
    )


def _ask_yes_no(
    question: str,
    *,
    default: bool,
) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(question + suffix).strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def _open_diff_window(
    repo: Path,
    patch_text: str,
    paths: set[str],
) -> None:
    try:
        preview_root = get_repo_workspace(repo) / "patch-preview"
        resolved_workspace = get_repo_workspace(repo).resolve()
        resolved_preview = preview_root.resolve()
        if resolved_preview.parent != resolved_workspace:
            raise OSError("Unsafe patch preview directory.")
        if preview_root.exists():
            shutil.rmtree(preview_root)
        before_root = preview_root / "before"
        after_root = preview_root / "after"
        before_root.mkdir(parents=True)
        after_root.mkdir(parents=True)
        preview_patch = preview_root / "canonical-preview.diff"
        atomic_write_text(preview_patch, patch_text, newline="\n")

        for raw_path in sorted(paths):
            relative = PurePosixPath(raw_path)
            source = repo.joinpath(*relative.parts)
            before = before_root.joinpath(*relative.parts)
            after = after_root.joinpath(*relative.parts)
            before.parent.mkdir(parents=True, exist_ok=True)
            after.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                shutil.copy2(source, before)
                shutil.copy2(source, after)

        run_git("apply", "--unsafe-paths", str(preview_patch), cwd=after_root)

        for raw_path in sorted(paths):
            relative = PurePosixPath(raw_path)
            before = before_root.joinpath(*relative.parts)
            after = after_root.joinpath(*relative.parts)
            if not before.exists():
                before.parent.mkdir(parents=True, exist_ok=True)
                before.write_bytes(b"")
            if not after.exists():
                after.parent.mkdir(parents=True, exist_ok=True)
                after.write_bytes(b"")
            open_code_diff(before.resolve(), after.resolve())
    except (OSError, GitError, HistoryError) as exc:
        raise PatchError(
            f"Could not open diff window: {exc}"
        ) from exc


def _show_test_result(test_result) -> None:
    status = "PASSED" if test_result.returncode == 0 else "FAILED"
    symbol = "[OK]" if test_result.returncode == 0 else "[FAIL]"
    color = "green" if test_result.returncode == 0 else "red"
    print(_status(
        f"{symbol} TESTS {status}: {test_result.command} "
        f"({test_result.duration_seconds:.2f}s)",
        color,
    ))
    print(f"Test report: {test_result.output_file}")


def _preserve_test_report(repo: Path, test_result, phase: str):
    """Keep phase output for repair context while ``latest.md`` is replaced."""
    destination = get_test_results_dir(repo) / f"{phase}.md"
    try:
        shutil.copyfile(test_result.output_file, destination)
        return replace(test_result, output_file=destination)
    except (OSError, TypeError):
        return test_result


def _clear_phase_test_reports(repo: Path) -> None:
    for name in ("baseline.md", "targeted.md", "full-suite.md"):
        try:
            (get_test_results_dir(repo) / name).unlink(missing_ok=True)
        except OSError:
            pass


def _run_background_baseline(repo: Path):
    """Run the existing project-wide test command without terminal output."""
    from .test_runner import run_project_tests
    return run_project_tests(repo)


def _verify_undo(
    repo: Path,
    undone_entry: Path,
) -> None:
    patch_file = get_history_patch_file(undone_entry)
    try:
        run_git(
            "apply",
            "--check",
            "--recount",
            str(patch_file),
            cwd=repo,
        )
    except GitError as exc:
        raise PatchUndoError(
            "Patchen backades, men återställningen "
            "kunde inte verifieras.\n"
            f"{exc}"
        ) from exc


CHECK_REPAIR_COMPANION_PROMPT = (
    "Use the attached CHECK_REPAIR_CONTEXT.md as the current repository context "
    "and source of truth. Fix the listed failing test(s) without weakening tests "
    "or reverting unrelated working-tree changes. Trace the failure through the "
    "supplied current implementation and repair the underlying behavior. Return "
    "exactly one complete unified diff with at least three unchanged context lines "
    "around each hunk and no explanation outside the diff."
)


def build_check_repair_context(repo: Path, test_result, selected_failures: frozenset[str]) -> Path:
    """Materialize a focused new task from a standalone health check."""
    selected = frozenset(selected_failures) & getattr(test_result, "failed_tests", frozenset())
    if not selected:
        raise PatchError("Choose at least one failing test from this health check.")
    try:
        output = test_result.output_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        output = "[Saved test output is unavailable.]"
    traceback_paths: list[Path] = []
    repo_root = repo.resolve()
    for raw_path in re.findall(r'^\s*File "([^"]+)", line \d+', output, re.MULTILINE):
        try:
            path = Path(raw_path).resolve()
            path.relative_to(repo_root)
        except (OSError, ValueError):
            continue
        traceback_paths.append(path)
    from .context_builder import build_context_from_test_roots
    source_context, selected_paths = build_context_from_test_roots(
        repo, selected, traceback_paths
    )
    try:
        dirty = get_status(repo) or "[Working tree clean.]"
    except GitError:
        dirty = "[Working-tree status unavailable.]"
    try:
        diff_paths = [path.relative_to(repo).as_posix() for path in selected_paths]
        current_diff = run_git("diff", "--no-renames", "--", *diff_paths, cwd=repo) if diff_paths else ""
    except GitError:
        current_diff = ""
    diagnostics = []
    for failure_id in sorted(selected):
        diagnostics.extend([f"### {failure_id}", "```text", _failure_output_excerpt(output, failure_id), "```", ""])
    content = "\n".join([
        "# ChatCode Check Repair Context", "",
        "Generated from `chatcode check` against the current working tree.",
        f"Repository: `{repo.resolve()}`", "",
        "## Repair targets", *(f"- {failure}" for failure in sorted(selected)), "",
        "## Relevant failure output", *diagnostics,
        "## Dirty working-tree state", "```text", dirty, "```", "",
        "## Directly relevant current diff", "```diff", current_diff or "[No unstaged diff for selected source files.]", "```", "",
        "## Exact current source and bounded dependencies",
        source_context,
        "## Repair success criterion",
        "The listed failing tests are the repair targets. The repair is successful when those tests pass without introducing new regressions.",
        "Do not weaken or remove tests. Do not revert unrelated working-tree changes.", "",
    ])
    destination = get_check_repair_context_file(repo)
    atomic_write_text(destination, content, newline="\n")
    return destination


def show_check_repair_send_instructions(context: Path) -> None:
    print(f"Check repair context created: {context}")
    show_chatgpt_upload_artifact(context)
    print("Attach that file and send this message with it:")
    print(CHECK_REPAIR_COMPANION_PROMPT)


EXISTING_FAILURE_COMPANION_PROMPT = (
    "Use the attached EXISTING_FAILURE_CONTEXT.md as the current repository "
    "context and source of truth. Fix the selected pre-existing test failure "
    "without reverting unrelated working-tree changes or the previously "
    "successful patch. Return exactly one complete unified diff with at least "
    "three unchanged context lines around each hunk and no explanation outside "
    "the diff."
)

FOLLOWUP_COMPANION_PROMPT = (
    "Use the attached FOLLOWUP_CONTEXT.md as the current repository context "
    "and source of truth. The previous patch applied successfully and automated "
    "validation passed, but the user reports that the original problem remains "
    "unresolved. Use the included user feedback to continue the fix without "
    "reverting unrelated working-tree changes or the previous patch. Return "
    "exactly one complete unified diff with at least three unchanged context "
    "lines around each hunk and no explanation outside the diff."
)

MAX_FOLLOWUP_ROUNDS = 5


def _load_followup_state(repo: Path) -> dict | None:
    try:
        state = json.loads(get_followup_state_file(repo).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) and state.get("version") == 1 else None


def _save_followup_state(repo: Path, state: dict) -> None:
    atomic_write_text(
        get_followup_state_file(repo),
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        newline="\n",
    )


def mark_followup_resolved(repo: Path) -> None:
    state = _load_followup_state(repo)
    if state is None or not state.get("unresolved"):
        return
    state["unresolved"] = False
    state["updated_at"] = time.time()
    _save_followup_state(repo, state)


def _followup_attempt(result: ApplyResult, validation: TestValidation, patch_summary: str) -> dict:
    return {
        "history_entry": str(result.history_entry.resolve()),
        "paths": sorted(result.paths),
        "patch_summary": patch_summary,
        "validation_status": validation.status,
    }


def _result_from_followup_attempt(attempt: dict) -> ApplyResult | None:
    history_entry = attempt.get("history_entry")
    paths = attempt.get("paths")
    if not isinstance(history_entry, str) or not isinstance(paths, list):
        return None
    return ApplyResult(
        {path for path in paths if isinstance(path, str)}, Path(history_entry)
    )


def _attempted_python_symbols(result: ApplyResult) -> set[str]:
    """Recover definitions changed by the successfully applied patch."""
    attempted: set[str] = set()
    for relative in result.paths:
        if not relative.casefold().endswith(".py"):
            continue
        parts = PurePosixPath(relative).parts
        snapshots: list[dict[str, str]] = []
        for state in ("before", "after"):
            path = result.history_entry.joinpath(state, *parts)
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (OSError, SyntaxError):
                snapshots.append({})
                continue
            snapshots.append({
                node.name: ast.get_source_segment(source, node) or ""
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            })
        before, after = snapshots
        attempted.update(
            name for name in before.keys() | after.keys()
            if before.get(name) != after.get(name)
        )
    return attempted


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
    existing_state = _load_followup_state(repo) if persist_state else None
    prior_feedback = list(feedback_history or [])
    prior_results = list(prior_attempts or [])
    if existing_state is not None and existing_state.get("unresolved"):
        prior_feedback = [
            item for item in existing_state.get("feedback", [])
            if isinstance(item, str) and item
        ]
        prior_results = [
            parsed for parsed in (
                _result_from_followup_attempt(item)
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
    from .context_builder import (
        _build_stable_patch_source_context,
        _ensure_explicit_task_files,
        collect_relevant_files,
    )
    from .retrieval.hybrid_retriever import resolve_alternative_callback_roots
    retrieved = collect_relevant_files(repo, retrieval_task, include_target_symbols=True)
    files, target_symbols = retrieved if isinstance(retrieved, tuple) else (retrieved, {})
    attempted_results = [*prior_results, result][-MAX_FOLLOWUP_ROUNDS:]
    attempted_symbols = {
        symbol for attempted in attempted_results
        for symbol in _attempted_python_symbols(attempted)
    }
    alternatives = resolve_alternative_callback_roots(
        repo, retrieval_task, attempted_symbols
    )
    # Follow-up-only priority: keep the attempted dirty path in normal
    # retrieval, but put structurally supported alternatives first so the
    # failed hypothesis cannot consume all patchable-source allowance.
    files = list(dict.fromkeys([*alternatives.files, *files]))
    target_symbols = {
        **alternatives.required_symbols,
        **{
            path: list(dict.fromkeys([
                *alternatives.required_symbols.get(path, []), *symbols,
            ]))
            for path, symbols in target_symbols.items()
        },
    }
    files = _ensure_explicit_task_files(repo, retrieval_task, files)
    source_context, source_hashes = _build_stable_patch_source_context(
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
        attempts.append(_followup_attempt(result, validation, patch_summary))
        _save_followup_state(repo, {
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
    state = _load_followup_state(repo)
    if state is None or not state.get("unresolved"):
        return None
    attempts = [item for item in state.get("attempts", []) if isinstance(item, dict)]
    feedback = [item for item in state.get("feedback", []) if isinstance(item, str) and item]
    if not attempts or not feedback or not isinstance(state.get("original_task"), str):
        return None
    latest = attempts[-1]
    result = _result_from_followup_attempt(latest)
    if result is None:
        return None
    prior_results = [
        parsed for parsed in (_result_from_followup_attempt(item) for item in attempts[:-1])
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


def show_followup_send_instructions(context: Path) -> None:
    print("Follow-up context created.")
    show_chatgpt_upload_artifact(context)
    print("Attach that file and send this message with it:")
    print(FOLLOWUP_COMPANION_PROMPT)


def _applied_patch_summary(result: ApplyResult) -> str:
    try:
        patch_text = get_history_patch_file(result.history_entry).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        patch_text = "[Applied patch text is unavailable; changed-file summary follows.]"
    return _fallback_patch_summary(patch_text, result.paths)


def _failure_output_excerpt(output: str, failure_id: str) -> str:
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
        from .indexing.project_graph import load_map
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
        section = _working_tree_section(repo, path, full_limit=35_000)
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
            "```text", _failure_output_excerpt(baseline_output, failure_id), "```",
            "#### Current post-patch failure output",
            "```text", _failure_output_excerpt(post_output, failure_id), "```", "",
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
        get_context_task(repo),
        "",
        "## Selected pre-existing failure(s)",
        *failure_sections,
        "## Recently applied patch summary",
        _fallback_patch_summary(patch_text, result.paths),
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


def show_existing_failure_send_instructions(context: Path) -> None:
    print(f"Existing failure context created: {context}")
    show_chatgpt_upload_artifact(context)
    print("Attach that file and send this message with it:")
    print(EXISTING_FAILURE_COMPANION_PROMPT)


def _choose_existing_failures(failures: frozenset[str]) -> frozenset[str]:
    ordered = sorted(failures)
    if len(ordered) == 1:
        print(f"Selected pre-existing failure: {ordered[0]}")
        return frozenset(ordered)
    print("Select pre-existing failure(s) for a new fix task:")
    for index, failure in enumerate(ordered, start=1):
        print(f"[{index}] {failure}")
    print("Enter comma-separated numbers, or A for all (one is recommended).")
    answer = input("> ").strip().lower()
    if answer == "a":
        return frozenset(ordered)
    try:
        selected = {ordered[int(part.strip()) - 1] for part in answer.split(",") if part.strip()}
    except (IndexError, ValueError):
        selected = set()
    if not selected:
        print("No valid failure selected.")
    return frozenset(selected)


def _post_apply_choice(
    repo: Path,
    result: ApplyResult,
    test_result,
    test_error: Exception | None,
    *,
    recommend_undo: bool = False,
    recommend_keep_for_repair: bool = False,
    validation: TestValidation | None = None,
) -> None:
    from .history import open_history_review

    failed = (
        test_result is not None
        and test_result.returncode != 0
    )
    pre_existing_only = validation is not None and validation.status == "existing"
    followup_eligible = validation is not None and validation.status in {"passed", "repair_passed"}
    followup_feedback: str | None = None

    if followup_eligible:
        while True:
            answer = input("Did the patch solve your problem? [Y/n] ").strip().lower()
            if answer in {"", "y", "yes"}:
                if get_context_kind(repo) == "followup":
                    mark_followup_resolved(repo)
                print("Changes kept.")
                return
            if answer in {"n", "no"}:
                break
            print("Choose Y or N.")
        while not followup_feedback:
            followup_feedback = input("What is still not working?\n> ").strip()
            if not followup_feedback:
                print("Please describe what is still not working.")
        try:
            context = build_followup_context(
                repo, result, validation, followup_feedback, _applied_patch_summary(result),
            )
            show_followup_send_instructions(context)
            from .cli import open_folder
            open_folder(context.parent)
            print("Changes kept.")
            return
        except (OSError, PatchError) as exc:
            print(f"Could not create follow-up context: {exc}")

    while True:
        if failed:
            undo_label = "[U] Undo (recommended)" if recommend_undo else "[U] Undo"
            keep_label = (
                "[K] Keep changes (recommended for repair)"
                if recommend_keep_for_repair else "[K] Keep changes"
            )
            if pre_existing_only:
                print(f"{keep_label}  [F] Keep changes + create fix context for existing failure  {undo_label}  [R] Review diff  [T] Show test output")
            else:
                print(f"{keep_label}  {undo_label}  [R] Review diff  [T] Show test output")
            choice = input("> ").strip().lower()
            if not choice and recommend_undo:
                choice = "u"
            elif not choice and recommend_keep_for_repair:
                choice = "k"
        else:
            retry = "  [F] Retry follow-up context" if followup_feedback else ""
            print("[K] Keep  [U] Undo  [R] Review diff" + retry)
            choice = input("> ").strip().lower() or "k"

        if choice == "k":
            print("Changes kept.")
            return

        if choice == "r":
            try:
                open_history_review(
                    result.history_entry
                )
            except HistoryError as exc:
                print(f"Could not open review: {exc}")
            continue

        if choice == "t" and failed and test_result is not None:
            try:
                print(
                    test_result.output_file.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                )
            except OSError as exc:
                print(f"Could not read test output: {exc}")
            continue

        if choice == "f" and followup_feedback and validation is not None:
            try:
                context = build_followup_context(
                    repo, result, validation, followup_feedback, _applied_patch_summary(result),
                )
                show_followup_send_instructions(context)
                from .cli import open_folder
                open_folder(context.parent)
                print("Changes kept.")
                return
            except (OSError, PatchError) as exc:
                print(f"Could not create follow-up context: {exc}")
                continue

        if choice == "f" and pre_existing_only and validation is not None:
            selected = _choose_existing_failures(validation.existing_failures)
            if not selected:
                continue
            try:
                context = build_existing_failure_context(repo, result, validation, selected)
                show_existing_failure_send_instructions(context)
                # Reuse the same cross-platform folder opener used for normal
                # generated contexts. Importing here avoids a module import
                # cycle between the CLI and patch workflow modules.
                from .cli import open_folder
                open_folder(context.parent)
            except OSError as exc:
                print(f"Could not create existing failure context: {exc}")
                continue
            print("Changes kept.")
            return

        if choice == "u":
            undone = undo_last_patch(repo)
            _verify_undo(
                repo,
                undone.history_entry,
            )
            print("Changes restored successfully.")
            if recommend_keep_for_repair or (
                validation is not None and validation.status == "repair_failed"
            ):
                _clear_repair_context(repo)
                print("Repair context invalidated because its working-tree baseline was undone.")
            return

        if test_error is not None and choice == "t":
            print(f"No test report is available: {test_error}")
            continue

        extra = ", T, or F." if (failed and pre_existing_only) or followup_feedback else ", or T." if failed else "."
        print("Choose K, U, R" + extra)


def _run_apply_flow(
    repo: Path,
    patch_file: Path,
    *,
    yes: bool = False,
) -> ApplyResult:
    repair_targets = get_active_repair_targets(repo)
    preview = _apply_patch_core(
        repo,
        patch_file,
        dry_run=True,
    )
    assert isinstance(preview, PatchPreview)

    # Start only after the candidate has passed all non-mutating applicability
    # checks. The snapshot is rechecked immediately before mutation below.
    baseline_snapshot = _capture_repository_snapshot(repo)
    _clear_phase_test_reports(repo)
    cached_baseline = _load_verified_baseline(repo, baseline_snapshot)
    baseline_executor: ThreadPoolExecutor | None = None
    baseline_future: Future | None = None
    if cached_baseline is not None:
        print("Pre-patch baseline: using verified cached result.")
        print("[OK] Repository state unchanged since last full-suite validation." if cached_baseline.returncode == 0
              else f"[WARN] {len(cached_baseline.failed_tests)} known pre-existing failure(s).")
    else:
        baseline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatcode-baseline")
        baseline_future = baseline_executor.submit(_run_background_baseline, repo)
        print("Starting pre-patch test baseline in background...")

    summary = _build_patch_summary(
        repo,
        preview,
    ) if not repair_targets else "\n".join([
        "Repair candidate (summary derived from the diff, not AI):",
        _fallback_patch_summary(preview.patch_text, preview.paths),
        "Repair targets:",
        *(f"  - {target}" for target in sorted(repair_targets)),
    ])
    print("Patch is technically applicable.\n")
    print("Planned changes:")
    print(summary)

    interactive = (
        sys.stdin.isatty()
        and sys.stdout.isatty()
    )
    if not interactive and not yes:
        if baseline_future is not None:
            baseline_future.cancel()
        if baseline_executor is not None:
            baseline_executor.shutdown(wait=False, cancel_futures=True)
        raise PatchError(
            "Non-interactive apply requires --yes. "
            "No repository files were changed."
        )

    if not yes:
        if _ask_yes_no(
            "View full diff before applying?",
            default=True,
        ):
            _open_diff_window(repo, preview.patch_text, preview.paths)

        if not _ask_yes_no(
            "Apply these changes?",
            default=False,
        ):
            if baseline_future is not None:
                baseline_future.cancel()
            if baseline_executor is not None:
                baseline_executor.shutdown(wait=False, cancel_futures=True)
            print("Apply cancelled. No repository files were changed.")
            raise PatchError(
                "Apply cancelled by user."
            )

    baseline = None
    baseline_error: Exception | None = None
    try:
        current_snapshot = _capture_repository_snapshot(repo)
        if cached_baseline is not None and current_snapshot == baseline_snapshot:
            baseline = cached_baseline
        elif current_snapshot != baseline_snapshot:
            # Do not overlap a replacement suite with the discarded suite:
            # project test tools often share caches and report locations.
            print("Repository changed during review; discarding pre-patch baseline.")
            if baseline_future is not None and baseline_executor is not None:
                baseline_future.cancel()
                try:
                    baseline_future.result()
                except Exception:
                    pass
                baseline_executor.shutdown(wait=True, cancel_futures=True)
            print("Running fresh pre-patch test baseline...")
            baseline = _run_background_baseline(repo)
        else:
            assert baseline_future is not None and baseline_executor is not None
            if not baseline_future.done():
                print("Waiting for pre-patch baseline...")
            baseline = baseline_future.result()
            baseline_executor.shutdown(wait=True, cancel_futures=True)
        baseline = _preserve_test_report(repo, baseline, "baseline")
        _show_test_result(baseline)
    except Exception as exc:
        # Match the established baseline-infrastructure-error path. Test
        # failures are TestResult values and never arrive here as exceptions.
        baseline_error = exc
        if baseline_executor is not None:
            baseline_executor.shutdown(wait=False, cancel_futures=True)
        print(f"Pre-patch tests could not be run: {exc}")

    result = _apply_patch_core(
        repo,
        patch_file,
    )
    assert isinstance(result, ApplyResult)

    print(_status("\n[OK] PATCH APPLIED", "green"))
    print("Running relevant tests, then the full suite...")

    test_result = None
    relevant_result = None
    test_error: Exception | None = baseline_error
    try:
        from .history import update_history_test_result
        from .test_runner import TestError, run_project_tests, run_relevant_tests

        try:
            relevant_result = run_relevant_tests(repo, result.paths)
            if relevant_result is not None:
                relevant_result = _preserve_test_report(repo, relevant_result, "targeted")
                _show_test_result(relevant_result)
        except TestError as exc:
            # A narrow selection is only an optimization.  It must never
            # prevent the required full-suite validation from running.
            test_error = exc
            print(f"Relevant tests could not be run: {exc}")

        remaining_after_targeted = (
            getattr(relevant_result, "failed_tests", frozenset())
            & repair_targets
            if relevant_result is not None and relevant_result.returncode != 0
            else frozenset()
        )
        if remaining_after_targeted:
            update_history_test_result(result.history_entry, "REPAIR_FAILED")
            print(_status(
                "[FAIL] Repair candidate rejected by relevant tests; "
                "the full suite was skipped.",
                "red",
            ))
        else:
            try:
                full_snapshot = _capture_repository_snapshot(repo)
                test_result = run_project_tests(repo)
                test_result = _preserve_test_report(repo, test_result, "full-suite")
                if _capture_repository_snapshot(repo) == full_snapshot:
                    save_verified_baseline(repo, full_snapshot, test_result)
                update_history_test_result(
                    result.history_entry,
                    "PASSED" if test_result.returncode == 0 else "FAILED",
                    command=test_result.command,
                    returncode=test_result.returncode,
                    duration_seconds=test_result.duration_seconds,
                )
                _show_test_result(test_result)
            except TestError as exc:
                test_error = exc
                update_history_test_result(
                    result.history_entry,
                    "ERROR",
                )
                print(f"Tests could not be run: {exc}")
    except HistoryError as exc:
        test_error = exc
        print(f"Could not save test status: {exc}")

    validation = _classify_test_validation(
        baseline,
        relevant_result,
        test_result,
        repair_targets,
    )
    _show_test_validation(validation)
    validation_passed = validation.status in {"passed", "repair_passed", "existing"}
    if not validation_passed:
        try:
            repair_context = build_test_failure_repair_context(
                repo, result, validation
            )
            color = "red" if validation.status == "regressions" else "yellow"
            print(_status("Repair context created.", color))
            show_repair_send_instructions(repair_context)
        except OSError as exc:
            print(_status(f"Could not create repair context: {exc}", "yellow"))

    print("\nWhat changed:")
    print(summary)

    if interactive:
        _post_apply_choice(
            repo,
            result,
            (
                test_result
                if test_result is not None and test_result.returncode != 0
                else relevant_result
            ),
            test_error,
            recommend_keep_for_repair=not validation_passed,
            validation=validation,
        )
    elif not validation_passed:
        print(
            "Functional validation did not pass during non-interactive "
            "--yes apply. Changes were kept; review the test report above."
        )
    elif test_error is not None:
        print(
            "Tests were unavailable during non-interactive --yes apply. "
            "Changes were kept."
        )

    return result


def apply_patch(
    repo: Path,
    patch_file: Path,
) -> ApplyResult:
    if _CLI_APPLY_INVOCATION:
        _run_apply_flow(
            repo,
            patch_file,
            yes=_CLI_APPLY_YES,
        )
        raise SystemExit(0)

    result = _apply_patch_core(
        repo,
        patch_file,
    )
    assert isinstance(result, ApplyResult)
    return result


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
