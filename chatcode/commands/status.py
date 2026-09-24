"""CLI status and context commands."""
from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..context.errors import ContextBuildError
from ..git_utils import GitError
from ..patching.errors import PatchError


def command_status(
    reindex: bool = False,
    *,
    get_repo_root_fn: Callable[[], Path],
    update_project_map_fn: Callable,
    reporter_factory: Callable[[], object],
    get_branch_fn: Callable[[Path], str],
    build_safe_status_fn: Callable[[Path], str],
    get_background_full_suite_status_fn: Callable[[Path], dict | None],
    status_fn: Callable[[str, str | None], str],
    ensure_background_failure_repair_context_fn: Callable,
    show_repair_send_instructions_fn: Callable[[Path], None],
) -> None:
    repo = get_repo_root_fn()

    if reindex:
        print("Forcing a full project-index rebuild...")
        update_project_map_fn(
            repo,
            progress=reporter_factory(),
            force_rebuild=True,
        )
        print()

    print(f"Repository: {repo}")
    print(
        f"Branch:     {get_branch_fn(repo)}"
    )
    print()

    status = build_safe_status_fn(repo)

    if status:
        print("Changes:")
        print(status)
    else:
        print("Working tree clean.")

    print()
    background = get_background_full_suite_status_fn(repo)
    if background is None:
        print("Background full suite: no run recorded.")
        return

    state = background.get("state")
    if state == "running":
        print(f"Background full suite: {status_fn('RUNNING', 'yellow')}")
        if background.get("started_at") is not None:
            started = datetime.fromtimestamp(
                float(background["started_at"])
            ).astimezone().isoformat(timespec="seconds")
            print(f"Started:  {started}")
    elif state == "completed":
        returncode = int(background.get("returncode", 1))
        label = "PASSED" if returncode == 0 else "FAILED"
        color = "green" if returncode == 0 else "red"
        print(f"Background full suite: {status_fn(label, color)}")
        if background.get("command"):
            print(f"Command:  {background['command']}")
        if background.get("duration_seconds") is not None:
            print(f"Duration: {float(background['duration_seconds']):.2f}s")
        if background.get("completed_at") is not None:
            completed = datetime.fromtimestamp(
                float(background["completed_at"])
            ).astimezone().isoformat(timespec="seconds")
            print(f"Completed: {completed}")
        failures = background.get("failed_tests", [])
        if failures:
            print(f"Failures: {len(failures)}")
            for failure in failures[:10]:
                print(f"  {failure}")
        if background.get("report"):
            print(f"Report:   {background['report']}")
        if returncode != 0:
            try:
                repair_context, created = ensure_background_failure_repair_context_fn(
                    repo,
                    background,
                )
            except (OSError, PatchError, GitError) as exc:
                print(status_fn(
                    f"Repair context could not be created: {exc}",
                    "yellow",
                ))
            else:
                if created:
                    print(status_fn(
                        "Repair context created from the failed background suite.",
                        "yellow",
                    ))
                    show_repair_send_instructions_fn(repair_context)
                else:
                    print("Repair context: " + status_fn(
                        str(repair_context),
                        "cyan",
                    ))
    elif state == "error":
        print("Background full suite: ERROR")
        print(f"Error: {background.get('error', 'Unknown background test error')}")
    else:
        print("Background full suite: UNKNOWN")


def command_context(
    task: str,
    patch_oriented: bool = False,
    *,
    get_repo_root_fn: Callable[[], Path],
    reporter_factory: Callable[[], object],
    build_patch_context_fn: Callable,
    build_context_fn: Callable,
    get_default_patch_file_fn: Callable[[Path], Path],
    show_chatgpt_upload_artifact_fn: Callable[[Path], None],
) -> None:
    repo = get_repo_root_fn()
    reporter = reporter_factory()

    try:
        if patch_oriented:
            output = build_patch_context_fn(
                repo,
                task,
                index_progress=reporter,
            )
        else:
            output = build_context_fn(
                repo,
                task,
                index_progress=reporter,
            )
    except ContextBuildError as exc:
        print(f"Context not created: {exc}", file=sys.stderr)
        return

    patch_file = get_default_patch_file_fn(
        repo
    )

    patch_file.write_text(
        "",
        encoding="utf-8",
    )

    print("Context created:")
    show_chatgpt_upload_artifact_fn(output)
    print()
    print("Workspace directory:")
    print(output.parent)
