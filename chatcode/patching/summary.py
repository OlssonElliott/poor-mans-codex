"""Human-readable and optional Qwen patch summaries."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path


def fallback_patch_summary(
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


def qwen_patch_summary(
    repo: Path,
    patch_text: str,
    *,
    which_fn: Callable[[str], str | None],
    run_fn: Callable,
    getenv_fn: Callable[[str, str], str],
    get_context_task_fn: Callable[[Path], str | None],
) -> str | None:
    if which_fn("ollama") is None:
        return None

    model = getenv_fn(
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
        get_context_task_fn(repo) or "",
        "",
        "Patch:",
        patch_text[:60_000],
    ])

    try:
        process = run_fn(
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
