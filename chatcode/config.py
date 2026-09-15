from __future__ import annotations

import os
from pathlib import Path


def get_setting(name: str) -> str | None:
    """Read a process setting first, then ChatCode's local .env file."""
    configured = os.environ.get(name)
    if configured:
        return configured

    env_file = Path(__file__).resolve().parents[1] / ".env"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        return value or None
    return None


def get_boolean_setting(name: str, default: bool = False) -> bool:
    value = get_setting(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_index_mode() -> str:
    """Resolve the explicit index mode with backwards-compatible Qwen settings."""
    configured = get_setting("CHATCODE_INDEX_MODE")
    if configured is not None:
        mode = configured.strip().lower()
        if mode not in {"ai", "static"}:
            raise ValueError(
                "CHATCODE_INDEX_MODE must be either 'ai' or 'static'"
            )
        return mode

    model = get_setting("CHATCODE_QWEN_MODEL")
    legacy_toggle = get_setting("CHATCODE_QWEN_ENABLED")
    if legacy_toggle is not None:
        enabled = legacy_toggle.strip().lower() in {"1", "true", "yes", "on"}
        return "ai" if enabled and model else "static"
    return "ai" if model else "static"
