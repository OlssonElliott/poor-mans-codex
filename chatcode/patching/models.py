"""Shared patch workflow data models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    __test__ = False

    baseline: object | None
    targeted: object | None
    full: object | None
    status: str
    new_failures: frozenset[str] = frozenset()
    existing_failures: frozenset[str] = frozenset()
    fixed_failures: frozenset[str] = frozenset()
    repair_targets: frozenset[str] = frozenset()
    remaining_repair_failures: frozenset[str] = frozenset()
