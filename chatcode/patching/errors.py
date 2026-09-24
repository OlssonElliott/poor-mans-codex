"""Patch workflow exceptions."""
from __future__ import annotations

from pathlib import Path


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
