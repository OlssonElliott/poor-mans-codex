"""Background full-suite orchestration with explicit compatibility hooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from .baseline import RepositorySnapshot
from ..models import ApplyResult
from . import background_tests


@dataclass(frozen=True)
class BackgroundHooks:
    capture_snapshot_fn: Callable[[Path], RepositorySnapshot]
    save_baseline_fn: Callable
    get_status_fn: Callable[[Path], dict | None]
    cleanup_fn: Callable[..., None]
    snapshot_payload_fn: Callable[[RepositorySnapshot], dict]
    popen_fn: Callable


def complete_full_suite(
    hooks: BackgroundHooks,
    repo_name: str,
    snapshot_payload: dict,
    run_id: str,
    application_payload: dict | None = None,
) -> None:
    return background_tests.complete_full_suite(
        repo_name,
        snapshot_payload,
        run_id,
        application_payload,
        capture_snapshot_fn=hooks.capture_snapshot_fn,
        save_baseline_fn=hooks.save_baseline_fn,
        get_status_fn=hooks.get_status_fn,
        cleanup_fn=hooks.cleanup_fn,
    )


def complete_full_suite_from_status(
    repo_name: str,
    run_id: str,
    *,
    complete_fn: Callable,
) -> None:
    return background_tests.complete_full_suite_from_status(
        repo_name,
        run_id,
        complete_fn=complete_fn,
    )


def start_full_suite(
    hooks: BackgroundHooks,
    repo: Path,
    snapshot: RepositorySnapshot,
    application: ApplyResult | None = None,
) -> None:
    return background_tests.start_full_suite(
        repo,
        snapshot,
        application,
        snapshot_payload_fn=hooks.snapshot_payload_fn,
        cleanup_fn=hooks.cleanup_fn,
        popen=hooks.popen_fn,
    )

def detached_full_suite_from_status(
    repo_name: str,
    run_id: str,
) -> None:
    """Run the detached worker without importing the compatibility patch facade."""
    from . import baseline

    def save_baseline(repo: Path, snapshot: RepositorySnapshot, result) -> None:
        return baseline.save_verified_baseline(
            repo,
            snapshot,
            result,
            test_config_identity_fn=baseline.test_config_identity,
        )

    hooks = BackgroundHooks(
        capture_snapshot_fn=baseline.capture_repository_snapshot,
        save_baseline_fn=save_baseline,
        get_status_fn=background_tests.get_status,
        cleanup_fn=background_tests.cleanup_artifacts,
        snapshot_payload_fn=baseline.snapshot_payload,
        popen_fn=background_tests.subprocess.Popen,
    )

    return complete_full_suite_from_status(
        repo_name,
        run_id,
        complete_fn=lambda name, payload, current_run_id, application_payload=None: (
            complete_full_suite(
                hooks,
                name,
                payload,
                current_run_id,
                application_payload,
            )
        ),
    )
