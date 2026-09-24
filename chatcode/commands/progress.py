"""Console rendering for project-index progress events."""
from __future__ import annotations

import sys

from ..indexing.index_manager import IndexProgress


class ConsoleIndexReporter:
    def __init__(self) -> None:
        self._progress_line = False
        self._initial = False

    def _finish_progress_line(self) -> None:
        if self._progress_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_line = False

    def __call__(self, event: IndexProgress) -> None:
        if event.phase == "index_mode" and event.status == "selected":
            print(f"Index mode: {event.reason}.")
        elif event.phase == "index" and event.status == "initial":
            self._initial = True
            print("Project map not found. Building initial project index...")
        elif event.phase == "index" and event.status == "forced":
            self._initial = True
            print("Cached project map discarded. Rebuilding from source...")
        elif event.phase == "scan" and event.status == "started":
            print("Scanning project files...")
        elif event.phase == "index" and event.status == "updating":
            print("Project changes detected. Updating project index...")
        elif event.phase == "index" and event.status == "up_to_date":
            print("Project index up to date.")
        elif event.phase == "static" and event.status == "complete":
            print(f"Static analysis complete: {event.total} source files.")
        elif event.phase == "static" and event.status == "saved":
            print("Project map saved.")
            if self._initial:
                print(f"Project index created: {event.total} source files.")
            else:
                print(
                    "Project index updated: "
                    f"{event.changed} changed, {event.added} new, "
                    f"{event.deleted} removed."
                )
        elif event.phase == "semantic" and event.status == "started":
            print(
                "Semantic cache: "
                f"{event.reused} reused, {event.invalidated} invalidated, "
                f"{event.pending} pending."
            )
            print(f"Running semantic analysis with {event.model}...")
        elif event.phase == "semantic" and event.status == "progress":
            width = 20
            filled = round(width * event.completed / event.total) if event.total else width
            bar = "█" * filled + "░" * (width - filled)
            details = (
                f"Semantic indexing [{bar}] "
                f"{event.completed}/{event.total}"
            )
            if event.current_file:
                details += f" | {event.current_file}"
            if event.eta_seconds is not None:
                if event.eta_seconds < 60:
                    details += f" | ~{round(event.eta_seconds)}s remaining"
                else:
                    details += f" | ~{round(event.eta_seconds / 60)}m remaining"
            sys.stdout.write("\r" + details.ljust(120))
            sys.stdout.flush()
            self._progress_line = True
        elif event.phase == "semantic" and event.status == "complete":
            self._finish_progress_line()
            if self._initial:
                print(f"Semantic analysis complete: {event.completed} files.")
            else:
                print(f"Semantic analysis updated: {event.processed} files.")
            if event.failed:
                labels = {
                    "timeout": "timeout",
                    "ollama_not_found": "Ollama not found",
                    "model_not_found": "model not found",
                    "ollama_process_error": "Ollama process error",
                    "empty_response": "empty response",
                    "invalid_json": "invalid JSON",
                    "schema_validation_error": "schema validation",
                    "unexpected_exception": "unexpected exception",
                }
                print("Qwen semantic analysis failures:")
                for reason, count in event.failure_counts:
                    print(f"  {labels.get(reason, reason)}: {count}")
                print("Example semantic failures:")
                for example in event.failure_examples:
                    print(f"  file: {example.file}")
                    print(f"  reason: {example.reason}")
                    print(f"  error: {example.error}")
                    if example.response:
                        response = example.response.replace("\n", "\\n")
                        print(f"  response: {response}")
                print("Continuing with static index.")
            if event.stopped_early:
                print(
                    "Semantic circuit breaker stopped indexing after "
                    f"{event.processed} files ({event.reason}); "
                    f"{event.remaining} files remain pending."
                )
                print("Continuing with static retrieval for this run.")
        elif event.phase == "semantic_preflight" and event.status == "started":
            print(f"Checking semantic model {event.model}...")
        elif event.phase == "semantic_preflight" and event.status == "complete":
            print("Semantic model check passed.")
        elif event.phase == "semantic_preflight" and event.status == "failed":
            print(f"Semantic model check failed ({event.reason}): {event.error}")
            print("Skipping AI semantic indexing.")
            print("Continuing with static retrieval.")
        elif event.phase == "semantic" and event.status == "interrupted":
            self._finish_progress_line()
            print("Semantic indexing interrupted by user.")
            print(
                f"Progress saved: {event.completed}/{event.total} files completed."
            )
