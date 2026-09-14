from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath

from .workspace import (
    get_applied_history_dir,
    get_history_dir,
    get_undone_history_dir,
)


class HistoryError(RuntimeError):
    pass


def _now_iso() -> str:
    return (
        datetime.now()
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _new_history_id() -> str:
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )


def _metadata_file(
    entry_dir: Path,
) -> Path:
    return entry_dir / "metadata.json"


def _save_metadata(
    entry_dir: Path,
    metadata: dict,
) -> None:
    _metadata_file(entry_dir).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_metadata(
    entry_dir: Path,
) -> dict:
    try:
        return json.loads(
            _metadata_file(
                entry_dir
            ).read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise HistoryError(
            "Kunde inte läsa "
            f"historiken: {entry_dir}"
        ) from exc


def _repo_path(
    repo: Path,
    relative_path: str,
) -> Path:
    posix_path = PurePosixPath(
        relative_path
    )

    return repo.joinpath(
        *posix_path.parts
    )


def _snapshot_path(
    root: Path,
    relative_path: str,
) -> Path:
    posix_path = PurePosixPath(
        relative_path
    )

    return root.joinpath(
        *posix_path.parts
    )


def _capture_file(
    repo: Path,
    relative_path: str,
    destination_root: Path,
) -> bool:
    source = _repo_path(
        repo,
        relative_path,
    )

    destination = _snapshot_path(
        destination_root,
        relative_path,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = (
        source.exists()
        and source.is_file()
    )

    if exists:
        shutil.copy2(
            source,
            destination,
        )
    else:
        destination.write_bytes(b"")

    return exists


def begin_history_entry(
    repo: Path,
    patch_text: str,
    paths: set[str],
) -> Path:
    history_id = _new_history_id()

    pending_root = (
        get_history_dir(repo)
        / ".pending"
    )

    pending_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    entry_dir = (
        pending_root / history_id
    )

    entry_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    patch_file = (
        entry_dir / "patch.diff"
    )

    patch_file.write_text(
        patch_text,
        encoding="utf-8",
        newline="\n",
    )

    before_dir = (
        entry_dir / "before"
    )

    before_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files: list[dict] = []

    for relative_path in sorted(paths):
        before_exists = _capture_file(
            repo,
            relative_path,
            before_dir,
        )

        files.append({
            "path": relative_path,
            "before_exists": before_exists,
            "after_exists": None,
        })

    metadata = {
        "id": history_id,
        "status": "PENDING",
        "applied_at": _now_iso(),
        "undone_at": None,
        "test_status": "NOT_RUN",
        "test_command": None,
        "test_returncode": None,
        "test_duration_seconds": None,
        "files": files,
    }

    _save_metadata(
        entry_dir,
        metadata,
    )

    return entry_dir


def finalize_history_entry(
    repo: Path,
    pending_entry: Path,
) -> Path:
    metadata = _load_metadata(
        pending_entry
    )

    after_dir = (
        pending_entry / "after"
    )

    after_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_info in metadata["files"]:
        relative_path = file_info["path"]

        after_exists = _capture_file(
            repo,
            relative_path,
            after_dir,
        )

        file_info[
            "after_exists"
        ] = after_exists

    metadata["status"] = "APPLIED"
    metadata["completed_at"] = _now_iso()

    _save_metadata(
        pending_entry,
        metadata,
    )

    destination = (
        get_applied_history_dir(repo)
        / pending_entry.name
    )

    try:
        shutil.move(
            str(pending_entry),
            str(destination),
        )
    except OSError as exc:
        raise HistoryError(
            "Patchen applicerades, men "
            "historiken kunde inte "
            "slutföras."
        ) from exc

    return destination


def discard_pending_entry(
    entry_dir: Path,
) -> None:
    try:
        shutil.rmtree(
            entry_dir,
        )
    except OSError:
        pass


def get_history_patch_file(
    entry: Path,
) -> Path:
    if entry.is_dir():
        return entry / "patch.diff"

    return entry


def get_latest_applied_entry(
    repo: Path,
) -> Path:
    applied_dir = (
        get_applied_history_dir(repo)
    )

    entries = [
        path
        for path in applied_dir.iterdir()
        if (
            path.is_dir()
            or path.suffix.lower()
            == ".diff"
        )
    ]

    if not entries:
        raise HistoryError(
            "Det finns ingen tidigare "
            "ChatCode patch att backa."
        )

    entries.sort(
        key=lambda path: path.name,
        reverse=True,
    )

    return entries[0]


def move_entry_to_undone(
    repo: Path,
    entry: Path,
) -> Path:
    destination = (
        get_undone_history_dir(repo)
        / entry.name
    )

    if entry.is_dir():
        metadata = _load_metadata(
            entry
        )

        metadata["status"] = "UNDONE"
        metadata["undone_at"] = (
            _now_iso()
        )

        _save_metadata(
            entry,
            metadata,
        )

    try:
        shutil.move(
            str(entry),
            str(destination),
        )
    except OSError as exc:
        raise HistoryError(
            "Ändringen backades, men "
            "historiken kunde inte "
            "flyttas till undone."
        ) from exc

    return destination


def update_history_test_result(
    entry: Path,
    status: str,
    command: str | None = None,
    returncode: int | None = None,
    duration_seconds: float | None = None,
) -> None:
    if not entry.is_dir():
        return

    metadata_file = _metadata_file(
        entry
    )

    if not metadata_file.exists():
        return

    metadata = _load_metadata(
        entry
    )

    metadata["test_status"] = status
    metadata["test_command"] = command
    metadata[
        "test_returncode"
    ] = returncode
    metadata[
        "test_duration_seconds"
    ] = duration_seconds

    _save_metadata(
        entry,
        metadata,
    )


def _extract_legacy_paths(
    patch_file: Path,
) -> list[str]:
    try:
        content = patch_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return []

    paths: set[str] = set()

    for line in content.splitlines():
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

    return sorted(paths)


def _legacy_timestamp(
    name: str,
) -> str:
    base = Path(name).stem

    try:
        parsed = datetime.strptime(
            base[:15],
            "%Y%m%d_%H%M%S",
        )

        return parsed.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return base


def list_history_records(
    repo: Path,
) -> list[dict]:
    records: list[dict] = []

    locations = [
        (
            "APPLIED",
            get_applied_history_dir(repo),
        ),
        (
            "UNDONE",
            get_undone_history_dir(repo),
        ),
    ]

    for status, directory in locations:
        for entry in directory.iterdir():
            if entry.name.startswith("."):
                continue

            if entry.is_dir():
                try:
                    metadata = (
                        _load_metadata(entry)
                    )
                except HistoryError:
                    continue

                records.append({
                    **metadata,
                    "_entry_path": str(entry),
                })

            elif (
                entry.suffix.lower()
                == ".diff"
            ):
                records.append({
                    "id": entry.stem,
                    "status": status,
                    "applied_at": None,
                    "test_status": "UNKNOWN",
                    "files": [
                        {
                            "path": path,
                        }
                        for path
                        in _extract_legacy_paths(
                            entry
                        )
                    ],
                    "legacy": True,
                    "_entry_path": str(entry),
                })

    records.sort(
        key=lambda record: record["id"],
        reverse=True,
    )

    return records


def format_history(
    repo: Path,
) -> str:
    records = list_history_records(
        repo
    )

    if not records:
        return (
            "No ChatCode history found."
        )

    lines = [
        "ChatCode history",
        "",
    ]

    for index, record in enumerate(
        records,
        start=1,
    ):
        applied_at = record.get(
            "applied_at"
        )

        if applied_at:
            try:
                timestamp = (
                    datetime.fromisoformat(
                        applied_at
                    ).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )
            except ValueError:
                timestamp = applied_at
        else:
            timestamp = _legacy_timestamp(
                record["id"]
            )

        status = record.get(
            "status",
            "UNKNOWN",
        )

        test_status = record.get(
            "test_status",
            "UNKNOWN",
        )

        lines.append(
            f"{index}. {timestamp}"
        )

        lines.append(
            f"   {status}"
            f" | Tests: {test_status}"
        )

        files = record.get(
            "files",
            [],
        )

        if files:
            for file_info in files:
                lines.append(
                    "   "
                    + file_info["path"]
                )
        else:
            lines.append(
                "   (no files)"
            )

        lines.append("")

    return "\n".join(lines).rstrip()


def _open_code_diff(
    before: Path,
    after: Path,
) -> None:
    code = shutil.which("code")

    if code is None:
        raise HistoryError(
            "VS Code kommandot 'code' "
            "kunde inte hittas i PATH."
        )

    if os.name == "nt":
        command_line = (
            subprocess.list2cmdline([
                "code",
                "--diff",
                str(before),
                str(after),
            ])
        )

        subprocess.Popen([
            "cmd.exe",
            "/d",
            "/s",
            "/c",
            command_line,
        ])

    else:
        subprocess.Popen([
            code,
            "--diff",
            str(before),
            str(after),
        ])


def open_history_review(
    entry: Path,
) -> None:
    if not entry.is_dir():
        raise HistoryError(
            "Den här äldre historikposten "
            "saknar before/after snapshots "
            "och kan därför inte öppnas "
            "som review."
        )

    metadata = _load_metadata(
        entry
    )

    files = metadata.get(
        "files",
        [],
    )

    if not files:
        raise HistoryError(
            "Historikposten innehåller "
            "inga filer att granska."
        )

    before_root = (
        entry / "before"
    )

    after_root = (
        entry / "after"
    )

    for file_info in files:
        relative_path = (
            file_info["path"]
        )

        before = _snapshot_path(
            before_root,
            relative_path,
        )

        after = _snapshot_path(
            after_root,
            relative_path,
        )

        if not before.exists():
            before.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            before.write_bytes(b"")

        if not after.exists():
            after.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            after.write_bytes(b"")

        _open_code_diff(
            before.resolve(),
            after.resolve(),
        )


def review_history_by_index(
    repo: Path,
    index: int,
) -> dict:
    records = list_history_records(
        repo
    )

    if not records:
        raise HistoryError(
            "Det finns ingen ChatCode "
            "historik att granska."
        )

    if index < 1 or index > len(records):
        raise HistoryError(
            "Historiknummer "
            f"{index} finns inte."
        )

    record = records[index - 1]

    entry = Path(
        record["_entry_path"]
    )

    open_history_review(
        entry
    )

    return record