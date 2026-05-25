"""Core populate logic for the session event store."""
from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..event_packages.package import (
    completion_rows_from_index_and_shards,
    load_event_package,
)
from ..orchestration.manifest import ArchiveEntity, load_manifest
from .models import SessionEventsRecord
from .schema import PARQUET_SCHEMA, build_session_record, session_fingerprint

LOGGER = logging.getLogger(__name__)
LOCK_FILENAME = ".populate.lock"
LOCK_TIMEOUT_MINUTES = 60.0


@dataclass(frozen=True)
class ReadySession:
    """A session whose archives are all complete per the completion index."""

    session_id: str
    archive_ids: list[str]
    dates: list[str]
    expected_fingerprint: str


@dataclass
class PopulateSummary:
    """Mutable accumulator for the populate run summary."""

    dates_processed: int = 0
    dates_skipped_no_delta: int = 0
    sessions_written: int = 0
    sessions_updated: int = 0
    sessions_unchanged: int = 0
    sessions_incomplete: int = 0
    sessions_load_failed: int = 0

    def as_dict(self) -> dict:
        return {
            "dates_processed": self.dates_processed,
            "dates_skipped_no_delta": self.dates_skipped_no_delta,
            "sessions_written": self.sessions_written,
            "sessions_updated": self.sessions_updated,
            "sessions_unchanged": self.sessions_unchanged,
            "sessions_incomplete": self.sessions_incomplete,
            "sessions_load_failed": self.sessions_load_failed,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def populate_session_store(
    *,
    manifest_path: str | Path,
    events_output: str | Path,
    store_output: str | Path,
    dates: Sequence[str] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """Populate date-partitioned session event parquet files.

    Returns a summary dict with counts of written, updated, unchanged,
    incomplete, and failed sessions.
    """
    events_output = Path(events_output)
    store_output = Path(store_output)
    store_output.mkdir(parents=True, exist_ok=True)

    lock_path = _acquire_lock(store_output)
    try:
        return _run_populate(
            manifest_path=Path(manifest_path),
            events_output=events_output,
            store_output=store_output,
            dates=set(dates) if dates else None,
            force=force,
            verbose=verbose,
        )
    finally:
        _release_lock(lock_path)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _run_populate(
    *,
    manifest_path: Path,
    events_output: Path,
    store_output: Path,
    dates: set[str] | None,
    force: bool,
    verbose: bool,
) -> dict:
    summary = PopulateSummary()

    entities = load_manifest(manifest_path)
    LOGGER.info("Loaded manifest: %d archive entities", len(entities))

    completion_state = _load_completion_state(events_output)
    LOGGER.info("Loaded completion state: %d completed archives", len(completion_state))

    session_map = _group_manifest_by_session(entities)
    ready_sessions = _identify_ready_sessions(session_map, completion_state)
    summary.sessions_incomplete = len(session_map) - len(ready_sessions)
    LOGGER.info(
        "Sessions: %d total, %d ready, %d incomplete",
        len(session_map),
        len(ready_sessions),
        summary.sessions_incomplete,
    )

    date_to_sessions = _group_by_date(ready_sessions, dates)
    if dates:
        LOGGER.info("Date filter active: processing %d dates", len(date_to_sessions))

    for date, session_ids in sorted(date_to_sessions.items()):
        result = _process_date_partition(
            date=date,
            session_ids=session_ids,
            ready_sessions=ready_sessions,
            events_output=events_output,
            store_output=store_output,
            force=force,
            verbose=verbose,
        )
        summary.dates_processed += result.written > 0 or result.updated > 0
        summary.dates_skipped_no_delta += result.written == 0 and result.updated == 0
        summary.sessions_written += result.written
        summary.sessions_updated += result.updated
        summary.sessions_unchanged += result.unchanged
        summary.sessions_load_failed += result.load_failed

    LOGGER.info("Populate complete: %s", json.dumps(summary.as_dict(), sort_keys=True))
    return summary.as_dict()


def _load_completion_state(events_output: Path) -> dict[tuple[str, str], str]:
    """Map (session_id, archive_id) -> event_package_fingerprint from completion index/shards."""
    try:
        rows = completion_rows_from_index_and_shards(events_output)
    except Exception:
        LOGGER.warning("Failed to read completion index; treating as empty", exc_info=True)
        return {}
    return {
        (str(row["session_id"]), str(row["archive_id"])): str(row.get("event_package_fingerprint", ""))
        for row in rows
    }


def _group_manifest_by_session(
    entities: list[ArchiveEntity],
) -> dict[str, list[ArchiveEntity]]:
    """Group manifest entities by session_id."""
    by_session: dict[str, list[ArchiveEntity]] = {}
    for entity in entities:
        by_session.setdefault(entity.session_id, []).append(entity)
    return by_session


def _identify_ready_sessions(
    session_map: dict[str, list[ArchiveEntity]],
    completion_state: dict[tuple[str, str], str],
) -> dict[str, ReadySession]:
    """Identify sessions where all archives have completion rows."""
    ready: dict[str, ReadySession] = {}
    for session_id, archives in session_map.items():
        fingerprints: list[tuple[str, str]] = []
        all_complete = True
        for entity in archives:
            key = (entity.session_id, entity.archive_id)
            fp = completion_state.get(key)
            if fp is None:
                all_complete = False
                break
            fingerprints.append((entity.archive_id, fp))
        if not all_complete:
            continue
        dates = sorted({e.date for e in archives if e.date})
        ready[session_id] = ReadySession(
            session_id=session_id,
            archive_ids=[e.archive_id for e in archives],
            dates=dates,
            expected_fingerprint=session_fingerprint(fingerprints),
        )
    return ready


def _group_by_date(
    ready_sessions: dict[str, ReadySession],
    date_filter: set[str] | None,
) -> dict[str, list[str]]:
    """Map each date to the list of ready session_ids that belong to it."""
    date_to_sessions: dict[str, list[str]] = {}
    for session_id, info in ready_sessions.items():
        for date in info.dates:
            if date_filter and date not in date_filter:
                continue
            date_to_sessions.setdefault(date, []).append(session_id)
    return date_to_sessions


@dataclass
class _DateResult:
    written: int = 0
    updated: int = 0
    unchanged: int = 0
    load_failed: int = 0


def _process_date_partition(
    *,
    date: str,
    session_ids: list[str],
    ready_sessions: dict[str, ReadySession],
    events_output: Path,
    store_output: Path,
    force: bool,
    verbose: bool,
) -> _DateResult:
    """Process one date partition: compute delta, load events, write if needed."""
    result = _DateResult()
    parquet_path = store_output / f"{date}.parquet"

    existing_fingerprints: dict[str, str] = {}
    existing_rows: list[dict] = []
    if parquet_path.is_file():
        table = pq.read_table(parquet_path)
        for i in range(table.num_rows):
            sid = table.column("session_id")[i].as_py()
            fp = table.column("session_fingerprint")[i].as_py()
            existing_fingerprints[sid] = fp
            existing_rows.append({
                col: table.column(col)[i].as_py()
                for col in PARQUET_SCHEMA.names
            })

    sessions_to_load: list[str] = []
    for session_id in session_ids:
        info = ready_sessions[session_id]
        stored_fp = existing_fingerprints.get(session_id)
        if force:
            sessions_to_load.append(session_id)
        elif stored_fp is None:
            sessions_to_load.append(session_id)
        elif stored_fp != info.expected_fingerprint:
            sessions_to_load.append(session_id)
        else:
            result.unchanged += 1
            if verbose:
                LOGGER.debug("session_id=%s status=unchanged", session_id)

    if not sessions_to_load:
        return result

    new_rows: dict[str, dict] = {}
    for session_id in sessions_to_load:
        info = ready_sessions[session_id]
        row = _load_and_build_row(
            session_id=session_id,
            archive_ids=info.archive_ids,
            date=date,
            events_output=events_output,
        )
        if row is None:
            result.load_failed += 1
            if verbose:
                LOGGER.debug("session_id=%s status=load_failed", session_id)
            continue

        is_update = session_id in existing_fingerprints
        if is_update:
            result.updated += 1
            if verbose:
                LOGGER.debug("session_id=%s status=updated reason=fingerprint_changed", session_id)
        else:
            result.written += 1
            if verbose:
                LOGGER.debug("session_id=%s status=written reason=new", session_id)
        new_rows[session_id] = row

    if not new_rows:
        return result

    merged = [row for row in existing_rows if row["session_id"] not in new_rows]
    merged.extend(new_rows.values())
    merged.sort(key=lambda r: r["session_id"])

    _atomic_write_parquet(parquet_path, merged)
    return result


def _load_and_build_row(
    *,
    session_id: str,
    archive_ids: list[str],
    date: str,
    events_output: Path,
) -> dict | None:
    """Load event packages for all archives and build a parquet row dict.

    Returns None if any archive package fails to load.
    """
    archive_events: list[tuple[str, list[dict]]] = []
    archive_fingerprints: list[tuple[str, str]] = []

    for archive_id in archive_ids:
        package_dir = events_output / session_id / archive_id
        try:
            package = load_event_package(package_dir)
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "Failed to load package session_id=%s archive_id=%s: %s",
                session_id, archive_id, exc,
            )
            return None

        events_path = package.events_path
        try:
            raw_events = _read_events_jsonl(events_path)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "Failed to read events session_id=%s archive_id=%s path=%s: %s",
                session_id, archive_id, events_path, exc,
            )
            return None

        archive_events.append((archive_id, raw_events))
        archive_fingerprints.append((archive_id, package.event_package_fingerprint))

    record = build_session_record(
        session_id=session_id,
        date=date,
        archive_events=archive_events,
    )
    fp = session_fingerprint(archive_fingerprints)
    total_events = sum(len(item.events_data) for item in record.event_items)

    data_json = json.dumps(
        record.model_dump(exclude_none=True),
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "session_id": session_id,
        "date": date,
        "archive_count": len(archive_ids),
        "event_count": total_events,
        "session_fingerprint": fp,
        "data": data_json,
    }


def _read_events_jsonl(path: Path) -> list[dict]:
    """Read all events from a JSONL file."""
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Atomic parquet writes
# ---------------------------------------------------------------------------


def _atomic_write_parquet(path: Path, rows: list[dict]) -> None:
    """Write rows to a parquet file atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f".tmp.{os.getpid()}",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Lock management
# ---------------------------------------------------------------------------


def _acquire_lock(store_output: Path) -> Path:
    """Acquire a global populate lock. Raises if already locked."""
    lock_path = store_output / LOCK_FILENAME
    if lock_path.is_file():
        _try_reclaim_stale_lock(lock_path)
    lock_data = {
        "pid": os.getpid(),
        "hostname": _hostname(),
        "started_at": time.time(),
    }
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump(lock_data, f)
    except FileExistsError:
        existing = _read_lock(lock_path)
        raise RuntimeError(
            f"Session store is locked by another process: "
            f"pid={existing.get('pid')} host={existing.get('hostname')} "
            f"since={existing.get('started_at')}"
        ) from None
    return lock_path


def _release_lock(lock_path: Path) -> None:
    """Release the populate lock."""
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def _try_reclaim_stale_lock(lock_path: Path) -> None:
    """Reclaim a lock if it is stale (time-based or same-host dead PID)."""
    lock_data = _read_lock(lock_path)
    if not lock_data:
        lock_path.unlink(missing_ok=True)
        return

    age_minutes = (time.time() - lock_data.get("started_at", 0)) / 60.0
    lock_host = lock_data.get("hostname", "")
    lock_pid = lock_data.get("pid", 0)

    if age_minutes >= LOCK_TIMEOUT_MINUTES:
        LOGGER.warning(
            "Reclaiming stale lock (age=%.1f min > %.1f min timeout)",
            age_minutes, LOCK_TIMEOUT_MINUTES,
        )
        lock_path.unlink(missing_ok=True)
        return

    if lock_host == _hostname() and lock_pid and not _pid_alive(lock_pid):
        LOGGER.warning(
            "Reclaiming lock from dead process pid=%d on this host", lock_pid,
        )
        lock_path.unlink(missing_ok=True)
        return


def _read_lock(lock_path: Path) -> dict:
    try:
        with lock_path.open("r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _hostname() -> str:
    return os.environ.get("HOSTNAME", "") or os.uname().nodename


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return True
