"""Read, write, and index compact atomic event packages."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..composition.jsonutil import canonical_json_bytes, jsonable, pretty_json_text


EVENT_PACKAGE_SCHEMA = "event_package.v1"
PACKAGE_JSON = "package.json"
EVENTS_JSONL = "events.jsonl"
COMPLETED_SHARDS_DIR = "_meta/completed_shards"
INDEX_DIR = "_index"
COMPLETED_PARQUET = "completed.parquet"
COMPACTION_STATE_JSON = "compaction_state.json"


@dataclass(frozen=True)
class EventPackage:
    """Loaded view of one event package directory."""

    path: Path
    package: dict

    @property
    def events_path(self) -> Path:
        return self.path / self.package.get("files", {}).get("events", EVENTS_JSONL)

    @property
    def event_package_fingerprint(self) -> str:
        return str(self.package.get("event_package_fingerprint", ""))

    @property
    def input_fingerprint(self) -> str:
        return str(self.package.get("input_fingerprint", ""))

    @property
    def event_config_fingerprint(self) -> str:
        return str(self.package.get("event_config_fingerprint", ""))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def event_package_complete(package_dir: str | Path) -> bool:
    """Return True when a package has a complete manifest and events file."""
    try:
        load_event_package(package_dir)
        return True
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        return False


def load_event_package(package_dir: str | Path) -> EventPackage:
    path = Path(package_dir)
    package_path = path / PACKAGE_JSON
    if not package_path.is_file():
        raise FileNotFoundError(f"event package manifest not found: {package_path}")
    with package_path.open("r", encoding="utf-8") as f:
        package = json.load(f)
    if package.get("schema") != EVENT_PACKAGE_SCHEMA:
        raise ValueError(f"Unsupported event package schema: {package.get('schema')!r}")
    if package.get("status") != "complete":
        raise ValueError(f"Event package is not complete: {package_path}")
    events_rel = package.get("files", {}).get("events", EVENTS_JSONL)
    events_path = path / events_rel
    if not events_path.is_file():
        raise FileNotFoundError(f"event package events file not found: {events_path}")
    return EventPackage(path=path, package=package)


def event_config_fingerprint(
    *,
    producer_configs: Mapping,
    event_policy: Mapping,
    normalizer_version: str,
) -> str:
    payload = {
        "producer_configs": producer_configs,
        "event_policy": event_policy,
        "normalizer_version": normalizer_version,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def input_fingerprint(
    *,
    source_artifacts: Mapping,
    event_config_fingerprint_value: str,
) -> str:
    payload = {
        "source_artifacts": _fingerprint_payload(source_artifacts),
        "event_config_fingerprint": event_config_fingerprint_value,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def event_package_fingerprint(package_payload: Mapping, events: Sequence[Mapping]) -> str:
    payload = {
        "package": _fingerprint_payload(package_payload),
        "events": [_fingerprint_payload(event) for event in events],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_event_package_payload(
    *,
    session_id: str,
    archive_id: str,
    date: str,
    audio: Mapping,
    source_artifacts: Mapping,
    producer_runs: Sequence[Mapping],
    event_policy: Mapping,
    producer_configs: Mapping,
    event_config_fingerprint_value: str,
    input_fingerprint_value: str,
    events: Sequence[Mapping],
    created_at: str | None = None,
) -> dict:
    """Build a deterministic event package manifest."""
    counts = _event_counts(events)
    payload = {
        "schema": EVENT_PACKAGE_SCHEMA,
        "status": "complete",
        "session_id": str(session_id),
        "archive_id": str(archive_id),
        "date": str(date or ""),
        "created_at": created_at or utc_now_iso(),
        "audio": dict(audio),
        "source_artifacts": dict(source_artifacts),
        "producer_runs": list(producer_runs),
        "producer_configs": dict(producer_configs),
        "event_policy": dict(event_policy),
        "event_config_fingerprint": event_config_fingerprint_value,
        "input_fingerprint": input_fingerprint_value,
        "counts": counts,
        "files": {"events": EVENTS_JSONL},
    }
    payload["event_package_fingerprint"] = event_package_fingerprint(payload, events)
    return jsonable(payload)


def write_event_package(
    *,
    package_dir: str | Path,
    package_payload: Mapping,
    events: Sequence[Mapping],
) -> Path:
    """Atomically write events first, then the complete package manifest."""
    path = Path(package_dir)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(path / EVENTS_JSONL, events)
    _atomic_write_text(path / PACKAGE_JSON, pretty_json_text(package_payload))
    return path


def completion_row(package_dir: str | Path, package_payload: Mapping) -> dict:
    payload = dict(package_payload)
    audio = dict(payload.get("audio", {}))
    counts = dict(payload.get("counts", {}))
    return {
        "date": payload.get("date", ""),
        "session_id": payload["session_id"],
        "archive_id": payload["archive_id"],
        "package_path": str(Path(package_dir).resolve()),
        "event_config_fingerprint": payload.get("event_config_fingerprint", ""),
        "input_fingerprint": payload.get("input_fingerprint", ""),
        "event_package_fingerprint": payload.get("event_package_fingerprint", ""),
        "audio_sha256": audio.get("sha256", ""),
        "event_count": int(counts.get("events_total", 0)),
        "counts_by_task": counts.get("by_task", {}),
        "counts_by_label": counts.get("by_label", {}),
        "completed_at": payload.get("created_at") or utc_now_iso(),
    }


def append_completion_row(
    *,
    events_base: str | Path,
    worker_id: str,
    package_dir: str | Path,
    package_payload: Mapping,
) -> Path:
    """Append to a process-unique completion shard."""
    row = completion_row(package_dir, package_payload)
    date = str(row.get("date") or "unknown")
    shard = Path(events_base) / COMPLETED_SHARDS_DIR / f"date={_safe_part(date)}" / f"{worker_id}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with shard.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")
    return shard


def compact_completion_index(events_base: str | Path) -> Path:
    """Incrementally compact completion shards into ``_index/completed.parquet``."""
    base = Path(events_base)
    index_dir = base / INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = index_dir / COMPLETED_PARQUET
    state_path = index_dir / COMPACTION_STATE_JSON

    state = _read_json(state_path) if state_path.is_file() else {"offsets": {}}
    offsets: dict[str, int] = {str(k): int(v) for k, v in state.get("offsets", {}).items()}

    rows = _read_existing_completion_rows(parquet_path)
    new_offsets = dict(offsets)
    for shard in sorted((base / COMPLETED_SHARDS_DIR).rglob("*.jsonl")):
        key = str(shard.relative_to(base))
        new_rows, next_offset = _read_newline_terminated_rows(shard, offsets.get(key, 0))
        rows.extend(new_rows)
        new_offsets[key] = next_offset

    rows = _dedupe_rows(rows)
    _atomic_write_parquet(parquet_path, rows)
    _atomic_write_text(state_path, pretty_json_text({"offsets": new_offsets}))
    return parquet_path


def completion_rows_from_index_and_shards(events_base: str | Path) -> list[dict]:
    """Return compacted rows plus uncompacted shard deltas."""
    base = Path(events_base)
    index_dir = base / INDEX_DIR
    parquet_path = index_dir / COMPLETED_PARQUET
    state_path = index_dir / COMPACTION_STATE_JSON
    rows = _read_existing_completion_rows(parquet_path)
    state = _read_json(state_path) if state_path.is_file() else {"offsets": {}}
    offsets = {str(k): int(v) for k, v in state.get("offsets", {}).items()}
    for shard in sorted((base / COMPLETED_SHARDS_DIR).rglob("*.jsonl")):
        key = str(shard.relative_to(base))
        new_rows, _ = _read_newline_terminated_rows(shard, offsets.get(key, 0))
        rows.extend(new_rows)
    return _dedupe_rows(rows)


def _event_counts(events: Sequence[Mapping]) -> dict:
    by_task: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for event in events:
        task = str(event.get("task", ""))
        label = str(event.get("label", ""))
        by_task[task] = by_task.get(task, 0) + 1
        by_label[label] = by_label.get(label, 0) + 1
    return {
        "events_total": len(events),
        "by_task": dict(sorted(by_task.items())),
        "by_label": dict(sorted(by_label.items())),
    }


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    lines = [
        json.dumps(jsonable(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in rows
    ]
    _atomic_write_text(path, "".join(line + "\n" for line in lines))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f".tmp.{os.getpid()}",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_write_parquet(path: Path, rows: Sequence[Mapping]) -> None:
    table = pa.Table.from_pylist([jsonable(row) for row in rows])
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f".tmp.{os.getpid()}",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_existing_completion_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    table = pq.read_table(path)
    return [dict(row) for row in table.to_pylist()]


def _read_newline_terminated_rows(path: Path, offset: int) -> tuple[list[dict], int]:
    try:
        with path.open("rb") as f:
            f.seek(max(0, int(offset)))
            data = f.read()
    except OSError:
        return [], offset
    if not data:
        return [], offset
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        return [], offset
    complete = data[: last_newline + 1]
    next_offset = offset + last_newline + 1
    rows: list[dict] = []
    for line in complete.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, next_offset


def _dedupe_rows(rows: Iterable[Mapping]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("session_id", "")), str(row.get("archive_id", "")))
        if not key[0] or not key[1]:
            continue
        by_key[key] = dict(row)
    return [by_key[key] for key in sorted(by_key)]


def _fingerprint_payload(value):
    if isinstance(value, Mapping):
        return {
            str(k): _fingerprint_payload(v)
            for k, v in value.items()
            if k not in {"created_at", "completed_at", "package_path", "manifest_path", "path"}
        }
    if isinstance(value, list):
        return [_fingerprint_payload(item) for item in value]
    return jsonable(value)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_part(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value) or "unknown"
