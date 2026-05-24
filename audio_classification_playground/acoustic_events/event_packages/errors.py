"""Structured error records for event package workers."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
import uuid


PACKAGING_ERRORS_DIR = "_meta/packaging_errors"
PERMANENT_ERROR_NAMES = (
    "ValueError",
    "KeyError",
    "FileNotFoundError",
    "JSONDecodeError",
)


def append_packaging_error(
    events_base: str | Path,
    *,
    session_id: str,
    archive_id: str,
    error: Exception,
    source_artifacts: dict | None = None,
    is_permanent: bool | None = None,
) -> Path:
    """Persist one packaging failure as a UUID JSON file."""
    if is_permanent is None:
        is_permanent = _is_permanent_packaging_error(error)
    payload = {
        "session_id": session_id,
        "archive_id": archive_id,
        "error_type": type(error).__name__,
        "detail": str(error)[:2000],
        "is_permanent": bool(is_permanent),
        "timestamp": time.time(),
        "worker": os.environ.get("HOSTNAME", "unknown"),
        "source_artifacts": source_artifacts or {},
    }
    directory = Path(events_base) / PACKAGING_ERRORS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.json"
    tmp = directory / f".{name}.tmp"
    target = directory / name
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def load_packaging_attempt_counts(events_base: str | Path) -> Counter[tuple[str, str]]:
    errors_dir = Path(events_base) / PACKAGING_ERRORS_DIR
    counts: Counter[tuple[str, str]] = Counter()
    if not errors_dir.is_dir():
        return counts
    for path in errors_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = (payload["session_id"], payload["archive_id"])
            if payload.get("is_permanent", False):
                counts[key] = 9999
            else:
                counts[key] += 1
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return counts


def packaging_error_archives(events_base: str | Path) -> set[tuple[str, str]]:
    return set(load_packaging_attempt_counts(events_base))


def _is_permanent_packaging_error(error: Exception) -> bool:
    return type(error).__name__ in PERMANENT_ERROR_NAMES
