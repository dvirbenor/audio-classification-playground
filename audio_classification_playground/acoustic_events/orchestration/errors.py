"""Durable, contention-free error logging for audio and inference failures.

Each error event is written as an individual JSON file with a UUID name
under ``_meta/audio_errors/`` or ``_meta/inference_errors/``.  This avoids
any ``flock`` contention across pods.

Audio errors are categorised as **permanent** (``no_matching_file``) or
**transient** (``glacier_storage_class``, ``download_failed``).
``glacier_storage_class`` is transient because objects may be restored from
Glacier, at which point the next retry succeeds automatically.
Inference errors track per-archive attempt counts so a max-retry policy
can be enforced.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .audio_resolver import AudioResolutionError

LOGGER = logging.getLogger(__name__)

AUDIO_ERRORS_DIR = "_meta/audio_errors"
INFERENCE_ERRORS_DIR = "_meta/inference_errors"

PERMANENT_AUDIO_ERROR_TYPES = frozenset({"no_matching_file"})

DETERMINISTIC_ERRORS: tuple[str, ...] = (
    "SndfileError",
    "LibsndfileError",
    "NoBackendError",
    "AudioFormatError",
    "IsADirectoryError",
)


def _write_error_json(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.json"
    target = directory / filename
    tmp = directory / f".{filename}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(str(tmp), str(target))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def _stable_error_json(directory: Path, filename: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.is_file():
        return target
    tmp = directory / f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        try:
            fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            tmp.unlink(missing_ok=True)
            return target
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(tmp.read_text(encoding="utf-8"))
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def append_audio_error(
    output_base: Path,
    error: AudioResolutionError,
) -> Path:
    """Persist an audio resolution error, deduped by archive/error type."""
    payload = {
        "session_id": error.session_id,
        "archive_id": error.archive_id,
        "file_parent_dir": error.file_parent_dir,
        "error_type": error.error_type,
        "detail": error.detail,
        "s3_key": error.s3_key,
        "is_permanent": error.is_permanent,
        "timestamp": time.time(),
        "worker": os.environ.get("HOSTNAME", "unknown"),
    }
    filename = f"{error.session_id}__{error.archive_id}__{error.error_type}.json"
    path = _stable_error_json(output_base / AUDIO_ERRORS_DIR, filename, payload)
    LOGGER.info(
        "Audio error [%s]: %s/%s — %s",
        error.error_type, error.session_id, error.archive_id, error.detail,
    )
    return path


def append_inference_error(
    output_base: Path,
    session_id: str,
    archive_id: str,
    error: Exception,
    is_deterministic: bool = False,
    task_group: str | None = None,
) -> Path:
    """Persist a single inference failure as a JSON file."""
    error_type_name = type(error).__name__
    payload = {
        "session_id": session_id,
        "archive_id": archive_id,
        "error_type": error_type_name,
        "detail": str(error)[:2000],
        "is_deterministic": is_deterministic,
        "task_group": task_group or "all",
        "timestamp": time.time(),
        "worker": os.environ.get("HOSTNAME", "unknown"),
    }
    directory = output_base / INFERENCE_ERRORS_DIR
    if task_group and task_group != "all":
        directory = directory / task_group
    path = _write_error_json(directory, payload)
    LOGGER.info(
        "Inference error [%s]: %s/%s — %s",
        error_type_name, session_id, archive_id, str(error)[:200],
    )
    return path


def load_permanent_error_set(output_base: Path) -> set[tuple[str, str]]:
    """Return ``{(session_id, archive_id)}`` for all permanent audio errors.

    Duplicates are naturally deduplicated by the set.
    """
    errors_dir = output_base / AUDIO_ERRORS_DIR
    if not errors_dir.is_dir():
        return set()
    result: set[tuple[str, str]] = set()
    for f in errors_dir.rglob("*.json"):
        if not f.name.endswith(".json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("is_permanent", False):
                result.add((data["session_id"], data["archive_id"]))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    LOGGER.info("Loaded %d permanent audio errors", len(result))
    return result


def load_inference_attempt_counts(
    output_base: Path,
    *,
    task_group: str | None = None,
) -> Counter[tuple[str, str] | tuple[str, str, str]]:
    """Count inference error files per archive or task group.

    Each JSON file represents one attempt.  Deterministic errors are counted
    with a high sentinel (9999) to ensure they exceed any max-retry threshold.
    """
    errors_dir = output_base / INFERENCE_ERRORS_DIR
    if not errors_dir.is_dir():
        return Counter()
    counts: Counter[tuple[str, str] | tuple[str, str, str]] = Counter()
    if task_group is None:
        files = errors_dir.rglob("*.json")
    elif task_group != "all":
        files = (errors_dir / task_group).glob("*.json")
    else:
        files = errors_dir.glob("*.json")
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            group = str(data.get("task_group") or task_group or "all")
            key = (
                (data["session_id"], data["archive_id"])
                if task_group is None
                else (data["session_id"], data["archive_id"], group)
            )
            if data.get("is_deterministic", False):
                counts[key] = 9999
            else:
                counts[key] += 1
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    LOGGER.info("Loaded inference attempt counts for %d archives", len(counts))
    return counts


def count_inference_attempts_for(
    output_base: Path,
    session_id: str,
    archive_id: str,
    *,
    task_group: str | None = None,
) -> int:
    """Authoritative attempt count for a specific archive.

    Re-reads the error directory to avoid stale in-memory caches across
    pods.  Called after an archive is claimed (lock held).
    """
    errors_dir = output_base / INFERENCE_ERRORS_DIR
    if not errors_dir.is_dir():
        return 0
    count = 0
    if task_group is None:
        files = errors_dir.rglob("*.json")
    elif task_group != "all":
        files = (errors_dir / task_group).glob("*.json")
    else:
        files = errors_dir.glob("*.json")
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if (
                data["session_id"] == session_id
                and data["archive_id"] == archive_id
                and (
                    task_group is None
                    or str(data.get("task_group") or task_group) == task_group
                )
            ):
                if data.get("is_deterministic", False):
                    return 9999
                count += 1
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return count


def is_deterministic_error(error: Exception) -> bool:
    """Return True if *error* is known to be non-retryable."""
    error_name = type(error).__name__
    return any(det in error_name for det in DETERMINISTIC_ERRORS)


# ---------------------------------------------------------------------------
# Grouped error summary
# ---------------------------------------------------------------------------


@dataclass
class ErrorGroup:
    """One error type's aggregate stats across all error JSON files."""

    error_type: str
    record_count: int = 0
    unique_archives: set[tuple[str, str]] = field(default_factory=set)
    is_permanent: bool | None = None
    example_detail: str = ""
    example_archive: tuple[str, str] = ("", "")


def summarize_errors_grouped(errors_dir: Path) -> list[ErrorGroup]:
    """Read all JSON error files in *errors_dir* and group by ``error_type``.

    Returns groups sorted by ``record_count`` descending, then
    ``error_type`` ascending.  Malformed JSON files are silently skipped.
    """
    if not errors_dir.is_dir():
        return []

    groups: dict[str, ErrorGroup] = {}

    for f in errors_dir.rglob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            etype = data.get("error_type", "unknown")
            sid = data["session_id"]
            aid = data["archive_id"]
            detail = str(data.get("detail", ""))[:120].replace("\n", " ")
            is_perm = data.get("is_permanent")  # None for inference errors
        except (OSError, json.JSONDecodeError, KeyError):
            continue

        if etype not in groups:
            groups[etype] = ErrorGroup(
                error_type=etype,
                is_permanent=is_perm,
                example_detail=detail,
                example_archive=(sid, aid),
            )

        grp = groups[etype]
        grp.record_count += 1
        grp.unique_archives.add((sid, aid))

    return sorted(
        groups.values(),
        key=lambda g: (-g.record_count, g.error_type),
    )
