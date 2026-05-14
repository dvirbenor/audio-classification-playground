"""Durable, contention-free error logging for audio and inference failures.

Each error event is written as an individual JSON file with a UUID name
under ``_meta/audio_errors/`` or ``_meta/inference_errors/``.  This avoids
any ``flock`` contention across pods.

Audio errors are categorised as **permanent** (``no_matching_file``,
``glacier_storage_class``) or **transient** (``download_failed``).
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
from pathlib import Path

from .audio_resolver import AudioResolutionError

LOGGER = logging.getLogger(__name__)

AUDIO_ERRORS_DIR = "_meta/audio_errors"
INFERENCE_ERRORS_DIR = "_meta/inference_errors"

PERMANENT_AUDIO_ERROR_TYPES = frozenset({"no_matching_file", "glacier_storage_class"})

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


def append_audio_error(
    output_base: Path,
    error: AudioResolutionError,
) -> Path:
    """Persist a single audio resolution error as a JSON file."""
    payload = {
        "session_id": error.session_id,
        "archive_id": error.archive_id,
        "file_parent_dir": error.file_parent_dir,
        "error_type": error.error_type,
        "detail": error.detail,
        "is_permanent": error.is_permanent,
        "timestamp": time.time(),
        "worker": os.environ.get("HOSTNAME", "unknown"),
    }
    path = _write_error_json(output_base / AUDIO_ERRORS_DIR, payload)
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
) -> Path:
    """Persist a single inference failure as a JSON file."""
    error_type_name = type(error).__name__
    payload = {
        "session_id": session_id,
        "archive_id": archive_id,
        "error_type": error_type_name,
        "detail": str(error)[:2000],
        "is_deterministic": is_deterministic,
        "timestamp": time.time(),
        "worker": os.environ.get("HOSTNAME", "unknown"),
    }
    path = _write_error_json(output_base / INFERENCE_ERRORS_DIR, payload)
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
    for f in errors_dir.iterdir():
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


def load_inference_attempt_counts(output_base: Path) -> Counter[tuple[str, str]]:
    """Count inference error files per ``(session_id, archive_id)``.

    Each JSON file represents one attempt.  Deterministic errors are counted
    with a high sentinel (9999) to ensure they exceed any max-retry threshold.
    """
    errors_dir = output_base / INFERENCE_ERRORS_DIR
    if not errors_dir.is_dir():
        return Counter()
    counts: Counter[tuple[str, str]] = Counter()
    for f in errors_dir.iterdir():
        if not f.name.endswith(".json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            key = (data["session_id"], data["archive_id"])
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
) -> int:
    """Authoritative attempt count for a specific archive.

    Re-reads the error directory to avoid stale in-memory caches across
    pods.  Called after an archive is claimed (lock held).
    """
    errors_dir = output_base / INFERENCE_ERRORS_DIR
    if not errors_dir.is_dir():
        return 0
    count = 0
    for f in errors_dir.iterdir():
        if not f.name.endswith(".json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data["session_id"] == session_id and data["archive_id"] == archive_id:
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
