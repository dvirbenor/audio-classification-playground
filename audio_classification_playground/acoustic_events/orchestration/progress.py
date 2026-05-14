"""Progress scanning and completion checks for the orchestration pipeline.

Two flavours of completion check:

* **File-existence** (``is_archive_complete``): fast, used by the CLI
  ``progress`` subcommand and by ``reclaim_stale``.
* **Config-aware** (``is_archive_complete_for_config``): reads manifests and
  validates ``inference_config_hash`` against the current run config.  Used
  by the worker loop to avoid reusing stale artifacts produced with a
  different backbone, batch size, or model.

Progress scanning uses a single ``os.scandir`` walk of the output tree
rather than probing every expected path.  This keeps cost proportional to
*work completed on disk* instead of total dataset size, which is critical
on network filesystems like EFS where each ``stat()`` is a round-trip.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..inference.artifacts import (
    MANIFEST_FILENAME,
    PREDICTIONS_FILENAME,
    inference_configs_match,
)

LOGGER = logging.getLogger(__name__)

TASKS = ("vad", "affect", "disfluency", "emotion")
_TASKS_SET = frozenset(TASKS)
_COMPLETE_FILES = frozenset({MANIFEST_FILENAME, PREDICTIONS_FILENAME})

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProgressSummary:
    total_entities: int = 0
    complete: int = 0
    partial: int = 0
    permanent_audio_errors: int = 0
    inference_errors_by_archive: int = 0
    locked: int = 0
    remaining: int = 0
    task_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class QuickSummary:
    """Parquet-free progress snapshot derived purely from disk state."""

    complete: int = 0
    partial: int = 0
    task_counts: dict[str, int] = field(default_factory=dict)
    lock_count: int = 0
    audio_error_records: int = 0
    permanent_audio_error_archives: int = 0
    inference_error_records: int = 0
    inference_error_archives: int = 0


# ---------------------------------------------------------------------------
# Single-archive checks (used by worker and reclaim-stale)
# ---------------------------------------------------------------------------


def is_task_complete(task_dir: Path) -> bool:
    """Check whether a single task directory contains a valid artifact."""
    return (
        (task_dir / MANIFEST_FILENAME).is_file()
        and (task_dir / PREDICTIONS_FILENAME).is_file()
    )


def is_archive_complete(output_base: Path, session_id: str, archive_id: str) -> bool:
    """Fast file-existence check: all four tasks present."""
    archive_dir = output_base / session_id / archive_id
    return all(is_task_complete(archive_dir / task) for task in TASKS)


def is_archive_complete_for_config(
    output_base: Path,
    session_id: str,
    archive_id: str,
    expected_config_hashes: dict[str, str],
    expected_configs: dict[str, dict] | None = None,
    ignore_batch_size: bool = False,
) -> bool:
    """Config-aware completion check.

    First performs a fast file-existence check, then opens each manifest
    and validates that ``inference_config_hash`` matches the expected value.
    Short-circuits on the first missing or mismatched task.
    """
    archive_dir = output_base / session_id / archive_id
    for task in TASKS:
        task_dir = archive_dir / task
        if not is_task_complete(task_dir):
            return False
        manifest_path = task_dir / MANIFEST_FILENAME
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            expected = expected_config_hashes.get(task)
            if expected is not None and manifest.get("inference_config_hash") == expected:
                continue
            if ignore_batch_size and expected_configs is not None:
                observed_config = manifest.get("inference_config")
                expected_config = expected_configs.get(task)
                if (
                    isinstance(observed_config, dict)
                    and expected_config is not None
                    and inference_configs_match(
                        observed_config,
                        expected_config,
                        ignore_batch_size=True,
                    )
                ):
                    continue
            if expected is not None:
                return False
        except (OSError, json.JSONDecodeError, KeyError):
            return False
    return True


def is_task_complete_for_config(
    output_base: Path,
    session_id: str,
    archive_id: str,
    task: str,
    expected_config_hash: str,
    expected_config: dict | None = None,
    ignore_batch_size: bool = False,
) -> bool:
    """Return True when one task artifact exists and matches config."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    task_dir = output_base / session_id / archive_id / task
    if not is_task_complete(task_dir):
        return False
    manifest_path = task_dir / MANIFEST_FILENAME
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("inference_config_hash") == expected_config_hash:
            return True
        observed_config = manifest.get("inference_config")
        return (
            ignore_batch_size
            and isinstance(observed_config, dict)
            and expected_config is not None
            and inference_configs_match(
                observed_config,
                expected_config,
                ignore_batch_size=True,
            )
        )
    except (OSError, json.JSONDecodeError, KeyError):
        return False


# ---------------------------------------------------------------------------
# Filesystem walker
# ---------------------------------------------------------------------------

_META_DIR = "_meta"


def _is_task_complete_via_scandir(task_path: str) -> bool:
    """Check task completion with one ``readdir`` instead of two ``stat()``."""
    try:
        with os.scandir(task_path) as entries:
            names = {e.name for e in entries if e.is_file(follow_symlinks=False)}
    except (OSError, FileNotFoundError):
        return False
    return _COMPLETE_FILES <= names


def _walk_completed_tasks(output_base: Path) -> dict[tuple[str, str], set[str]]:
    """Walk the output tree and return ``{(session_id, archive_id): {task_names}}``.

    Uses ``os.scandir`` context managers throughout and tolerates
    ``OSError`` / ``FileNotFoundError`` per-entry so that concurrent
    workers writing or removing directories do not corrupt results.
    Only skips the literal ``_meta`` directory (session IDs may legally
    start with ``_``).  Returns an empty dict when *output_base* does not
    exist.
    """
    if not output_base.is_dir():
        return {}
    result: dict[tuple[str, str], set[str]] = {}
    try:
        with os.scandir(output_base) as sessions:
            for session_entry in sessions:
                if session_entry.name == _META_DIR:
                    continue
                try:
                    if not session_entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                try:
                    with os.scandir(session_entry.path) as archives:
                        for archive_entry in archives:
                            try:
                                if not archive_entry.is_dir(follow_symlinks=False):
                                    continue
                            except OSError:
                                continue
                            tasks_done: set[str] = set()
                            try:
                                with os.scandir(archive_entry.path) as task_entries:
                                    for task_entry in task_entries:
                                        try:
                                            if (
                                                task_entry.name in _TASKS_SET
                                                and task_entry.is_dir(follow_symlinks=False)
                                                and _is_task_complete_via_scandir(
                                                    task_entry.path,
                                                )
                                            ):
                                                tasks_done.add(task_entry.name)
                                        except OSError:
                                            continue
                            except (OSError, FileNotFoundError):
                                pass  # preserve tasks_done collected before the error
                            if tasks_done:
                                key = (session_entry.name, archive_entry.name)
                                result[key] = tasks_done
                except (OSError, FileNotFoundError):
                    continue
    except (OSError, FileNotFoundError):
        pass
    return result


# ---------------------------------------------------------------------------
# Full progress scan (manifest-backed)
# ---------------------------------------------------------------------------


def scan_progress(
    output_base: Path,
    entities: list,
    permanent_audio_errors: set[tuple[str, str]] | None = None,
    inference_error_counts: dict[tuple[str, str], int] | None = None,
) -> ProgressSummary:
    """Generate a comprehensive progress summary over all entities.

    Walks the output tree once, then classifies each entity via in-memory
    set lookups.  Semantics (skip order, field definitions) are identical
    to the original per-entity probe implementation.
    """
    from .locking import LOCKS_DIR

    perm_errors = permanent_audio_errors or set()
    inf_errors = inference_error_counts or {}

    locks_dir = output_base / LOCKS_DIR
    locked_set: set[str] = set()
    if locks_dir.is_dir():
        for lf in locks_dir.iterdir():
            if lf.name.endswith(".lock"):
                locked_set.add(lf.stem)

    completed_map = _walk_completed_tasks(output_base)

    summary = ProgressSummary(total_entities=len(entities))
    task_counts: dict[str, int] = {t: 0 for t in TASKS}

    for entity in entities:
        sid, aid = entity.session_id, entity.archive_id

        if (sid, aid) in perm_errors:
            summary.permanent_audio_errors += 1
            continue

        if (sid, aid) in inf_errors:
            summary.inference_errors_by_archive += 1

        done_tasks = completed_map.get((sid, aid), set())
        completed_tasks = len(done_tasks)
        for task in done_tasks:
            task_counts[task] += 1

        if completed_tasks == len(TASKS):
            summary.complete += 1
        elif completed_tasks > 0:
            summary.partial += 1
        else:
            lock_stem = f"{sid}__{aid}"
            if lock_stem in locked_set:
                summary.locked += 1

    summary.task_counts = task_counts
    summary.remaining = (
        summary.total_entities
        - summary.complete
        - summary.permanent_audio_errors
    )
    return summary


# ---------------------------------------------------------------------------
# Quick disk summary (parquet-free)
# ---------------------------------------------------------------------------


def quick_disk_summary(output_base: Path) -> QuickSummary:
    """Fast, parquet-free progress snapshot from disk state alone.

    Walks the output tree, ``_meta/locks/``, ``_meta/audio_errors/``, and
    ``_meta/inference_errors/`` to produce a pulse-check summary without
    needing the archive manifest.  Cannot report totals or remaining counts.
    """
    from .locking import LOCKS_DIR
    from .errors import AUDIO_ERRORS_DIR, INFERENCE_ERRORS_DIR

    completed_map = _walk_completed_tasks(output_base)

    summary = QuickSummary()
    task_counts: dict[str, int] = {t: 0 for t in TASKS}

    for done_tasks in completed_map.values():
        for task in done_tasks:
            task_counts[task] += 1
        if len(done_tasks) == len(TASKS):
            summary.complete += 1
        else:
            summary.partial += 1

    summary.task_counts = task_counts

    # Locks
    locks_dir = output_base / LOCKS_DIR
    if locks_dir.is_dir():
        summary.lock_count = sum(
            1 for f in locks_dir.iterdir() if f.name.endswith(".lock")
        )

    # Audio errors
    audio_dir = output_base / AUDIO_ERRORS_DIR
    if audio_dir.is_dir():
        permanent_archives: set[tuple[str, str]] = set()
        for f in audio_dir.iterdir():
            if not f.name.endswith(".json"):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                summary.audio_error_records += 1
                if data.get("is_permanent", False):
                    permanent_archives.add(
                        (data["session_id"], data["archive_id"]),
                    )
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        summary.permanent_audio_error_archives = len(permanent_archives)

    # Inference errors
    inference_dir = output_base / INFERENCE_ERRORS_DIR
    if inference_dir.is_dir():
        inf_archives: set[tuple[str, str]] = set()
        for f in inference_dir.iterdir():
            if not f.name.endswith(".json"):
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                summary.inference_error_records += 1
                inf_archives.add((data["session_id"], data["archive_id"]))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        summary.inference_error_archives = len(inf_archives)

    return summary
