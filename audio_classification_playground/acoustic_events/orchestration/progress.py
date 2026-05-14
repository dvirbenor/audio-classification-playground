"""Progress scanning and completion checks for the orchestration pipeline.

Two flavours of completion check:

* **File-existence** (``is_archive_complete``): fast, used by the CLI
  ``progress`` subcommand and by ``reclaim_stale``.
* **Config-aware** (``is_archive_complete_for_config``): reads manifests and
  validates ``inference_config_hash`` against the current run config.  Used
  by the worker loop to avoid reusing stale artifacts produced with a
  different backbone, batch size, or model.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..inference.artifacts import MANIFEST_FILENAME, PREDICTIONS_FILENAME

LOGGER = logging.getLogger(__name__)

TASKS = ("vad", "affect", "disfluency", "emotion")


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
            if expected is not None and manifest.get("inference_config_hash") != expected:
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
        return manifest.get("inference_config_hash") == expected_config_hash
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def scan_progress(
    output_base: Path,
    entities: list,
    permanent_audio_errors: set[tuple[str, str]] | None = None,
    inference_error_counts: dict[tuple[str, str], int] | None = None,
) -> ProgressSummary:
    """Generate a comprehensive progress summary over all entities."""
    from .locking import LOCKS_DIR

    perm_errors = permanent_audio_errors or set()
    inf_errors = inference_error_counts or {}

    locks_dir = output_base / LOCKS_DIR
    locked_set: set[str] = set()
    if locks_dir.is_dir():
        for lf in locks_dir.iterdir():
            if lf.name.endswith(".lock"):
                locked_set.add(lf.stem)

    summary = ProgressSummary(total_entities=len(entities))
    task_counts: dict[str, int] = {t: 0 for t in TASKS}

    for entity in entities:
        sid, aid = entity.session_id, entity.archive_id

        if (sid, aid) in perm_errors:
            summary.permanent_audio_errors += 1
            continue

        if (sid, aid) in inf_errors:
            summary.inference_errors_by_archive += 1

        archive_dir = output_base / sid / aid
        completed_tasks = 0
        for task in TASKS:
            if is_task_complete(archive_dir / task):
                completed_tasks += 1
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
