"""Distributed archive locking via atomic file creation on EFS.

Archive-level lock files live under ``<output_base>/_meta/locks/`` with names
of the form ``<session_id>__<archive_id>.lock``.  Task-fleet lock files live
under ``<output_base>/_meta/locks/<namespace>/``.  Atomicity is guaranteed by
``os.open(..., O_CREAT | O_EXCL)``, which fails if the file already exists.

Locks are released (deleted) on both success and failure, and a
``reclaim_stale`` sweep can recover orphan locks left by crashed pods.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

from .manifest import ArchiveEntity

LOGGER = logging.getLogger(__name__)

LOCKS_DIR = "_meta/locks"


def _lock_path(
    output_base: Path,
    entity: ArchiveEntity,
    *,
    namespace: str | None = None,
) -> Path:
    locks_dir = output_base / LOCKS_DIR
    if namespace:
        locks_dir = locks_dir / namespace
    return locks_dir / f"{entity.session_id}__{entity.archive_id}.lock"


def iter_lock_files(output_base: Path) -> list[Path]:
    """Return all flat and task-scoped lock files."""
    locks_dir = output_base / LOCKS_DIR
    if not locks_dir.is_dir():
        return []
    return sorted(p for p in locks_dir.rglob("*.lock") if p.is_file())


def flat_lock_files(output_base: Path) -> list[Path]:
    locks_dir = output_base / LOCKS_DIR
    if not locks_dir.is_dir():
        return []
    return sorted(p for p in locks_dir.glob("*.lock") if p.is_file())


def nested_lock_files(output_base: Path) -> list[Path]:
    locks_dir = output_base / LOCKS_DIR
    if not locks_dir.is_dir():
        return []
    return sorted(
        p for p in locks_dir.rglob("*.lock")
        if p.is_file() and p.parent != locks_dir
    )


def try_claim(
    output_base: Path,
    entity: ArchiveEntity,
    *,
    namespace: str | None = None,
    task_group: str | None = None,
) -> bool:
    """Attempt to atomically create a lock file for *entity*.

    Returns ``True`` if the lock was acquired, ``False`` if another worker
    already holds it.
    """
    lock = _lock_path(output_base, entity, namespace=namespace)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(
                f"worker={os.environ.get('HOSTNAME', 'unknown')}\n"
                f"pid={os.getpid()}\n"
                f"time={time.time()}\n"
                f"task_group={task_group or namespace or 'all'}\n"
            )
        return True
    except FileExistsError:
        return False


def release_claim(
    output_base: Path,
    entity: ArchiveEntity,
    *,
    namespace: str | None = None,
) -> None:
    """Remove the lock file for *entity*.  Idempotent."""
    lock = _lock_path(output_base, entity, namespace=namespace)
    try:
        lock.unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.warning("Failed to release lock %s: %s", lock, exc)


def reclaim_stale(
    output_base: Path,
    older_than_minutes: float = 60.0,
    is_complete_fn: Callable[[str, str], bool] | None = None,
) -> int:
    """Remove lock files that are older than *older_than_minutes*.

    Also removes locks for archives that are already complete (determined
    by *is_complete_fn(session_id, archive_id)*), regardless of age.

    Returns the number of reclaimed locks.
    """
    locks_dir = output_base / LOCKS_DIR
    if not locks_dir.is_dir():
        return 0

    cutoff = time.time() - older_than_minutes * 60
    reclaimed = 0
    for lock_file in iter_lock_files(output_base):
        if not lock_file.name.endswith(".lock"):
            continue
        stem = lock_file.stem
        if "__" not in stem:
            continue

        session_id, _, archive_id = stem.partition("__")
        stale = False
        try:
            mtime = lock_file.stat().st_mtime
            if mtime < cutoff:
                stale = True
        except OSError:
            stale = True

        if not stale and is_complete_fn is not None:
            if is_complete_fn(session_id, archive_id):
                stale = True

        if stale:
            try:
                lock_file.unlink(missing_ok=True)
                reclaimed += 1
                LOGGER.info("Reclaimed stale lock: %s", lock_file.name)
            except OSError as exc:
                LOGGER.warning("Could not reclaim %s: %s", lock_file.name, exc)

    LOGGER.info("Reclaimed %d stale locks", reclaimed)
    return reclaimed
