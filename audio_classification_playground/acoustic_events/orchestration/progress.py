"""Progress scanning and completion checks for the orchestration pipeline.

Two flavours of completion check:

* **File-existence** (``is_archive_complete``): fast, used by the CLI
  ``progress`` subcommand and by ``reclaim_stale``.
* **Config-aware** (``is_archive_complete_for_config``): reads manifests and
  validates ``inference_config_hash`` against the current run config.  Used
  by the worker loop to avoid reusing stale artifacts produced with a
  different backbone, batch size, or model.

Manifest-backed progress scanning (``scan_progress``) checks only the
entities listed in the parquet via a thread pool of concurrent
``stat`` / ``readdir`` calls.  This avoids walking the entire output tree
and parallelises I/O against EFS.

The parquet-free ``quick_disk_summary`` still uses ``find(1)`` (falling
back to ``os.scandir``) since it has no entity list to target.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
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

# Per-worker completion logs written by workers after each archive completes.
# One file per worker avoids any concurrent-write races on NFS/EFS
# (O_APPEND is not atomic on NFS — lseek+write is not a single syscall).
# Total progress = sum of line counts across all files in this directory.
COMPLETIONS_DIR = "_meta/progress_completions"


def filter_entities_needing_work(
    output_base: Path,
    entities: list,
    required_tasks: tuple[str, ...],
    require_vad: bool = False,
    permanent_errors: set | None = None,
) -> list:
    """Return the subset of entities that still have work to do.

    Runs a parallel bulk EFS scan (64 threads) once at startup, then filters
    in memory — far faster than checking each archive sequentially in the
    main loop when most archives are already done or VAD-less.

    Args:
        required_tasks: tasks the worker needs to run (e.g. affect/disfluency/emotion).
        require_vad: if True, drop archives whose vad/ artifact is not complete.
        permanent_errors: set of (session_id, archive_id) to exclude.
    """
    perm = permanent_errors or set()
    entity_keys = [
        (e.session_id, e.archive_id)
        for e in entities
        if (e.session_id, e.archive_id) not in perm
    ]

    LOGGER.info(
        "Pre-scan: checking %d entities (require_vad=%s, tasks=%s) …",
        len(entity_keys),
        require_vad,
        required_tasks,
    )
    completed_map = completed_tasks_for_entity_keys(output_base, entity_keys)

    required_set = frozenset(required_tasks)
    filtered = []
    skipped_complete = 0
    skipped_no_vad = 0

    for entity in entities:
        key = (entity.session_id, entity.archive_id)
        if key in perm:
            continue
        done = completed_map.get(key, frozenset())
        if require_vad and "vad" not in done:
            skipped_no_vad += 1
            continue
        if required_set <= done:
            skipped_complete += 1
            continue
        filtered.append(entity)

    LOGGER.info(
        "Pre-scan done: %d need work, %d already complete, %d missing VAD (skipped)",
        len(filtered),
        skipped_complete,
        skipped_no_vad,
    )
    return filtered


def record_archive_complete(
    output_base: Path,
    session_id: str,
    archive_id: str,
    tasks: tuple[str, ...],
    worker_id: str,
    ts: str,
) -> None:
    """Append one completion record to this worker's progress log.

    Each worker writes only to its own file (keyed by worker_id), so no
    locking or atomic-append guarantees are needed — safe on EFS/NFS.
    Silently ignores write errors so a blip never kills a worker.
    """
    log_path = output_base / COMPLETIONS_DIR / f"{worker_id}.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = (
            json.dumps(
                {
                    "session_id": session_id,
                    "archive_id": archive_id,
                    "tasks": list(tasks),
                    "worker_id": worker_id,
                    "ts": ts,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass

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


def is_task_artifact_complete(task_dir: Path) -> bool:
    """Check file existence plus manifest complete status."""
    if not is_task_complete(task_dir):
        return False
    try:
        with open(task_dir / MANIFEST_FILENAME, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def is_task_artifact_complete_for_archive(
    output_base: Path,
    session_id: str,
    archive_id: str,
    task: str,
) -> bool:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    return is_task_artifact_complete(output_base / session_id / archive_id / task)


def are_tasks_complete_by_artifact(
    output_base: Path,
    session_id: str,
    archive_id: str,
    tasks: tuple[str, ...],
) -> bool:
    return all(
        is_task_artifact_complete_for_archive(output_base, session_id, archive_id, task)
        for task in tasks
    )


def incomplete_tasks_by_artifact(
    output_base: Path,
    session_id: str,
    archive_id: str,
    tasks: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        task for task in tasks
        if not is_task_artifact_complete_for_archive(output_base, session_id, archive_id, task)
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


def _walk_completed_tasks_find(output_base: Path) -> dict[tuple[str, str], set[str]]:
    """Walk via ``find(1)`` — single subprocess, minimal EFS round-trips.

    Locates ``manifest.json`` and ``predictions.npz`` at exactly depth 4
    (``session/archive/task/file``), excludes ``_meta/``, and returns the
    same ``{(session_id, archive_id): {task_names}}`` as the scandir
    variant.  Raises ``RuntimeError`` if ``find`` fails so the caller can
    fall back.
    """
    base_str = str(output_base)
    prefix = base_str + "/"

    proc = subprocess.run(
        [
            "find", base_str,
            "-mindepth", "4", "-maxdepth", "4",
            "-type", "f",
            "(", "-name", MANIFEST_FILENAME,
            "-o", "-name", PREDICTIONS_FILENAME, ")",
            "-not", "-path", "*/_meta/*",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())

    # (session, archive, task) -> set of filenames found
    task_files: dict[tuple[str, str, str], set[str]] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith(prefix):
            continue
        rel = line[len(prefix):]
        parts = rel.split("/")
        if len(parts) != 4 or parts[2] not in _TASKS_SET:
            continue
        key = (parts[0], parts[1], parts[2])
        task_files.setdefault(key, set()).add(parts[3])

    result: dict[tuple[str, str], set[str]] = {}
    for (sid, aid, task), files in task_files.items():
        if _COMPLETE_FILES <= files:
            result.setdefault((sid, aid), set()).add(task)
    return result


def _walk_completed_tasks_scandir(
    output_base: Path,
) -> dict[tuple[str, str], set[str]]:
    """Walk via nested ``os.scandir`` — portable fallback.

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
                                                and task_entry.is_dir(
                                                    follow_symlinks=False,
                                                )
                                                and _is_task_complete_via_scandir(
                                                    task_entry.path,
                                                )
                                            ):
                                                tasks_done.add(task_entry.name)
                                        except OSError:
                                            continue
                            except (OSError, FileNotFoundError):
                                pass
                            if tasks_done:
                                key = (session_entry.name, archive_entry.name)
                                result[key] = tasks_done
                except (OSError, FileNotFoundError):
                    continue
    except (OSError, FileNotFoundError):
        pass
    return result


def _walk_completed_tasks(output_base: Path) -> dict[tuple[str, str], set[str]]:
    """Walk the output tree and return ``{(session_id, archive_id): {task_names}}``.

    Tries ``find(1)`` first for speed on network filesystems, falls back
    to ``os.scandir`` if ``find`` is unavailable or fails.
    """
    try:
        return _walk_completed_tasks_find(output_base)
    except (RuntimeError, FileNotFoundError, OSError):
        LOGGER.debug("find-based walk failed, falling back to scandir")
        return _walk_completed_tasks_scandir(output_base)


# ---------------------------------------------------------------------------
# Targeted parallel entity check (manifest-backed progress)
# ---------------------------------------------------------------------------

_PARALLEL_WORKERS = 64


def _check_entity_completion(
    base_str: str,
    sid: str,
    aid: str,
) -> tuple[str, str, frozenset[str]]:
    """Check which tasks are complete for one ``(session_id, archive_id)``.

    Uses string paths and ``os.path`` to avoid ``Path`` object overhead
    when called from a thread pool over many entities.
    """
    archive_path = f"{base_str}/{sid}/{aid}"
    try:
        if not os.path.isdir(archive_path):
            return (sid, aid, frozenset())
    except OSError:
        return (sid, aid, frozenset())
    done: set[str] = set()
    for task in TASKS:
        if _is_task_complete_via_scandir(f"{archive_path}/{task}"):
            done.add(task)
    return (sid, aid, frozenset(done))


def _check_entities_parallel(
    output_base: Path,
    entity_keys: list[tuple[str, str]],
    max_workers: int = _PARALLEL_WORKERS,
) -> dict[tuple[str, str], set[str]]:
    """Check task completion for known entities using parallel I/O.

    Issues concurrent ``stat`` / ``readdir`` calls against EFS via a thread
    pool, replacing the single-threaded ``find(1)`` tree walk.  Only
    probes paths for entities listed in the manifest — avoids scanning
    the entire output tree.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not entity_keys:
        return {}

    base_str = str(output_base)
    result: dict[tuple[str, str], set[str]] = {}

    pool_size = min(max_workers, len(entity_keys))
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        futures = {
            pool.submit(_check_entity_completion, base_str, sid, aid): (sid, aid)
            for sid, aid in entity_keys
        }
        for future in as_completed(futures):
            try:
                sid, aid, done = future.result()
                if done:
                    result[(sid, aid)] = set(done)
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Completion cache
# ---------------------------------------------------------------------------

_PROGRESS_CACHE = "_meta/progress_complete.txt"


def _load_completion_cache(output_base: Path) -> set[tuple[str, str]]:
    """Load the set of ``(session_id, archive_id)`` previously verified as
    fully complete (all 4 tasks present).

    The cache is a simple line-oriented text file stored inside the output
    tree.  Returns an empty set on any read error.
    """
    cache_path = output_base / _PROGRESS_CACHE
    try:
        text = cache_path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return set()
    result: set[tuple[str, str]] = set()
    for line in text.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            result.add((parts[0], parts[1]))
    return result


def _save_completion_cache(
    output_base: Path,
    complete: set[tuple[str, str]],
) -> None:
    """Atomically persist the set of fully-complete archives."""
    cache_path = output_base / _PROGRESS_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=str(cache_path.parent),
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for sid, aid in sorted(complete):
                f.write(f"{sid}\t{aid}\n")
        os.replace(str(tmp), str(cache_path))
    except OSError as exc:
        LOGGER.warning("Could not write progress cache: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def completed_tasks_for_entity_keys(
    output_base: Path,
    entity_keys: list[tuple[str, str]],
    *,
    use_cache: bool = True,
) -> dict[tuple[str, str], set[str]]:
    """Return completed task names for known ``(session_id, archive_id)`` keys.

    This is the manifest-targeted completion scan used by full progress and
    cache warming.  It avoids walking the whole output tree, which matters on
    EFS once the shared audio cache adds additional state below ``_meta/``.
    """
    entity_key_set = set(entity_keys)

    all_cached_complete: set[tuple[str, str]] = set()
    cached_complete: set[tuple[str, str]] = set()
    if use_cache:
        all_cached_complete = _load_completion_cache(output_base)
        cached_complete = all_cached_complete & entity_key_set

    keys_to_check = [k for k in entity_keys if k not in cached_complete]
    LOGGER.info(
        "Progress scan: %d entities, %d cached-complete, %d to verify on disk",
        len(entity_keys),
        len(cached_complete),
        len(keys_to_check),
    )

    checked_map = _check_entities_parallel(output_base, keys_to_check)

    completed_map: dict[tuple[str, str], set[str]] = {
        key: set(TASKS) for key in cached_complete
    }
    completed_map.update(checked_map)

    newly_complete = {k for k, v in checked_map.items() if len(v) == len(TASKS)}
    if use_cache and newly_complete:
        _save_completion_cache(output_base, all_cached_complete | newly_complete)
        LOGGER.info(
            "Progress cache updated: %d newly complete, %d total cached",
            len(newly_complete),
            len(all_cached_complete | newly_complete),
        )

    return completed_map


# ---------------------------------------------------------------------------
# Full progress scan (manifest-backed)
# ---------------------------------------------------------------------------


def scan_progress(
    output_base: Path,
    entities: list,
    permanent_audio_errors: set[tuple[str, str]] | None = None,
    inference_error_counts: dict[tuple[str, str], int] | None = None,
    *,
    use_cache: bool = True,
) -> ProgressSummary:
    """Generate a comprehensive progress summary over all entities.

    Checks only the entities listed in the manifest using parallel I/O,
    then classifies each via in-memory set lookups.  When *use_cache* is
    True (default), archives previously verified as fully complete are
    read from a lightweight disk cache and not re-checked on EFS — this
    makes repeated progress calls progressively faster as work completes.

    Pass ``use_cache=False`` (CLI ``--no-cache``) to force a full re-scan.
    """
    from .locking import iter_lock_files

    perm_errors = permanent_audio_errors or set()
    inf_errors = inference_error_counts or {}

    locked_set: set[str] = set()
    for lf in iter_lock_files(output_base):
        locked_set.add(lf.stem)

    entity_keys = [(e.session_id, e.archive_id) for e in entities]
    completed_map = completed_tasks_for_entity_keys(
        output_base,
        entity_keys,
        use_cache=use_cache,
    )

    # --- classify entities --------------------------------------------------
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
    from .locking import iter_lock_files

    summary.lock_count = len(iter_lock_files(output_base))

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
