"""Warm the shared decoded-audio cache ahead of task-fleet workers."""
from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from ..inference.artifacts import SAMPLE_RATE
from .audio_cache import SharedAudioCache
from .audio_resolver import AudioResolutionError, BUCKET
from .errors import load_inference_attempt_counts, load_permanent_error_set
from .locking import iter_lock_files
from .manifest import ArchiveEntity, load_manifest, sort_manifest_by_session
from .progress import TASKS, completed_tasks_for_entity_keys
from .task_groups import (
    TASK_GROUP_AFFECT,
    TASK_GROUP_DISFLUENCY,
    TASK_GROUP_EMOTION,
    TASK_GROUP_VAD,
)

LOGGER = logging.getLogger(__name__)

WARMER_SCAN_INTERVAL_SEC = 60.0
WARMER_RESUME_RATIO = 0.80
WARMER_WORKERS = 4
WARMER_PROGRESS_INTERVAL_SEC = 30.0
CAPACITY_FALLBACK_BREAK_THRESHOLD = 20
_PRESSURE_FALLBACK_REASONS = frozenset({"capacity", "cache_full"})


@dataclass(frozen=True)
class WarmCacheSummary:
    warmed: int = 0
    cache_hits: int = 0
    errors: int = 0
    fallbacks: int = 0
    complete: bool = False


def warm_cache(
    *,
    parquet_path: str | Path,
    output_base: str | Path,
    audio_cache_dir: str | Path,
    max_cache_bytes: int,
    seed: int | None = None,
    max_inference_attempts: int = 3,
    audio_cache_lock_stale_minutes: float = 60.0,
    warm_workers: int = WARMER_WORKERS,
    s3_max_pool_connections: int | None = None,
    sample_rate: int = SAMPLE_RATE,
    scan_interval_sec: float = WARMER_SCAN_INTERVAL_SEC,
    once: bool = False,
) -> WarmCacheSummary:
    """Run the decoded-cache warmer.

    ``once`` is intended for tests and short validation runs.  Production CLI
    usage leaves it ``False`` so the warmer keeps rescanning until the
    manifest is terminal.
    """
    if warm_workers < 1:
        raise ValueError("warm_workers must be >= 1")
    if s3_max_pool_connections is None:
        s3_max_pool_connections = max(64, warm_workers * 4)
    output = Path(output_base)
    entities = load_manifest(parquet_path)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(entities)
        LOGGER.info("Entity ordering: shuffled (seed=%d)", seed)
    else:
        entities = sort_manifest_by_session(entities)
        LOGGER.info("Entity ordering: session-grouped sort (date, session_id)")
    cache = SharedAudioCache(
        audio_cache_dir,
        sample_rate=sample_rate,
        max_cache_bytes=max_cache_bytes,
        stale_lock_minutes=audio_cache_lock_stale_minutes,
        bucket=BUCKET,
        s3_max_pool_connections=s3_max_pool_connections,
    )
    LOGGER.info(
        "Cache warmer starting: entities=%d output=%s cache_dir=%s "
        "max_cache_bytes=%d resume_below_bytes=%d seed=%s workers=%d "
        "s3_max_pool_connections=%d stale_lock_minutes=%.1f scan_interval=%.1fs "
        "once=%s",
        len(entities),
        output,
        audio_cache_dir,
        max_cache_bytes,
        int(max_cache_bytes * WARMER_RESUME_RATIO),
        seed,
        warm_workers,
        s3_max_pool_connections,
        audio_cache_lock_stale_minutes,
        scan_interval_sec,
        once,
    )

    entities_by_key: dict[tuple[str, str], ArchiveEntity] = {
        (e.session_id, e.archive_id): e for e in entities
    }

    stop_cleanup = threading.Event()
    cleanup_thread = threading.Thread(
        target=_run_cleanup_loop,
        kwargs={
            "cache": cache,
            "output_base": output,
            "entities_by_key": entities_by_key,
            "max_inference_attempts": max_inference_attempts,
            "stop_event": stop_cleanup,
            "interval_sec": scan_interval_sec,
        },
        daemon=True,
        name="cache-cleanup",
    )
    cleanup_thread.start()

    total = WarmCacheSummary()
    cycle_idx = 0
    try:
        while True:
            cycle_idx += 1
            LOGGER.info(
                "Cache warmer scan %d starting: cache_bytes=%d/%d",
                cycle_idx,
                cache.cache_bytes(),
                max_cache_bytes,
            )
            cycle = _warm_one_cycle(
                cache=cache,
                output_base=output,
                entities=entities,
                max_cache_bytes=max_cache_bytes,
                max_inference_attempts=max_inference_attempts,
                warm_workers=warm_workers,
            )
            total = WarmCacheSummary(
                warmed=total.warmed + cycle.warmed,
                cache_hits=total.cache_hits + cycle.cache_hits,
                errors=total.errors + cycle.errors,
                fallbacks=total.fallbacks + cycle.fallbacks,
                complete=cycle.complete,
            )
            LOGGER.info(
                "Cache warmer scan %d complete: warmed=%d hits=%d errors=%d "
                "fallbacks=%d complete=%s cache_bytes=%d/%d totals="
                "warmed:%d hits:%d errors:%d fallbacks:%d",
                cycle_idx,
                cycle.warmed,
                cycle.cache_hits,
                cycle.errors,
                cycle.fallbacks,
                cycle.complete,
                cache.cache_bytes(),
                max_cache_bytes,
                total.warmed,
                total.cache_hits,
                total.errors,
                total.fallbacks,
            )
            if cycle.complete or once:
                LOGGER.info(
                    "Cache warmer exiting: complete=%s once=%s warmed=%d hits=%d "
                    "errors=%d fallbacks=%d cache_bytes=%d/%d",
                    cycle.complete,
                    once,
                    total.warmed,
                    total.cache_hits,
                    total.errors,
                    total.fallbacks,
                    cache.cache_bytes(),
                    max_cache_bytes,
                )
                return total
            cache_bytes = cache.cache_bytes()
            if cache_bytes >= max_cache_bytes:
                LOGGER.info(
                    "Cache warmer paused above cap: cache_bytes=%d/%d; "
                    "cleaning until <=%d",
                    cache_bytes,
                    max_cache_bytes,
                    int(max_cache_bytes * WARMER_RESUME_RATIO),
                )
                while cache.cache_bytes() > int(max_cache_bytes * WARMER_RESUME_RATIO):
                    pressure_permanent = load_permanent_error_set(output)
                    pressure_completed = completed_tasks_for_entity_keys(
                        output,
                        [(e.session_id, e.archive_id) for e in entities],
                    )
                    pressure_terminal = _terminal_entities(
                        output,
                        entities,
                        pressure_completed,
                        pressure_permanent,
                        max_inference_attempts=max_inference_attempts,
                    )
                    pressure_locks = _active_locks(output)
                    pressure_protected = _protected_s3_keys(
                        cache, entities, pressure_locks
                    )
                    cleanup = cache.cleanup(
                        output_base=output,
                        terminal_entities=pressure_terminal,
                        protected_s3_keys=pressure_protected,
                        target_bytes=int(max_cache_bytes * WARMER_RESUME_RATIO),
                    )
                    LOGGER.info(
                        "Cache warmer pressure cleanup: removed_objects=%d "
                        "removed_bytes=%d removed_locks=%d removed_temps=%d "
                        "cache_bytes=%d/%d",
                        cleanup.removed_objects,
                        cleanup.removed_bytes,
                        cleanup.removed_locks,
                        cleanup.removed_temps,
                        cache.cache_bytes(),
                        max_cache_bytes,
                    )
                    time.sleep(scan_interval_sec)
            else:
                LOGGER.info(
                    "Cache warmer sleeping %.1fs before next scan",
                    scan_interval_sec,
                )
                time.sleep(scan_interval_sec)
    finally:
        stop_cleanup.set()
        cleanup_thread.join(timeout=30)
        if cleanup_thread.is_alive():
            LOGGER.warning("Background cache cleaner did not stop within 30s")


def _warm_one_cycle(
    *,
    cache: SharedAudioCache,
    output_base: Path,
    entities: list[ArchiveEntity],
    max_cache_bytes: int,
    max_inference_attempts: int,
    warm_workers: int,
) -> WarmCacheSummary:
    permanent_errors = load_permanent_error_set(output_base)
    completed = completed_tasks_for_entity_keys(
        output_base,
        [(entity.session_id, entity.archive_id) for entity in entities],
    )
    active_locks = _active_locks(output_base)
    protected_s3_keys = _protected_s3_keys(cache, entities, active_locks)
    terminal_entities = _terminal_entities(
        output_base,
        entities,
        completed,
        permanent_errors,
        max_inference_attempts=max_inference_attempts,
    )
    LOGGER.info(
        "Cache warmer scan state: terminal=%d/%d permanent_errors=%d "
        "active_locks=%d protected_s3_keys=%d cache_bytes=%d/%d",
        len(terminal_entities),
        len(entities),
        len(permanent_errors),
        sum(len(values) for values in active_locks.values()),
        len(protected_s3_keys),
        cache.cache_bytes(),
        max_cache_bytes,
    )
    cleanup = cache.cleanup(
        output_base=output_base,
        terminal_entities=terminal_entities,
        protected_s3_keys=protected_s3_keys,
        target_bytes=(
            int(max_cache_bytes * WARMER_RESUME_RATIO)
            if cache.cache_bytes() > max_cache_bytes
            else None
        ),
    )
    LOGGER.info(
        "Cache cleaner cycle: removed_objects=%d removed_bytes=%d "
        "removed_locks=%d removed_temps=%d cache_bytes=%d/%d",
        cleanup.removed_objects,
        cleanup.removed_bytes,
        cleanup.removed_locks,
        cleanup.removed_temps,
        cache.cache_bytes(),
        max_cache_bytes,
    )

    if len(terminal_entities) >= len(entities):
        return WarmCacheSummary(complete=True)

    frontier_by_task = {
        task: _frontier_for_task(
            entities,
            task,
            completed,
            terminal_entities,
            active_locks,
        )
        for task in TASKS
    }
    start = min((idx for idx in frontier_by_task.values() if idx is not None), default=None)
    LOGGER.info(
        "Cache warmer frontiers: %s; slowest_start_index=%s",
        _format_frontiers(frontier_by_task),
        "none" if start is None else start,
    )
    if start is None:
        return WarmCacheSummary(complete=True)

    warmed = 0
    hits = 0
    errors = 0
    fallbacks = 0
    submitted = 0
    completed_futures = 0
    last_progress_log = time.monotonic()
    futures: set[Future] = set()

    consecutive_pressure = 0
    capacity_stalled = False

    def collect_and_log(done: set[Future]) -> None:
        nonlocal warmed
        nonlocal hits
        nonlocal errors
        nonlocal fallbacks
        nonlocal completed_futures
        nonlocal last_progress_log
        nonlocal consecutive_pressure
        nonlocal capacity_stalled
        w, h, e, f, pf = _collect(done)
        warmed += w
        hits += h
        errors += e
        fallbacks += f
        completed_futures += len(done)
        if pf > 0 and w == 0 and h == 0:
            consecutive_pressure += pf
        else:
            consecutive_pressure = 0
        if consecutive_pressure >= CAPACITY_FALLBACK_BREAK_THRESHOLD:
            capacity_stalled = True
        now = time.monotonic()
        if now - last_progress_log >= WARMER_PROGRESS_INTERVAL_SEC:
            LOGGER.info(
                "Cache warmer progress: submitted=%d completed=%d warmed=%d "
                "hits=%d errors=%d fallbacks=%d cache_bytes=%d/%d",
                submitted,
                completed_futures,
                warmed,
                hits,
                errors,
                fallbacks,
                cache.cache_bytes(),
                max_cache_bytes,
            )
            last_progress_log = now

    min_writeable_bytes = min(1_000_000, max_cache_bytes // 1000)

    with ThreadPoolExecutor(max_workers=warm_workers, thread_name_prefix="cache-warm") as pool:
        for entity in entities[start:]:
            if capacity_stalled:
                LOGGER.info(
                    "Cache warmer breaking: %d consecutive capacity fallbacks "
                    "submitted=%d completed=%d cache_bytes=%d/%d",
                    consecutive_pressure,
                    submitted,
                    completed_futures,
                    cache.cache_bytes(),
                    max_cache_bytes,
                )
                break
            if cache.cache_bytes() >= max_cache_bytes - min_writeable_bytes:
                LOGGER.info(
                    "Cache warmer reached cap during submission: "
                    "submitted=%d completed=%d cache_bytes=%d/%d",
                    submitted,
                    completed_futures,
                    cache.cache_bytes(),
                    max_cache_bytes,
                )
                break
            key = (entity.session_id, entity.archive_id)
            if key in terminal_entities or key in permanent_errors:
                continue
            if completed.get(key, set()) >= set(TASKS):
                continue
            futures.add(pool.submit(cache.get_decoded_audio, entity))
            submitted += 1
            while len(futures) >= warm_workers:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                collect_and_log(done)
                if capacity_stalled:
                    break
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            collect_and_log(done)

    LOGGER.info(
        "Cache warmer submissions complete: start_index=%d submitted=%d "
        "completed=%d warmed=%d hits=%d errors=%d fallbacks=%d "
        "cache_bytes=%d/%d",
        start,
        submitted,
        completed_futures,
        warmed,
        hits,
        errors,
        fallbacks,
        cache.cache_bytes(),
        max_cache_bytes,
    )
    return WarmCacheSummary(
        warmed=warmed,
        cache_hits=hits,
        errors=errors,
        fallbacks=fallbacks,
        complete=False,
    )


def _collect(done: set[Future]) -> tuple[int, int, int, int, int]:
    """Return (warmed, hits, errors, fallbacks, pressure_fallbacks)."""
    warmed = hits = errors = fallbacks = pressure_fallbacks = 0
    for future in done:
        try:
            result = future.result()
        except Exception:
            LOGGER.warning("Cache warmer task failed", exc_info=True)
            errors += 1
            continue
        if isinstance(result, AudioResolutionError):
            LOGGER.debug(
                "Cache warmer audio error: %s/%s type=%s detail=%s",
                result.session_id,
                result.archive_id,
                result.error_type,
                result.detail,
            )
            errors += 1
            continue
        if result.stats.object_cache_hit:
            hits += 1
        elif result.stats.cache_write:
            warmed += 1
        if result.stats.cache_fallback:
            fallbacks += 1
            if result.stats.cache_fallback_reason in _PRESSURE_FALLBACK_REASONS:
                pressure_fallbacks += 1
    return warmed, hits, errors, fallbacks, pressure_fallbacks


def _frontier_for_task(
    entities: list[ArchiveEntity],
    task: str,
    completed: dict[tuple[str, str], set[str]],
    terminal_entities: set[tuple[str, str]],
    active_locks: dict[str, set[tuple[str, str]]],
) -> int | None:
    locked_for_task = active_locks.get(task, set()) | active_locks.get("all", set())
    for idx, entity in enumerate(entities):
        key = (entity.session_id, entity.archive_id)
        if key in terminal_entities or key in locked_for_task:
            continue
        if task not in completed.get(key, set()):
            return idx
    return None


def _terminal_entities(
    output_base: Path,
    entities: list[ArchiveEntity],
    completed: dict[tuple[str, str], set[str]],
    permanent_errors: set[tuple[str, str]],
    *,
    max_inference_attempts: int,
) -> set[tuple[str, str]]:
    terminal = set(permanent_errors)
    all_tasks = set(TASKS)
    for key, done in completed.items():
        if done >= all_tasks:
            terminal.add(key)

    group_counts = {
        group: load_inference_attempt_counts(output_base, task_group=group)
        for group in (
            TASK_GROUP_AFFECT,
            TASK_GROUP_DISFLUENCY,
            TASK_GROUP_EMOTION,
            TASK_GROUP_VAD,
        )
    }
    all_mode_counts = load_inference_attempt_counts(output_base, task_group="all")

    for entity in entities:
        key = (entity.session_id, entity.archive_id)
        if all_mode_counts.get((entity.session_id, entity.archive_id, "all"), 0) >= max_inference_attempts:
            terminal.add(key)
            continue
        if all(
            group_counts[group].get((entity.session_id, entity.archive_id, group), 0)
            >= max_inference_attempts
            for group in group_counts
        ):
            terminal.add(key)
    return terminal


def _active_locks(output_base: Path) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = {}
    locks_root = output_base / "_meta" / "locks"
    for lock_file in iter_lock_files(output_base):
        stem = lock_file.stem
        if "__" not in stem:
            continue
        sid, _, aid = stem.partition("__")
        if lock_file.parent == locks_root:
            namespace = "all"
        else:
            namespace = lock_file.parent.name
        result.setdefault(namespace, set()).add((sid, aid))
    return result


def _protected_s3_keys(
    cache: SharedAudioCache,
    entities: list[ArchiveEntity],
    active_locks: dict[str, set[tuple[str, str]]],
) -> set[str]:
    locked_entities: set[tuple[str, str]] = set()
    for keys in active_locks.values():
        locked_entities |= keys
    if not locked_entities:
        return set()
    by_key = {
        (entity.session_id, entity.archive_id): entity
        for entity in entities
    }
    protected: set[str] = set()
    for key in locked_entities:
        entity = by_key.get(key)
        if entity is None:
            continue
        s3_key = cache.cached_s3_key_for_entity(entity)
        if s3_key:
            protected.add(s3_key)
    return protected


def _format_frontiers(frontier_by_task: dict[str, int | None]) -> str:
    return ", ".join(
        f"{task}={'done' if idx is None else idx}"
        for task, idx in frontier_by_task.items()
    )


def _run_cleanup_loop(
    *,
    cache: SharedAudioCache,
    output_base: Path,
    entities_by_key: dict[tuple[str, str], ArchiveEntity],
    max_inference_attempts: int,
    stop_event: threading.Event,
    interval_sec: float,
) -> None:
    """Background thread: periodically evict terminal entities from the cache."""
    first = True
    while not stop_event.is_set():
        if not first:
            stop_event.wait(interval_sec)
            if stop_event.is_set():
                break
        first = False
        try:
            entries = cache._object_entries()
            cached_keys = [
                e["entity_key"] for e in entries if e["entity_key"] != ("", "")
            ]
            if not cached_keys:
                continue
            cached_key_set = set(cached_keys)

            entities_subset = [
                entities_by_key[k] for k in cached_key_set if k in entities_by_key
            ]

            completed = completed_tasks_for_entity_keys(output_base, cached_keys)
            permanent_errors = load_permanent_error_set(output_base)
            terminal = _terminal_entities(
                output_base,
                entities_subset,
                completed,
                permanent_errors,
                max_inference_attempts=max_inference_attempts,
            )
            terminal = terminal & cached_key_set

            if not terminal:
                continue

            active_locks = _active_locks(output_base)
            protected = _protected_s3_keys(
                cache, list(entities_by_key.values()), active_locks
            )

            cleanup = cache.cleanup(
                output_base=output_base,
                terminal_entities=terminal,
                protected_s3_keys=protected,
                target_bytes=None,
            )
            LOGGER.info(
                "Background cache cleaner: removed_objects=%d removed_bytes=%d "
                "removed_locks=%d removed_temps=%d terminal=%d/%d "
                "cache_bytes=%d",
                cleanup.removed_objects,
                cleanup.removed_bytes,
                cleanup.removed_locks,
                cleanup.removed_temps,
                len(terminal),
                len(cached_keys),
                cache.cache_bytes(),
            )
        except Exception:
            LOGGER.warning(
                "Background cache cleaner iteration failed", exc_info=True
            )
            stop_event.wait(min(interval_sec * 2, 300))
