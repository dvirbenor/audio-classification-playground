"""Warm the shared decoded-audio cache ahead of task-fleet workers."""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from ..inference.artifacts import SAMPLE_RATE
from .audio_cache import SharedAudioCache
from .audio_resolver import AudioResolutionError, BUCKET
from .errors import load_inference_attempt_counts, load_permanent_error_set
from .locking import iter_lock_files
from .manifest import ArchiveEntity, load_manifest
from .progress import TASKS, _walk_completed_tasks
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
    sample_rate: int = SAMPLE_RATE,
    scan_interval_sec: float = WARMER_SCAN_INTERVAL_SEC,
    once: bool = False,
) -> WarmCacheSummary:
    """Run the decoded-cache warmer.

    ``once`` is intended for tests and short validation runs.  Production CLI
    usage leaves it ``False`` so the warmer keeps rescanning until the
    manifest is terminal.
    """
    output = Path(output_base)
    entities = load_manifest(parquet_path)
    rng = random.Random(seed)
    rng.shuffle(entities)
    cache = SharedAudioCache(
        audio_cache_dir,
        sample_rate=sample_rate,
        max_cache_bytes=max_cache_bytes,
        stale_lock_minutes=audio_cache_lock_stale_minutes,
        bucket=BUCKET,
    )

    total = WarmCacheSummary()
    while True:
        cycle = _warm_one_cycle(
            cache=cache,
            output_base=output,
            entities=entities,
            max_cache_bytes=max_cache_bytes,
            max_inference_attempts=max_inference_attempts,
        )
        total = WarmCacheSummary(
            warmed=total.warmed + cycle.warmed,
            cache_hits=total.cache_hits + cycle.cache_hits,
            errors=total.errors + cycle.errors,
            fallbacks=total.fallbacks + cycle.fallbacks,
            complete=cycle.complete,
        )
        if cycle.complete or once:
            return total
        cache_bytes = cache.cache_bytes()
        LOGGER.info(
            "Cache warmer cycle: warmed=%d hits=%d errors=%d fallbacks=%d "
            "cache_bytes=%d/%d",
            cycle.warmed,
            cycle.cache_hits,
            cycle.errors,
            cycle.fallbacks,
            cache_bytes,
            max_cache_bytes,
        )
        if cache_bytes >= max_cache_bytes:
            while cache.cache_bytes() > int(max_cache_bytes * WARMER_RESUME_RATIO):
                cache.cleanup(
                    output_base=output,
                    target_bytes=int(max_cache_bytes * WARMER_RESUME_RATIO),
                )
                time.sleep(scan_interval_sec)
        else:
            time.sleep(scan_interval_sec)


def _warm_one_cycle(
    *,
    cache: SharedAudioCache,
    output_base: Path,
    entities: list[ArchiveEntity],
    max_cache_bytes: int,
    max_inference_attempts: int,
) -> WarmCacheSummary:
    permanent_errors = load_permanent_error_set(output_base)
    completed = _walk_completed_tasks(output_base)
    active_locks = _active_locks(output_base)
    protected_s3_keys = _protected_s3_keys(cache, entities, active_locks)
    terminal_entities = _terminal_entities(
        output_base,
        entities,
        completed,
        permanent_errors,
        max_inference_attempts=max_inference_attempts,
    )
    cache.cleanup(
        output_base=output_base,
        terminal_entities=terminal_entities,
        protected_s3_keys=protected_s3_keys,
        target_bytes=(
            int(max_cache_bytes * WARMER_RESUME_RATIO)
            if cache.cache_bytes() > max_cache_bytes
            else None
        ),
    )

    if len(terminal_entities) >= len(entities):
        return WarmCacheSummary(complete=True)

    frontiers = [
        _frontier_for_task(
            entities,
            task,
            completed,
            terminal_entities,
            active_locks,
        )
        for task in TASKS
    ]
    start = min((idx for idx in frontiers if idx is not None), default=None)
    if start is None:
        return WarmCacheSummary(complete=True)

    warmed = 0
    hits = 0
    errors = 0
    fallbacks = 0
    futures: set[Future] = set()

    with ThreadPoolExecutor(max_workers=WARMER_WORKERS, thread_name_prefix="cache-warm") as pool:
        for entity in entities[start:]:
            if cache.cache_bytes() >= max_cache_bytes:
                break
            key = (entity.session_id, entity.archive_id)
            if key in terminal_entities or key in permanent_errors:
                continue
            if completed.get(key, set()) >= set(TASKS):
                continue
            futures.add(pool.submit(cache.get_decoded_audio, entity))
            while len(futures) >= WARMER_WORKERS:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                w, h, e, f = _collect(done)
                warmed += w
                hits += h
                errors += e
                fallbacks += f
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            w, h, e, f = _collect(done)
            warmed += w
            hits += h
            errors += e
            fallbacks += f

    return WarmCacheSummary(
        warmed=warmed,
        cache_hits=hits,
        errors=errors,
        fallbacks=fallbacks,
        complete=False,
    )


def _collect(done: set[Future]) -> tuple[int, int, int, int]:
    warmed = hits = errors = fallbacks = 0
    for future in done:
        try:
            result = future.result()
        except Exception:
            LOGGER.warning("Cache warmer task failed", exc_info=True)
            errors += 1
            continue
        if isinstance(result, AudioResolutionError):
            errors += 1
            continue
        if result.stats.object_cache_hit:
            hits += 1
        elif result.stats.cache_write:
            warmed += 1
        if result.stats.cache_fallback:
            fallbacks += 1
    return warmed, hits, errors, fallbacks


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
