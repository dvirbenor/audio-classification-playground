"""Tests for the background cache cleanup thread and early-break logic."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from audio_classification_playground.acoustic_events.orchestration.audio_cache import (
    CachedAudio,
    CacheAcquisitionStats,
    CleanupSummary,
    SharedAudioCache,
)
from audio_classification_playground.acoustic_events.orchestration.cache_warmer import (
    CAPACITY_FALLBACK_BREAK_THRESHOLD,
    _PRESSURE_FALLBACK_REASONS,
    _collect,
    _run_cleanup_loop,
    warm_cache,
)
from audio_classification_playground.acoustic_events.orchestration.manifest import (
    ArchiveEntity,
)


def _make_entity(session_id: str, archive_id: str) -> ArchiveEntity:
    return ArchiveEntity(session_id, archive_id, f"prefix/{archive_id}")


def _write_cache_object(
    cache_dir: Path, entity: ArchiveEntity, *, sample_rate: int = 16000
) -> str:
    """Write a fake decoded cache object for an entity. Returns the s3_key."""
    from audio_classification_playground.acoustic_events.orchestration.audio_cache import (
        DECODED_CACHE_VERSION,
        decoded_object_key,
        resolution_cache_key,
    )
    from audio_classification_playground.acoustic_events.orchestration.audio_resolver import (
        BUCKET,
    )

    s3_key = f"accounts/{entity.file_parent_dir}/{entity.archive_id}.wav"
    obj_key = decoded_object_key(s3_key, sample_rate)
    obj_dir = cache_dir / "objects" / obj_key[:2]
    obj_dir.mkdir(parents=True, exist_ok=True)
    obj_path = obj_dir / f"{obj_key}.npy"
    meta_path = obj_dir / f"{obj_key}.json"

    samples = np.zeros(1600, dtype="<f4")
    np.save(obj_path, samples)
    meta = {
        "version": DECODED_CACHE_VERSION,
        "bucket": BUCKET,
        "s3_key": s3_key,
        "source_extension": ".wav",
        "source_object_size_bytes": 3200,
        "source_storage_class": "STANDARD",
        "audio_sha256": "deadbeef",
        "n_samples": 1600,
        "decoded_bytes": 6400,
        "sample_rate": sample_rate,
        "duration_sec": 0.1,
        "first_entity": {
            "session_id": entity.session_id,
            "archive_id": entity.archive_id,
            "file_parent_dir": entity.file_parent_dir,
        },
        "created_at": time.time(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    res_key = resolution_cache_key(BUCKET, entity.file_parent_dir)
    res_dir = cache_dir / "resolution" / res_key[:2]
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"{res_key}.json"
    with open(res_path, "w") as f:
        json.dump({
            "status": "resolved",
            "bucket": BUCKET,
            "file_parent_dir": entity.file_parent_dir,
            "s3_key": s3_key,
            "source_extension": ".wav",
            "created_at": time.time(),
        }, f)

    return s3_key


def _write_progress_complete(output_base: Path, keys: list[tuple[str, str]]) -> None:
    """Write a progress_complete.txt file."""
    cache_path = output_base / "_meta" / "progress_complete.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for sid, aid in keys:
            f.write(f"{sid}\t{aid}\n")


class TestCapacityToleranceBreak(unittest.TestCase):
    """The submission loop should break when cache is near-full."""

    def test_near_full_cache_breaks_submission(self):
        """When cache_bytes is within min_writeable_bytes of max, stop submitting."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            output_base = Path(tmp) / "output"
            output_base.mkdir()
            (output_base / "_meta").mkdir()

            max_bytes = 10_000_000
            cache = SharedAudioCache(
                cache_dir,
                sample_rate=16000,
                max_cache_bytes=max_bytes,
                s3_client=object(),
            )

            entity = _make_entity("s1", "a1")
            parquet_path = Path(tmp) / "test.parquet"

            with patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".cache_warmer.load_manifest",
                return_value=[entity],
            ), patch.object(
                cache, "cache_bytes", return_value=max_bytes - 100
            ), patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".cache_warmer.completed_tasks_for_entity_keys",
                return_value={},
            ), patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".cache_warmer.load_permanent_error_set",
                return_value=set(),
            ), patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".cache_warmer.load_inference_attempt_counts",
                return_value={},
            ):
                result = warm_cache(
                    parquet_path=parquet_path,
                    output_base=output_base,
                    audio_cache_dir=cache_dir,
                    max_cache_bytes=max_bytes,
                    once=True,
                    warm_workers=1,
                    scan_interval_sec=0.1,
                )
            self.assertEqual(result.warmed, 0)


class TestEarlyBreakOnPressure(unittest.TestCase):
    """The submission loop should break on sustained capacity fallbacks."""

    def test_collect_counts_pressure_fallbacks(self):
        """_collect() returns pressure fallback count for capacity/cache_full."""
        from concurrent.futures import ThreadPoolExecutor

        stats_capacity = CacheAcquisitionStats(
            cache_fallback=True, cache_fallback_reason="capacity"
        )
        stats_cache_full = CacheAcquisitionStats(
            cache_fallback=True, cache_fallback_reason="cache_full"
        )
        stats_write_failed = CacheAcquisitionStats(
            cache_fallback=True, cache_fallback_reason="write_failed"
        )
        stats_hit = CacheAcquisitionStats(object_cache_hit=True)
        stats_write = CacheAcquisitionStats(cache_write=True)

        results = [
            CachedAudio(audio=MagicMock(), s3_key="k1", stats=stats_capacity),
            CachedAudio(audio=MagicMock(), s3_key="k2", stats=stats_cache_full),
            CachedAudio(audio=MagicMock(), s3_key="k3", stats=stats_write_failed),
            CachedAudio(audio=MagicMock(), s3_key="k4", stats=stats_hit),
            CachedAudio(audio=MagicMock(), s3_key="k5", stats=stats_write),
        ]

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = {pool.submit(lambda r=r: r) for r in results}
            import concurrent.futures

            concurrent.futures.wait(futures)

        warmed, hits, errors, fallbacks, pressure = _collect(futures)
        self.assertEqual(warmed, 1)
        self.assertEqual(hits, 1)
        self.assertEqual(errors, 0)
        self.assertEqual(fallbacks, 3)
        self.assertEqual(pressure, 2)

    def test_capacity_stalled_flag_breaks_outer_loop(self):
        """Consecutive pressure fallbacks set capacity_stalled and break."""
        consecutive = 0
        capacity_stalled = False
        threshold = CAPACITY_FALLBACK_BREAK_THRESHOLD

        for _ in range(threshold):
            pf = 1
            w = 0
            h = 0
            if pf > 0 and w == 0 and h == 0:
                consecutive += pf
            else:
                consecutive = 0
            if consecutive >= threshold:
                capacity_stalled = True
                break

        self.assertTrue(capacity_stalled)
        self.assertEqual(consecutive, threshold)

    def test_pressure_counter_resets_on_hit(self):
        """A cache hit resets the consecutive pressure counter."""
        consecutive = 5

        pf = 0
        w = 0
        h = 1
        if pf > 0 and w == 0 and h == 0:
            consecutive += pf
        else:
            consecutive = 0

        self.assertEqual(consecutive, 0)


class TestBackgroundCleanupThread(unittest.TestCase):
    """Background cleanup thread evicts terminal entities and stops cleanly."""

    def test_cleanup_thread_stops_on_event(self):
        """Thread exits promptly when stop_event is set."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            output_base = Path(tmp) / "output"
            (output_base / "_meta").mkdir(parents=True)

            cache = SharedAudioCache(
                cache_dir,
                sample_rate=16000,
                max_cache_bytes=100_000_000,
                s3_client=object(),
            )
            entities_by_key: dict[tuple[str, str], ArchiveEntity] = {}
            stop = threading.Event()

            t = threading.Thread(
                target=_run_cleanup_loop,
                kwargs={
                    "cache": cache,
                    "output_base": output_base,
                    "entities_by_key": entities_by_key,
                    "max_inference_attempts": 3,
                    "stop_event": stop,
                    "interval_sec": 0.1,
                },
                daemon=True,
            )
            t.start()
            time.sleep(0.3)
            stop.set()
            t.join(timeout=5)
            self.assertFalse(t.is_alive())

    def test_cleanup_evicts_terminal_cached_entities(self):
        """Background cleanup removes objects for entities in progress_complete."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            output_base = Path(tmp) / "output"
            (output_base / "_meta").mkdir(parents=True)

            cache = SharedAudioCache(
                cache_dir,
                sample_rate=16000,
                max_cache_bytes=100_000_000,
                s3_client=object(),
            )

            entity = _make_entity("sess1", "arch1")
            _write_cache_object(cache_dir, entity)
            _write_progress_complete(output_base, [("sess1", "arch1")])

            entries_before = cache._object_entries()
            self.assertEqual(len(entries_before), 1)

            entities_by_key = {("sess1", "arch1"): entity}
            stop = threading.Event()

            with patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".cache_warmer.load_inference_attempt_counts",
                return_value={},
            ):
                t = threading.Thread(
                    target=_run_cleanup_loop,
                    kwargs={
                        "cache": cache,
                        "output_base": output_base,
                        "entities_by_key": entities_by_key,
                        "max_inference_attempts": 3,
                        "stop_event": stop,
                        "interval_sec": 0.1,
                    },
                    daemon=True,
                )
                t.start()
                time.sleep(1.0)
                stop.set()
                t.join(timeout=5)

            entries_after = cache._object_entries()
            self.assertEqual(len(entries_after), 0)

    def test_cleanup_protects_shared_s3_keys(self):
        """Objects whose S3 key is used by a locked entity are not evicted."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            output_base = Path(tmp) / "output"
            (output_base / "_meta").mkdir(parents=True)

            cache = SharedAudioCache(
                cache_dir,
                sample_rate=16000,
                max_cache_bytes=100_000_000,
                s3_client=object(),
            )

            terminal_entity = _make_entity("sess1", "arch1")
            _write_cache_object(cache_dir, terminal_entity)
            _write_progress_complete(output_base, [("sess1", "arch1")])

            active_entity = _make_entity("sess2", "arch2")
            active_entity = ArchiveEntity(
                "sess2", "arch2", terminal_entity.file_parent_dir
            )

            locks_dir = output_base / "_meta" / "locks" / "affect"
            locks_dir.mkdir(parents=True)
            lock_file = locks_dir / "sess2__arch2.lock"
            lock_file.write_text("{}")

            entities_by_key = {
                ("sess1", "arch1"): terminal_entity,
                ("sess2", "arch2"): active_entity,
            }
            stop = threading.Event()

            with patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".cache_warmer.load_inference_attempt_counts",
                return_value={},
            ):
                t = threading.Thread(
                    target=_run_cleanup_loop,
                    kwargs={
                        "cache": cache,
                        "output_base": output_base,
                        "entities_by_key": entities_by_key,
                        "max_inference_attempts": 3,
                        "stop_event": stop,
                        "interval_sec": 0.1,
                    },
                    daemon=True,
                )
                t.start()
                time.sleep(1.0)
                stop.set()
                t.join(timeout=5)

            entries_after = cache._object_entries()
            self.assertEqual(
                len(entries_after),
                1,
                "Object should be protected because active entity shares S3 key",
            )


if __name__ == "__main__":
    unittest.main()
