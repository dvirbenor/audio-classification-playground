import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_classification_playground.acoustic_events.inference.audio import AudioData
from audio_classification_playground.acoustic_events.orchestration.audio_cache import (
    CachedAudio,
    CacheAcquisitionStats,
    SharedAudioCache,
    decoded_object_key,
    resolution_cache_key,
)
from audio_classification_playground.acoustic_events.orchestration.audio_resolver import (
    AudioDownloadResult,
)
from audio_classification_playground.acoustic_events.orchestration.manifest import (
    ArchiveEntity,
)
from audio_classification_playground.acoustic_events.orchestration.prefetch import (
    Prefetcher,
)


class SharedAudioCacheTest(unittest.TestCase):
    def test_hash_keys_are_filename_safe_for_weird_inputs(self):
        resolution = resolution_cache_key("bucket", "prefix/with spaces/יוניקוד")
        decoded = decoded_object_key("a/b/c+weird name.wav", 16000)

        self.assertRegex(resolution, r"^[0-9a-f]{32}$")
        self.assertRegex(decoded, r"^[0-9a-f]{32}$")

    def test_resolution_cache_avoids_repeated_s3_list(self):
        entity = ArchiveEntity("s1", "a1", "prefix/a1")
        calls = []

        def fake_resolve(s3_client, file_parent_dir, bucket):
            calls.append((file_parent_dir, bucket))
            return "accounts/a1.wav"

        with tempfile.TemporaryDirectory() as tmp:
            cache = SharedAudioCache(
                tmp,
                sample_rate=16000,
                max_cache_bytes=10_000_000,
                s3_client=object(),
                poll_sec=0.01,
            )
            with patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".audio_cache.resolve_audio_key",
                side_effect=fake_resolve,
            ):
                first = cache.resolve(entity)
                second = cache.resolve(entity)

        self.assertEqual(first.s3_key, "accounts/a1.wav")
        self.assertEqual(second.s3_key, "accounts/a1.wav")
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(calls), 1)

    def test_decoded_cache_hit_skips_download_and_decode(self):
        entity = ArchiveEntity("s1", "a1", "prefix/a1")
        calls = {"download": 0, "decode": 0}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            source = tmp_root / "source.wav"

            def fake_download(s3_client, key, bucket, tmp_dir=None):
                calls["download"] += 1
                source.write_bytes(b"fake")
                return AudioDownloadResult(
                    local_path=source,
                    s3_key=key,
                    object_size_bytes=4,
                    storage_class="STANDARD",
                    source_extension=".wav",
                    head_sec=0.1,
                    download_sec=0.2,
                )

            def fake_load_audio(path, *, sample_rate, recording_id):
                calls["decode"] += 1
                samples = np.arange(sample_rate, dtype=np.float32)
                return AudioData(
                    path=Path(path),
                    recording_id=recording_id,
                    samples=samples,
                    sample_rate=sample_rate,
                    duration_sec=1.0,
                    audio_sha256="sha-a1",
                )

            cache = SharedAudioCache(
                tmp_root / "cache",
                sample_rate=16000,
                max_cache_bytes=10_000_000,
                s3_client=object(),
                poll_sec=0.01,
            )
            with patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".audio_cache.resolve_audio_key",
                return_value="accounts/a1.wav",
            ), patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".audio_cache.download_audio",
                side_effect=fake_download,
            ), patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".audio_cache.load_audio",
                side_effect=fake_load_audio,
            ):
                first = cache.get_decoded_audio(entity)
                second = cache.get_decoded_audio(entity)

        self.assertFalse(first.stats.object_cache_hit)
        self.assertTrue(first.stats.cache_write)
        self.assertTrue(second.stats.object_cache_hit)
        self.assertEqual(second.audio.audio_sha256, "sha-a1")
        self.assertEqual(calls, {"download": 1, "decode": 1})

    def test_cache_pressure_falls_back_without_writing_shared_object(self):
        entity = ArchiveEntity("s1", "a1", "prefix/a1")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            source = tmp_root / "source.wav"

            def fake_download(s3_client, key, bucket, tmp_dir=None):
                source.write_bytes(b"fake")
                return AudioDownloadResult(local_path=source, s3_key=key)

            def fake_load_audio(path, *, sample_rate, recording_id):
                samples = np.ones(sample_rate, dtype=np.float32)
                return AudioData(
                    path=Path(path),
                    recording_id=recording_id,
                    samples=samples,
                    sample_rate=sample_rate,
                    duration_sec=1.0,
                    audio_sha256="sha-a1",
                )

            cache = SharedAudioCache(
                tmp_root / "cache",
                sample_rate=16000,
                max_cache_bytes=1,
                s3_client=object(),
                poll_sec=0.01,
            )
            with patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".audio_cache.resolve_audio_key",
                return_value="accounts/a1.wav",
            ), patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".audio_cache.download_audio",
                side_effect=fake_download,
            ), patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".audio_cache.load_audio",
                side_effect=fake_load_audio,
            ):
                result = cache.get_decoded_audio(entity)

        self.assertTrue(result.stats.cache_fallback)
        self.assertEqual(result.stats.cache_fallback_reason, "capacity")
        self.assertEqual(cache.cache_bytes(), 0)

    def test_reclaim_stale_removes_object_lock_and_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SharedAudioCache(
                tmp,
                sample_rate=16000,
                max_cache_bytes=10_000_000,
                stale_lock_minutes=0.001,
                poll_sec=0.01,
            )
            key = decoded_object_key("accounts/a1.wav", 16000)
            lock = cache._object_lock_path(key)
            object_path, _ = cache._object_paths(key)
            lock.parent.mkdir(parents=True, exist_ok=True)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            lock.write_text("stale", encoding="utf-8")
            tmp_file = object_path.parent / ".tmp.test.npy"
            tmp_file.write_bytes(b"partial")
            old = time.time() - 120
            os.utime(lock, (old, old))
            os.utime(tmp_file, (old, old))

            locks, temps, reservations = cache.reclaim_stale()

        self.assertEqual(locks, 1)
        self.assertGreaterEqual(temps, 1)
        self.assertEqual(reservations, 0)
        self.assertFalse(lock.exists())
        self.assertFalse(tmp_file.exists())

    def test_reclaim_stale_removes_stale_reservation_and_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SharedAudioCache(
                tmp,
                sample_rate=16000,
                max_cache_bytes=10_000_000,
                stale_lock_minutes=0.001,
                poll_sec=0.01,
            )
            key = decoded_object_key("accounts/a1.wav", 16000)
            reservation = cache._reservation_path(key)
            reservation.parent.mkdir(parents=True, exist_ok=True)
            reservation.write_text(
                '{"object_key":"%s","bytes":64000}' % key,
                encoding="utf-8",
            )
            tmp_file = reservation.parent / ".tmp.reservation.json"
            tmp_file.write_text("partial", encoding="utf-8")
            old = time.time() - 120
            os.utime(reservation, (old, old))
            os.utime(tmp_file, (old, old))

            locks, temps, reservations = cache.reclaim_stale()

        self.assertEqual(locks, 0)
        self.assertGreaterEqual(temps, 1)
        self.assertEqual(reservations, 1)
        self.assertFalse(reservation.exists())
        self.assertFalse(tmp_file.exists())

    def test_reclaim_stale_removes_reservation_for_ready_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = SharedAudioCache(
                tmp,
                sample_rate=16000,
                max_cache_bytes=10_000_000,
                stale_lock_minutes=60.0,
                poll_sec=0.01,
            )
            key = decoded_object_key("accounts/a1.wav", 16000)
            reservation = cache._reservation_path(key)
            object_path, metadata_path = cache._object_paths(key)
            reservation.parent.mkdir(parents=True, exist_ok=True)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            reservation.write_text(
                '{"object_key":"%s","bytes":64000}' % key,
                encoding="utf-8",
            )
            object_path.write_bytes(b"ready")
            metadata_path.write_text("{}", encoding="utf-8")

            locks, temps, reservations = cache.reclaim_stale()

        self.assertEqual(locks, 0)
        self.assertEqual(temps, 0)
        self.assertEqual(reservations, 1)
        self.assertFalse(reservation.exists())

    def test_cleaner_refuses_to_evict_active_task_lock(self):
        entity = ArchiveEntity("s1", "a1", "prefix/a1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = SharedAudioCache(
                root / "cache",
                sample_rate=16000,
                max_cache_bytes=10_000_000,
                poll_sec=0.01,
            )
            key = decoded_object_key("accounts/a1.wav", 16000)
            object_path, metadata_path = cache._object_paths(key)
            cache._write_decoded_object(
                object_path,
                metadata_path,
                samples=np.zeros(16000, dtype=np.float32),
                metadata={
                    "version": "decoded-f32-mono-v1",
                    "s3_key": "accounts/a1.wav",
                    "audio_sha256": "sha-a1",
                    "sample_rate": 16000,
                    "duration_sec": 1.0,
                    "decoded_bytes": 64000,
                    "first_entity": {
                        "session_id": entity.session_id,
                        "archive_id": entity.archive_id,
                    },
                },
            )
            locks_dir = root / "out" / "_meta" / "locks" / "vad"
            locks_dir.mkdir(parents=True)
            (locks_dir / "s1__a1.lock").write_text("active", encoding="utf-8")

            summary = cache.cleanup(
                output_base=root / "out",
                terminal_entities={("s1", "a1")},
                target_bytes=0,
            )

            self.assertEqual(summary.removed_objects, 0)
            self.assertTrue(object_path.exists())


class PrefetcherSharedCacheOwnershipTest(unittest.TestCase):
    def test_discard_does_not_unlink_shared_cache_path(self):
        entity = ArchiveEntity("s1", "a1", "prefix/a1")

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cached.npy"
            cache_path.write_bytes(b"shared")

            class FakeCache:
                def get_decoded_audio(self, entity, *, tmp_dir=None):
                    samples = np.zeros(16000, dtype=np.float32)
                    return CachedAudio(
                        audio=AudioData(
                            path=cache_path,
                            recording_id=entity.archive_id,
                            samples=samples,
                            sample_rate=16000,
                            duration_sec=1.0,
                            audio_sha256="sha-a1",
                        ),
                        s3_key="accounts/a1.wav",
                        source_extension=".wav",
                        cache_path=cache_path,
                        stats=CacheAcquisitionStats(object_cache_hit=True),
                    )

            pf = Prefetcher(max_workers=1, vad_workers=0, audio_cache=FakeCache())
            try:
                pf.submit(entity)
                result = pf.get(entity)
                self.assertTrue(result.object_cache_hit)
                pf.discard(entity)
            finally:
                pf.shutdown()

            self.assertTrue(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
