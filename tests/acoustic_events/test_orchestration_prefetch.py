import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_classification_playground.acoustic_events.inference.audio import AudioData
from audio_classification_playground.acoustic_events.orchestration.audio_resolver import (
    AudioDownloadResult,
    AudioResolutionError,
)
from audio_classification_playground.acoustic_events.orchestration.manifest import (
    ArchiveEntity,
)
from audio_classification_playground.acoustic_events.orchestration.prefetch import (
    PrefetchResult,
    Prefetcher,
)


class PrefetcherTest(unittest.TestCase):
    def test_vad_runs_only_when_requested_and_returns_timings(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        vad_calls = []

        def vad_factory():
            def detector(samples, sample_rate):
                vad_calls.append((len(samples), sample_rate))
                return [(0.0, 0.25)]

            return detector

        with tempfile.TemporaryDirectory() as tmp:
            with _patched_audio(tmp):
                pf = Prefetcher(
                    max_workers=1,
                    vad_workers=1,
                    vad_detector_factory=vad_factory,
                    prewarm_vad=False,
                )
                try:
                    pf.submit(entity, precompute_vad=False)
                    result = pf.get(entity)
                    self.assertIsInstance(result, PrefetchResult)
                    self.assertIsNone(result.vad_intervals)
                    self.assertEqual(vad_calls, [])
                    self.assertGreaterEqual(result.timings.download_decode_sec, 0.0)
                    self.assertGreaterEqual(result.timings.decode_queue_wait_sec, 0.0)
                    self.assertGreaterEqual(result.timings.vad_queue_wait_sec, 0.0)
                    self.assertGreaterEqual(
                        result.timings.prefetch_submit_to_ready_sec, 0.0,
                    )
                    self.assertGreaterEqual(result.ready_time, 0.0)

                    pf.discard(entity)
                    pf.submit(entity, precompute_vad=True)
                    result = pf.get(entity)
                    self.assertEqual(result.vad_intervals, [(0.0, 0.25)])
                    self.assertEqual(vad_calls, [(16000, 16000)])
                    self.assertGreaterEqual(result.timings.vad_sec, 0.0)
                finally:
                    pf.shutdown()

    def test_is_ready_and_wait_any_handle_missing_or_empty_inputs(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        pf = Prefetcher(max_workers=1, vad_workers=0)
        try:
            self.assertFalse(pf.is_ready(entity))
            self.assertFalse(pf.wait_any([]))
            self.assertFalse(pf.wait_any([entity], timeout_sec=0.0))
        finally:
            pf.shutdown()

    def test_decode_queue_wait_is_recorded_when_pool_is_backed_up(self):
        entities = [
            ArchiveEntity("s1", "a1", "prefix/a1"),
            ArchiveEntity("s1", "a2", "prefix/a2"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)

            def fake_resolve_and_download(**kwargs):
                archive_id = kwargs["archive_id"]
                path = tmp_root / f"{archive_id}.wav"
                path.write_bytes(b"fake")
                return path, f"{archive_id}.wav"

            def fake_load_audio(path, *, sample_rate, recording_id):
                if recording_id == "a1":
                    time.sleep(0.05)
                samples = np.zeros(sample_rate, dtype=np.float32)
                return AudioData(
                    path=Path(path),
                    recording_id=recording_id,
                    samples=samples,
                    sample_rate=sample_rate,
                    duration_sec=1.0,
                    audio_sha256=f"hash-{recording_id}",
                )

            with patch.multiple(
                "audio_classification_playground.acoustic_events.orchestration.prefetch",
                resolve_and_download=fake_resolve_and_download,
                load_audio=fake_load_audio,
            ):
                pf = Prefetcher(max_workers=1, vad_workers=0)
                try:
                    for entity in entities:
                        pf.submit(entity, precompute_vad=False)
                    results = [pf.get(entity) for entity in entities]
                finally:
                    pf.shutdown()

        waits = [result.timings.decode_queue_wait_sec for result in results]
        self.assertGreater(max(waits), 0.0)

    def test_vad_queue_wait_is_recorded_when_vad_pool_is_backed_up(self):
        entities = [
            ArchiveEntity("s1", "a1", "prefix/a1"),
            ArchiveEntity("s1", "a2", "prefix/a2"),
        ]

        def vad_factory():
            def detector(samples, sample_rate):
                time.sleep(0.05)
                return [(0.0, 0.1)]

            return detector

        with tempfile.TemporaryDirectory() as tmp:
            with _patched_audio(tmp):
                pf = Prefetcher(
                    max_workers=2,
                    vad_workers=1,
                    vad_detector_factory=vad_factory,
                    prewarm_vad=False,
                )
                try:
                    for entity in entities:
                        pf.submit(entity, precompute_vad=True)
                    results = [pf.get(entity) for entity in entities]
                finally:
                    pf.shutdown()

        waits = [result.timings.vad_queue_wait_sec for result in results]
        self.assertGreater(max(waits), 0.0)

    def test_audio_download_result_metadata_reaches_prefetch_result(self):
        entity = ArchiveEntity("s1", "a1", "prefix")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            audio_path = tmp_root / "a1.wav"
            audio_path.write_bytes(b"fake")

            def fake_resolve(**kwargs):
                return AudioDownloadResult(
                    local_path=audio_path,
                    s3_key="accounts/studio/takes/a1.wav",
                    object_size_bytes=123,
                    storage_class="STANDARD",
                    source_extension=".wav",
                )

            def fake_load_audio(path, *, sample_rate, recording_id):
                samples = np.zeros(sample_rate, dtype=np.float32)
                return AudioData(
                    path=Path(path),
                    recording_id=recording_id,
                    samples=samples,
                    sample_rate=sample_rate,
                    duration_sec=1.0,
                    audio_sha256=f"hash-{recording_id}",
                )

            with patch.multiple(
                "audio_classification_playground.acoustic_events.orchestration.prefetch",
                resolve_and_download=fake_resolve,
                load_audio=fake_load_audio,
            ):
                pf = Prefetcher(max_workers=1, vad_workers=0)
                try:
                    pf.submit(entity, precompute_vad=False)
                    result = pf.get(entity)
                finally:
                    pf.shutdown()

        self.assertEqual(result.s3_key, "accounts/studio/takes/a1.wav")
        self.assertEqual(result.audio_source_extension, ".wav")
        self.assertEqual(result.audio_object_size_bytes, 123)
        self.assertEqual(result.audio_storage_class, "STANDARD")

    def test_audio_download_result_tuple_unpacking_is_legacy_compatible(self):
        local_path = Path("/tmp/audio.wav")
        result = AudioDownloadResult(local_path=local_path, s3_key="key.wav")
        path, key = result
        self.assertEqual(path, local_path)
        self.assertEqual(key, "key.wav")

    def test_vad_detector_factory_is_thread_local(self):
        entities = [
            ArchiveEntity("s1", "a1", "prefix/a1"),
            ArchiveEntity("s1", "a2", "prefix/a2"),
        ]
        factory_threads = []

        def vad_factory():
            factory_threads.append(threading.get_ident())

            def detector(samples, sample_rate):
                return [(0.0, 0.1)]

            return detector

        with tempfile.TemporaryDirectory() as tmp:
            with _patched_audio(tmp):
                pf = Prefetcher(
                    max_workers=2,
                    vad_workers=1,
                    vad_detector_factory=vad_factory,
                    prewarm_vad=True,
                )
                try:
                    for entity in entities:
                        pf.submit(entity, precompute_vad=True)
                    for entity in entities:
                        result = pf.get(entity)
                        self.assertEqual(result.vad_intervals, [(0.0, 0.1)])
                finally:
                    pf.shutdown()

        self.assertEqual(len(factory_threads), 2)
        self.assertEqual(len(set(factory_threads)), 2)

    def test_vad_failure_raises_from_get(self):
        entity = ArchiveEntity("s1", "a1", "prefix")

        def vad_factory():
            def detector(samples, sample_rate):
                raise RuntimeError("vad boom")

            return detector

        with tempfile.TemporaryDirectory() as tmp:
            with _patched_audio(tmp):
                pf = Prefetcher(
                    max_workers=1,
                    vad_workers=1,
                    vad_detector_factory=vad_factory,
                    prewarm_vad=False,
                )
                try:
                    pf.submit(entity, precompute_vad=True)
                    with self.assertRaisesRegex(RuntimeError, "vad boom"):
                        pf.get(entity)
                finally:
                    pf.shutdown()

    def test_audio_resolution_error_is_returned(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        error = AudioResolutionError("s1", "a1", "prefix", "no_matching_file", "missing")

        with patch(
            "audio_classification_playground.acoustic_events.orchestration.prefetch.resolve_and_download",
            return_value=error,
        ):
            pf = Prefetcher(max_workers=1, vad_workers=0)
            try:
                pf.submit(entity, precompute_vad=False)
                result = pf.get(entity)
                self.assertIsInstance(result, AudioResolutionError)
                self.assertEqual(result.error_type, error.error_type)
                self.assertEqual(result.detail, error.detail)
                self.assertGreater(result.ready_time, 0.0)
            finally:
                pf.shutdown()


def _patched_audio(tmp_dir: str):
    tmp_root = Path(tmp_dir)

    def fake_resolve_and_download(**kwargs):
        archive_id = kwargs["archive_id"]
        path = tmp_root / f"{archive_id}.wav"
        path.write_bytes(b"fake")
        return path, f"{archive_id}.wav"

    def fake_load_audio(path, *, sample_rate, recording_id):
        samples = np.zeros(sample_rate, dtype=np.float32)
        return AudioData(
            path=Path(path),
            recording_id=recording_id,
            samples=samples,
            sample_rate=sample_rate,
            duration_sec=1.0,
            audio_sha256=f"hash-{recording_id}",
        )

    return patch.multiple(
        "audio_classification_playground.acoustic_events.orchestration.prefetch",
        resolve_and_download=fake_resolve_and_download,
        load_audio=fake_load_audio,
    )


# ===========================================================================
# AudioResolutionError classification and s3_key propagation
# ===========================================================================


class AudioResolutionErrorClassificationTest(unittest.TestCase):

    def test_no_matching_file_is_permanent(self):
        err = AudioResolutionError("s1", "a1", "pfx", "no_matching_file")
        self.assertTrue(err.is_permanent)

    def test_glacier_storage_class_is_transient(self):
        err = AudioResolutionError("s1", "a1", "pfx", "glacier_storage_class")
        self.assertFalse(err.is_permanent)

    def test_download_failed_is_transient(self):
        err = AudioResolutionError("s1", "a1", "pfx", "download_failed")
        self.assertFalse(err.is_permanent)

    def test_s3_key_defaults_to_empty(self):
        err = AudioResolutionError("s1", "a1", "pfx", "no_matching_file")
        self.assertEqual(err.s3_key, "")

    def test_s3_key_is_preserved(self):
        err = AudioResolutionError(
            "s1", "a1", "pfx", "glacier_storage_class",
            detail="...", s3_key="path/to/file.wav",
        )
        self.assertEqual(err.s3_key, "path/to/file.wav")


class PrefetchS3KeyPropagationTest(unittest.TestCase):
    """Verify s3_key is propagated through prefetch error paths."""

    def test_discard_before_decode_includes_s3_key(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        s3_key_value = "accounts/studio/takes/file.wav"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            audio_path = tmp_root / "a1.wav"
            audio_path.write_bytes(b"fake")

            def fake_resolve(**kwargs):
                return audio_path, s3_key_value

            with patch(
                "audio_classification_playground.acoustic_events.orchestration"
                ".prefetch.resolve_and_download",
                side_effect=fake_resolve,
            ):
                pf = Prefetcher(max_workers=1, vad_workers=0)
                try:
                    pf.discard(entity)
                    result = pf.get(entity)
                    self.assertIsInstance(result, AudioResolutionError)
                    self.assertEqual(result.s3_key, s3_key_value)
                    self.assertEqual(result.error_type, "download_failed")
                finally:
                    pf.shutdown()

    def test_decode_failure_includes_s3_key(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        s3_key_value = "accounts/studio/takes/file.wav"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            audio_path = tmp_root / "a1.wav"
            audio_path.write_bytes(b"fake")

            def fake_resolve(**kwargs):
                return audio_path, s3_key_value

            def fake_load_audio(path, *, sample_rate, recording_id):
                raise RuntimeError("corrupt audio")

            with patch.multiple(
                "audio_classification_playground.acoustic_events.orchestration.prefetch",
                resolve_and_download=fake_resolve,
                load_audio=fake_load_audio,
            ):
                pf = Prefetcher(max_workers=1, vad_workers=0)
                try:
                    pf.submit(entity, precompute_vad=False)
                    result = pf.get(entity)
                    self.assertIsInstance(result, AudioResolutionError)
                    self.assertEqual(result.s3_key, s3_key_value)
                    self.assertEqual(result.error_type, "download_failed")
                    self.assertIn("corrupt audio", result.detail)
                finally:
                    pf.shutdown()


if __name__ == "__main__":
    unittest.main()
