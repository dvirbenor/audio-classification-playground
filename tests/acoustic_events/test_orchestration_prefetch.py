import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_classification_playground.acoustic_events.inference.audio import AudioData
from audio_classification_playground.acoustic_events.orchestration.audio_resolver import (
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

                    pf.discard(entity)
                    pf.submit(entity, precompute_vad=True)
                    result = pf.get(entity)
                    self.assertEqual(result.vad_intervals, [(0.0, 0.25)])
                    self.assertEqual(vad_calls, [(16000, 16000)])
                    self.assertGreaterEqual(result.timings.vad_sec, 0.0)
                finally:
                    pf.shutdown()

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
                self.assertIs(pf.get(entity), error)
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


if __name__ == "__main__":
    unittest.main()
