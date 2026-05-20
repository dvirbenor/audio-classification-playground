import json
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audio_classification_playground.acoustic_events.inference.artifacts import (
    InferenceRunResult,
)
from audio_classification_playground.acoustic_events.inference.audio import AudioData
from audio_classification_playground.acoustic_events.orchestration import worker
from audio_classification_playground.acoustic_events.orchestration.manifest import (
    ArchiveEntity,
)
from audio_classification_playground.acoustic_events.orchestration.prefetch import (
    PrefetchResult,
)
from audio_classification_playground.acoustic_events.inference.wavlm_runtime import (
    WAVLM_COMPILED_STATIC_BATCH_SIZE,
    WavLMRuntimeSettings,
)

_TASKS = ("vad", "affect", "disfluency", "emotion")


def _fake_inference_result():
    """Fresh InferenceRunResult with all keys the worker expects."""
    return InferenceRunResult(
        artifacts={},
        reused={t: False for t in _TASKS},
        task_elapsed_sec={t: 0.0 for t in _TASKS},
    )


class WorkerAsyncVadTest(unittest.TestCase):
    def test_per_task_batch_sizes_are_passed_to_models_and_run_all(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        model_kwargs = []
        run_kwargs = []

        class FakeModels(_FakeModels):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                model_kwargs.append(kwargs)

        def fake_run_all(*args, **kwargs):
            run_kwargs.append(kwargs)
            return _fake_inference_result()

        with _worker_patches(
            [entity],
            model_suite_cls=FakeModels,
            run_all_fn=fake_run_all,
        ):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                batch_size=512,
                affect_batch_size=384,
                emotion_batch_size=64,
                wavlm_autocast_dtype="bf16",
                wavlm_compile=True,
                wavlm_compile_dynamic=True,
                wavlm_stream_layer_sum=True,
                emotion_autocast_dtype="bf16",
                emotion_compile=True,
                emotion_runtime_mode="custom",
                allow_tf32=True,
                prefetch_lookahead=1,
                vad_prefetch_workers=1,
            )

        self.assertEqual(model_kwargs[0]["affect_batch_size"], 384)
        self.assertEqual(model_kwargs[0]["disfluency_batch_size"], 512)
        self.assertEqual(model_kwargs[0]["emotion_batch_size"], 64)
        self.assertEqual(model_kwargs[0]["wavlm_autocast_dtype"], "bf16")
        self.assertTrue(model_kwargs[0]["wavlm_compile"])
        self.assertTrue(model_kwargs[0]["wavlm_compile_dynamic"])
        self.assertTrue(model_kwargs[0]["wavlm_stream_layer_sum"])
        self.assertEqual(model_kwargs[0]["emotion_autocast_dtype"], "bf16")
        self.assertTrue(model_kwargs[0]["emotion_compile"])
        self.assertEqual(model_kwargs[0]["emotion_runtime_mode"], "custom")
        self.assertTrue(model_kwargs[0]["allow_tf32"])
        self.assertEqual(run_kwargs[0]["affect_batch_size"], 384)
        self.assertEqual(run_kwargs[0]["disfluency_batch_size"], 512)
        self.assertEqual(run_kwargs[0]["emotion_batch_size"], 64)
        self.assertEqual(run_kwargs[0]["wavlm_autocast_dtype"], "bf16")
        self.assertTrue(run_kwargs[0]["wavlm_compile"])
        self.assertTrue(run_kwargs[0]["wavlm_compile_dynamic"])
        self.assertTrue(run_kwargs[0]["wavlm_stream_layer_sum"])
        self.assertEqual(run_kwargs[0]["emotion_autocast_dtype"], "bf16")
        self.assertTrue(run_kwargs[0]["emotion_compile"])
        self.assertEqual(run_kwargs[0]["emotion_runtime_mode"], "custom")
        self.assertTrue(run_kwargs[0]["allow_tf32"])

    def test_orchestration_emotion_batch_defaults_to_64(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        model_kwargs = []

        class FakeModels(_FakeModels):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                model_kwargs.append(kwargs)

        with _worker_patches([entity], model_suite_cls=FakeModels):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                wavlm_runtime_preset="fast_exact",
                prefetch_lookahead=1,
                vad_prefetch_workers=1,
            )

        self.assertEqual(model_kwargs[0]["affect_batch_size"], 512)
        self.assertEqual(model_kwargs[0]["disfluency_batch_size"], 512)
        self.assertEqual(model_kwargs[0]["emotion_batch_size"], 64)

    def test_expected_configs_auto_runtime_records_optimized_emotion_on_cuda(self):
        with patch("torch.cuda.is_available", return_value=True), patch.object(
            worker,
            "resolve_wavlm_runtime_settings",
            return_value=_fast_exact_settings(device="cuda"),
        ):
            configs = worker.build_expected_configs(
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                batch_size=512,
                device="cuda",
            )

        emotion = configs["emotion"]
        self.assertEqual(emotion["batch_size"], 64)
        self.assertTrue(emotion["torch_compile"])
        self.assertEqual(emotion["torch_compile_mode"], "default")
        self.assertTrue(emotion["torch_allow_tf32"])
        self.assertFalse(configs["affect"]["torch_allow_tf32"])
        self.assertNotIn("torch_allow_tf32", configs["disfluency"])

    def test_compiled_static_wavlm_preset_uses_256_batches_and_warmup(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        model_kwargs = []
        run_kwargs = []

        class FakeModels(_FakeModels):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                model_kwargs.append(kwargs)

        def fake_run_all(*args, **kwargs):
            run_kwargs.append(kwargs)
            return _fake_inference_result()

        with _worker_patches(
            [entity],
            model_suite_cls=FakeModels,
            run_all_fn=fake_run_all,
        ), patch.object(
            worker,
            "resolve_wavlm_runtime_settings",
            return_value=_compiled_static_settings(),
        ), patch.object(
            worker,
            "configure_inductor_cache_namespace",
            return_value={"configured": True, "writable": True, "path": "/tmp/cache"},
        ):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="wavlm",
                prefetch_lookahead=1,
                vad_prefetch_workers=1,
            )

        self.assertEqual(
            model_kwargs[0]["affect_batch_size"],
            WAVLM_COMPILED_STATIC_BATCH_SIZE,
        )
        self.assertEqual(
            model_kwargs[0]["disfluency_batch_size"],
            WAVLM_COMPILED_STATIC_BATCH_SIZE,
        )
        self.assertTrue(model_kwargs[0]["wavlm_compile"])
        self.assertEqual(model_kwargs[0]["wavlm_compile_mode"], "default")
        self.assertTrue(model_kwargs[0]["wavlm_static_batch"])
        self.assertTrue(model_kwargs[0]["wavlm_warmup"])
        self.assertEqual(model_kwargs[0]["wavlm_runtime_preset"], "compiled_static")
        self.assertEqual(
            run_kwargs[0]["affect_batch_size"],
            WAVLM_COMPILED_STATIC_BATCH_SIZE,
        )
        self.assertEqual(
            run_kwargs[0]["disfluency_batch_size"],
            WAVLM_COMPILED_STATIC_BATCH_SIZE,
        )
        self.assertTrue(run_kwargs[0]["wavlm_static_batch"])
        self.assertEqual(run_kwargs[0]["wavlm_runtime_preset"], "compiled_static")

    def test_expected_configs_fp32_eager_runtime_has_empty_emotion_extras(self):
        configs = worker.build_expected_configs(
            affect_backbone="wavlm",
            disfluency_backbone="whisper",
            batch_size=512,
            emotion_runtime_mode="fp32-eager",
            device="cpu",
        )

        emotion = configs["emotion"]
        self.assertEqual(emotion["batch_size"], 64)
        self.assertNotIn("torch_compile", emotion)
        self.assertNotIn("torch_allow_tf32", emotion)

    def test_claim_happens_before_prefetch_and_intervals_are_passed(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        events = []
        run_calls = []

        def try_claim(output_base, claimed_entity):
            events.append(("claim", claimed_entity.archive_id))
            return True

        class FakePrefetcher(_FakePrefetcher):
            def submit(self, submitted_entity, *, precompute_vad=False):
                events.append(("submit", submitted_entity.archive_id, precompute_vad))
                super().submit(submitted_entity, precompute_vad=precompute_vad)

        def fake_run_all(*args, **kwargs):
            run_calls.append(kwargs["vad_detector"](np.zeros(4, dtype=np.float32), 16000))
            return _fake_inference_result()

        with _worker_patches(
            [entity],
            prefetcher_cls=FakePrefetcher,
            try_claim_fn=try_claim,
            run_all_fn=fake_run_all,
        ):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                prefetch_lookahead=1,
                vad_prefetch_workers=1,
            )

        self.assertEqual(events, [("claim", "a1"), ("submit", "a1", True)])
        self.assertEqual(run_calls, [[(0.0, 0.5)]])

    def test_claimed_inflight_count_is_bounded(self):
        entities = [ArchiveEntity("s1", f"a{i}", f"prefix/{i}") for i in range(5)]
        active = set()
        max_active = 0

        def try_claim(output_base, entity):
            nonlocal max_active
            active.add(entity.archive_id)
            max_active = max(max_active, len(active))
            return True

        def release_claim(output_base, entity):
            active.discard(entity.archive_id)

        with _worker_patches(
            entities,
            try_claim_fn=try_claim,
            release_claim_fn=release_claim,
        ):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                prefetch_lookahead=2,
                vad_prefetch_workers=1,
                seed=5,
            )

        self.assertLessEqual(max_active, 2)
        self.assertEqual(active, set())

    def test_cached_vad_skips_async_vad_and_does_not_load_sync_vad(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        submit_args = []
        vad_detectors = []

        class FakePrefetcher(_FakePrefetcher):
            def submit(self, submitted_entity, *, precompute_vad=False):
                submit_args.append(precompute_vad)
                super().submit(submitted_entity, precompute_vad=precompute_vad)

        def fake_run_all(*args, **kwargs):
            vad_detectors.append(kwargs["vad_detector"])
            return _fake_inference_result()

        with _worker_patches(
            [entity],
            prefetcher_cls=FakePrefetcher,
            task_complete_fn=lambda *args, **kwargs: args[3] == "vad",
            run_all_fn=fake_run_all,
        ):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                prefetch_lookahead=1,
                vad_prefetch_workers=1,
                completion_policy="config",
            )

        self.assertEqual(submit_args, [False])
        self.assertEqual(vad_detectors, [None])

    def test_shutdown_releases_current_and_queued_claims(self):
        entities = [ArchiveEntity("s1", f"a{i}", f"prefix/{i}") for i in range(3)]
        released = []

        def fake_run_all(*args, **kwargs):
            raise worker.ShutdownRequested("stop")

        with _worker_patches(
            entities,
            release_claim_fn=lambda output_base, entity: released.append(entity.archive_id),
            run_all_fn=fake_run_all,
        ):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                prefetch_lookahead=2,
                vad_prefetch_workers=1,
                seed=5,
            )

        self.assertEqual(Counter(released), Counter({"a0": 1, "a1": 1}))

    def test_one_archive_failure_does_not_cancel_siblings(self):
        entities = [ArchiveEntity("s1", f"a{i}", f"prefix/{i}") for i in range(3)]
        calls = []
        errors = []

        def fake_run_all(*args, **kwargs):
            archive_id = Path(kwargs["audio_source_key"]).stem
            calls.append(archive_id)
            if archive_id == "a0":
                raise RuntimeError("boom")
            return _fake_inference_result()

        with _worker_patches(
            entities,
            run_all_fn=fake_run_all,
            handle_error_fn=lambda exc, output_base, entity, attempts, max_attempts: errors.append(
                entity.archive_id
            ),
        ):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                prefetch_lookahead=2,
                vad_prefetch_workers=1,
                seed=5,
            )

        self.assertEqual(calls, ["a0", "a1", "a2"])
        self.assertEqual(errors, ["a0"])

    def test_sync_vad_fallback_passes_model_vad(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        vad_detectors = []

        def fake_run_all(*args, **kwargs):
            vad_detectors.append(kwargs["vad_detector"])
            return _fake_inference_result()

        with _worker_patches([entity], run_all_fn=fake_run_all):
            worker.run_worker(
                parquet_path="manifest.parquet",
                output_base=tempfile.mkdtemp(),
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                prefetch_lookahead=1,
                vad_prefetch_workers=0,
            )

        self.assertEqual(vad_detectors, ["sync-vad"])

    def test_timing_jsonl_written_after_inference(self):
        entity = ArchiveEntity("s1", "a1", "prefix")
        with tempfile.TemporaryDirectory() as tmpdir:
            with _worker_patches([entity]):
                worker.run_worker(
                    parquet_path="manifest.parquet",
                    output_base=tmpdir,
                    affect_backbone="wavlm",
                    disfluency_backbone="whisper",
                    prefetch_lookahead=1,
                    vad_prefetch_workers=1,
                )

            timings_dir = Path(tmpdir) / "_meta" / "timings"
            jsonl_files = list(timings_dir.glob("*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)

            with open(jsonl_files[0], encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 1)

            record = json.loads(lines[0])
            self.assertEqual(record["session_id"], "s1")
            self.assertEqual(record["archive_id"], "a1")
            expected_fields = {
                "worker_id", "session_id", "archive_id", "ts",
                "audio_duration_sec", "prefetch_wait_sec",
                "download_decode_sec", "vad_precompute_sec",
                "precomputed_vad", "vad_reused", "affect_reused",
                "disfluency_reused", "emotion_reused",
                "vad_sec", "affect_sec", "disfluency_sec", "emotion_sec",
                "inference_sec", "total_sec",
            }
            self.assertTrue(expected_fields <= set(record.keys()))
            for f in ("vad_sec", "affect_sec", "disfluency_sec", "emotion_sec",
                       "inference_sec", "total_sec"):
                self.assertIsInstance(record[f], (int, float))


class _FakePrefetcher:
    def __init__(self, *args, **kwargs):
        self._precompute: dict[tuple[str, str], bool] = {}

    def submit(self, entity, *, precompute_vad=False):
        self._precompute[(entity.session_id, entity.archive_id)] = precompute_vad

    def get(self, entity):
        precompute_vad = self._precompute[(entity.session_id, entity.archive_id)]
        return PrefetchResult(
            audio=_fake_audio(entity.archive_id),
            s3_key=f"{entity.archive_id}.wav",
            vad_intervals=[(0.0, 0.5)] if precompute_vad else None,
        )

    def discard(self, entity):
        pass

    def shutdown(self):
        pass


class _FakeModels:
    affect = object()
    disfluency = object()
    emotion = object()
    vad = "sync-vad"

    def __init__(self, *args, **kwargs):
        self.load_vad = kwargs.get("load_vad")


def _fast_exact_settings(*, device: str = "cpu") -> WavLMRuntimeSettings:
    return WavLMRuntimeSettings(
        requested_preset=None,
        preset="fast_exact",
        device=device,
        task_batch_size=None,
        autocast_dtype=None,
        compile_model=False,
        compile_mode="reduce-overhead",
        compile_dynamic=False,
        stream_layer_sum=False,
        allow_tf32=False,
        static_batch=False,
        warmup=False,
    )


def _compiled_static_settings() -> WavLMRuntimeSettings:
    return WavLMRuntimeSettings(
        requested_preset=None,
        preset="compiled_static",
        device="cuda",
        task_batch_size=WAVLM_COMPILED_STATIC_BATCH_SIZE,
        autocast_dtype=None,
        compile_model=True,
        compile_mode="default",
        compile_dynamic=False,
        stream_layer_sum=False,
        allow_tf32=False,
        static_batch=True,
        warmup=True,
    )


@contextmanager
def _worker_patches(
    entities,
    *,
    model_suite_cls=_FakeModels,
    prefetcher_cls=_FakePrefetcher,
    try_claim_fn=None,
    release_claim_fn=None,
    task_complete_fn=None,
    run_all_fn=None,
    handle_error_fn=None,
):
    try_claim_fn = try_claim_fn or (lambda output_base, entity: True)
    release_claim_fn = release_claim_fn or (lambda output_base, entity: None)
    task_complete_fn = task_complete_fn or (lambda *args, **kwargs: False)
    run_all_fn = run_all_fn or (lambda *args, **kwargs: _fake_inference_result())
    handle_error_fn = handle_error_fn or (lambda *args, **kwargs: None)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "audio_classification_playground.acoustic_events.inference.models.ModelSuite",
                model_suite_cls,
            )
        )
        stack.enter_context(
            patch.multiple(
                worker,
                load_manifest=lambda parquet_path: list(entities),
                load_permanent_error_set=lambda output_base: set(),
                load_inference_attempt_counts=lambda *args, **kwargs: Counter(),
                is_task_artifact_complete_for_archive=task_complete_fn,
                is_task_complete_for_config=task_complete_fn,
                try_claim=try_claim_fn,
                release_claim=release_claim_fn,
                count_inference_attempts_for=lambda *args, **kwargs: 0,
                Prefetcher=prefetcher_cls,
                run_all_inference=run_all_fn,
                _handle_inference_error=lambda *args, **kwargs: handle_error_fn(*args),
            )
        )
        yield


def _fake_audio(recording_id: str) -> AudioData:
    return AudioData(
        path=Path(f"/tmp/{recording_id}.wav"),
        recording_id=recording_id,
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration_sec=1.0,
        audio_sha256=f"hash-{recording_id}",
    )


if __name__ == "__main__":
    unittest.main()
