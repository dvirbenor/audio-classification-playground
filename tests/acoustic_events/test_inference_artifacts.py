import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_classification_playground.acoustic_events.inference import (
    artifact_to_affect_signals,
    artifact_to_disfluency_logits,
    artifact_to_emotion_probabilities,
    artifact_to_vad,
    decoded_audio_sha256,
    inference_config_hash,
    list_cached_artifacts,
    load_prediction_artifact,
    run_affect_inference,
    run_all_inference,
    run_vad,
)
from audio_classification_playground.acoustic_events.inference.runners import (
    DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD,
    DEFAULT_VAD_MIN_SILENCE_SEC,
    DEFAULT_VAD_MIN_SPEECH_SEC,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    emotion2vec_scores_to_probabilities,
)
from audio_classification_playground.acoustic_events.producers.affect import (
    Config as AffectConfig,
    extract_events,
)
from audio_classification_playground.acoustic_events.producers.disfluency import (
    produce_disfluency_events,
)
from audio_classification_playground.acoustic_events.producers.emotion import (
    run_from_probabilities,
)


EMOTION2VEC_LABELS = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]


class InferenceArtifactTest(unittest.TestCase):
    def test_decoded_audio_hash_and_config_hash_are_stable_and_sensitive(self):
        audio = np.asarray([0.0, 0.5, -0.25], dtype=np.float32)
        self.assertEqual(decoded_audio_sha256(audio), decoded_audio_sha256(audio.copy()))

        base = {
            "task": "affect",
            "model_id": "model",
            "backbone": "wavlm",
            "sample_rate": 16000,
            "window_sec": 3.5,
            "hop_sec": 0.25,
            "batch_size": 16,
            "transform_policy": "test",
        }
        changed = dict(base, batch_size=32)
        self.assertNotEqual(inference_config_hash(base), inference_config_hash(changed))

    def test_single_task_write_load_and_explicit_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = _write_audio(Path(tmp) / "clip.wav")
            out_dir = Path(tmp) / "artifacts"
            calls = []

            def predictor(windows):
                calls.append(len(windows))
                return {
                    "arousal": np.linspace(0.1, 0.2, len(windows)),
                    "valence": np.linspace(0.2, 0.3, len(windows)),
                    "dominance": np.linspace(0.3, 0.4, len(windows)),
                }

            first = run_affect_inference(
                audio_path,
                out_dir=out_dir,
                backbone="wavlm",
                predictor=predictor,
                progress=_quiet,
            )
            second = run_affect_inference(
                audio_path,
                out_dir=out_dir,
                backbone="wavlm",
                predictor=predictor,
                reuse_cache=True,
                progress=_quiet,
            )

            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertEqual(calls, [1])
            loaded = load_prediction_artifact(first.artifact.path)
            self.assertEqual(loaded.task, "affect")
            self.assertEqual(set(loaded.arrays), {"arousal", "valence", "dominance"})
            self.assertEqual(len(list_cached_artifacts(out_dir, task="affect")), 1)

    def test_run_all_returns_task_keyed_artifacts_and_reuses_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = _write_audio(Path(tmp) / "clip.wav")
            out_dir = Path(tmp) / "artifacts"
            cleanup_calls = []
            predictors = {
                "affect": _fake_affect,
                "disfluency": _fake_disfluency,
                "emotion": _fake_emotion,
            }

            first = run_all_inference(
                audio_path,
                out_dir=out_dir,
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                predictors=predictors,
                vad_detector=_fake_vad,
                cleanup_cuda=lambda: cleanup_calls.append("cleanup"),
                progress=_quiet,
            )
            second = run_all_inference(
                audio_path,
                out_dir=out_dir,
                affect_backbone="wavlm",
                disfluency_backbone="whisper",
                predictors=predictors,
                vad_detector=_fake_vad,
                cleanup_cuda=lambda: cleanup_calls.append("cleanup"),
                reuse_cache=True,
                progress=_quiet,
            )

            self.assertEqual(set(first.artifacts), {"vad", "affect", "disfluency", "emotion"})
            self.assertEqual(first.reused, {task: False for task in first.artifacts})
            self.assertEqual(second.reused, {task: True for task in second.artifacts})
            self.assertEqual(len(cleanup_calls), 4)

    def test_vad_manifest_defaults_match_vox_profile_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = _write_audio(Path(tmp) / "clip.wav")
            result = run_vad(
                audio_path,
                out_dir=Path(tmp) / "artifacts",
                detector=_fake_vad,
                progress=_quiet,
            )

            config = result.artifact.manifest["inference_config"]
            self.assertEqual(config["threshold"], DEFAULT_VAD_SPEECH_THRESHOLD)
            self.assertEqual(config["speech_threshold"], DEFAULT_VAD_SPEECH_THRESHOLD)
            self.assertEqual(config["min_speech_sec"], DEFAULT_VAD_MIN_SPEECH_SEC)
            self.assertEqual(config["min_silence_sec"], DEFAULT_VAD_MIN_SILENCE_SEC)
            self.assertEqual(
                config["frame_speech_ratio_threshold"],
                DEFAULT_VAD_FRAME_SPEECH_RATIO_THRESHOLD,
            )

    def test_run_all_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = _write_audio(Path(tmp) / "clip.wav")
            called = []

            def failing_disfluency(windows):
                called.append("disfluency")
                raise RuntimeError("boom")

            def emotion(windows):
                called.append("emotion")
                return _fake_emotion(windows)

            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_all_inference(
                    audio_path,
                    out_dir=Path(tmp) / "artifacts",
                    affect_backbone="wavlm",
                    disfluency_backbone="wavlm",
                    predictors={
                        "affect": _fake_affect,
                        "disfluency": failing_disfluency,
                        "emotion": emotion,
                    },
                    vad_detector=_fake_vad,
                    progress=_quiet,
                )
            self.assertEqual(called, ["disfluency"])

    def test_orphan_predictions_without_manifest_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            orphan_dir = root / "rec" / "hash" / "affect" / "deadbeef"
            orphan_dir.mkdir(parents=True)
            np.savez_compressed(orphan_dir / "predictions.npz", arousal=np.array([1.0]))

            self.assertEqual(list_cached_artifacts(root), [])

    def test_emotion_transform_is_deterministic_and_rejects_zero_rows(self):
        scores = np.array([[1, 1, 1, 1, 2, 1, 1, 1, 1]], dtype=np.float32)
        probs, labels = emotion2vec_scores_to_probabilities(scores, EMOTION2VEC_LABELS)

        self.assertIn("other", labels)
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertGreater(float(probs[0, labels.index("other")]), 0.1)
        with self.assertRaisesRegex(ValueError, "zero-score"):
            emotion2vec_scores_to_probabilities(np.zeros((1, len(EMOTION2VEC_LABELS))), EMOTION2VEC_LABELS)

    def test_artifacts_adapt_to_existing_producers(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = _write_audio(Path(tmp) / "clip.wav")
            result = run_all_inference(
                audio_path,
                out_dir=Path(tmp) / "artifacts",
                affect_backbone="wavlm",
                disfluency_backbone="wavlm",
                predictors={
                    "affect": _fake_affect,
                    "disfluency": _fake_disfluency,
                    "emotion": _fake_emotion,
                },
                vad_detector=_fake_vad,
                progress=_quiet,
            )

            vad = artifact_to_vad(result.artifacts["vad"])
            affect_signals = artifact_to_affect_signals(result.artifacts["affect"])
            affect_events = extract_events(
                affect_signals,
                vad,
                AffectConfig(
                    radius_sec=2.0,
                    min_context_sec=0.0,
                    seed_min_width_sec=0.25,
                    min_duration_sec=0.25,
                ),
            )
            self.assertIsInstance(affect_events, list)

            fluency, types, hop, window, duration = artifact_to_disfluency_logits(
                result.artifacts["disfluency"]
            )
            disfluency_run, _, _ = produce_disfluency_events(
                fluency_logits=fluency,
                disfluency_type_logits=types,
                hop_sec=hop,
                window_sec=window,
                audio_duration_sec=duration,
            )
            self.assertEqual(disfluency_run.task, "disfluency")

            probabilities, labels, hop, window, duration = artifact_to_emotion_probabilities(
                result.artifacts["emotion"]
            )
            emotion_run, _, _ = run_from_probabilities(
                probabilities,
                labels,
                hop_sec=hop,
                window_sec=window,
                audio_duration_sec=duration,
                vad_intervals=vad.intervals,
            )
            self.assertEqual(emotion_run.task, "emotion")


def _write_audio(path: Path) -> Path:
    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)
    sf.write(path, samples, sr)
    return path


def _quiet(message):
    pass


def _fake_vad(samples, sample_rate):
    return [(0.0, max(0.25, len(samples) / sample_rate))]


def _fake_affect(windows):
    n = len(windows)
    return {
        "arousal": np.linspace(0.0, 0.2, n, dtype=np.float32),
        "valence": np.linspace(0.2, 0.0, n, dtype=np.float32),
        "dominance": np.zeros(n, dtype=np.float32),
    }


def _fake_disfluency(windows):
    n = len(windows)
    fluency = np.column_stack([
        np.ones(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ])
    types = np.zeros((n, 5), dtype=np.float32)
    return {
        "fluency_logits": fluency,
        "disfluency_type_logits": types,
    }


def _fake_emotion(windows):
    n = len(windows)
    scores = np.tile(
        np.array([[0.05, 0.02, 0.02, 0.1, 0.6, 0.1, 0.03, 0.03, 0.05]], dtype=np.float32),
        (n, 1),
    )
    return scores, EMOTION2VEC_LABELS


if __name__ == "__main__":
    unittest.main()
