"""Runner-level tests for VAD-gated inference.

These prove the production guarantee end-to-end: gated arrays are full-length,
*bit-identical to the full timeline at every kept (speech) frame*, filled at
non-speech frames, computed on fewer windows, and stored under a distinct
config hash. Uses content-deterministic fake predictors so a given window's
output is independent of whether it was run in a subset or the full batch.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_classification_playground.acoustic_events.inference.artifacts import (
    decoded_audio_sha256,
    load_prediction_artifact,
)
from audio_classification_playground.acoustic_events.inference.audio import AudioData
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    EMOTION_WINDOW_SEC,
    DEFAULT_HOP_SEC,
    run_affect_inference,
    run_all_inference,
    run_disfluency_inference,
)
from audio_classification_playground.acoustic_events.inference.vad_gating import (
    VadGating,
    speech_window_mask,
)

SR = 16_000
DUR = 40.0
INTERVALS = [(10.0, 20.0)]
BRIDGE = 1.5

EMOTION2VEC_LABELS = [
    "生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy", "中立/neutral",
    "其他/other", "难过/sad", "吃惊/surprised", "<unk>",
]


def _audio(tmp: Path) -> AudioData:
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(int(SR * DUR)).astype(np.float32)
    return AudioData(
        path=tmp / "x.wav", recording_id="rec", samples=samples,
        sample_rate=SR, duration_sec=DUR, audio_sha256=decoded_audio_sha256(samples),
    )


def _affect_fake():
    calls = []

    def fn(windows):
        w = np.asarray(windows, dtype=np.float32)
        calls.append(len(w))
        return {
            "arousal": w.mean(axis=1).astype(np.float32),
            "valence": w.std(axis=1).astype(np.float32),
            "dominance": w[:, 0].astype(np.float32),
        }

    return fn, calls


def _disfluency_fake():
    calls = []

    def fn(windows):
        w = np.asarray(windows, dtype=np.float32)
        calls.append(len(w))
        fl = np.stack([w.mean(axis=1), w.std(axis=1)], axis=1).astype(np.float32)
        ty = np.stack(
            [w.mean(axis=1), w.std(axis=1), w.max(axis=1), w.min(axis=1), w[:, 0]],
            axis=1,
        ).astype(np.float32)
        return {"fluency_logits": fl, "disfluency_type_logits": ty}

    return fn, calls


def _emotion_fake():
    calls = []
    row = np.array([0.05, 0.02, 0.02, 0.9, 0.05, 0.02, 0.02, 0.02, 0.0], dtype=np.float32)

    def fn(windows):
        n = len(windows)
        calls.append(n)
        return np.tile(row, (n, 1)), EMOTION2VEC_LABELS

    return fn, calls


class AffectGatingRunnerTest(unittest.TestCase):
    def test_gated_matches_full_at_kept_and_fills_rest(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = _audio(tmp)

            full_fn, full_calls = _affect_fake()
            run_affect_inference(
                audio, out_dir=tmp, backbone="wavlm", predictor=full_fn,
                artifact_path=tmp / "affect_full", reuse_cache=False,
            )
            gated_fn, gated_calls = _affect_fake()
            run_affect_inference(
                audio, out_dir=tmp, backbone="wavlm", predictor=gated_fn,
                artifact_path=tmp / "affect_gated", reuse_cache=False,
                vad_intervals=INTERVALS,
                vad_gating=VadGating(enabled=True, bridge_sec=BRIDGE, tasks=("affect",)),
            )

            full = load_prediction_artifact(tmp / "affect_full")
            gated = load_prediction_artifact(tmp / "affect_gated")
            n = full.arrays["arousal"].shape[0]
            mask = speech_window_mask(n, DEFAULT_HOP_SEC, AFFECT_WINDOW_SEC, INTERVALS, bridge_sec=BRIDGE)
            kept = np.nonzero(mask)[0]

            self.assertEqual(gated.arrays["arousal"].shape, full.arrays["arousal"].shape)
            self.assertGreater(kept.size, 0)
            self.assertLess(kept.size, n)
            for name in ("arousal", "valence", "dominance"):
                np.testing.assert_array_equal(
                    gated.arrays[name][kept], full.arrays[name][kept]
                )
                self.assertTrue((gated.arrays[name][~mask] == 0.0).all())

            # fewer windows computed, and distinct artifact lineage
            self.assertEqual(full_calls, [n])
            self.assertEqual(gated_calls, [int(kept.size)])
            self.assertNotEqual(
                gated.manifest["inference_config_hash"],
                full.manifest["inference_config_hash"],
            )
            self.assertIn("vad_gating", gated.manifest["inference_config"])
            self.assertNotIn("vad_gating", full.manifest["inference_config"])


class DisfluencyGatingRunnerTest(unittest.TestCase):
    def test_gated_matches_full_at_kept(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = _audio(tmp)
            full_fn, _ = _disfluency_fake()
            run_disfluency_inference(
                audio, out_dir=tmp, backbone="wavlm", predictor=full_fn,
                artifact_path=tmp / "disf_full", reuse_cache=False,
            )
            gated_fn, gated_calls = _disfluency_fake()
            run_disfluency_inference(
                audio, out_dir=tmp, backbone="wavlm", predictor=gated_fn,
                artifact_path=tmp / "disf_gated", reuse_cache=False,
                vad_intervals=INTERVALS,
                vad_gating=VadGating(enabled=True, bridge_sec=BRIDGE, tasks=("disfluency",)),
            )
            full = load_prediction_artifact(tmp / "disf_full")
            gated = load_prediction_artifact(tmp / "disf_gated")
            n = full.arrays["fluency_logits"].shape[0]
            mask = speech_window_mask(n, DEFAULT_HOP_SEC, DISFLUENCY_WINDOW_SEC, INTERVALS, bridge_sec=BRIDGE)
            kept = np.nonzero(mask)[0]
            for name in ("fluency_logits", "disfluency_type_logits"):
                self.assertEqual(gated.arrays[name].shape, full.arrays[name].shape)
                np.testing.assert_array_equal(
                    gated.arrays[name][kept], full.arrays[name][kept]
                )
                self.assertTrue((gated.arrays[name][~mask] == 0.0).all())
            self.assertEqual(gated_calls, [int(kept.size)])


class RunAllGatingIntegrationTest(unittest.TestCase):
    def test_end_to_end_resolves_intervals_from_vad_step(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = _audio(tmp)

            def fake_vad(samples, sample_rate):
                return list(INTERVALS)

            def run(out_root: Path, gating):
                affect_fn, _ = _affect_fake()
                disf_fn, _ = _disfluency_fake()
                emo_fn, emo_calls = _emotion_fake()
                run_all_inference(
                    audio,
                    out_dir=out_root,
                    affect_backbone="wavlm",
                    disfluency_backbone="wavlm",
                    predictors={"affect": affect_fn, "disfluency": disf_fn, "emotion": emo_fn},
                    vad_detector=fake_vad,
                    artifact_path_fn=lambda task: out_root / task,
                    vad_gating=gating,  # NOTE: no vad_intervals -> resolved from the vad step
                )
                return emo_calls

            full_root = tmp / "full"
            gated_root = tmp / "gated"
            run(full_root, None)
            gated_emo_calls = run(
                gated_root, VadGating(enabled=True, bridge_sec=BRIDGE, tasks=("affect", "emotion")),
            )

            # affect: gated == full at kept speech frames, full-length, filled elsewhere
            af_full = load_prediction_artifact(full_root / "affect")
            af_gated = load_prediction_artifact(gated_root / "affect")
            n = af_full.arrays["arousal"].shape[0]
            mask = speech_window_mask(n, DEFAULT_HOP_SEC, AFFECT_WINDOW_SEC, INTERVALS, bridge_sec=BRIDGE)
            kept = np.nonzero(mask)[0]
            self.assertGreater(kept.size, 0)
            self.assertLess(kept.size, n)
            np.testing.assert_array_equal(af_gated.arrays["arousal"][kept], af_full.arrays["arousal"][kept])
            self.assertTrue((af_gated.arrays["arousal"][~mask] == 0.0).all())

            # emotion: full-length valid probabilities, gated computed on a subset
            em_full = load_prediction_artifact(full_root / "emotion")
            em_gated = load_prediction_artifact(gated_root / "emotion")
            probs = em_gated.arrays["probabilities"]
            self.assertEqual(probs.shape, em_full.arrays["probabilities"].shape)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
            em_mask = speech_window_mask(
                probs.shape[0], DEFAULT_HOP_SEC, EMOTION_WINDOW_SEC, INTERVALS, bridge_sec=BRIDGE
            )
            em_kept = np.nonzero(em_mask)[0]
            np.testing.assert_allclose(
                probs[em_kept], em_full.arrays["probabilities"][em_kept], atol=1e-6
            )
            self.assertEqual(gated_emo_calls, [int(em_kept.size)])

            # vad artifact is unaffected by gating
            self.assertEqual(
                load_prediction_artifact(full_root / "vad").arrays["intervals_sec"].shape,
                load_prediction_artifact(gated_root / "vad").arrays["intervals_sec"].shape,
            )


if __name__ == "__main__":
    unittest.main()
