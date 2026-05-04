import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_classification_playground.acoustic_events.producers.affect import Vad
from audio_classification_playground.acoustic_events.producers.emotion import (
    CANONICAL_CHANNELS,
    Config,
    TRACK_ID,
    canonicalize_probabilities,
    normalize_label,
    run_from_probabilities,
)
from audio_classification_playground.acoustic_events.review.storage import save_session


def probs(**values):
    row = {label: 0.0 for label in CANONICAL_CHANNELS}
    row.update(values)
    total = sum(row.values())
    if total < 1.0:
        row["other"] += 1.0 - total
    return [row[label] for label in CANONICAL_CHANNELS]


class EmotionProducerTest(unittest.TestCase):
    def test_label_normalization_and_unk_folding(self):
        labels = [
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
        raw = np.array([[0.10, 0.05, 0.04, 0.20, 0.15, 0.06, 0.30, 0.03, 0.07]])

        folded, channels = canonicalize_probabilities(raw, labels)

        self.assertEqual(channels, CANONICAL_CHANNELS)
        self.assertEqual(normalize_label("开心/happy"), "happiness")
        self.assertAlmostEqual(folded[0, channels.index("other")], 0.13)
        self.assertAlmostEqual(float(folded.sum()), 1.0)

    def test_rejects_non_probability_rows(self):
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            canonicalize_probabilities(
                np.array([[0.8, 0.8]]),
                ["angry", "neutral"],
            )

    def test_track_and_outputs_are_contract_shaped(self):
        matrix = np.array([
            probs(happiness=0.8, neutral=0.1, other=0.1),
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(happiness=0.8, neutral=0.1, other=0.1),
            probs(happiness=0.7, neutral=0.2, other=0.1),
        ])
        cfg = Config(
            absolute_min_probability=0.6,
            class_quantile=0.0,
            min_duration_sec=1.0,
            support_close_gap_sec=0.0,
        )

        run, tracks, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            vad_intervals=((0.0, 99.0),),
            config=cfg,
        )

        self.assertEqual(run.producer_id, "emotion.categorical.v1")
        self.assertEqual(tracks[0].track_id, TRACK_ID)
        self.assertEqual(tracks[0].renderer, "multi_probability")
        self.assertEqual(tracks[0].channels, CANONICAL_CHANNELS)
        self.assertEqual(len(events), 1)
        self.assertIn("class_occupancy", run.outputs)
        self.assertIn("resolved_thresholds", run.outputs)
        self.assertIn("event_summary", run.outputs)
        self.assertIn("label_mapping", run.outputs)
        self.assertIn("vad", run.outputs)
        self.assertTrue(run.config_hash)

    def test_missing_vad_default_is_tracks_only_and_debug_hash_differs(self):
        matrix = np.array([
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(happiness=0.9, neutral=0.05, other=0.05),
        ])
        cfg = Config(class_quantile=0.0, min_duration_sec=0.25)

        run, tracks, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            config=cfg,
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(events, [])
        self.assertEqual(run.outputs["resolved_thresholds"], {})
        self.assertEqual(run.outputs["class_occupancy"], {})
        self.assertEqual(run.outputs["vad"]["thresholds_computed_on"], "none_missing_vad")
        self.assertEqual(run.outputs["vad"]["no_event_reason"], "vad_required_but_missing")
        self.assertEqual(run.outputs["suppressed_non_vad_emotion_frames"], {})

        debug_run, _, debug_events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            config=cfg,
            require_vad_for_events=False,
        )
        self.assertEqual(len(debug_events), 1)
        self.assertEqual(debug_run.outputs["vad"]["thresholds_computed_on"], "all_frames_debug")
        self.assertNotEqual(run.config_hash, debug_run.config_hash)

    def test_neutral_and_other_never_emit_events(self):
        matrix = np.array([
            probs(neutral=0.9, sadness=0.1, other=0.0),
            probs(neutral=0.95, sadness=0.05, other=0.0),
            probs(other=0.9, sadness=0.1, neutral=0.0),
            probs(other=0.95, sadness=0.05, neutral=0.0),
        ])
        _, _, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            vad_intervals=((0.0, 99.0),),
            config=Config(class_quantile=0.0, min_duration_sec=0.25),
        )
        self.assertEqual(events, [])

    def test_vad_controls_threshold_denominator_and_no_speech_case(self):
        matrix = np.array([
            probs(anger=0.8, neutral=0.1, other=0.1),
            probs(anger=0.9, neutral=0.05, other=0.05),
            probs(anger=0.1, neutral=0.8, other=0.1),
            probs(anger=0.1, neutral=0.8, other=0.1),
        ])
        cfg = Config(
            absolute_min_probability=0.0,
            class_quantile=0.5,
            min_duration_sec=0.25,
        )

        run, _, _ = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=1.0,
            window_sec=1.0,
            vad_intervals=((0.0, 2.0),),
            config=cfg,
        )

        self.assertAlmostEqual(run.outputs["resolved_thresholds"]["anger"], 0.85)
        self.assertEqual(run.outputs["vad"]["valid_frame_count"], 2)
        self.assertEqual(run.outputs["vad"]["thresholds_computed_on"], "speech_frames")
        self.assertAlmostEqual(run.outputs["class_occupancy"]["anger"]["top1_share"], 1.0)

        no_speech_run, _, no_speech_events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=1.0,
            window_sec=1.0,
            vad_intervals=((10.0, 11.0),),
            config=cfg,
        )
        self.assertEqual(no_speech_events, [])
        self.assertEqual(no_speech_run.outputs["resolved_thresholds"], {})
        self.assertEqual(no_speech_run.outputs["class_occupancy"], {})
        self.assertEqual(no_speech_run.outputs["vad"]["thresholds_computed_on"], "none_no_speech")
        self.assertEqual(no_speech_run.outputs["vad"]["no_event_reason"], "vad_found_no_speech")

    def test_vad_overlap_mask_catches_short_intervals_between_frame_centers(self):
        matrix = np.array([
            probs(anger=0.9, neutral=0.05, other=0.05),
            probs(anger=0.9, neutral=0.05, other=0.05),
            probs(neutral=0.9, anger=0.05, other=0.05),
        ])

        run, _, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=1.0,
            window_sec=1.0,
            vad_intervals=((0.9, 1.1),),
            config=Config(
                absolute_min_probability=0.6,
                class_quantile=0.0,
                min_duration_sec=1.0,
                support_close_gap_sec=0.0,
            ),
        )

        self.assertEqual(run.outputs["vad"]["valid_frame_count"], 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "anger")

    def test_high_confidence_non_vad_emotion_is_suppressed_and_counted(self):
        matrix = np.array([
            probs(anger=0.9, neutral=0.05, other=0.05),
            probs(sadness=0.95, neutral=0.03, other=0.02),
            probs(sadness=0.7, neutral=0.2, other=0.1),
        ])

        run, _, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=1.0,
            window_sec=1.0,
            vad_intervals=((0.0, 1.0),),
            config=Config(
                absolute_min_probability=0.6,
                class_quantile=0.0,
                min_duration_sec=1.0,
                support_close_gap_sec=0.0,
            ),
        )

        self.assertEqual([event.label for event in events], ["anger"])
        suppressed = run.outputs["suppressed_non_vad_emotion_frames"]
        self.assertEqual(suppressed["sadness"]["frame_count"], 2)
        self.assertAlmostEqual(suppressed["sadness"]["max_probability"], 0.95)

    def test_closing_does_not_bridge_across_non_speech(self):
        matrix = np.array([
            probs(anger=0.9, neutral=0.05, other=0.05),
            probs(neutral=0.9, anger=0.05, other=0.05),
            probs(neutral=0.9, anger=0.05, other=0.05),
            probs(neutral=0.9, anger=0.05, other=0.05),
            probs(anger=0.9, neutral=0.05, other=0.05),
        ])
        cfg = Config(
            absolute_min_probability=0.6,
            class_quantile=0.0,
            min_duration_sec=2.0,
            support_close_gap_sec=10.0,
        )

        _, _, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=1.0,
            window_sec=1.0,
            vad_intervals=((0.0, 1.0), (4.0, 5.0)),
            config=cfg,
        )

        self.assertEqual(events, [])

    def test_isolated_spike_is_noop_but_clustered_spikes_close_into_event(self):
        isolated = np.array([
            probs(anger=0.9, neutral=0.05, other=0.05),
            probs(neutral=0.9, anger=0.05, other=0.05),
            probs(neutral=0.9, anger=0.05, other=0.05),
            probs(neutral=0.9, anger=0.05, other=0.05),
        ])
        cfg = Config(
            absolute_min_probability=0.6,
            class_quantile=0.0,
            min_duration_sec=1.0,
            support_close_gap_sec=0.5,
        )
        _, _, isolated_events = run_from_probabilities(
            isolated,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            vad_intervals=((0.0, 99.0),),
            config=cfg,
        )
        self.assertEqual(isolated_events, [])

        clustered = np.array([
            probs(anger=0.9, neutral=0.05, other=0.05),
            probs(neutral=0.8, anger=0.1, other=0.1),
            probs(neutral=0.8, anger=0.1, other=0.1),
            probs(anger=0.7, neutral=0.2, other=0.1),
        ])
        _, _, clustered_events = run_from_probabilities(
            clustered,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            vad_intervals=((0.0, 99.0),),
            config=cfg,
        )

        self.assertEqual(len(clustered_events), 1)
        event = clustered_events[0]
        self.assertEqual(event.label, "anger")
        self.assertEqual(event.extra["raw_support_frame_count"], 2)
        self.assertEqual(event.extra["closed_support_frame_count"], 4)
        self.assertEqual(event.extra["bridge_frame_count"], 2)
        self.assertAlmostEqual(event.evidence["mean_probability"], 0.8)

    def test_background_margin_suppresses_ambiguous_support(self):
        matrix = np.array([
            probs(sadness=0.62, neutral=0.55, other=0.0),
            probs(sadness=0.62, neutral=0.55, other=0.0),
            probs(sadness=0.62, neutral=0.55, other=0.0),
            probs(sadness=0.62, neutral=0.55, other=0.0),
        ])
        # Normalize the intentionally ambiguous rows.
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

        _, _, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            vad_intervals=((0.0, 99.0),),
            config=Config(
                absolute_min_probability=0.5,
                class_quantile=0.0,
                background_margin=0.15,
                min_duration_sec=0.25,
            ),
        )

        self.assertEqual(events, [])

    def test_center_support_boundaries_clipping_and_stable_ids(self):
        matrix = np.array([
            probs(sadness=0.9, neutral=0.05, other=0.05),
            probs(neutral=0.9, sadness=0.05, other=0.05),
            probs(neutral=0.9, sadness=0.05, other=0.05),
            probs(anger=0.9, neutral=0.05, other=0.05),
        ])
        cfg = Config(
            absolute_min_probability=0.6,
            class_quantile=0.0,
            min_duration_sec=0.25,
            support_close_gap_sec=0.0,
        )

        _, _, events = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=0.25,
            audio_duration_sec=0.8,
            vad_intervals=((0.0, 99.0),),
            config=cfg,
        )

        self.assertEqual([e.label for e in events], ["sadness", "anger"])
        self.assertEqual(events[0].event_id, "emotion.categorical.v1.categorical.000000")
        self.assertEqual(events[1].event_id, "emotion.categorical.v1.categorical.000001")
        self.assertAlmostEqual(events[0].start_sec, 0.0)
        self.assertAlmostEqual(events[0].end_sec, 0.25)
        self.assertAlmostEqual(events[1].end_sec, 0.8)

    def test_config_hash_is_deterministic(self):
        matrix = np.array([
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(happiness=0.9, neutral=0.05, other=0.05),
        ])
        kwargs = dict(
            probabilities=matrix,
            labels=CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            vad_intervals=((0.0, 99.0),),
            config=Config(class_quantile=0.0),
        )
        run1, _, _ = run_from_probabilities(**kwargs)
        run2, _, _ = run_from_probabilities(**kwargs)
        self.assertEqual(run1.config_hash, run2.config_hash)

    def test_tracks_only_session_round_trip(self):
        matrix = np.array([
            probs(happiness=0.9, neutral=0.05, other=0.05),
            probs(neutral=0.9, happiness=0.05, other=0.05),
        ])
        run, tracks, _ = run_from_probabilities(
            matrix,
            CANONICAL_CHANNELS,
            hop_sec=0.25,
            window_sec=1.0,
            vad_intervals=((0.0, 1.0),),
            config=Config(class_quantile=0.0),
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "sample.wav"
            sf.write(str(audio), np.zeros(16000, dtype=np.float32), 16000)
            path = save_session(
                events=[],
                tracks=tracks,
                producer_runs=[run],
                vad=Vad(intervals=((0.0, 1.0),)),
                audio_path=audio,
                session_dir=root / "sessions",
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["events"], [])
            self.assertIn(TRACK_ID, data["tracks_meta"])


if __name__ == "__main__":
    unittest.main()
