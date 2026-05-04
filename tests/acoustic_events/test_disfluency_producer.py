import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_classification_playground.acoustic_events.producers.affect import Vad
from audio_classification_playground.acoustic_events.producers.disfluency import (
    DISFLUENCY_TYPE_LABELS,
    FLUENCY_TRACK_ID,
    TYPE_TRACK_ID,
    DisfluencyConfig,
    extract_events,
    produce_disfluency_events,
    tracks_from_logits,
)
from audio_classification_playground.acoustic_events.review.storage import save_session


def binary_logits(p_disfluent):
    p = np.asarray(p_disfluent, dtype=np.float64)
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    return np.column_stack([np.zeros_like(p), np.log(p / (1.0 - p))])


def type_logits(rows):
    matrix = np.zeros((len(rows), len(DISFLUENCY_TYPE_LABELS)), dtype=np.float64)
    for i, row in enumerate(rows):
        for label, p in row.items():
            eps = 1e-6
            p = min(max(float(p), eps), 1.0 - eps)
            matrix[i, DISFLUENCY_TYPE_LABELS.index(label)] = math.log(p / (1.0 - p))
    return matrix


class DisfluencyProducerTest(unittest.TestCase):
    def test_validation_rejects_bad_shapes_and_duration(self):
        good_fluency = binary_logits([0.1, 0.2])
        good_types = type_logits([{}, {}])

        with self.assertRaisesRegex(ValueError, "shape"):
            tracks_from_logits(np.zeros((2, 3)), good_types, hop_sec=0.25, window_sec=3.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            tracks_from_logits(good_fluency, np.zeros((2, 4)), hop_sec=0.25, window_sec=3.0)
        with self.assertRaisesRegex(ValueError, "same frame count"):
            tracks_from_logits(good_fluency, type_logits([{}]), hop_sec=0.25, window_sec=3.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            tracks_from_logits([[0, np.nan]], type_logits([{}]), hop_sec=0.25, window_sec=3.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            tracks_from_logits(good_fluency, good_types, hop_sec=0.0, window_sec=3.0)
        with self.assertRaisesRegex(ValueError, "extends beyond audio duration"):
            tracks_from_logits(
                good_fluency,
                good_types,
                hop_sec=1.0,
                window_sec=3.0,
                audio_duration_sec=3.5,
            )

    def test_tracks_are_contract_shaped(self):
        fluency = binary_logits([0.1, 0.8])
        types = type_logits([
            {"Block": 0.2},
            {"Block": 0.9, "Word Repetition": 0.7},
        ])

        tracks = tracks_from_logits(fluency, types, hop_sec=0.25, window_sec=3.0)

        self.assertEqual([track.track_id for track in tracks], [FLUENCY_TRACK_ID, TYPE_TRACK_ID])
        self.assertEqual(tracks[0].producer_id, "disfluency.default")
        self.assertEqual(tracks[0].renderer, "probability")
        self.assertEqual(tracks[0].values.shape, (2,))
        self.assertAlmostEqual(float(tracks[0].values[1]), 0.8)
        self.assertEqual(tracks[0].meta["activation"], "softmax_class_1")
        self.assertEqual(tracks[1].renderer, "multi_probability")
        self.assertEqual(tracks[1].values.shape, (2, 5))
        self.assertEqual(tracks[1].channels, DISFLUENCY_TYPE_LABELS)
        self.assertEqual(tracks[1].meta["activation"], "sigmoid")

    def test_center_support_bounds_and_peak_timing(self):
        fluency = binary_logits([0.1, 0.72, 0.9, 0.76, 0.2])
        types = type_logits([
            {},
            {"Block": 0.8},
            {"Block": 0.9},
            {"Block": 0.7},
            {},
        ])
        events = extract_events(
            fluency,
            types,
            hop_sec=0.25,
            window_sec=3.0,
            config=DisfluencyConfig(
                seed_threshold=0.70,
                shoulder_threshold=0.50,
                min_support_sec=0.50,
                merge_gap_sec=0.0,
                type_threshold=0.70,
            ),
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        # centers are 1.75, 2.0, 2.25 for frames 1:4
        self.assertAlmostEqual(event.start_sec, 1.625)
        self.assertAlmostEqual(event.end_sec, 2.375)
        self.assertAlmostEqual(event.evidence["peak_time_sec"], 2.0)
        self.assertEqual(event.extra["peak_frame"], 2)
        self.assertEqual(event.extra["support_start_frame"], 1)
        self.assertEqual(event.extra["support_end_frame"], 4)
        self.assertAlmostEqual(event.extra["full_receptive_window_bounds"]["start_sec"], 0.25)
        self.assertAlmostEqual(event.extra["full_receptive_window_bounds"]["end_sec"], 3.75)

    def test_hop_aware_support_keeps_short_valid_region_and_filters_spike(self):
        cfg = DisfluencyConfig(
            seed_threshold=0.70,
            shoulder_threshold=0.50,
            min_support_sec=0.50,
            merge_gap_sec=0.0,
            type_threshold=0.70,
        )
        two_frame = extract_events(
            binary_logits([0.1, 0.8, 0.8, 0.1]),
            type_logits([{}, {"Block": 0.9}, {"Block": 0.9}, {}]),
            hop_sec=0.25,
            window_sec=3.0,
            config=cfg,
        )
        one_frame = extract_events(
            binary_logits([0.1, 0.8, 0.1]),
            type_logits([{}, {"Block": 0.9}, {}]),
            hop_sec=0.25,
            window_sec=3.0,
            config=cfg,
        )
        coarse_hop = extract_events(
            binary_logits([0.1, 0.8, 0.1]),
            type_logits([{}, {"Block": 0.9}, {}]),
            hop_sec=1.0,
            window_sec=3.0,
            config=cfg,
        )

        self.assertEqual(len(two_frame), 1)
        self.assertAlmostEqual(two_frame[0].duration_sec, 0.5)
        self.assertEqual(one_frame, [])
        self.assertEqual(len(coarse_hop), 1)

    def test_merge_happens_before_support_filtering(self):
        cfg = DisfluencyConfig(
            seed_threshold=0.70,
            shoulder_threshold=0.50,
            min_support_sec=0.50,
            merge_gap_sec=0.25,
            type_threshold=0.70,
        )
        events = extract_events(
            binary_logits([0.8, 0.1, 0.8]),
            type_logits([{"Block": 0.9}, {}, {"Block": 0.9}]),
            hop_sec=0.25,
            window_sec=3.0,
            config=cfg,
        )
        separated = extract_events(
            binary_logits([0.8, 0.1, 0.1, 0.8]),
            type_logits([{"Block": 0.9}, {}, {}, {"Block": 0.9}]),
            hop_sec=0.25,
            window_sec=3.0,
            config=cfg,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].extra["support_start_frame"], 0)
        self.assertEqual(events[0].extra["support_end_frame"], 3)
        self.assertEqual(separated, [])

    def test_multilabel_suppression_keeps_useful_cofiring_type(self):
        cfg = DisfluencyConfig(
            seed_threshold=0.70,
            shoulder_threshold=0.50,
            min_support_sec=0.50,
            merge_gap_sec=0.0,
            type_threshold=0.70,
        )
        events = extract_events(
            binary_logits([0.8, 0.9]),
            type_logits([
                {"Sound Repetition": 0.95, "Word Repetition": 0.90},
                {"Sound Repetition": 0.95, "Word Repetition": 0.85},
            ]),
            hop_sec=0.25,
            window_sec=3.0,
            config=cfg,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "word_repetition")
        self.assertEqual(events[0].evidence["active_types"][0]["name"], "Word Repetition")
        self.assertEqual(events[0].evidence["suppressed_active_types"][0]["name"], "Sound Repetition")

    def test_pure_sound_repetition_suppressed_by_default_and_unspecified_optional(self):
        fluency = binary_logits([0.8, 0.9])
        types = type_logits([
            {"Sound Repetition": 0.95},
            {"Sound Repetition": 0.95},
        ])
        default_run, _, default_events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=0.25,
            window_sec=3.0,
        )
        unspecified_run, _, unspecified_events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=0.25,
            window_sec=3.0,
            config=DisfluencyConfig(emit_unspecified=True),
        )

        self.assertEqual(default_events, [])
        self.assertEqual(default_run.outputs["suppressed_pure_sound_repetition_count"], 1)
        self.assertEqual(default_run.outputs["unspecified_region_count"], 1)
        self.assertEqual(len(unspecified_events), 1)
        self.assertEqual(unspecified_events[0].label, "disfluent")
        self.assertEqual(
            unspecified_events[0].evidence["suppressed_active_types"][0]["name"],
            "Sound Repetition",
        )
        self.assertEqual(unspecified_run.outputs["emitted_unspecified_event_count"], 1)

    def test_deterministic_ties_use_model_label_order(self):
        events = extract_events(
            binary_logits([0.8, 0.9]),
            type_logits([
                {"Block": 0.8, "Word Repetition": 0.8},
                {"Block": 0.8, "Word Repetition": 0.8},
            ]),
            hop_sec=0.25,
            window_sec=3.0,
            config=DisfluencyConfig(type_threshold=0.70),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "block")

    def test_event_contract_and_save_session_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "sample.wav"
            sf.write(str(audio), np.zeros(80000, dtype=np.float32), 16000)
            run, tracks, events = produce_disfluency_events(
                fluency_logits=binary_logits([0.1, 0.8, 0.85, 0.1]),
                disfluency_type_logits=type_logits([
                    {},
                    {"Interjection": 0.8},
                    {"Interjection": 0.9},
                    {},
                ]),
                hop_sec=0.25,
                window_sec=3.0,
                audio_duration_sec=5.0,
            )

            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertRegex(event.event_id, r"^disfluency\.default\.instance\.\d{6}$")
            self.assertEqual(event.score_name, "probability")
            self.assertEqual(event.source_track_ids, (FLUENCY_TRACK_ID, TYPE_TRACK_ID))
            self.assertAlmostEqual(event.duration_sec, event.end_sec - event.start_sec)
            self.assertTrue(run.config_hash)
            self.assertEqual(run.outputs["label_counts"], {"interjection": 1})

            path = save_session(
                events=events,
                tracks=tracks,
                producer_runs=[run],
                vad=Vad(intervals=((0.0, 5.0),)),
                audio_path=audio,
                session_dir=root / "sessions",
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["event_schema"], "acoustic_events.v1")
            self.assertEqual(data["producer_runs"][0]["producer_id"], "disfluency.default")
            self.assertIn(FLUENCY_TRACK_ID, data["tracks_meta"])
            self.assertEqual(data["tracks_meta"][TYPE_TRACK_ID]["channels"], list(DISFLUENCY_TYPE_LABELS))
            self.assertEqual(data["events"][0]["label"], "interjection")


if __name__ == "__main__":
    unittest.main()
