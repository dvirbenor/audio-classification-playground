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
            require_vad_for_events=False,
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
            require_vad_for_events=False,
            config=cfg,
        )
        one_frame = extract_events(
            binary_logits([0.1, 0.8, 0.1]),
            type_logits([{}, {"Block": 0.9}, {}]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=cfg,
        )
        coarse_hop = extract_events(
            binary_logits([0.1, 0.8, 0.1]),
            type_logits([{}, {"Block": 0.9}, {}]),
            hop_sec=1.0,
            window_sec=3.0,
            require_vad_for_events=False,
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
            require_vad_for_events=False,
            config=cfg,
        )
        separated = extract_events(
            binary_logits([0.8, 0.1, 0.1, 0.8]),
            type_logits([{"Block": 0.9}, {}, {}, {"Block": 0.9}]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=cfg,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].extra["support_start_frame"], 0)
        self.assertEqual(events[0].extra["support_end_frame"], 3)
        self.assertEqual(separated, [])

    def test_useful_type_wins_over_sound_repetition_at_peak(self):
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
                {"Sound Repetition": 0.75, "Word Repetition": 0.85},
            ]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=cfg,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "word_repetition")
        self.assertEqual(events[0].evidence["active_types"][0]["name"], "Word Repetition")
        self.assertEqual(events[0].evidence["suppressed_active_types"][0]["name"], "Sound Repetition")

    def test_sound_repetition_dominant_at_peak_still_emits_non_suppressed_label(self):
        cfg = DisfluencyConfig(
            seed_threshold=0.70,
            shoulder_threshold=0.50,
            min_support_sec=0.50,
            merge_gap_sec=0.0,
            type_threshold=0.70,
        )
        run, _, events = produce_disfluency_events(
            fluency_logits=binary_logits([0.8, 0.9]),
            disfluency_type_logits=type_logits([
                {"Sound Repetition": 0.8, "Word Repetition": 0.9},
                {"Sound Repetition": 0.9, "Word Repetition": 0.8},
            ]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=cfg,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "word_repetition")
        self.assertTrue(
            any(t["name"] == "Sound Repetition" for t in events[0].evidence["suppressed_active_types"]),
        )

    def test_sound_repetition_high_overall_still_labels_non_suppressed(self):
        events = extract_events(
            binary_logits([0.8, 0.9]),
            type_logits([
                {"Sound Repetition": 0.95, "Word Repetition": 0.72},
                {"Sound Repetition": 0.20, "Word Repetition": 0.85},
            ]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=DisfluencyConfig(
                seed_threshold=0.70,
                shoulder_threshold=0.50,
                min_support_sec=0.50,
                merge_gap_sec=0.0,
                type_threshold=0.70,
            ),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "word_repetition")
        self.assertGreater(events[0].evidence["type_max"]["Sound Repetition"], 0.9)
        self.assertGreater(
            events[0].evidence["type_at_peak"]["Word Repetition"],
            events[0].evidence["type_at_peak"]["Sound Repetition"],
        )

    def test_pure_sound_repetition_suppressed_by_default_and_not_unspecified(self):
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
            require_vad_for_events=False,
        )
        unspecified_run, _, unspecified_events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=DisfluencyConfig(emit_unspecified=True),
        )

        self.assertEqual(default_events, [])
        self.assertEqual(default_run.outputs["suppressed_pure_count"], 1)
        self.assertEqual(default_run.outputs["unspecified_region_count"], 1)
        self.assertEqual(unspecified_events, [])
        self.assertEqual(unspecified_run.outputs["suppressed_pure_count"], 1)
        self.assertEqual(unspecified_run.outputs["unspecified_region_count"], 1)
        self.assertEqual(unspecified_run.outputs["emitted_unspecified_event_count"], 0)

    def test_unspecified_can_emit_when_no_type_is_active(self):
        run, _, events = produce_disfluency_events(
            fluency_logits=binary_logits([0.8, 0.9]),
            disfluency_type_logits=type_logits([{}, {}]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=DisfluencyConfig(emit_unspecified=True),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "disfluent")
        self.assertEqual(events[0].evidence["active_types"], [])
        self.assertEqual(events[0].evidence["suppressed_active_types"], [])
        self.assertEqual(run.outputs["unspecified_region_count"], 1)
        self.assertEqual(run.outputs["emitted_unspecified_event_count"], 1)

    def test_empty_suppressed_types_allows_sound_repetition_events(self):
        _, _, events = produce_disfluency_events(
            fluency_logits=binary_logits([0.8, 0.9]),
            disfluency_type_logits=type_logits([
                {"Sound Repetition": 0.9},
                {"Sound Repetition": 0.95},
            ]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
            config=DisfluencyConfig(suppressed_types=()),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "sound_repetition")
        self.assertEqual(events[0].evidence["active_types"][0]["name"], "Sound Repetition")
        self.assertEqual(events[0].evidence["suppressed_active_types"], [])

    def test_deterministic_ties_use_model_label_order(self):
        events = extract_events(
            binary_logits([0.8, 0.9]),
            type_logits([
                {"Block": 0.8, "Word Repetition": 0.8},
                {"Block": 0.8, "Word Repetition": 0.8},
            ]),
            hop_sec=0.25,
            window_sec=3.0,
            require_vad_for_events=False,
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
                fluency_logits=binary_logits([0.1, 0.90, 0.95, 0.1]),
                disfluency_type_logits=type_logits([
                    {},
                    {"Interjection": 0.92},
                    {"Interjection": 0.95},
                    {},
                ]),
                hop_sec=0.25,
                window_sec=3.0,
                audio_duration_sec=5.0,
                require_vad_for_events=False,
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


class DisfluencyVadGatingTest(unittest.TestCase):
    """VAD-based speech gating for disfluency event extraction."""

    CFG = DisfluencyConfig(
        seed_threshold=0.70,
        shoulder_threshold=0.50,
        min_support_sec=0.50,
        merge_gap_sec=0.0,
        type_threshold=0.70,
    )
    HOP = 0.25
    WINDOW = 3.0

    def _two_frame_disfluent(self):
        """Two adjacent high-confidence disfluent frames (min_support_frames=2)."""
        return (
            binary_logits([0.1, 0.8, 0.9, 0.1]),
            type_logits([{}, {"Block": 0.9}, {"Block": 0.9}, {}]),
        )

    def test_vad_required_but_missing_emits_no_events(self):
        fluency, types = self._two_frame_disfluent()
        run, tracks, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            config=self.CFG,
        )

        self.assertEqual(events, [])
        self.assertEqual(len(tracks), 2)
        vad = run.outputs["vad"]
        self.assertFalse(vad["provided"])
        self.assertTrue(vad["required_for_events"])
        self.assertEqual(vad["no_event_reason"], "vad_required_but_missing")

    def test_vad_found_no_speech_emits_no_events(self):
        fluency, types = self._two_frame_disfluent()
        run, tracks, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            vad_intervals=(),
            config=self.CFG,
        )

        self.assertEqual(events, [])
        self.assertEqual(len(tracks), 2)
        vad = run.outputs["vad"]
        self.assertTrue(vad["provided"])
        self.assertEqual(vad["speech_frame_count"], 0)
        self.assertEqual(vad["no_event_reason"], "vad_found_no_speech")

    def test_require_vad_false_skips_speech_filter(self):
        fluency, types = self._two_frame_disfluent()
        run, _, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            require_vad_for_events=False,
            config=self.CFG,
        )

        self.assertEqual(len(events), 1)
        vad = run.outputs["vad"]
        self.assertFalse(vad["provided"])
        self.assertFalse(vad["required_for_events"])
        self.assertIsNone(vad["no_event_reason"])
        self.assertNotIn("speech_support_frames", events[0].extra)

    def test_region_in_speech_passes(self):
        fluency, types = self._two_frame_disfluent()
        run, _, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            vad_intervals=((0.0, 10.0),),
            config=self.CFG,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].extra["speech_support_frames"], 2)
        self.assertAlmostEqual(events[0].extra["speech_ratio"], 1.0)
        self.assertEqual(run.outputs["suppressed_insufficient_speech_count"], 0)

    def test_region_with_insufficient_speech_support_suppressed(self):
        fluency, types = self._two_frame_disfluent()
        # Frame centers: [1.5, 1.75, 2.0, 2.25] (i*0.25 + 1.5)
        # p_disfluent peaks at frames 1 (speech) and 2 (NON-speech).
        # Frame bins: [1.625..1.875, 1.875..2.125]
        # VAD (1.6, 1.87): overlaps frame 1 (1.625<1.87, 1.6<1.875) but
        #   not frame 2 (1.875<2.125 yes, but 1.87>1.875 no → False)
        # Region geometry is built over speech-masked confidence, so the
        # non-speech peak at frame 2 is zeroed and never seeds/expands a region.
        # Only frame 1 survives as a seed (1 frame < min_support_frames=2), so
        # no candidate region forms at all — suppression happens at the
        # speech-scoped geometry stage, not the later speech-support filter.
        run, _, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            vad_intervals=((1.6, 1.87),),
            config=self.CFG,
        )

        self.assertEqual(events, [])
        self.assertEqual(run.outputs["candidate_region_count"], 0)
        self.assertEqual(run.outputs["suppressed_insufficient_speech_count"], 0)

    def test_region_at_speech_boundary_trims_to_speech(self):
        fluency = binary_logits([0.1, 0.8, 0.9, 0.8, 0.1])
        types = type_logits([
            {},
            {"Block": 0.9},
            {"Block": 0.9},
            {"Block": 0.9},
            {},
        ])
        # Over the full timeline this region would be frames [1,4) (centers
        # [1.75, 2.0, 2.25]) — shoulder-expanding into frame 3.
        # Frame bins: [1.625..1.875, 1.875..2.125, 2.125..2.375]
        # VAD (0.0, 2.1): overlaps frames 1 and 2, not frame 3 (2.1 < 2.125).
        # Geometry is built over speech-masked confidence, so frame 3 is zeroed
        # and the region trims to [1,3) at the speech boundary instead of
        # leaking into non-speech: full speech support, no non-speech tail.
        run, _, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            vad_intervals=((0.0, 2.1),),
            config=self.CFG,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].extra["support_start_frame"], 1)
        self.assertEqual(events[0].extra["support_end_frame"], 3)
        self.assertEqual(events[0].extra["speech_support_frames"], 2)
        self.assertAlmostEqual(events[0].extra["speech_ratio"], 1.0)

    def test_short_nonspeech_gap_still_bridges_two_speech_bumps(self):
        # Two disfluent bumps on speech (frames [1,3) and [4,6)) separated by a
        # single non-speech frame 3. Speech-scoped geometry zeroes frame 3 so it
        # neither seeds nor expands, but the index gap (0.25s) is within
        # merge_gap_sec, so the bumps still merge into one region [1,6).
        # Aggregation then reads the real (un-masked) confidence at the bridged
        # frame 3 — the value VAD gating keeps, since merge_gap_sec <= bridge_sec.
        cfg = DisfluencyConfig(
            seed_threshold=0.70,
            shoulder_threshold=0.50,
            min_support_sec=0.50,
            merge_gap_sec=0.25,
            type_threshold=0.70,
        )
        fluency = binary_logits([0.1, 0.8, 0.9, 0.1, 0.9, 0.8])
        types = type_logits([
            {},
            {"Block": 0.9},
            {"Block": 0.9},
            {},
            {"Block": 0.9},
            {"Block": 0.9},
        ])
        # VAD covers frames 1,2 and 4,5 but leaves frame 3 (center 2.25,
        # bin [2.125, 2.375]) non-speech.
        run, _, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            vad_intervals=((1.6, 2.12), (2.38, 3.0)),
            config=cfg,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].extra["support_start_frame"], 1)
        self.assertEqual(events[0].extra["support_end_frame"], 6)
        # 4 speech frames (1,2,4,5); the bridged non-speech frame 3 is excluded.
        self.assertEqual(events[0].extra["speech_support_frames"], 4)
        self.assertAlmostEqual(events[0].extra["speech_ratio"], 4.0 / 5.0)
        self.assertEqual(run.outputs["candidate_region_count"], 1)

    def test_empty_vad_suppresses_even_when_not_required(self):
        fluency, types = self._two_frame_disfluent()
        run, _, events = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            vad_intervals=(),
            require_vad_for_events=False,
            config=self.CFG,
        )

        self.assertEqual(events, [])
        self.assertEqual(run.outputs["vad"]["no_event_reason"], "vad_found_no_speech")

    def test_invalid_vad_interval_rejected(self):
        fluency, types = self._two_frame_disfluent()
        with self.assertRaisesRegex(ValueError, "non-positive duration"):
            extract_events(
                fluency, types,
                hop_sec=self.HOP,
                window_sec=self.WINDOW,
                vad_intervals=((3.0, 2.0),),
                config=self.CFG,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            extract_events(
                fluency, types,
                hop_sec=self.HOP,
                window_sec=self.WINDOW,
                vad_intervals=((float("nan"), 1.0),),
                config=self.CFG,
            )

    def test_config_hash_reflects_require_vad_for_events(self):
        fluency, types = self._two_frame_disfluent()
        with_vad, _, _ = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            require_vad_for_events=True,
            config=self.CFG,
        )
        without_vad, _, _ = produce_disfluency_events(
            fluency_logits=fluency,
            disfluency_type_logits=types,
            hop_sec=self.HOP,
            window_sec=self.WINDOW,
            require_vad_for_events=False,
            config=self.CFG,
        )

        self.assertNotEqual(with_vad.config_hash, without_vad.config_hash)
        self.assertTrue(with_vad.config["require_vad_for_events"])
        self.assertFalse(without_vad.config["require_vad_for_events"])


if __name__ == "__main__":
    unittest.main()
