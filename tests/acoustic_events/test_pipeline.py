import json
import os
import unittest
from pathlib import Path

import numpy as np

from audio_classification_playground.acoustic_events import (
    Config,
    Signal,
    Vad,
    extract_events,
    tracks_from_signals,
)
from audio_classification_playground.acoustic_events.producers.affect.config import value_for_signal
from audio_classification_playground.acoustic_events.producers.affect.preprocessing import assign_frame_blocks, build_blocks


class PipelineTest(unittest.TestCase):
    def test_long_shifted_block_with_internal_peak_is_one_long_event(self):
        hop = 1.0
        window = 1.0
        values = np.zeros(270, dtype=float)
        values[30:230:2] = 0.14
        values[31:230:2] = 0.16
        values[100:106] = 0.45
        # Keep context non-perfectly-flat but far below the shifted block.
        values[0:20:2] = 0.02
        values[1:20:2] = -0.02
        values[240:260:2] = 0.02
        values[241:260:2] = -0.02

        config = Config(
            radius_sec=120.0,
            min_context_sec=2.0,
            z_seed=1.75,
            seed_min_width_sec=1.0,
            z_return=0.5,
            min_duration_sec=2.5,
            merge_gap_sec=0.5,
        )
        events = extract_events(
            [Signal("arousal", values, hop, window)],
            Vad(intervals=((0, 20), (30, 230), (240, 260))),
            config,
        )
        leaves = [
            e for e in events
            if e.source_track_ids == ("affect.arousal",) and e.event_type == "deviation"
        ]

        self.assertTrue(
            any(e.extra["frame_start"] == 30 and e.extra["frame_end"] == 230 for e in leaves),
            [e.as_dict() for e in leaves],
        )
        long_event = next(e for e in leaves if e.extra["frame_start"] == 30 and e.extra["frame_end"] == 230)
        self.assertRegex(long_event.event_id, r"^affect\.default\.deviation\.\d{6}$")
        self.assertEqual(long_event.task, "affect")
        self.assertEqual(long_event.label, "arousal_deviation")
        self.assertEqual(long_event.source_track_ids, ("affect.arousal",))
        self.assertEqual(long_event.score_name, "peak_z")
        self.assertIn("baseline_at_peak", long_event.evidence)
        self.assertEqual(long_event.direction, "+")
        self.assertIn(long_event.extra["peak_frame"], range(100, 106))

    def test_tracks_from_signals_maps_affect_to_regular_grid_tracks(self):
        track = tracks_from_signals([
            Signal("valence", np.array([0.1, 0.2]), hop_sec=0.25, window_sec=3.5)
        ])[0]

        self.assertEqual(track.track_id, "affect.valence")
        self.assertEqual(track.producer_id, "affect.default")
        self.assertEqual(track.task, "affect")
        self.assertEqual(track.value_type, "continuous")
        self.assertEqual(track.renderer, "line")

    def test_synthetic_pipeline_invariants(self):
        hop = 0.25
        window = 1.0
        values = np.zeros(220, dtype=float)
        values[40:47] = [0.0, 0.08, 0.14, 0.30, 0.30, 0.14, 0.08]
        values[130:137] = [0.0, -0.08, -0.14, -0.30, -0.30, -0.14, -0.08]
        values[0:30:2] = 0.02
        values[1:30:2] = -0.02
        values[180:210:2] = 0.02
        values[181:210:2] = -0.02
        config = Config(
            radius_sec=80.0,
            min_context_sec=2.0,
            z_seed=2.0,
            seed_min_width_sec=0.5,
            z_return=0.5,
            min_duration_sec=1.0,
            merge_gap_sec=0.5,
        )
        vad = Vad(intervals=((0, 55), (80, 155), (180, 210)))
        events = extract_events([Signal("arousal", values, hop, window)], vad, config)
        leaves = [e for e in events if e.event_type == "deviation"]
        self.assertGreaterEqual(len(leaves), 2)

        blocks = build_blocks(vad, config)
        frame_block = assign_frame_blocks(len(values), hop, window, blocks)
        interior = frame_block >= 0
        _assert_invariants(self, leaves, interior, config, "arousal", hop)

    def test_golden_recording_if_fixture_is_available(self):
        root = Path(os.environ.get(
            "AFFECTIVE_EVENTS_GOLDEN_DIR",
            "/workspace/labeling/jamespiper-2026-2-9__14-44-54",
        ))
        json_path = root / "8c8d972078e96826__20260429T134229Z.json"
        npz_path = root / "8c8d972078e96826__20260429T134229Z.npz"
        if not json_path.exists() or not npz_path.exists():
            self.skipTest("golden recording fixture is not available")

        with json_path.open() as f:
            metadata = json.load(f)
        arrays = np.load(npz_path)
        signals = [
            Signal("arousal", arrays["arousal"], 0.25, 3.5),
            Signal("valence", arrays["valence"], 0.25, 3.5),
            Signal("dominance", arrays["dominance"], 0.25, 3.5),
        ]
        events = extract_events(
            signals,
            Vad(intervals=tuple(tuple(x) for x in metadata["vad_intervals"])),
            Config.balanced(),
        )
        counts = {
            name: sum(1 for e in events if e.event_type == "deviation" and e.source_track_ids == (f"affect.{name}",))
            for name in ("arousal", "valence", "dominance")
        }
        for count in counts.values():
            self.assertGreaterEqual(count, 30)
            self.assertLessEqual(count, 90)


def _assert_invariants(testcase, leaves, interior, config, signal_name, hop):
    z_return = value_for_signal(config.z_return, signal_name)
    min_duration_sec = value_for_signal(config.min_duration_sec, signal_name)
    merge_gap_frames = int(round(value_for_signal(config.merge_gap_sec, signal_name) / hop))

    for event in leaves:
        frame_start = event.extra["frame_start"]
        frame_end = event.extra["frame_end"]
        testcase.assertTrue(interior[frame_start])
        testcase.assertTrue(interior[frame_end - 1])
        testcase.assertTrue(interior[frame_start:frame_end].all())
        testcase.assertGreaterEqual(event.duration_sec, min_duration_sec)
        testcase.assertGreaterEqual(event.score, value_for_signal(config.z_seed, signal_name))
        testcase.assertGreaterEqual(event.extra["seed_width_frames"] * hop, value_for_signal(config.seed_min_width_sec, signal_name))
        testcase.assertGreaterEqual(event.extra["shoulder_start_z"], z_return)
        testcase.assertGreaterEqual(event.extra["shoulder_end_z"], z_return)
        if event.extra["next_left_z"] is not None and frame_start > 0 and interior[frame_start - 1]:
            testcase.assertLess(event.extra["next_left_z"], z_return)
        if event.extra["next_right_z"] is not None and frame_end < len(interior) and interior[frame_end]:
            testcase.assertLess(event.extra["next_right_z"], z_return)

    for prev, cur in zip(leaves, leaves[1:]):
        if prev.source_track_ids == cur.source_track_ids and prev.direction == cur.direction:
            testcase.assertGreater(cur.extra["frame_start"] - prev.extra["frame_end"], merge_gap_frames)


if __name__ == "__main__":
    unittest.main()
