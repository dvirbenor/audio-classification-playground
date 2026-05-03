import unittest

import numpy as np

from audio_classification_playground.acoustic_events.producers.affect.config import Config
from audio_classification_playground.acoustic_events.producers.affect.detector import detect_prominence


class DetectorTest(unittest.TestCase):
    def test_seed_width_filter_drops_single_frame_crossing(self):
        z = np.zeros(30)
        z[3] = 3.0
        z[12:17] = 2.2
        interior = np.ones_like(z, dtype=bool)

        events = detect_prominence(
            z,
            interior,
            Config(
                z_seed=2.0,
                seed_min_width_sec=1.0,
                z_return=0.5,
                min_duration_sec=0.5,
                merge_gap_sec=0.0,
            ),
            "arousal",
            hop_sec=0.25,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["peak_i"], 12)

    def test_shoulder_boundary_matches_return_threshold(self):
        z = np.array([0.0, 0.4, 0.6, 1.0, 2.2, 1.0, 0.6, 0.4, 0.0])
        interior = np.ones_like(z, dtype=bool)

        events = detect_prominence(
            z,
            interior,
            Config(
                z_seed=2.0,
                seed_min_width_sec=0.25,
                z_return=0.5,
                min_duration_sec=0.5,
                merge_gap_sec=0.0,
            ),
            "arousal",
            hop_sec=0.25,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["frame_start"], 2)
        self.assertEqual(events[0]["frame_end"], 7)

    def test_shoulder_trace_stops_at_non_interior_frame(self):
        z = np.array([0.0, 0.4, 0.6, 1.0, 2.2, 1.0, 0.8, 0.6, 0.0])
        interior = np.ones_like(z, dtype=bool)
        interior[6] = False

        events = detect_prominence(
            z,
            interior,
            Config(
                z_seed=2.0,
                seed_min_width_sec=0.25,
                z_return=0.5,
                min_duration_sec=0.5,
                merge_gap_sec=0.0,
            ),
            "arousal",
            hop_sec=0.25,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["frame_start"], 2)
        self.assertEqual(events[0]["frame_end"], 6)

    def test_same_sign_merges_but_opposite_sign_splits(self):
        z = np.zeros(40)
        z[4:8] = [0.6, 2.2, 2.1, 0.6]
        z[10:14] = [0.6, 2.3, 2.1, 0.6]
        z[16:20] = [-0.6, -2.4, -2.1, -0.6]
        interior = np.ones_like(z, dtype=bool)

        events = detect_prominence(
            z,
            interior,
            Config(
                z_seed=2.0,
                seed_min_width_sec=0.5,
                z_return=0.5,
                min_duration_sec=0.5,
                merge_gap_sec=0.5,
            ),
            "arousal",
            hop_sec=0.25,
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["direction"], "+")
        self.assertEqual(events[0]["frame_start"], 4)
        self.assertEqual(events[0]["frame_end"], 14)
        self.assertEqual(events[1]["direction"], "-")


if __name__ == "__main__":
    unittest.main()
