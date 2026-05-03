import unittest

import numpy as np

from audio_classification_playground.affective_events import (
    Config,
    Signal,
    Vad,
    extract_events,
)


class SimpleAffectiveEventsTest(unittest.TestCase):
    def test_obvious_shifted_speech_block_emits_single_block_deviation(self):
        hop = 0.25
        window = 3.5
        n_frames = 420
        rng = np.random.default_rng(0)
        values = rng.normal(0.0, 0.04, n_frames)
        centers = np.arange(n_frames) * hop + window / 2

        vad = Vad(intervals=((5, 18), (30, 44), (58, 70), (86, 100)))
        values[(centers >= 58) & (centers <= 70)] += 0.55

        events = extract_events(
            [Signal("arousal", values, hop_sec=hop, window_sec=window)],
            vad,
            Config.balanced(),
        )

        leaves = [e for e in events if e.event_type == "block_deviation"]
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].block_ids, (2,))
        self.assertGreater(leaves[0].delta_z, Config.balanced().baseline_departure_z)

    def test_v2_backend_is_available_alongside_v1(self):
        from audio_classification_playground.affective_events.v2 import Config as V2Config

        self.assertEqual(V2Config.balanced().z_seed, 1.75)


if __name__ == "__main__":
    unittest.main()
