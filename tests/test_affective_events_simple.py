import unittest

import numpy as np

from audio_classification_playground.affective_events import (
    Config,
    Signal,
    Vad,
    extract_events,
    tracks_from_signals,
)
from audio_classification_playground.affective_events.config import value_for_signal


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

        leaves = [e for e in events if e.event_type == "deviation"]
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].evidence["block_ids"], (2,))
        self.assertGreater(leaves[0].score, value_for_signal(Config.balanced().z_seed, "arousal"))

    def test_tracks_from_signals_uses_canonical_track_schema(self):
        track = tracks_from_signals([
            Signal("arousal", np.zeros(4), hop_sec=0.25, window_sec=1.0)
        ])[0]

        self.assertEqual(track.track_id, "affect.arousal")
        self.assertEqual(track.task, "affect")
        self.assertEqual(track.renderer, "line")


if __name__ == "__main__":
    unittest.main()
