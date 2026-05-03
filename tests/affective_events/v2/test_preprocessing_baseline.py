import unittest

import numpy as np

from audio_classification_playground.affective_events.v2.baseline import (
    block_aware_baseline_scale,
)
from audio_classification_playground.affective_events.v2.config import Config
from audio_classification_playground.affective_events.v2.preprocessing import (
    assign_frame_blocks,
    build_blocks,
    global_stats,
)
from audio_classification_playground.affective_events.v2.types import Vad


class PreprocessingBaselineTest(unittest.TestCase):
    def test_assign_frame_blocks_uses_fully_interior_receptive_window(self):
        blocks = build_blocks(Vad(intervals=((2.0, 6.0),)), Config())
        frame_block = assign_frame_blocks(
            n_frames=8, hop_sec=1.0, window_sec=2.0, blocks=blocks
        )

        self.assertEqual(frame_block.tolist(), [-1, -1, 0, 0, 0, -1, -1, -1])

    def test_block_baseline_excludes_own_block(self):
        hop = 1.0
        window = 1.0
        values = np.zeros(40, dtype=float)
        values[0:10] = 0.2
        blocks = build_blocks(Vad(intervals=((0, 10), (20, 30))), Config())
        frame_block = assign_frame_blocks(len(values), hop, window, blocks)
        interior = frame_block >= 0
        g_med, g_mad = global_stats(values, interior)

        baseline, _ = block_aware_baseline_scale(
            values,
            frame_block,
            blocks,
            hop,
            window,
            Config(radius_sec=40.0, min_context_sec=1.0),
            global_median=g_med,
            global_mad=g_mad,
        )

        block0 = frame_block == 0
        self.assertTrue(np.allclose(baseline[block0], 0.0))

    def test_long_block_uses_per_frame_context(self):
        hop = 1.0
        window = 1.0
        values = np.zeros(110, dtype=float)
        values[30:70] = 0.5
        values[80:100] = 1.0
        blocks = build_blocks(
            Vad(intervals=((0, 20), (30, 70), (80, 100))),
            Config(min_speech_block_sec=0.1),
        )
        frame_block = assign_frame_blocks(len(values), hop, window, blocks)
        interior = frame_block >= 0
        g_med, g_mad = global_stats(values, interior)

        baseline, _ = block_aware_baseline_scale(
            values,
            frame_block,
            blocks,
            hop,
            window,
            Config(radius_sec=35.0, min_context_sec=1.0),
            global_median=g_med,
            global_mad=g_mad,
        )

        self.assertAlmostEqual(float(baseline[30]), 0.0)
        self.assertAlmostEqual(float(baseline[69]), 1.0)


if __name__ == "__main__":
    unittest.main()
