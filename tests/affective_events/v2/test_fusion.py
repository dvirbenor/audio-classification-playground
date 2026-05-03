import unittest
from itertools import count

from audio_classification_playground.affective_events.v2.config import Config
from audio_classification_playground.affective_events.v2.fusion import (
    attach_parent_ids,
    merge_cross_signal,
)
from audio_classification_playground.affective_events.v2.types import Event


def leaf(event_id, signal, start, end, direction="+", peak_z=2.0):
    return Event(
        event_id=event_id,
        signal_name=signal,
        event_type="deviation",
        start_sec=start,
        end_sec=end,
        duration_sec=end - start,
        frame_start=0,
        frame_end=1,
        direction=direction,
        peak_z=peak_z,
        peak_time_sec=(start + end) / 2,
        baseline_at_peak=0.0,
        scale_at_peak=1.0,
        delta=peak_z,
    )


class FusionTest(unittest.TestCase):
    def test_joint_construction_and_parent_ids(self):
        leaves = [
            leaf("a", "arousal", 10, 15, "+", 2.0),
            leaf("v", "valence", 11, 16, "-", 3.0),
            leaf("d", "dominance", 12, 14, "+", 4.0),
        ]

        parents = merge_cross_signal(leaves, Config(cross_signal_min_overlap_sec=1.0), count())
        leaves_with_parent = attach_parent_ids(leaves, parents)

        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0].event_type, "joint")
        self.assertEqual(set(parents[0].children), {"a", "v", "d"})
        self.assertEqual(parents[0].direction, "A+ V- D+")
        self.assertAlmostEqual(parents[0].peak_z, (2.0**2 + 3.0**2 + 4.0**2) ** 0.5)
        self.assertTrue(all(e.parent_id == parents[0].event_id for e in leaves_with_parent))

    def test_joint_requires_pairwise_overlap(self):
        leaves = [
            leaf("a", "arousal", 0, 3),
            leaf("v", "valence", 2, 5),
            leaf("d", "dominance", 4, 7),
        ]

        parents = merge_cross_signal(leaves, Config(cross_signal_min_overlap_sec=1.0), count())

        self.assertEqual(len(parents), 1)
        self.assertEqual(set(parents[0].children), {"a", "v"})


if __name__ == "__main__":
    unittest.main()
