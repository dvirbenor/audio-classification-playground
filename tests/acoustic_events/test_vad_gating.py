"""Unit tests for VAD-gated inference (inference/vad_gating.py).

The keystone is the *superset* property: the set of frames the affect producer
actually reads (frames contained in merged VAD blocks) must be a subset of the
gate's keep-mask, for any intervals — otherwise gating could change events.
"""
import random
import unittest

import numpy as np

from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    DEFAULT_HOP_SEC,
)
from audio_classification_playground.acoustic_events.inference.vad_gating import (
    DEFAULT_BRIDGE_SEC,
    VadGating,
    bridge_intervals,
    gate_window_arrays,
    intervals_from_array,
    scatter_fill,
    speech_window_mask,
)
from audio_classification_playground.acoustic_events.producers.affect.config import Config
from audio_classification_playground.acoustic_events.producers.affect.preprocessing import (
    assign_frame_blocks,
    build_blocks,
)
from audio_classification_playground.acoustic_events.producers.affect.types import Vad


class SpeechWindowMaskTest(unittest.TestCase):
    def test_single_interval_overlap_bounds(self):
        # hop=0.25, window=3.5, interval [10, 12]: windows whose span touches it.
        mask = speech_window_mask(
            n_frames=200, hop_sec=0.25, window_sec=3.5,
            vad_intervals=[(10.0, 12.0)], bridge_sec=0.0,
        )
        kept = np.nonzero(mask)[0]
        # window 27 starts at 6.75 -> [6.75, 10.25] overlaps; 26 -> [6.5, 10.0] touches only.
        self.assertEqual(int(kept.min()), 27)
        # window 47 starts at 11.75 < 12 overlaps; 48 starts at 12.0 -> touches only.
        self.assertEqual(int(kept.max()), 47)

    def test_overlap_not_containment_keeps_straddlers(self):
        # A window straddling the speech onset must be kept (overlap semantics).
        mask = speech_window_mask(
            n_frames=200, hop_sec=0.25, window_sec=3.5,
            vad_intervals=[(10.0, 12.0)], bridge_sec=0.0,
        )
        # window 28 spans [7.0, 10.5]: mostly silence, 0.5s speech -> kept.
        self.assertTrue(mask[28])

    def test_empty_intervals_all_false(self):
        mask = speech_window_mask(100, 0.25, 3.5, [], bridge_sec=1.5)
        self.assertEqual(mask.shape, (100,))
        self.assertFalse(mask.any())

    def test_zero_frames(self):
        self.assertEqual(speech_window_mask(0, 0.25, 3.5, [(1.0, 2.0)]).shape, (0,))

    def test_invalid_geometry_raises(self):
        with self.assertRaises(ValueError):
            speech_window_mask(10, 0.0, 3.5, [(1.0, 2.0)])


class BridgeIntervalsTest(unittest.TestCase):
    def test_merges_within_gap(self):
        merged = bridge_intervals([(0.0, 1.0), (1.4, 2.0)], bridge_sec=0.5)
        self.assertEqual(merged, [(0.0, 2.0)])

    def test_keeps_separate_beyond_gap(self):
        merged = bridge_intervals([(0.0, 1.0), (1.6, 2.0)], bridge_sec=0.5)
        self.assertEqual(merged, [(0.0, 1.0), (1.6, 2.0)])

    def test_sorts_and_drops_nonpositive(self):
        merged = bridge_intervals([(5.0, 6.0), (2.0, 2.0), (0.0, 1.0)], bridge_sec=0.0)
        self.assertEqual(merged, [(0.0, 1.0), (5.0, 6.0)])

    def test_larger_bridge_is_superset_mask(self):
        # Gap (3.0s) wider than the window's forward-reach, so only a bridge
        # that spans it adds windows in the gap region.
        intervals = [(0.0, 1.0), (4.0, 5.0)]  # gap 3.0
        narrow = speech_window_mask(400, 0.25, 3.0, intervals, bridge_sec=0.5)
        wide = speech_window_mask(400, 0.25, 3.0, intervals, bridge_sec=3.0)
        # wide bridges the gap, narrow doesn't -> wide keeps >= narrow everywhere
        self.assertTrue(np.all(wide[narrow]))
        self.assertGreater(wide.sum(), narrow.sum())


class SupersetPropertyTest(unittest.TestCase):
    """Every frame the affect producer reads must be inside the gate keep-mask."""

    def _interior_frames(self, intervals, n_frames, window_sec):
        cfg = Config.balanced()
        vad = Vad(intervals=tuple(intervals))
        blocks = build_blocks(vad, cfg)
        frame_block = assign_frame_blocks(n_frames, DEFAULT_HOP_SEC, window_sec, blocks)
        return frame_block >= 0

    def test_affect_interior_subset_of_gate_random(self):
        rng = random.Random(1234)
        for trial in range(200):
            # build random non-degenerate intervals over ~120 s
            intervals = []
            t = rng.uniform(0.0, 5.0)
            while t < 115.0:
                dur = rng.uniform(0.05, 6.0)
                intervals.append((t, t + dur))
                t += dur + rng.uniform(0.05, 4.0)
            if not intervals:
                continue
            n_frames = int(120.0 / DEFAULT_HOP_SEC)
            for window_sec in (AFFECT_WINDOW_SEC, DISFLUENCY_WINDOW_SEC):
                interior = self._interior_frames(intervals, n_frames, window_sec)
                mask = speech_window_mask(
                    n_frames, DEFAULT_HOP_SEC, window_sec, intervals,
                    bridge_sec=DEFAULT_BRIDGE_SEC,
                )
                missed = np.nonzero(interior & ~mask)[0]
                self.assertEqual(
                    missed.size, 0,
                    msg=f"trial {trial} window {window_sec}: producer reads "
                        f"{missed.size} frames the gate dropped",
                )

    def test_gate_default_bridge_covers_emotion_close_gap(self):
        # emotion closes holes up to 1.0s; default bridge (1.5) must cover that.
        self.assertGreaterEqual(DEFAULT_BRIDGE_SEC, 1.0)


class ScatterFillTest(unittest.TestCase):
    def test_1d_scatter(self):
        kept = np.array([5.0, 7.0], dtype=np.float32)
        idx = np.array([1, 3])
        out = scatter_fill(kept, idx, 5, fill=0.0)
        np.testing.assert_array_equal(out, [0.0, 5.0, 0.0, 7.0, 0.0])
        self.assertEqual(out.dtype, np.float32)

    def test_2d_scatter(self):
        kept = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        idx = np.array([0, 2])
        out = scatter_fill(kept, idx, 4, fill=-1.0)
        self.assertEqual(out.shape, (4, 2))
        np.testing.assert_array_equal(out[0], [1.0, 2.0])
        np.testing.assert_array_equal(out[1], [-1.0, -1.0])
        np.testing.assert_array_equal(out[2], [3.0, 4.0])

    def test_empty_kept(self):
        out = scatter_fill(np.zeros((0, 2), np.float32), np.array([], dtype=int), 3, fill=0.0)
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue((out == 0.0).all())


class GateWindowArraysTest(unittest.TestCase):
    @staticmethod
    def _predict(windows):
        # deterministic per-window scalar = mean of the window
        return {"value": windows.mean(axis=1).astype(np.float32)}

    def _windows(self, n, win=8):
        rng = np.random.default_rng(0)
        return rng.standard_normal((n, win)).astype(np.float32)

    def test_inactive_gating_is_passthrough(self):
        w = self._windows(40)
        arrays, mask = gate_window_arrays(
            self._predict, w, hop_sec=0.25, window_sec=3.5,
            vad_intervals=[(1.0, 2.0)], gating=VadGating(enabled=False),
        )
        self.assertIsNone(mask)
        np.testing.assert_array_equal(arrays["value"], self._predict(w)["value"])

    def test_none_intervals_is_passthrough(self):
        w = self._windows(40)
        arrays, mask = gate_window_arrays(
            self._predict, w, hop_sec=0.25, window_sec=3.5,
            vad_intervals=None, gating=VadGating(enabled=True),
        )
        self.assertIsNone(mask)
        np.testing.assert_array_equal(arrays["value"], self._predict(w)["value"])

    def test_gated_matches_full_at_kept_and_fills_rest(self):
        n = 200
        w = self._windows(n)
        intervals = [(10.0, 12.0)]
        full = self._predict(w)["value"]
        arrays, mask = gate_window_arrays(
            self._predict, w, hop_sec=0.25, window_sec=3.5,
            vad_intervals=intervals, gating=VadGating(enabled=True, bridge_sec=0.0),
            fill=0.0,
        )
        self.assertEqual(arrays["value"].shape, (n,))
        kept = np.nonzero(mask)[0]
        self.assertGreater(kept.size, 0)
        self.assertLess(kept.size, n)
        # kept frames are bit-identical to the full-timeline computation
        np.testing.assert_array_equal(arrays["value"][kept], full[kept])
        # non-kept frames are the fill sentinel
        non_kept = np.nonzero(~mask)[0]
        self.assertTrue((arrays["value"][non_kept] == 0.0).all())

    def test_all_silence_skips_compute(self):
        calls = []

        def predict(windows):
            calls.append(len(windows))
            return {"value": windows.mean(axis=1).astype(np.float32)}

        w = self._windows(50)
        arrays, mask = gate_window_arrays(
            predict, w, hop_sec=0.25, window_sec=3.5,
            vad_intervals=[], gating=VadGating(enabled=True),
        )
        self.assertFalse(mask.any())
        self.assertEqual(arrays["value"].shape, (50,))
        self.assertTrue((arrays["value"] == 0.0).all())
        self.assertEqual(calls, [0])  # predictor still called, but on zero windows


class VadGatingConfigTest(unittest.TestCase):
    def test_config_extra_in_hash_surface(self):
        extra = VadGating(enabled=True, bridge_sec=1.5).config_extra()
        self.assertEqual(extra, {"vad_gating": {"policy": "overlap_v1", "bridge_sec": 1.5}})

    def test_rejects_bad_policy_and_bridge(self):
        with self.assertRaises(ValueError):
            VadGating(policy="nope")
        with self.assertRaises(ValueError):
            VadGating(bridge_sec=-1.0)
        with self.assertRaises(ValueError):
            VadGating(tasks=("affect", "bogus"))

    def test_gates_default_includes_all_gatable_tasks(self):
        g = VadGating(enabled=True)
        self.assertTrue(g.gates("affect"))
        self.assertTrue(g.gates("emotion"))
        # disfluency region detection is speech-scoped, so gating it is
        # event-identical and it is gated by default.
        self.assertTrue(g.gates("disfluency"))
        # disabled never gates; explicit task subset works
        self.assertFalse(VadGating(enabled=False, tasks=("affect",)).gates("affect"))
        self.assertFalse(VadGating(enabled=True, tasks=("affect",)).gates("disfluency"))

    def test_intervals_from_array_roundtrip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.5]], dtype=np.float32)
        self.assertEqual(intervals_from_array(arr), [(1.0, 2.0), (3.0, 4.5)])
        self.assertEqual(intervals_from_array(None), [])
        self.assertEqual(intervals_from_array(np.zeros((0, 2))), [])


if __name__ == "__main__":
    unittest.main()
