"""Block construction, per-frame speech coverage, smoothing, boundary flags.

Everything here is purely a function of the inputs (signal grid + VAD); no
detection logic lives in this module.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .config import Config
from .types import Block, Signal, Vad


# ---------------------------------------------------------------------------
# VAD → analysis blocks
# ---------------------------------------------------------------------------


def build_blocks(vad: Vad, config: Config) -> list[Block]:
    """Merge raw VAD intervals into analysis blocks and attach gap metadata."""
    merged = vad.merged(
        max_gap_sec=config.vad_merge_gap_sec,
        min_duration_sec=config.min_speech_block_sec,
    )
    intervals = merged.intervals
    blocks: list[Block] = []
    for i, (s, e) in enumerate(intervals):
        gap_before = s - intervals[i - 1][1] if i > 0 else float("inf")
        gap_after = intervals[i + 1][0] - e if i + 1 < len(intervals) else float("inf")
        blocks.append(
            Block(
                block_id=i,
                start_sec=float(s),
                end_sec=float(e),
                gap_before_sec=float(gap_before),
                gap_after_sec=float(gap_after),
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Speech coverage per signal frame
# ---------------------------------------------------------------------------


def _cumulative_speech(t: np.ndarray, iv_starts: np.ndarray, iv_ends: np.ndarray) -> np.ndarray:
    """Total speech seconds in ``[0, t]`` for each query time ``t``.

    Vectorized via a piecewise-linear cumulative function over sorted
    non-overlapping intervals: O((N + V) log V) thanks to ``searchsorted``.
    """
    if iv_starts.size == 0:
        return np.zeros_like(t, dtype=np.float64)
    durations = iv_ends - iv_starts
    cum_before = np.concatenate([[0.0], np.cumsum(durations)])  # len V + 1
    idx = np.searchsorted(iv_starts, t, side="right") - 1  # index of last started interval, or -1
    out = np.zeros_like(t, dtype=np.float64)
    valid = idx >= 0
    if not np.any(valid):
        return out
    v_idx = idx[valid]
    v_t = t[valid]
    inside = v_t < iv_ends[v_idx]
    out[valid] = cum_before[v_idx] + np.where(
        inside,
        v_t - iv_starts[v_idx],
        iv_ends[v_idx] - iv_starts[v_idx],
    )
    return out


def compute_coverage(signal: Signal, vad: Vad) -> np.ndarray:
    """Fraction of each frame's receptive window that overlaps speech, in ``[0, 1]``."""
    starts, ends = signal.frame_audio_extent()
    iv_s = vad.starts
    iv_e = vad.ends
    speech = _cumulative_speech(ends, iv_s, iv_e) - _cumulative_speech(starts, iv_s, iv_e)
    return np.clip(speech / signal.window_sec, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Frame → block assignment
# ---------------------------------------------------------------------------


def assign_blocks(signal: Signal, blocks: list[Block]) -> np.ndarray:
    """Map each frame to a block id, or ``-1`` if its center falls outside any block."""
    centers = signal.frame_centers()
    if not blocks:
        return np.full(signal.n_frames, -1, dtype=np.int32)
    starts = np.array([b.start_sec for b in blocks])
    ends = np.array([b.end_sec for b in blocks])
    idx = np.searchsorted(starts, centers, side="right") - 1
    out = np.full(signal.n_frames, -1, dtype=np.int32)
    valid = (idx >= 0) & (idx < len(blocks))
    inside = valid & (centers <= ends[np.clip(idx, 0, len(blocks) - 1)])
    out[inside] = idx[inside]
    return out


# ---------------------------------------------------------------------------
# Boundary flags
# ---------------------------------------------------------------------------


def boundary_flags(
    signal: Signal, blocks: list[Block], frame_block: np.ndarray, margin_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(near_start, near_end)`` boolean arrays per frame.

    Only frames that are inside a block are flagged; silence frames stay False.
    """
    centers = signal.frame_centers()
    near_start = np.zeros(signal.n_frames, dtype=bool)
    near_end = np.zeros(signal.n_frames, dtype=bool)
    for b in blocks:
        idx = np.where(frame_block == b.block_id)[0]
        if idx.size == 0:
            continue
        ts = centers[idx]
        near_start[idx] = ts - b.start_sec <= margin_sec
        near_end[idx] = b.end_sec - ts <= margin_sec
    return near_start, near_end


# ---------------------------------------------------------------------------
# Smoothing within blocks only
# ---------------------------------------------------------------------------


def smooth_within_blocks(
    values: np.ndarray, frame_block: np.ndarray, hop_sec: float, median_sec: float
) -> np.ndarray:
    """Apply a per-block median filter; silence frames are passed through unchanged."""
    if median_sec <= 0:
        return values.astype(np.float64, copy=True)
    width = max(1, int(round(median_sec / hop_sec)))
    if width % 2 == 0:
        width += 1
    out = values.astype(np.float64, copy=True)
    for bid in np.unique(frame_block):
        if bid < 0:
            continue
        idx = np.where(frame_block == bid)[0]
        if idx.size >= 3:
            w = min(width, idx.size if idx.size % 2 == 1 else idx.size - 1)
            out[idx] = median_filter(values[idx], size=w, mode="nearest")
    return out
