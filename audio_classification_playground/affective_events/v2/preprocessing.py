"""Pre-processing for v2 affective event detection."""
from __future__ import annotations

import numpy as np

from .config import Config
from .types import Block, Vad


_MAD_TO_SIGMA = 1.4826


def build_blocks(vad: Vad, config: Config) -> list[Block]:
    merged = vad.merged(
        max_gap_sec=config.vad_merge_gap_sec,
        min_duration_sec=config.min_speech_block_sec,
    )
    return [
        Block(block_id=i, start_sec=float(s), end_sec=float(e))
        for i, (s, e) in enumerate(merged.intervals)
    ]


def assign_frame_blocks(
    n_frames: int, hop_sec: float, window_sec: float, blocks: list[Block]
) -> np.ndarray:
    """Assign each fully interior frame to a block, or -1 otherwise."""
    starts = np.arange(n_frames, dtype=np.float64) * hop_sec
    ends = starts + window_sec
    out = np.full(n_frames, -1, dtype=np.int32)
    for block in blocks:
        mask = (starts >= block.start_sec) & (ends <= block.end_sec)
        out[mask] = block.block_id
    return out


def global_stats(values: np.ndarray, interior: np.ndarray) -> tuple[float, float]:
    """Return robust (median, scaled MAD) over interior frames."""
    sample = values[interior]
    if sample.size == 0:
        sample = values[np.isfinite(values)]
    if sample.size == 0:
        return 0.0, 1e-9
    med = float(np.median(sample))
    mad = float(_MAD_TO_SIGMA * np.median(np.abs(sample - med)))
    return med, max(mad, 1e-9)
