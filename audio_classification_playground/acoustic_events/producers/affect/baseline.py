"""Block-aware baseline and scale estimation."""
from __future__ import annotations

import numpy as np

from .config import Config
from .types import Block


_MAD_TO_SIGMA = 1.4826


def block_aware_baseline_scale(
    values: np.ndarray,
    frame_block: np.ndarray,
    blocks: list[Block],
    hop_sec: float,
    window_sec: float,
    config: Config,
    *,
    global_median: float,
    global_mad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame baseline and scale; NaN outside interior frames."""
    n_frames = len(values)
    centers = np.arange(n_frames, dtype=np.float64) * hop_sec + window_sec / 2
    baseline = np.full(n_frames, np.nan, dtype=np.float64)
    scale = np.full(n_frames, np.nan, dtype=np.float64)

    block_indices = {
        block.block_id: np.where(frame_block == block.block_id)[0]
        for block in blocks
    }
    floor = max(config.scale_floor_frac * global_mad, 1e-9)

    for block in blocks:
        own = block_indices.get(block.block_id)
        if own is None or own.size == 0:
            continue

        if block.duration_sec <= config.radius_sec:
            ctx = _context_for_time(
                block.center_sec,
                frame_block,
                centers,
                block.block_id,
                config.radius_sec,
            )
            med, sc = _baseline_scale_from_context(
                values, ctx, hop_sec, config.min_context_sec,
                global_median, global_mad, floor,
            )
            baseline[own] = med
            scale[own] = sc
            continue

        for idx in own:
            ctx = _context_for_time(
                centers[idx],
                frame_block,
                centers,
                block.block_id,
                config.radius_sec,
            )
            baseline[idx], scale[idx] = _baseline_scale_from_context(
                values, ctx, hop_sec, config.min_context_sec,
                global_median, global_mad, floor,
            )

    return baseline, scale


def _context_for_time(
    t_sec: float,
    frame_block: np.ndarray,
    centers: np.ndarray,
    own_block_id: int,
    radius_sec: float,
) -> np.ndarray:
    return np.where(
        (frame_block >= 0)
        & (frame_block != own_block_id)
        & (np.abs(centers - t_sec) <= radius_sec)
    )[0]


def _baseline_scale_from_context(
    values: np.ndarray,
    ctx: np.ndarray,
    hop_sec: float,
    min_context_sec: float,
    global_median: float,
    global_mad: float,
    scale_floor: float,
) -> tuple[float, float]:
    if ctx.size * hop_sec < min_context_sec:
        return float(global_median), float(max(global_mad, 1e-9))
    ctx_vals = values[ctx]
    med = float(np.median(ctx_vals))
    mad = float(_MAD_TO_SIGMA * np.median(np.abs(ctx_vals - med)))
    return med, max(mad, scale_floor)
