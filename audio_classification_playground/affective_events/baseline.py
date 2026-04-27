"""Robust normalization and local-context baseline estimation.

The local baseline is computed from speech-core frames inside a context
window centered on the candidate, with the candidate's own neighborhood
excluded. This is the central guard against baseline-chasing and scale
inflation that fragile rolling-MAD pipelines suffer from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import median_abs_deviation

from .config import Config


_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class RobustStats:
    center: float
    scale: float


@dataclass(frozen=True)
class Baseline:
    """Local baseline estimate around a candidate region."""

    value: float
    context_speech_sec: float
    source: str  # "local" or "global"


def compute_global_stats(values: np.ndarray, mask: np.ndarray) -> RobustStats:
    """Median / scaled-MAD over masked frames.

    Falls back to all frames if the mask is empty (degenerate VAD case).
    """
    sample = values[mask] if mask.any() else values
    center = float(np.median(sample))
    mad = float(median_abs_deviation(sample, scale=1.0))
    scale = max(_MAD_TO_SIGMA * mad, 1e-9)
    return RobustStats(center=center, scale=scale)


def local_baseline(
    candidate_start_sec: float,
    candidate_end_sec: float,
    frame_centers: np.ndarray,
    values: np.ndarray,
    core_mask: np.ndarray,
    hop_sec: float,
    global_stats: RobustStats,
    config: Config,
) -> Baseline:
    """Robust median over speech-core frames near the candidate, excluding it.

    Returns the global median (with ``source="global"``) when the local
    context is too sparse, so the caller never has to handle ``None``.
    """
    ctx_lo = candidate_start_sec - config.local_context_radius_sec
    ctx_hi = candidate_end_sec + config.local_context_radius_sec
    ex_lo = candidate_start_sec - config.exclude_candidate_radius_sec
    ex_hi = candidate_end_sec + config.exclude_candidate_radius_sec

    in_context = (frame_centers >= ctx_lo) & (frame_centers <= ctx_hi)
    not_excluded = (frame_centers < ex_lo) | (frame_centers > ex_hi)
    mask = core_mask & in_context & not_excluded
    n = int(mask.sum())
    context_speech_sec = n * hop_sec

    if context_speech_sec < config.min_context_speech_sec:
        return Baseline(value=global_stats.center, context_speech_sec=context_speech_sec, source="global")
    return Baseline(value=float(np.median(values[mask])), context_speech_sec=context_speech_sec, source="local")


def local_scale(
    candidate_start_sec: float,
    candidate_end_sec: float,
    frame_centers: np.ndarray,
    values: np.ndarray,
    core_mask: np.ndarray,
    global_stats: RobustStats,
    config: Config,
) -> float:
    """Scaled local MAD with a floor relative to the global scale.

    The floor prevents tiny-variance pockets from inflating residual z-scores.
    """
    ctx_lo = candidate_start_sec - config.local_context_radius_sec
    ctx_hi = candidate_end_sec + config.local_context_radius_sec
    ex_lo = candidate_start_sec - config.exclude_candidate_radius_sec
    ex_hi = candidate_end_sec + config.exclude_candidate_radius_sec

    in_context = (frame_centers >= ctx_lo) & (frame_centers <= ctx_hi)
    not_excluded = (frame_centers < ex_lo) | (frame_centers > ex_hi)
    mask = core_mask & in_context & not_excluded

    floor = config.scale_floor_frac * global_stats.scale
    if mask.sum() < 8:
        return global_stats.scale
    mad = float(median_abs_deviation(values[mask], scale=1.0))
    return max(_MAD_TO_SIGMA * mad, floor)
