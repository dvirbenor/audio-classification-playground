"""VAD-gated inference: skip GPU compute on non-speech windows.

The pipeline runs affect/disfluency/emotion on the full timeline, but the
producers only ever read frames inside (merged/closed) VAD speech regions
(affect uses *containment* of a frame's window in a merged block; disfluency
and emotion use *overlap* with their own merge/close gaps). So if we compute
only the windows that overlap speech — using a gate that is a **superset** of
every producer's consumed frame set — and fill the rest with a task-specific
"no signal" sentinel, the emitted events are unchanged while the GPU skips the
silence.

Superset rule: keep any window that *overlaps* a VAD interval after bridging
gaps by ``bridge_sec`` (default 1.5 s ≥ the largest producer gap, emotion's
``support_close_gap_sec`` = 1.0 s). Overlap — not containment — so windows that
straddle a speech boundary (which the producers read) are kept.

Gating is recorded in ``inference_config`` (``VadGating.config_extra``) so the
artifact hash reflects it: gated runs are a new artifact lineage, leaving
existing full-timeline artifacts untouched.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_BRIDGE_SEC = 1.5  # >= emotion support_close_gap_sec (1.0) + margin
GATING_POLICY = "overlap_v1"

GATABLE_TASKS = ("affect", "disfluency", "emotion")
# Default-gated set is the empirically event-identical one. disfluency is
# EXCLUDED by default: its candidate-region detection (`_support_regions`) runs
# over the full timeline *before* the speech-support filter, so filled
# non-speech frames shift region boundaries (~1 borderline event / archive).
# Gating disfluency identically needs a speech-scoped producer change first
# (see VAD_GATING_IMPLEMENTATION_PLAN.md §5.5); opt in via ``tasks`` once done.
DEFAULT_GATED_TASKS = ("affect", "emotion")

Intervals = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class VadGating:
    """Settings for VAD-gated inference."""

    enabled: bool = False
    bridge_sec: float = DEFAULT_BRIDGE_SEC
    policy: str = GATING_POLICY
    tasks: tuple[str, ...] = DEFAULT_GATED_TASKS

    def __post_init__(self) -> None:
        if self.bridge_sec < 0.0:
            raise ValueError("bridge_sec must be non-negative")
        if self.policy != GATING_POLICY:
            raise ValueError(f"unknown gating policy {self.policy!r}")
        unknown = set(self.tasks) - set(GATABLE_TASKS)
        if unknown:
            raise ValueError(f"unknown gated tasks: {sorted(unknown)}")

    @property
    def active(self) -> bool:
        return bool(self.enabled)

    def gates(self, task: str) -> bool:
        """Whether *task*'s inference should be VAD-gated."""
        return bool(self.enabled) and task in self.tasks

    def config_extra(self) -> dict:
        """Descriptor merged into ``inference_config`` (only when enabled)."""
        return {
            "vad_gating": {
                "policy": self.policy,
                "bridge_sec": float(self.bridge_sec),
            }
        }


def intervals_from_array(arr) -> list[tuple[float, float]]:
    """Convert a stored ``intervals_sec`` [n, 2] array to a list of tuples."""
    if arr is None:
        return []
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return []
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"intervals array must be [n, 2], got {a.shape}")
    return [(float(s), float(e)) for s, e in a]


def bridge_intervals(intervals: Intervals, bridge_sec: float) -> list[tuple[float, float]]:
    """Sort and merge intervals whose gap is ``<= bridge_sec``.

    Positive-duration intervals only; the result is sorted and disjoint with
    gaps strictly greater than ``bridge_sec``.
    """
    cleaned = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not cleaned:
        return []
    out: list[list[float]] = [list(cleaned[0])]
    for s, e in cleaned[1:]:
        if s - out[-1][1] <= bridge_sec:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def speech_window_mask(
    n_frames: int,
    hop_sec: float,
    window_sec: float,
    vad_intervals: Intervals,
    *,
    bridge_sec: float = DEFAULT_BRIDGE_SEC,
) -> np.ndarray:
    """Boolean keep-mask over the sliding-window grid.

    Window ``i`` spans ``[i*hop, i*hop + window]`` and is kept iff it *overlaps*
    a bridged VAD interval, i.e. ``i*hop < e`` and ``i*hop + window > s``.
    """
    n = int(n_frames)
    mask = np.zeros(max(n, 0), dtype=bool)
    if n <= 0:
        return mask
    if hop_sec <= 0.0 or window_sec <= 0.0:
        raise ValueError("hop_sec and window_sec must be positive")
    for s, e in bridge_intervals(vad_intervals, bridge_sec):
        # i*hop + window > s  ->  i > (s - window) / hop
        i_lo = int(np.floor((s - window_sec) / hop_sec)) + 1
        # i*hop < e           ->  i < e / hop
        i_hi = int(np.ceil(e / hop_sec)) - 1
        i_lo = max(0, i_lo)
        i_hi = min(n - 1, i_hi)
        if i_hi >= i_lo:
            mask[i_lo : i_hi + 1] = True
    return mask


def scatter_fill(
    kept_values: np.ndarray,
    kept_idx: np.ndarray,
    n_frames: int,
    *,
    fill: float,
) -> np.ndarray:
    """Scatter ``kept_values`` (length K, with any trailing dims) into a
    full-length ``[n_frames, *trailing]`` array, filling the rest with ``fill``.
    """
    kv = np.asarray(kept_values)
    shape = (int(n_frames),) + kv.shape[1:]
    out = np.full(shape, fill, dtype=kv.dtype)
    if kept_idx.size:
        out[kept_idx] = kv
    return out


def gate_window_arrays(
    predict_fn: Callable[[np.ndarray], dict[str, np.ndarray]],
    windows: np.ndarray,
    *,
    hop_sec: float,
    window_sec: float,
    vad_intervals: Intervals | None,
    gating: VadGating | None,
    fill: float = 0.0,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Run a windows->dict predictor only on speech windows, scatter+fill.

    Returns ``(arrays, mask)``. When gating is inactive or no intervals are
    available, runs the full timeline and returns ``mask=None`` (no behaviour
    change). Otherwise every output array is full-length ``len(windows)`` with
    non-speech rows set to ``fill`` — preserving downstream frame alignment.
    """
    if gating is None or not gating.active or vad_intervals is None:
        return predict_fn(windows), None
    n_frames = len(windows)
    mask = speech_window_mask(
        n_frames, hop_sec, window_sec, vad_intervals, bridge_sec=gating.bridge_sec
    )
    kept_idx = np.nonzero(mask)[0]
    sub = predict_fn(windows[kept_idx])
    arrays = {
        name: scatter_fill(values, kept_idx, n_frames, fill=fill)
        for name, values in sub.items()
    }
    return arrays, mask
