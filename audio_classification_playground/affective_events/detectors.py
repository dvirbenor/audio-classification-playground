"""Per-signal candidate detectors.

All detectors share the same context object (:class:`SignalContext`) so each
function stays focused on its specific evidence. Every detector emits leaf
:class:`Event` instances; per-signal episode aggregation and cross-signal
joint merging happen in :mod:`fusion`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Iterable

import numpy as np
from scipy.stats import theilslopes

from .baseline import Baseline, RobustStats, local_baseline, local_scale
from .config import Config
from .scoring import (
    boundary_score,
    combine_confidence,
    context_score,
    coverage_score,
    direction_of,
    duration_score,
    strength_score,
)
from .types import Block, Event, Signal


# ---------------------------------------------------------------------------
# Per-signal context bundle (built once in pipeline, passed to each detector)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalContext:
    signal: Signal
    blocks: list[Block]
    coverage: np.ndarray
    usable: np.ndarray
    core: np.ndarray
    frame_block: np.ndarray
    near_start: np.ndarray
    near_end: np.ndarray
    smoothed: np.ndarray
    centers: np.ndarray
    global_stats: RobustStats
    config: Config
    id_counter: count

    def next_id(self, prefix: str) -> str:
        return f"{prefix}_{self.signal.name}_{next(self.id_counter):05d}"

    def block_frames(self, block_id: int, only_core: bool = True) -> np.ndarray:
        idx = np.where(self.frame_block == block_id)[0]
        if only_core and idx.size:
            idx = idx[self.core[idx]]
        return idx


# ---------------------------------------------------------------------------
# Shared event construction helper
# ---------------------------------------------------------------------------


def _make_event(
    ctx: SignalContext,
    *,
    event_id: str,
    event_type: str,
    start_sec: float,
    end_sec: float,
    block_ids: tuple[int, ...],
    delta: float,
    delta_z: float,
    baseline: Baseline,
    frames_for_quality: np.ndarray,
    extra_components: dict[str, float] | None = None,
    shape_score_value: float = 1.0,
    extra: dict | None = None,
) -> Event:
    cov = float(ctx.coverage[frames_for_quality].mean()) if frames_for_quality.size else 0.0
    near_s = bool(ctx.near_start[frames_for_quality].any()) if frames_for_quality.size else False
    near_e = bool(ctx.near_end[frames_for_quality].any()) if frames_for_quality.size else False

    components = {
        "strength": strength_score(abs(delta_z)),
        "duration": duration_score(end_sec - start_sec, ctx.signal.window_sec),
        "coverage": coverage_score(cov),
        "context": context_score(baseline.context_speech_sec),
        "boundary": boundary_score(near_s, near_e),
        "shape": shape_score_value,
    }
    if extra_components:
        components.update(extra_components)

    confidence = combine_confidence(components, ctx.config)
    pad = ctx.config.review_pad_sec
    return Event(
        event_id=event_id,
        signal_name=ctx.signal.name,
        event_type=event_type,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
        block_ids=block_ids,
        delta=float(delta),
        delta_z=float(delta_z),
        direction=direction_of(delta_z),
        baseline_value=baseline.value,
        baseline_context_speech_sec=baseline.context_speech_sec,
        baseline_source=baseline.source,
        mean_speech_coverage=cov,
        near_block_start=near_s,
        near_block_end=near_e,
        strength=float(abs(delta_z)),
        confidence=confidence,
        confidence_components=components,
        review_audio_start_sec=max(0.0, start_sec - pad),
        review_audio_end_sec=end_sec + pad,
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# Detector 1: speech-block deviation
# ---------------------------------------------------------------------------


def detect_block_deviations(ctx: SignalContext) -> list[Event]:
    out: list[Event] = []
    for b in ctx.blocks:
        idx = ctx.block_frames(b.block_id, only_core=True)
        core_dur = idx.size * ctx.signal.hop_sec
        if core_dur < ctx.config.min_block_for_deviation_sec:
            continue
        block_value = float(np.median(ctx.smoothed[idx]))
        baseline = local_baseline(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            ctx.signal.hop_sec, ctx.global_stats, ctx.config,
        )
        scale = local_scale(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            ctx.global_stats, ctx.config,
        )
        delta = block_value - baseline.value
        delta_z = delta / scale
        if abs(delta_z) < ctx.config.block_deviation_z_threshold:
            continue

        # Sign consistency across the block
        residuals = ctx.smoothed[idx] - baseline.value
        sign_match = float((np.sign(residuals) == np.sign(delta)).mean())

        out.append(
            _make_event(
                ctx,
                event_id=ctx.next_id("blkdev"),
                event_type="block_deviation",
                start_sec=b.start_sec,
                end_sec=b.end_sec,
                block_ids=(b.block_id,),
                delta=delta,
                delta_z=delta_z,
                baseline=baseline,
                frames_for_quality=idx,
                shape_score_value=sign_match,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Detector 2: within-block excursion (hysteresis on residual energy)
# ---------------------------------------------------------------------------


def _hysteresis_intervals(
    z_abs: np.ndarray, enter: float, exit_: float
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    in_ev = False
    start = 0
    for i, v in enumerate(z_abs):
        if not in_ev and v >= enter:
            start = i
            in_ev = True
        elif in_ev and v < exit_:
            out.append((start, i - 1))
            in_ev = False
    if in_ev:
        out.append((start, len(z_abs) - 1))
    return out


def _merge_close_intervals(
    intervals: list[tuple[int, int]], min_gap_frames: int
) -> list[tuple[int, int]]:
    if not intervals:
        return intervals
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= min_gap_frames:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def detect_within_block_excursions(ctx: SignalContext) -> list[Event]:
    out: list[Event] = []
    hop = ctx.signal.hop_sec
    min_dur_frames = max(1, int(round(ctx.config.excursion_min_duration_sec / hop)))
    merge_gap_frames = max(0, int(round(ctx.config.excursion_merge_gap_sec / hop)))

    for b in ctx.blocks:
        if b.duration_sec < ctx.config.min_block_for_excursion_sec:
            continue
        block_idx = ctx.block_frames(b.block_id, only_core=False)
        if block_idx.size == 0:
            continue

        baseline = local_baseline(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            hop, ctx.global_stats, ctx.config,
        )
        scale = local_scale(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            ctx.global_stats, ctx.config,
        )

        residuals = ctx.smoothed[block_idx] - baseline.value
        z_abs = np.abs(residuals / scale)
        # Don't trigger on frames with low coverage
        z_abs = np.where(ctx.usable[block_idx], z_abs, 0.0)

        intervals = _hysteresis_intervals(
            z_abs, ctx.config.excursion_enter_z, ctx.config.excursion_exit_z
        )
        intervals = _merge_close_intervals(intervals, merge_gap_frames)
        intervals = [(s, e) for s, e in intervals if (e - s + 1) >= min_dur_frames]

        for s_loc, e_loc in intervals:
            ev_idx = block_idx[s_loc : e_loc + 1]
            ev_start = float(ctx.centers[ev_idx[0]])
            ev_end = float(ctx.centers[ev_idx[-1]])
            seg = ctx.smoothed[ev_idx] - baseline.value
            # Direction = sign of the largest |residual|
            peak = seg[np.argmax(np.abs(seg))]
            delta = float(peak)
            delta_z = delta / scale

            block_dur = b.duration_sec
            covers_block = (ev_end - ev_start) / max(block_dur, 1e-9) >= ctx.config.excursion_to_block_ratio
            event_type = "block_deviation" if covers_block else "within_block_excursion"
            id_prefix = "blkdev" if covers_block else "excurs"

            # Shape: how peaked is it? Peak / mean ratio of |residuals| in [0,1].
            mean_abs = float(np.mean(np.abs(seg)))
            shape = float(min(1.0, mean_abs / max(abs(peak), 1e-9)))

            out.append(
                _make_event(
                    ctx,
                    event_id=ctx.next_id(id_prefix),
                    event_type=event_type,
                    start_sec=b.start_sec if covers_block else ev_start,
                    end_sec=b.end_sec if covers_block else ev_end,
                    block_ids=(b.block_id,),
                    delta=delta,
                    delta_z=delta_z,
                    baseline=baseline,
                    frames_for_quality=ev_idx,
                    shape_score_value=shape,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Detector 3: within-block regime shift (single split-point scan)
# ---------------------------------------------------------------------------


def detect_within_block_regime_shifts(ctx: SignalContext) -> list[Event]:
    out: list[Event] = []
    hop = ctx.signal.hop_sec
    min_pre = ctx.config.regime_shift_min_pre_sec
    min_post = ctx.config.regime_shift_min_post_sec
    edge = ctx.config.regime_shift_edge_margin_sec

    for b in ctx.blocks:
        if b.duration_sec < ctx.config.min_block_for_regime_shift_sec:
            continue
        idx = ctx.block_frames(b.block_id, only_core=True)
        if idx.size < 6:
            continue
        ts = ctx.centers[idx]
        vals = ctx.smoothed[idx]

        # Candidate split positions: indices in `idx` honoring edge margins.
        scale = local_scale(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            ctx.global_stats, ctx.config,
        )
        baseline = local_baseline(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            hop, ctx.global_stats, ctx.config,
        )

        best = None  # (abs_z, split_pos, pre_med, post_med)
        for k in range(1, idx.size):
            t_split = ts[k]
            if t_split - b.start_sec < edge or b.end_sec - t_split < edge:
                continue
            if t_split - ts[0] < min_pre or ts[-1] - t_split < min_post:
                continue
            pre_med = float(np.median(vals[:k]))
            post_med = float(np.median(vals[k:]))
            delta = post_med - pre_med
            abs_z = abs(delta) / scale
            if best is None or abs_z > best[0]:
                best = (abs_z, k, pre_med, post_med)

        if best is None:
            continue
        abs_z, k, pre_med, post_med = best
        if abs_z < ctx.config.regime_shift_min_effect_z:
            continue

        delta = post_med - pre_med
        delta_z = delta / scale
        ev_start = float(ts[0])
        ev_end = float(ts[-1])

        # Shape: stability of pre/post (lower internal MAD relative to global => higher score)
        pre_var = float(np.median(np.abs(vals[:k] - pre_med))) if k > 0 else 0.0
        post_var = float(np.median(np.abs(vals[k:] - post_med))) if k < idx.size else 0.0
        instab = (pre_var + post_var) / max(abs(delta), 1e-9)
        shape = float(max(0.0, 1.0 - instab))

        out.append(
            _make_event(
                ctx,
                event_id=ctx.next_id("regime"),
                event_type="within_block_regime_shift",
                start_sec=ev_start,
                end_sec=ev_end,
                block_ids=(b.block_id,),
                delta=delta,
                delta_z=delta_z,
                baseline=baseline,
                frames_for_quality=idx,
                shape_score_value=shape,
                extra={"split_sec": float(ts[k]), "pre_median": pre_med, "post_median": post_med},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Detector 4: within-block ramp (robust slope + monotonicity)
# ---------------------------------------------------------------------------


def detect_within_block_ramps(ctx: SignalContext) -> list[Event]:
    out: list[Event] = []
    hop = ctx.signal.hop_sec

    for b in ctx.blocks:
        if b.duration_sec < ctx.config.min_block_for_ramp_sec:
            continue
        idx = ctx.block_frames(b.block_id, only_core=True)
        if idx.size < 8:
            continue
        ts = ctx.centers[idx]
        vals = ctx.smoothed[idx]

        scale = local_scale(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            ctx.global_stats, ctx.config,
        )
        baseline = local_baseline(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            hop, ctx.global_stats, ctx.config,
        )

        # Theil-Sen slope on (time, value) over the whole block.
        slope, intercept, *_ = theilslopes(vals, ts)
        duration = float(ts[-1] - ts[0])
        if duration < ctx.config.ramp_min_duration_sec:
            continue
        total_change = slope * duration
        total_change_z = total_change / scale
        if abs(total_change_z) < ctx.config.ramp_min_total_change_z:
            continue

        # Monotonicity: fraction of consecutive deltas with the same sign as total_change.
        diffs = np.diff(vals)
        sign = np.sign(total_change) if total_change != 0 else 0
        if sign == 0:
            continue
        mono = float((np.sign(diffs) == sign).mean())
        if mono < ctx.config.ramp_min_monotonicity:
            continue

        out.append(
            _make_event(
                ctx,
                event_id=ctx.next_id("ramp"),
                event_type="within_block_ramp",
                start_sec=float(ts[0]),
                end_sec=float(ts[-1]),
                block_ids=(b.block_id,),
                delta=float(total_change),
                delta_z=float(total_change_z),
                baseline=baseline,
                frames_for_quality=idx,
                shape_score_value=mono,
                extra={"slope_per_sec": float(slope), "monotonicity": mono},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Detector 5: short-gap block transition (adjacent-block contrast)
# ---------------------------------------------------------------------------


def detect_short_gap_transitions(ctx: SignalContext) -> list[Event]:
    out: list[Event] = []
    hop = ctx.signal.hop_sec
    blocks = ctx.blocks
    if len(blocks) < 2:
        return out

    for i in range(len(blocks) - 1):
        b1, b2 = blocks[i], blocks[i + 1]
        gap = b2.start_sec - b1.end_sec
        if gap > ctx.config.short_gap_max_sec:
            continue
        idx1 = ctx.block_frames(b1.block_id, only_core=True)
        idx2 = ctx.block_frames(b2.block_id, only_core=True)
        if idx1.size == 0 or idx2.size == 0:
            continue
        if (idx1.size * hop) < ctx.config.min_block_for_deviation_sec / 2:
            continue
        if (idx2.size * hop) < ctx.config.min_block_for_deviation_sec / 2:
            continue

        m1 = float(np.median(ctx.smoothed[idx1]))
        m2 = float(np.median(ctx.smoothed[idx2]))
        # Baseline / scale anchored at the transition point (midpoint of gap)
        t_mid = 0.5 * (b1.end_sec + b2.start_sec)
        baseline = local_baseline(
            t_mid, t_mid, ctx.centers, ctx.smoothed, ctx.core,
            hop, ctx.global_stats, ctx.config,
        )
        scale = local_scale(
            t_mid, t_mid, ctx.centers, ctx.smoothed, ctx.core,
            ctx.global_stats, ctx.config,
        )
        delta = m2 - m1
        delta_z = delta / scale
        if abs(delta_z) < ctx.config.short_gap_min_delta_z:
            continue

        frames = np.concatenate([idx1, idx2])
        # Shape: gap continuity (smaller gap => higher score)
        shape = float(max(0.0, 1.0 - gap / max(ctx.config.short_gap_max_sec, 1e-9)))

        out.append(
            _make_event(
                ctx,
                event_id=ctx.next_id("trans"),
                event_type="short_gap_block_transition",
                start_sec=b1.start_sec,
                end_sec=b2.end_sec,
                block_ids=(b1.block_id, b2.block_id),
                delta=delta,
                delta_z=delta_z,
                baseline=baseline,
                frames_for_quality=frames,
                shape_score_value=shape,
                extra={"gap_sec": float(gap), "pre_median": m1, "post_median": m2},
            )
        )
    return out


# ---------------------------------------------------------------------------
# Top-level: run all detectors for one signal
# ---------------------------------------------------------------------------


def run_all_detectors(ctx: SignalContext) -> list[Event]:
    events: list[Event] = []
    events.extend(detect_block_deviations(ctx))
    events.extend(detect_within_block_excursions(ctx))
    events.extend(detect_within_block_regime_shifts(ctx))
    events.extend(detect_within_block_ramps(ctx))
    events.extend(detect_short_gap_transitions(ctx))
    return events
