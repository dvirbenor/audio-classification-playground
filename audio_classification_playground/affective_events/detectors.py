"""Per-signal candidate detectors.

All detectors share the same context object (:class:`SignalContext`) so each
function stays focused on its specific evidence. Every detector emits leaf
:class:`Event` instances; per-signal episode aggregation and cross-signal
joint merging happen in :mod:`fusion`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import count

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

    def interior_frames(self, block_id: int) -> np.ndarray:
        """Core frames whose full receptive field falls within the block."""
        idx = self.block_frames(block_id, only_core=True)
        if idx.size == 0:
            return idx
        half_win = self.signal.window_sec / 2
        centers = self.centers[idx]
        b = next(b for b in self.blocks if b.block_id == block_id)
        mask = (centers - half_win >= b.start_sec) & (centers + half_win <= b.end_sec)
        return idx[mask]


# ---------------------------------------------------------------------------
# Shadow model-selection diagnostics (Phase 1: compute-only, no gating)
# ---------------------------------------------------------------------------


def _fit_block_models(
    ts: np.ndarray,
    vals: np.ndarray,
    block_start: float,
    block_end: float,
    edge_margin: float,
    min_pre: float,
    min_post: float,
) -> dict:
    """Fit M0/M1/M2 to a block's interior core frames and return RSS values.

    Returns a dict with keys ``rss_m0``, ``rss_m1``, ``rss_m2``,
    ``m1_split_k``, ``m1_pre_median``, ``m1_post_median``,
    ``m2_slope``, ``m2_intercept``.
    """
    n = len(vals)
    m0_med = float(np.median(vals))
    rss_m0 = float(np.sum(np.abs(vals - m0_med)))

    best_rss_m1 = float("inf")
    best_k = -1
    best_pre_med = m0_med
    best_post_med = m0_med
    for k in range(1, n):
        t_split = ts[k]
        if t_split - block_start < edge_margin or block_end - t_split < edge_margin:
            continue
        if t_split - ts[0] < min_pre or ts[-1] - t_split < min_post:
            continue
        pre_med = float(np.median(vals[:k]))
        post_med = float(np.median(vals[k:]))
        rss = float(
            np.sum(np.abs(vals[:k] - pre_med))
            + np.sum(np.abs(vals[k:] - post_med))
        )
        if rss < best_rss_m1:
            best_rss_m1 = rss
            best_k = k
            best_pre_med = pre_med
            best_post_med = post_med
    if best_k < 0:
        best_rss_m1 = rss_m0

    slope = intercept = 0.0
    if n >= 2:
        slope, intercept, *_ = theilslopes(vals, ts)
        fitted = intercept + slope * ts
        rss_m2 = float(np.sum(np.abs(vals - fitted)))
    else:
        rss_m2 = rss_m0

    return {
        "rss_m0": rss_m0,
        "rss_m1": best_rss_m1,
        "rss_m2": rss_m2,
        "m1_split_k": best_k,
        "m1_pre_median": best_pre_med,
        "m1_post_median": best_post_med,
        "m2_slope": slope,
        "m2_intercept": intercept,
    }


def _model_selection(rss_m0: float, rss_m1: float, rss_m2: float,
                     scale: float, n: int,
                     c: float = 1.0) -> tuple[str, float]:
    """Return ``(winner, margin)`` under BIC-analog model selection.

    Under a Laplace residual assumption, ``c=1.0`` is the principled default.
    ``margin`` is the fractional score improvement of the winner over M0.
    """
    lam = c * scale * math.log(max(n, 2)) / 2
    score_m0 = rss_m0 + lam * 1
    score_m1 = rss_m1 + lam * 3
    score_m2 = rss_m2 + lam * 2
    scores = {"M0": score_m0, "M1": score_m1, "M2": score_m2}
    winner = min(scores, key=scores.get)
    margin = (score_m0 - scores[winner]) / max(score_m0, 1e-9)
    return winner, margin


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
# Detector: within-block excursion (hysteresis on residual energy)
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


def detect_within_block_excursions(
    ctx: SignalContext,
    block_residuals: dict[int, np.ndarray] | None = None,
) -> list[Event]:
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

        if block_residuals is not None and b.block_id in block_residuals:
            residuals = block_residuals[b.block_id]
        else:
            residuals = ctx.smoothed[block_idx] - baseline.value

        z_abs = np.abs(residuals / scale)
        z_abs = np.where(ctx.usable[block_idx], z_abs, 0.0)

        intervals = _hysteresis_intervals(
            z_abs, ctx.config.excursion_enter_z, ctx.config.excursion_exit_z
        )
        intervals = _merge_close_intervals(intervals, merge_gap_frames)
        intervals = [(s, e) for s, e in intervals if (e - s + 1) >= min_dur_frames]

        exit_z = ctx.config.excursion_exit_z

        for s_loc, e_loc in intervals:
            seg_det = residuals[s_loc : e_loc + 1]
            peak_pos_det = int(np.argmax(np.abs(seg_det)))
            peak_loc = s_loc + peak_pos_det

            left = peak_loc
            while left > 0 and z_abs[left - 1] >= exit_z:
                left -= 1
            right = peak_loc
            while right < len(z_abs) - 1 and z_abs[right + 1] >= exit_z:
                right += 1

            ev_idx = block_idx[left : right + 1]
            ev_start = float(ctx.centers[ev_idx[0]])
            ev_end = float(ctx.centers[ev_idx[-1]])
            seg = residuals[left : right + 1]
            peak_pos = int(np.argmax(np.abs(seg)))
            peak = seg[peak_pos]
            delta = float(peak)
            delta_z = delta / scale

            block_dur = b.duration_sec
            covers_block = (ev_end - ev_start) / max(block_dur, 1e-9) >= ctx.config.excursion_to_block_ratio
            event_type = "block_deviation" if covers_block else "within_block_excursion"
            id_prefix = "blkdev" if covers_block else "excurs"

            int_idx = ctx.interior_frames(b.block_id)
            if int_idx.size > 0:
                int_local = np.searchsorted(block_idx, int_idx)
                int_local = int_local[int_local < len(residuals)]
                block_mean_abs_z = float(np.mean(np.abs(residuals[int_local] / scale)))
            else:
                block_mean_abs_z = float(np.mean(np.abs(seg / scale)))
            peak_z_val = float(np.max(np.abs(seg / scale)))
            shape = max(0.0, 1.0 - block_mean_abs_z / max(peak_z_val, 1e-9))

            peak_time = float(ctx.centers[ev_idx[peak_pos]])
            half_peak = 0.5 * peak_z_val
            ev_z_abs_full = np.abs(seg / scale)
            fwhm_left = peak_pos
            while fwhm_left > 0 and ev_z_abs_full[fwhm_left - 1] >= half_peak:
                fwhm_left -= 1
            fwhm_right = peak_pos
            while fwhm_right < len(ev_z_abs_full) - 1 and ev_z_abs_full[fwhm_right + 1] >= half_peak:
                fwhm_right += 1

            extra: dict = {
                "peak_z": peak_z_val,
                "peak_time_sec": peak_time,
                "mean_abs_z": float(np.mean(np.abs(seg / scale))),
                "block_mean_abs_z": block_mean_abs_z,
                "n_core_frames": int(ev_idx.size),
                "fwhm_start_sec": float(ctx.centers[ev_idx[fwhm_left]]),
                "fwhm_end_sec": float(ctx.centers[ev_idx[fwhm_right]]),
            }

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
                    extra=extra,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Detector: short-gap block transition (adjacent-block contrast)
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
# Unified block characterization via model selection (Phase 5)
# ---------------------------------------------------------------------------


def characterize_block(
    ctx: SignalContext, b: Block,
) -> tuple[list[Event], np.ndarray]:
    """Fit M0/M1/M2 to one block, emit at most one block-level event, return residuals.

    Returns ``(events, residuals)`` where *residuals* are computed over **all**
    block frames (core=False) from the winning model's fit so that the
    excursion detector can operate on them.
    """
    hop = ctx.signal.hop_sec
    int_idx = ctx.interior_frames(b.block_id)
    core_idx = ctx.block_frames(b.block_id, only_core=True)
    all_idx = ctx.block_frames(b.block_id, only_core=False)

    if all_idx.size == 0:
        return [], np.array([])

    baseline = local_baseline(
        b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
        hop, ctx.global_stats, ctx.config,
    )
    scale = local_scale(
        b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
        ctx.global_stats, ctx.config,
    )

    fit_idx = int_idx if int_idx.size >= 6 else core_idx
    if fit_idx.size < 2:
        residuals = ctx.smoothed[all_idx] - baseline.value
        return [], residuals

    ts_fit = ctx.centers[fit_idx]
    vals_fit = ctx.smoothed[fit_idx]
    ts_all = ctx.centers[all_idx]
    vals_all = ctx.smoothed[all_idx]

    edge = ctx.config.regime_shift_edge_margin_sec
    min_pre = ctx.config.regime_shift_min_pre_sec
    min_post = ctx.config.regime_shift_min_post_sec

    models = _fit_block_models(
        ts_fit, vals_fit, b.start_sec, b.end_sec, edge, min_pre, min_post,
    )
    c = ctx.config.model_penalty_c
    winner, margin = _model_selection(
        models["rss_m0"], models["rss_m1"], models["rss_m2"],
        scale, fit_idx.size, c=c,
    )

    min_eff = ctx.config.min_absolute_effect
    events: list[Event] = []

    if winner == "M1" and b.duration_sec >= ctx.config.min_block_for_regime_shift_sec:
        k = models["m1_split_k"]
        pre_med = models["m1_pre_median"]
        post_med = models["m1_post_median"]
        delta = post_med - pre_med

        if min_eff and abs(delta) < min_eff.get(ctx.signal.name, 0.0):
            winner = "M0"
        else:
            delta_z = delta / scale
            split_t = float(ts_fit[k])
            tau = ctx.signal.window_sec / 2
            ev_start = max(split_t - tau, b.start_sec)
            ev_end = min(split_t + tau, b.end_sec)

            transition_mask = (ctx.centers[fit_idx] >= ev_start) & (ctx.centers[fit_idx] <= ev_end)
            transition_idx = fit_idx[transition_mask]
            if transition_idx.size == 0:
                transition_idx = fit_idx

            pre_var = float(np.median(np.abs(vals_fit[:k] - pre_med))) if k > 0 else 0.0
            post_var = float(np.median(np.abs(vals_fit[k:] - post_med))) if k < fit_idx.size else 0.0
            shape = max(0.0, 1.0 - models["rss_m1"] / max(models["rss_m0"], 1e-9))
            within_seg_z = abs(delta) / max(math.sqrt(pre_var**2 + post_var**2), 1e-9)

            extra: dict = {
                "split_sec": split_t,
                "pre_median": pre_med,
                "post_median": post_med,
                "pre_mad": pre_var,
                "post_mad": post_var,
                "within_segment_z": within_seg_z,
                "n_core_frames": int(core_idx.size),
                "n_interior_frames": int(int_idx.size),
                "rss_m0": models["rss_m0"],
                "rss_m1": models["rss_m1"],
                "rss_m2": models["rss_m2"],
                "model_selection_winner": winner,
                "model_selection_margin": margin,
            }

            events.append(
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
                    frames_for_quality=transition_idx,
                    shape_score_value=shape,
                    extra=extra,
                )
            )

            split_all = np.searchsorted(ts_all, split_t)
            residuals = vals_all.copy()
            residuals[:split_all] -= pre_med
            residuals[split_all:] -= post_med
            return events, residuals

    if winner == "M2" and b.duration_sec >= ctx.config.min_block_for_ramp_sec:
        slope = models["m2_slope"]
        intercept = models["m2_intercept"]
        duration = float(ts_fit[-1] - ts_fit[0])
        total_change = slope * duration

        if min_eff and abs(total_change) < min_eff.get(ctx.signal.name, 0.0):
            winner = "M0"
        elif duration < ctx.config.ramp_min_duration_sec:
            winner = "M0"
        else:
            total_change_z = total_change / scale
            shape = max(0.0, 1.0 - models["rss_m2"] / max(models["rss_m0"], 1e-9))

            diffs = np.diff(vals_fit)
            sign = np.sign(total_change)
            mono = float((np.sign(diffs) == sign).mean()) if sign != 0 else 0.0

            extra = {
                "slope_per_sec": float(slope),
                "monotonicity": mono,
                "n_core_frames": int(core_idx.size),
                "n_interior_frames": int(int_idx.size),
                "rss_m0": models["rss_m0"],
                "rss_m2": models["rss_m2"],
                "model_selection_winner": winner,
                "model_selection_margin": margin,
            }

            events.append(
                _make_event(
                    ctx,
                    event_id=ctx.next_id("ramp"),
                    event_type="within_block_ramp",
                    start_sec=float(ts_fit[0]),
                    end_sec=float(ts_fit[-1]),
                    block_ids=(b.block_id,),
                    delta=float(total_change),
                    delta_z=float(total_change_z),
                    baseline=baseline,
                    frames_for_quality=int_idx if int_idx.size >= 6 else core_idx,
                    shape_score_value=shape,
                    extra=extra,
                )
            )

            fitted_all = intercept + slope * ts_all
            residuals = vals_all - fitted_all
            return events, residuals

    m0_med = float(np.median(vals_fit))
    residuals = vals_all - m0_med

    min_dur = ctx.config.min_block_for_deviation_sec + ctx.signal.window_sec
    if b.duration_sec >= min_dur and int_idx.size > 0:
        delta = m0_med - baseline.value
        delta_z = delta / scale
        if abs(delta_z) >= ctx.config.baseline_departure_z:
            if not (min_eff and abs(delta) < min_eff.get(ctx.signal.name, 0.0)):
                res_int = ctx.smoothed[int_idx] - baseline.value
                sign_match = float((np.sign(res_int) == np.sign(delta)).mean())
                z_abs = np.abs(res_int / scale)
                block_mad = float(np.median(np.abs(ctx.smoothed[int_idx] - m0_med)))
                core_value = float(np.median(ctx.smoothed[core_idx])) if core_idx.size else m0_med
                edge_contribution = abs(core_value - m0_med) / max(abs(delta), 1e-9)

                extra = {
                    "peak_z": float(np.max(z_abs)),
                    "mean_abs_z": float(np.mean(z_abs)),
                    "n_core_frames": int(core_idx.size),
                    "n_interior_frames": int(int_idx.size),
                    "block_mad": block_mad,
                    "interior_median": m0_med,
                    "edge_contribution": edge_contribution,
                    "rss_m0": models["rss_m0"],
                    "rss_m1": models["rss_m1"],
                    "rss_m2": models["rss_m2"],
                    "model_selection_winner": winner,
                    "model_selection_margin": margin,
                }

                events.append(
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
                        frames_for_quality=int_idx,
                        shape_score_value=sign_match,
                        extra=extra,
                    )
                )

    return events, residuals


# ---------------------------------------------------------------------------
# Top-level: run all detectors for one signal
# ---------------------------------------------------------------------------


def run_all_detectors(
    ctx: SignalContext, *, diagnostics: bool = False,
) -> list[Event] | tuple[list[Event], list[dict]]:
    events: list[Event] = []

    block_residuals: dict[int, np.ndarray] = {}
    for b in ctx.blocks:
        block_events, resid = characterize_block(ctx, b)
        events.extend(block_events)
        if resid.size > 0:
            block_residuals[b.block_id] = resid
    events.extend(detect_within_block_excursions(ctx, block_residuals=block_residuals))
    events.extend(detect_short_gap_transitions(ctx))

    if not diagnostics:
        return events

    fired_by_block: dict[int, list[str]] = {}
    for e in events:
        for bid in e.block_ids:
            fired_by_block.setdefault(bid, []).append(e.event_type)

    block_diag: list[dict] = []
    for b in ctx.blocks:
        idx = ctx.block_frames(b.block_id, only_core=True)
        int_idx = ctx.interior_frames(b.block_id)
        if idx.size < 2:
            continue
        fit_idx = int_idx if int_idx.size >= 6 else idx
        ts = ctx.centers[fit_idx]
        vals = ctx.smoothed[fit_idx]
        scale = local_scale(
            b.start_sec, b.end_sec, ctx.centers, ctx.smoothed, ctx.core,
            ctx.global_stats, ctx.config,
        )
        models = _fit_block_models(
            ts, vals, b.start_sec, b.end_sec,
            ctx.config.regime_shift_edge_margin_sec,
            ctx.config.regime_shift_min_pre_sec,
            ctx.config.regime_shift_min_post_sec,
        )
        winner, margin = _model_selection(
            models["rss_m0"], models["rss_m1"], models["rss_m2"],
            scale, fit_idx.size, c=ctx.config.model_penalty_c,
        )
        int_med = float(np.median(ctx.smoothed[int_idx])) if int_idx.size else float(np.median(vals))
        block_diag.append({
            "block_id": b.block_id,
            "signal_name": ctx.signal.name,
            "block_start_sec": b.start_sec,
            "block_end_sec": b.end_sec,
            "block_duration_sec": b.duration_sec,
            "n_interior_frames": int(int_idx.size),
            "n_core_frames": int(idx.size),
            "interior_median": int_med,
            "rss_m0": models["rss_m0"],
            "rss_m1": models["rss_m1"],
            "rss_m2": models["rss_m2"],
            "model_selection_would_choose": winner,
            "model_selection_margin": margin,
            "current_detectors_fired": fired_by_block.get(b.block_id, []),
        })

    return events, block_diag
