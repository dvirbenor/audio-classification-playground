"""End-to-end orchestration: ``extract_events(signals, vad, config)``.

The pipeline runs each signal through an identical sub-pipeline, then fuses
events into per-signal episodes and cross-signal joint parents. Detection
state is never shared between signals, so a single signal can fail (empty
result) without affecting the others.
"""
from __future__ import annotations

from itertools import count
from typing import Sequence

import numpy as np
import pandas as pd

from .baseline import compute_global_stats
from .config import Config
from .detectors import SignalContext, run_all_detectors
from .fusion import aggregate_episodes, attach_parent_ids, merge_cross_signal
from .preprocessing import (
    assign_blocks,
    boundary_flags,
    build_blocks,
    compute_coverage,
    smooth_within_blocks,
)
from .types import Event, Signal, Vad


def extract_events(
    signals: Sequence[Signal],
    vad: Vad,
    config: Config | None = None,
) -> list[Event]:
    """Run the full per-signal detection + fusion pipeline.

    Parameters
    ----------
    signals
        One or more :class:`Signal` instances. Detection runs independently
        per signal; signals may share a hop/window or differ.
    vad
        Voice-activity intervals (canonical seconds form). Use the
        :class:`Vad` classmethod constructors for Silero output or per-frame
        probabilities on a different grid.
    config
        Optional :class:`Config`; defaults to :meth:`Config.balanced`.

    Returns
    -------
    list[Event]
        Leaf events (one per detection) plus episode parents (per signal,
        when applicable) and joint parents (cross-signal). Children carry
        their ``parent_id`` back-reference. The list is sorted by
        ``(start_sec, signal_name)``.
    """
    config = config or Config.balanced()
    blocks = build_blocks(vad, config)
    id_counter = count()

    per_signal_events: dict[str, list[Event]] = {}
    for sig in signals:
        ctx = _build_context(sig, blocks, config, id_counter)
        per_signal_events[sig.name] = run_all_detectors(ctx)

    leaves = [e for evs in per_signal_events.values() for e in evs]
    episode_parents = aggregate_episodes(per_signal_events, config, id_counter)
    joint_parents = merge_cross_signal(leaves, config, id_counter)

    parents = episode_parents + joint_parents
    leaves = attach_parent_ids(leaves, parents)

    all_events = leaves + parents
    all_events.sort(key=lambda e: (e.start_sec, e.signal_name, e.event_id))
    return all_events


def _build_context(
    signal: Signal, blocks, config: Config, id_counter: count
) -> SignalContext:
    coverage = compute_coverage(signal, _vad_for_context(blocks, signal))
    usable = coverage >= config.usable_speech_coverage
    core = coverage >= config.core_speech_coverage
    frame_block = assign_blocks(signal, blocks)
    near_start, near_end = boundary_flags(signal, blocks, frame_block, config.boundary_margin_sec)
    smoothed = smooth_within_blocks(signal.values, frame_block, signal.hop_sec, config.smooth_median_sec)
    global_stats = compute_global_stats(smoothed, core)
    centers = signal.frame_centers()
    return SignalContext(
        signal=signal,
        blocks=blocks,
        coverage=coverage,
        usable=usable,
        core=core,
        frame_block=frame_block,
        near_start=near_start,
        near_end=near_end,
        smoothed=smoothed,
        centers=centers,
        global_stats=global_stats,
        config=config,
        id_counter=id_counter,
    )


def _vad_for_context(blocks, signal: Signal) -> Vad:
    """Reconstruct VAD intervals from analysis blocks for coverage computation.

    We use the *merged* blocks (not the raw VAD) so frame quality reflects the
    same units used for detection. This is intentional: coverage computed
    against raw VAD would penalize frames inside blocks formed by bridging
    micro-gaps.
    """
    return Vad(intervals=tuple((b.start_sec, b.end_sec) for b in blocks))


# ---------------------------------------------------------------------------
# DataFrame helper for review tools
# ---------------------------------------------------------------------------


_FLAT_COLUMNS = (
    "event_id", "signal_name", "event_type",
    "start_sec", "end_sec", "duration_sec",
    "block_ids", "delta", "delta_z", "direction",
    "baseline_value", "baseline_context_speech_sec", "baseline_source",
    "mean_speech_coverage", "near_block_start", "near_block_end",
    "strength", "confidence",
    "review_audio_start_sec", "review_audio_end_sec",
    "parent_id", "children",
)


def to_dataframe(events: Sequence[Event]) -> pd.DataFrame:
    """Flatten events into a sortable, filterable DataFrame for review.

    The ``confidence_components`` and ``extra`` mappings are kept as dict
    columns so the table stays compact while remaining lossless.
    """
    rows = []
    for e in events:
        row = {col: getattr(e, col) for col in _FLAT_COLUMNS}
        row["confidence_components"] = e.confidence_components
        row["extra"] = e.extra
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["start_sec", "signal_name", "event_id"]).reset_index(drop=True)
