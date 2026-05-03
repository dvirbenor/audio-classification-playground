"""End-to-end orchestration for the canonical affective-events detector."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from itertools import count
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import ProducerRun, RegularGridTrack
from .baseline import block_aware_baseline_scale
from .config import Config
from .detector import detect_prominence
from .fusion import attach_parent_ids, merge_cross_signal
from .preprocessing import assign_frame_blocks, build_blocks, global_stats
from .types import Event, Signal, Vad


DEFAULT_PRODUCER_ID = "affect.default"


def extract_events(
    signals: Sequence[Signal],
    vad: Vad,
    config: Config | None = None,
    *,
    diagnostics: bool = False,
    producer_id: str = DEFAULT_PRODUCER_ID,
) -> list[Event] | tuple[list[Event], pd.DataFrame]:
    """Extract affect deviation and joint events. See METHODOLOGY.md."""
    config = config or Config.balanced()
    blocks = build_blocks(vad, config)
    id_counter = count()
    leaves: list[Event] = []
    diag_rows: list[dict] = []

    for signal in signals:
        signal_leaves, rows = _extract_signal(
            signal, blocks, config, id_counter, producer_id=producer_id
        )
        leaves.extend(signal_leaves)
        diag_rows.extend(rows)

    parents = merge_cross_signal(leaves, config, id_counter, producer_id=producer_id)
    leaves = attach_parent_ids(leaves, parents)
    events = leaves + parents
    events.sort(key=lambda e: (e.start_sec, e.label, e.event_id))

    if diagnostics:
        return events, pd.DataFrame(diag_rows)
    return events


def extract_events_with_tracks(
    signals: Sequence[Signal],
    vad: Vad,
    config: Config | None = None,
    *,
    diagnostics: bool = False,
    producer_id: str = DEFAULT_PRODUCER_ID,
) -> tuple[list[Event], list[RegularGridTrack]] | tuple[list[Event], list[RegularGridTrack], pd.DataFrame]:
    """Extract affect events and return the source tracks for review sessions."""
    result = extract_events(
        signals, vad, config, diagnostics=diagnostics, producer_id=producer_id
    )
    tracks = tracks_from_signals(signals, producer_id=producer_id)
    if diagnostics:
        events, diag = result
        return events, tracks, diag
    return result, tracks


def tracks_from_signals(
    signals: Sequence[Signal],
    *,
    producer_id: str = DEFAULT_PRODUCER_ID,
) -> list[RegularGridTrack]:
    """Represent A/V/D prediction arrays as generic regular-grid tracks."""
    return [
        RegularGridTrack(
            track_id=f"affect.{signal.name}",
            producer_id=producer_id,
            task="affect",
            name=signal.name,
            value_type="continuous",
            renderer="line",
            values=signal.values,
            hop_sec=signal.hop_sec,
            window_sec=signal.window_sec,
            meta={"window_semantics": "frame summarizes [i*hop, i*hop + window]"},
        )
        for signal in signals
    ]


def producer_run(
    config: Config | dict | None = None,
    *,
    blocks: Sequence | None = None,
    producer_id: str = DEFAULT_PRODUCER_ID,
    source_model: str = "affect-continuous",
) -> ProducerRun:
    """Build the affect producer metadata used by review sessions."""
    cfg = config or Config.balanced()
    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    outputs: dict = {}
    if blocks is not None:
        outputs["blocks"] = [
            {
                "block_id": int(b.block_id),
                "start_sec": float(b.start_sec),
                "end_sec": float(b.end_sec),
            }
            for b in blocks
        ]
    return ProducerRun(
        producer_id=producer_id,
        task="affect",
        source_model=source_model,
        config=cfg_dict,
        config_hash=_config_hash(cfg_dict),
        outputs=outputs,
    )


def _extract_signal(
    signal: Signal,
    blocks,
    config: Config,
    id_counter: count,
    *,
    producer_id: str,
) -> tuple[list[Event], list[dict]]:
    frame_block = assign_frame_blocks(
        signal.n_frames, signal.hop_sec, signal.window_sec, blocks
    )
    interior = frame_block >= 0
    global_median, global_mad = global_stats(signal.values, interior)
    baseline, scale = block_aware_baseline_scale(
        signal.values,
        frame_block,
        blocks,
        signal.hop_sec,
        signal.window_sec,
        config,
        global_median=global_median,
        global_mad=global_mad,
    )

    z = np.zeros(signal.n_frames, dtype=np.float64)
    valid_z = interior & np.isfinite(baseline) & np.isfinite(scale) & (scale > 0)
    z[valid_z] = (signal.values[valid_z] - baseline[valid_z]) / scale[valid_z]

    candidates = detect_prominence(
        z, valid_z, config, signal.name, signal.hop_sec
    )
    centers = signal.frame_centers()
    events = [
        _make_leaf_event(
            signal,
            centers,
            baseline,
            scale,
            z,
            frame_block,
            candidate,
            id_counter,
            producer_id=producer_id,
        )
        for candidate in candidates
    ]

    diag_rows = _diagnostics(
        signal.name, blocks, frame_block, interior,
        global_median, global_mad, baseline, scale,
    )
    return events, diag_rows


def _make_leaf_event(
    signal: Signal,
    centers: np.ndarray,
    baseline: np.ndarray,
    scale: np.ndarray,
    z: np.ndarray,
    frame_block: np.ndarray,
    candidate: dict,
    id_counter: count,
    *,
    producer_id: str,
) -> Event:
    frame_start = candidate["frame_start"]
    frame_end = candidate["frame_end"]
    peak_i = candidate["peak_i"]
    direction = candidate["direction"]
    signed_z = z if direction == "+" else -z

    start_sec = float(centers[frame_start])
    end_sec = float(centers[frame_end - 1])
    event_id = f"{producer_id}.deviation.{next(id_counter):06d}"
    shoulder_start_z = float(signed_z[frame_start])
    shoulder_end_z = float(signed_z[frame_end - 1])
    next_left = frame_start - 1
    next_right = frame_end
    signed_peak_z = float(z[peak_i])

    return Event(
        event_id=event_id,
        producer_id=producer_id,
        task="affect",
        event_type="deviation",
        label=f"{signal.name}_deviation",
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=float(end_sec - start_sec),
        source_track_ids=(f"affect.{signal.name}",),
        score=float(candidate["peak_z"]),
        score_name="peak_z",
        direction=direction,
        evidence={
            "signed_z": signed_peak_z,
            "peak_time_sec": float(centers[peak_i]),
            "baseline_at_peak": float(baseline[peak_i]),
            "scale_at_peak": float(scale[peak_i]),
            "delta": float(signal.values[peak_i] - baseline[peak_i]),
            "block_ids": tuple(sorted(set(frame_block[frame_start:frame_end].tolist()))),
        },
        extra={
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "peak_frame": int(peak_i),
            "seed_start": int(candidate["seed_start"]),
            "seed_end": int(candidate["seed_end"]),
            "seed_width_frames": int(candidate["seed_end"] - candidate["seed_start"]),
            "shoulder_start_z": shoulder_start_z,
            "shoulder_end_z": shoulder_end_z,
            "next_left_z": float(signed_z[next_left]) if next_left >= 0 else None,
            "next_right_z": float(signed_z[next_right]) if next_right < len(z) else None,
            "merged_children": candidate.get("merged_children", []),
        },
    )


def _diagnostics(
    signal_name: str,
    blocks,
    frame_block: np.ndarray,
    interior: np.ndarray,
    global_median: float,
    global_mad: float,
    baseline: np.ndarray,
    scale: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for block in blocks:
        idx = np.where(frame_block == block.block_id)[0]
        rows.append({
            "signal_name": signal_name,
            "block_id": block.block_id,
            "block_start_sec": block.start_sec,
            "block_end_sec": block.end_sec,
            "n_interior_frames": int(idx.size),
            "global_median": float(global_median),
            "global_mad": float(global_mad),
            "baseline_median": float(np.nanmedian(baseline[idx])) if idx.size else np.nan,
            "scale_median": float(np.nanmedian(scale[idx])) if idx.size else np.nan,
            "n_total_interior_frames": int(interior.sum()),
        })
    return rows


def to_dataframe(events: Sequence[Event]) -> pd.DataFrame:
    rows = [event.as_dict() for event in events]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["start_sec", "task", "label", "event_id"]
    ).reset_index(drop=True)


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
