"""Prominence detector for block-aware z-score timelines."""
from __future__ import annotations

import math

import numpy as np

from .config import Config, value_for_signal


def detect_prominence(
    z: np.ndarray,
    interior: np.ndarray,
    config: Config,
    signal_name: str,
    hop_sec: float,
) -> list[dict]:
    """Detect same-principle positive and negative deviation events."""
    z_seed = value_for_signal(config.z_seed, signal_name)
    z_return = value_for_signal(config.z_return, signal_name)
    seed_min_width_frames = _duration_to_frame_count(
        value_for_signal(config.seed_min_width_sec, signal_name), hop_sec
    )
    min_duration_frames = _center_span_to_frame_count(
        value_for_signal(config.min_duration_sec, signal_name), hop_sec
    )
    merge_gap_frames = max(
        0, int(round(value_for_signal(config.merge_gap_sec, signal_name) / hop_sec))
    )

    events: list[dict] = []
    for sign, direction in ((1.0, "+"), (-1.0, "-")):
        signed_z = sign * z
        signed_z_clean = np.where(interior, signed_z, -np.inf)
        seed_mask = signed_z_clean >= z_seed
        for seed_start, seed_end in _runs(seed_mask):
            if seed_end - seed_start < seed_min_width_frames:
                continue
            peak_i = seed_start + int(np.argmax(signed_z_clean[seed_start:seed_end]))
            peak_z = float(signed_z_clean[peak_i])

            left = peak_i
            while (
                left > 0
                and interior[left - 1]
                and signed_z[left - 1] >= z_return
            ):
                left -= 1

            right = peak_i
            while (
                right < len(z) - 1
                and interior[right + 1]
                and signed_z[right + 1] >= z_return
            ):
                right += 1

            events.append({
                "frame_start": int(left),
                "frame_end": int(right + 1),
                "direction": direction,
                "peak_i": int(peak_i),
                "peak_z": peak_z,
                "seed_start": int(seed_start),
                "seed_end": int(seed_end),
            })

    events.sort(key=lambda e: (e["frame_start"], e["frame_end"], e["direction"]))
    events = _merge_same_sign_events(events, merge_gap_frames)
    return [
        e for e in events
        if e["frame_end"] - e["frame_start"] >= min_duration_frames
    ]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0 or not bool(mask.any()):
        return []
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_same_sign_events(events: list[dict], merge_gap_frames: int) -> list[dict]:
    merged: list[dict] = []
    for event in events:
        if (
            merged
            and merged[-1]["direction"] == event["direction"]
            and event["frame_start"] - merged[-1]["frame_end"] <= merge_gap_frames
        ):
            current = merged[-1]
            current["frame_end"] = max(current["frame_end"], event["frame_end"])
            current.setdefault("merged_children", []).append(dict(event))
            if event["peak_z"] > current["peak_z"]:
                current["peak_z"] = event["peak_z"]
                current["peak_i"] = event["peak_i"]
                current["seed_start"] = event["seed_start"]
                current["seed_end"] = event["seed_end"]
            continue

        copied = dict(event)
        copied["merged_children"] = [dict(event)]
        merged.append(copied)
    return merged


def _duration_to_frame_count(duration_sec: float, hop_sec: float) -> int:
    return max(1, int(math.ceil(duration_sec / hop_sec)))


def _center_span_to_frame_count(duration_sec: float, hop_sec: float) -> int:
    return max(1, int(math.ceil(duration_sec / hop_sec)) + 1)
