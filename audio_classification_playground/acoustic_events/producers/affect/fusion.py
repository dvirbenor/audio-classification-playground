"""Cross-signal fusion for canonical affect events."""
from __future__ import annotations

from dataclasses import replace
from itertools import count

from .config import Config
from .types import Event


def merge_cross_signal(
    leaves: list[Event],
    config: Config,
    id_counter: count,
    *,
    producer_id: str = "affect.default",
) -> list[Event]:
    """Emit pairwise-overlapping joint parents across different signals."""
    parents: list[Event] = []
    used: set[str] = set()
    remaining = sorted(leaves, key=lambda e: (-e.score, e.start_sec))

    for seed in remaining:
        if seed.event_id in used:
            continue
        group = [seed]
        group_tracks = set(seed.source_track_ids)

        for candidate in remaining:
            if candidate.event_id == seed.event_id or candidate.event_id in used:
                continue
            if group_tracks.intersection(candidate.source_track_ids):
                continue
            if all(_overlap_sec(candidate, member) >= config.cross_signal_min_overlap_sec for member in group):
                group.append(candidate)
                group_tracks.update(candidate.source_track_ids)

        if len(group_tracks) >= 2:
            parents.append(_make_joint(group, config, id_counter, producer_id))
            used.update(e.event_id for e in group)

    return sorted(parents, key=lambda e: (e.start_sec, e.event_id))


def attach_parent_ids(leaves: list[Event], parents: list[Event]) -> list[Event]:
    parent_of: dict[str, str] = {}
    for parent in parents:
        for child_id in parent.children:
            parent_of[child_id] = parent.event_id
    return [
        replace(leaf, parent_id=parent_of[leaf.event_id])
        if leaf.event_id in parent_of else leaf
        for leaf in leaves
    ]


def _make_joint(
    group: list[Event],
    config: Config,
    id_counter: count,
    producer_id: str,
) -> Event:
    start_sec = min(e.start_sec for e in group)
    end_sec = max(e.end_sec for e in group)
    peak_child = max(group, key=lambda e: e.score)
    peak_z = float(sum(e.score ** 2 for e in group) ** 0.5)
    source_track_ids = tuple(sorted({tid for e in group for tid in e.source_track_ids}))
    return Event(
        event_id=f"{producer_id}.joint.{next(id_counter):06d}",
        producer_id=producer_id,
        task="affect",
        event_type="joint",
        label="joint",
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
        source_track_ids=source_track_ids,
        score=peak_z,
        score_name="peak_z",
        direction=None,
        children=tuple(e.event_id for e in sorted(group, key=lambda e: e.start_sec)),
        evidence={
            "peak_time_sec": peak_child.evidence.get("peak_time_sec"),
            "child_scores": {e.label: e.score for e in group},
            "child_directions": {e.label: e.direction for e in group},
            "signature": _signature(group, config.signature_z_threshold),
        },
        extra={
            "child_tracks": source_track_ids,
            "joint_strength_formula": "sqrt(sum(child.peak_z ** 2))",
        },
    )


def _signature(group: list[Event], threshold: float) -> str:
    by_signal: dict[str, list[Event]] = {}
    for event in group:
        signal = _affect_axis(event)
        if signal:
            by_signal.setdefault(signal, []).append(event)

    labels: list[str] = []
    short = {"arousal": "A", "valence": "V", "dominance": "D"}
    for signal in ("arousal", "valence", "dominance"):
        events = by_signal.get(signal, [])
        labels.append(f"{short[signal]}{_axis_direction(events, threshold)}")
    for signal in sorted(s for s in by_signal if s not in short):
        labels.append(f"{signal}{_axis_direction(by_signal[signal], threshold)}")
    return " ".join(labels)


def _axis_direction(events: list[Event], threshold: float) -> str:
    pos = max((e.score for e in events if e.direction == "+"), default=0.0)
    neg = max((e.score for e in events if e.direction == "-"), default=0.0)
    if pos < threshold and neg < threshold:
        return "0"
    if pos >= neg:
        return "+"
    return "-"


def _overlap_sec(a: Event, b: Event) -> float:
    return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))


def _affect_axis(event: Event) -> str | None:
    if event.label.endswith("_deviation"):
        return event.label[: -len("_deviation")]
    for track_id in event.source_track_ids:
        if track_id.startswith("affect."):
            return track_id.split(".", 1)[1]
    return None
