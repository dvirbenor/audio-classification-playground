"""Cross-signal fusion for v2 affective events."""
from __future__ import annotations

from dataclasses import replace
from itertools import count

from .config import Config
from .types import Event


def merge_cross_signal(
    leaves: list[Event],
    config: Config,
    id_counter: count,
) -> list[Event]:
    """Emit pairwise-overlapping joint parents across different signals."""
    parents: list[Event] = []
    used: set[str] = set()
    remaining = sorted(leaves, key=lambda e: (-e.peak_z, e.start_sec))

    for seed in remaining:
        if seed.event_id in used:
            continue
        group = [seed]
        group_signals = {seed.signal_name}

        for candidate in remaining:
            if candidate.event_id == seed.event_id or candidate.event_id in used:
                continue
            if candidate.signal_name in group_signals:
                continue
            if all(_overlap_sec(candidate, member) >= config.cross_signal_min_overlap_sec for member in group):
                group.append(candidate)
                group_signals.add(candidate.signal_name)

        if len(group_signals) >= 2:
            parents.append(_make_joint(group, config, id_counter))
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


def _make_joint(group: list[Event], config: Config, id_counter: count) -> Event:
    start_sec = min(e.start_sec for e in group)
    end_sec = max(e.end_sec for e in group)
    peak_child = max(group, key=lambda e: e.peak_z)
    peak_z = float(sum(e.peak_z ** 2 for e in group) ** 0.5)
    return Event(
        event_id=f"joint_{next(id_counter):05d}",
        signal_name="joint",
        event_type="joint",
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
        frame_start=min(e.frame_start for e in group),
        frame_end=max(e.frame_end for e in group),
        direction=_signature(group, config.signature_z_threshold),
        peak_z=peak_z,
        peak_time_sec=peak_child.peak_time_sec,
        baseline_at_peak=0.0,
        scale_at_peak=0.0,
        delta=0.0,
        children=tuple(e.event_id for e in sorted(group, key=lambda e: e.start_sec)),
        extra={
            "child_signals": [e.signal_name for e in group],
            "child_peak_z": {e.signal_name: e.peak_z for e in group},
            "joint_strength_formula": "sqrt(sum(child.peak_z ** 2))",
        },
    )


def _signature(group: list[Event], threshold: float) -> str:
    by_signal: dict[str, list[Event]] = {}
    for event in group:
        by_signal.setdefault(event.signal_name, []).append(event)

    labels: list[str] = []
    short = {"arousal": "A", "valence": "V", "dominance": "D"}
    for signal in ("arousal", "valence", "dominance"):
        events = by_signal.get(signal, [])
        labels.append(f"{short[signal]}{_axis_direction(events, threshold)}")
    for signal in sorted(s for s in by_signal if s not in short):
        labels.append(f"{signal}{_axis_direction(by_signal[signal], threshold)}")
    return " ".join(labels)


def _axis_direction(events: list[Event], threshold: float) -> str:
    pos = max((e.peak_z for e in events if e.direction == "+"), default=0.0)
    neg = max((e.peak_z for e in events if e.direction == "-"), default=0.0)
    if pos < threshold and neg < threshold:
        return "0"
    if pos >= neg:
        return "+"
    return "-"


def _overlap_sec(a: Event, b: Event) -> float:
    return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
