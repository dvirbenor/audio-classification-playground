"""Within-signal episode aggregation and cross-signal joint merging.

These are the only two stages that take leaf events and produce parents.
Both keep the children intact (children are what reviewers actually listen
to); parents are an annotation layer with their own ids and signatures.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import count
from typing import Iterable

from .config import Config
from .scoring import combine_confidence, signature_for
from .types import Event


# ---------------------------------------------------------------------------
# Within-signal: group consecutive same-direction block-deviation events
# ---------------------------------------------------------------------------


def aggregate_episodes(
    events_per_signal: dict[str, list[Event]],
    config: Config,
    id_counter: count,
) -> list[Event]:
    """Per signal, link nearby block-level events sharing direction into an episode."""
    parents: list[Event] = []
    if not config.enable_episode_aggregation:
        return parents

    for signal_name, events in events_per_signal.items():
        block_evts = [
            e for e in events
            if e.event_type == "block_deviation"
        ]
        block_evts.sort(key=lambda e: e.start_sec)

        i = 0
        while i < len(block_evts):
            group = [block_evts[i]]
            j = i + 1
            while j < len(block_evts):
                prev = group[-1]
                cur = block_evts[j]
                gap = cur.start_sec - prev.end_sec
                if gap > config.episode_max_inter_block_gap_sec:
                    break
                if cur.direction != prev.direction:
                    break
                group.append(cur)
                j += 1
            if len(group) >= config.episode_min_children:
                parents.append(_make_episode(group, signal_name, config, id_counter))
            i = j

    return parents


def _make_episode(
    group: list[Event], signal_name: str, config: Config, id_counter: count
) -> Event:
    parent_id = f"episode_{signal_name}_{next(id_counter):05d}"
    start_sec = min(e.start_sec for e in group)
    end_sec = max(e.end_sec for e in group)

    # Aggregate child fields (weighted by duration)
    total_dur = sum(e.duration_sec for e in group) or 1.0
    delta = sum(e.delta * e.duration_sec for e in group) / total_dur
    delta_z = sum(e.delta_z * e.duration_sec for e in group) / total_dur
    coverage = sum(e.mean_speech_coverage * e.duration_sec for e in group) / total_dur
    context_sec = max(e.baseline_context_speech_sec for e in group)
    baseline_value = sum(e.baseline_value * e.duration_sec for e in group) / total_dur

    components = {
        "strength": max(e.confidence_components.get("strength", 0.0) for e in group),
        "duration": min(1.0, (end_sec - start_sec) / (4.0 * config.review_pad_sec)),
        "coverage": coverage,
        "context": min(1.0, context_sec / 60.0),
        "shape": float(len(group)) / max(len(group), config.episode_min_children),
        "boundary": sum(e.confidence_components.get("boundary", 1.0) for e in group) / len(group),
    }
    confidence = combine_confidence(components, config)

    return Event(
        event_id=parent_id,
        signal_name=f"episode:{signal_name}",
        event_type="affective_episode",
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
        block_ids=tuple(sorted({bid for e in group for bid in e.block_ids})),
        delta=float(delta),
        delta_z=float(delta_z),
        direction=group[0].direction,
        baseline_value=float(baseline_value),
        baseline_context_speech_sec=float(context_sec),
        baseline_source="aggregated",
        mean_speech_coverage=float(coverage),
        near_block_start=False,
        near_block_end=False,
        strength=float(abs(delta_z)),
        confidence=confidence,
        confidence_components=components,
        review_audio_start_sec=max(0.0, start_sec - config.review_pad_sec),
        review_audio_end_sec=end_sec + config.review_pad_sec,
        children=tuple(e.event_id for e in group),
        extra={"n_children": len(group), "child_signals": [signal_name] * len(group)},
    )


# ---------------------------------------------------------------------------
# Cross-signal: group temporally co-occurring events from different signals
# ---------------------------------------------------------------------------


def merge_cross_signal(
    leaf_events: list[Event], config: Config, id_counter: count
) -> list[Event]:
    """Group overlapping leaves from different signals into joint parents.

    Builds an overlap graph (two leaves linked iff they overlap by at least
    ``cross_signal_min_overlap_sec`` and come from different signals) and
    emits one joint parent per connected component that spans >= 2 signals.
    """
    if not config.enable_cross_signal_merge:
        return []

    leaves = sorted(leaf_events, key=lambda e: e.start_sec)
    by_id = {e.event_id: e for e in leaves}
    threshold = config.cross_signal_min_overlap_sec

    adjacency: dict[str, list[str]] = {e.event_id: [] for e in leaves}
    for i, e in enumerate(leaves):
        for f in leaves[i + 1 :]:
            if f.start_sec >= e.end_sec:
                break  # leaves are start-sorted; nothing further can overlap e
            if f.signal_name == e.signal_name:
                continue
            overlap = min(e.end_sec, f.end_sec) - max(e.start_sec, f.start_sec)
            if overlap >= threshold:
                adjacency[e.event_id].append(f.event_id)
                adjacency[f.event_id].append(e.event_id)

    parents: list[Event] = []
    visited: set[str] = set()
    for e in leaves:
        if e.event_id in visited:
            continue
        component: list[Event] = []
        stack = [e.event_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            component.append(by_id[nid])
            stack.extend(adjacency[nid])
        if len({c.signal_name for c in component}) >= 2:
            parents.append(_make_joint(component, config, id_counter))

    return parents


def _make_joint(group: list[Event], config: Config, id_counter: count) -> Event:
    parent_id = f"joint_{next(id_counter):05d}"
    start_sec = min(e.start_sec for e in group)
    end_sec = max(e.end_sec for e in group)
    duration = end_sec - start_sec

    deltas_z = {e.signal_name: e.delta_z for e in group}
    sig_label = signature_for(deltas_z)

    components = {
        "strength": max(e.confidence_components.get("strength", 0.0) for e in group),
        "duration": min(1.0, duration / (4.0 * config.review_pad_sec)),
        "coverage": sum(e.mean_speech_coverage for e in group) / len(group),
        "context": sum(e.baseline_context_speech_sec for e in group) / len(group) / 60.0,
        "shape": float(len(group)) / 3.0,  # 1.0 when all three signals participate
        "boundary": sum(e.confidence_components.get("boundary", 1.0) for e in group) / len(group),
    }
    components["context"] = min(1.0, components["context"])
    confidence = combine_confidence(components, config)

    # Joint "strength" = quadrature sum of children's |z|
    strength = float(sum(e.delta_z ** 2 for e in group) ** 0.5)

    return Event(
        event_id=parent_id,
        signal_name="joint",
        event_type="joint",
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        duration_sec=float(duration),
        block_ids=tuple(sorted({bid for e in group for bid in e.block_ids})),
        delta=0.0,
        delta_z=0.0,
        direction=sig_label,
        baseline_value=0.0,
        baseline_context_speech_sec=sum(e.baseline_context_speech_sec for e in group) / len(group),
        baseline_source="aggregated",
        mean_speech_coverage=components["coverage"],
        near_block_start=any(e.near_block_start for e in group),
        near_block_end=any(e.near_block_end for e in group),
        strength=strength,
        confidence=confidence,
        confidence_components=components,
        review_audio_start_sec=max(0.0, start_sec - config.review_pad_sec),
        review_audio_end_sec=end_sec + config.review_pad_sec,
        children=tuple(e.event_id for e in group),
        extra={
            "signature": sig_label,
            "child_signals": [e.signal_name for e in group],
            "child_types": [e.event_type for e in group],
            "delta_z_per_signal": deltas_z,
        },
    )


# ---------------------------------------------------------------------------
# Parent back-references on children
# ---------------------------------------------------------------------------


def attach_parent_ids(leaves: list[Event], parents: list[Event]) -> list[Event]:
    """Return a new leaf list with ``parent_id`` populated from ``parents.children``."""
    parent_of: dict[str, str] = {}
    for p in parents:
        for cid in p.children:
            # Joint parents take precedence over episode parents on duplicate keys
            if p.event_type == "joint" or cid not in parent_of:
                parent_of[cid] = p.event_id
    return [
        replace(e, parent_id=parent_of[e.event_id]) if e.event_id in parent_of else e
        for e in leaves
    ]
