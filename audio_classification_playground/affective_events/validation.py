"""Non-causal cross-block validation (Phase 7).

Adjusts event confidence based on context **following** each event.
Only modifies confidence — never suppresses, merges, or changes event types.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import Config
from .scoring import combine_confidence
from .types import Block, Event


def cross_block_validate(
    events: list[Event],
    block_interior_medians: dict[tuple[str, int], float],
    blocks: list[Block],
    config: Config,
) -> list[Event]:
    """Adjust event confidence using non-causal cross-block context.

    Parameters
    ----------
    events
        Leaf events (already detected).
    block_interior_medians
        Mapping ``(signal_name, block_id) -> interior_median``.
    blocks
        The analysis blocks (sorted by start time).
    config
        Pipeline configuration.
    """
    if not blocks:
        return events

    block_by_id = {b.block_id: b for b in blocks}
    sorted_ids = [b.block_id for b in sorted(blocks, key=lambda b: b.start_sec)]
    next_block_id: dict[int, int | None] = {}
    prev_block_id: dict[int, int | None] = {}
    for i, bid in enumerate(sorted_ids):
        next_block_id[bid] = sorted_ids[i + 1] if i + 1 < len(sorted_ids) else None
        prev_block_id[bid] = sorted_ids[i - 1] if i > 0 else None

    k = config.cross_block_consistency_z
    max_gap = config.cross_block_max_gap_sec

    out: list[Event] = []
    for e in events:
        if e.event_type == "within_block_regime_shift":
            e = _validate_regime_shift(
                e, block_by_id, next_block_id,
                block_interior_medians, k, max_gap,
                config,
            )
        elif e.event_type == "block_deviation":
            e = _validate_block_deviation(
                e, block_by_id, next_block_id, prev_block_id,
                block_interior_medians, k, max_gap,
                config,
            )
        out.append(e)
    return out


def _validate_regime_shift(
    e: Event,
    block_by_id: dict[int, Block],
    next_block_id: dict[int, int | None],
    medians: dict[tuple[str, int], float],
    k: float,
    max_gap: float,
    config: Config,
) -> Event:
    if len(e.block_ids) != 1:
        return e
    bid = e.block_ids[0]
    nid = next_block_id.get(bid)
    if nid is None:
        return e

    this_block = block_by_id[bid]
    next_block = block_by_id[nid]
    gap = next_block.start_sec - this_block.end_sec
    if gap > max_gap:
        return e

    next_med = medians.get((e.signal_name, nid))
    if next_med is None:
        return e

    pre_med = e.extra.get("pre_median")
    post_med = e.extra.get("post_median")
    if pre_med is None or post_med is None:
        return e

    scale = abs(post_med - pre_med) if abs(post_med - pre_med) > 1e-9 else 1.0

    post_dist = abs(next_med - post_med) / scale
    pre_dist = abs(next_med - pre_med) / scale

    cross_block_score: float
    if post_dist < k:
        cross_block_score = 1.0
    elif pre_dist < k:
        cross_block_score = 0.3
    else:
        cross_block_score = 0.6

    return _adjust_confidence(e, "cross_block", cross_block_score, config)


def _validate_block_deviation(
    e: Event,
    block_by_id: dict[int, Block],
    next_block_id: dict[int, int | None],
    prev_block_id: dict[int, int | None],
    medians: dict[tuple[str, int], float],
    k: float,
    max_gap: float,
    config: Config,
) -> Event:
    if len(e.block_ids) != 1:
        return e
    bid = e.block_ids[0]
    this_block = block_by_id[bid]

    neighbors_near_baseline = 0
    neighbors_checked = 0

    for adj_id_map in (prev_block_id, next_block_id):
        adj_id = adj_id_map.get(bid)
        if adj_id is None:
            continue
        adj_block = block_by_id[adj_id]
        gap = abs(adj_block.start_sec - this_block.end_sec)
        if adj_id_map is prev_block_id:
            gap = this_block.start_sec - adj_block.end_sec
        if gap > max_gap:
            continue
        adj_med = medians.get((e.signal_name, adj_id))
        if adj_med is None:
            continue
        neighbors_checked += 1
        deviation_from_baseline = abs(adj_med - e.baseline_value)
        block_deviation = abs(e.extra.get("interior_median", e.baseline_value + e.delta) - e.baseline_value)
        if block_deviation > 1e-9 and deviation_from_baseline / block_deviation < k:
            neighbors_near_baseline += 1

    if neighbors_checked == 0:
        return e

    cross_block_score: float
    if neighbors_near_baseline == neighbors_checked:
        cross_block_score = 1.0
    elif neighbors_near_baseline > 0:
        cross_block_score = 0.7
    else:
        cross_block_score = 0.4

    return _adjust_confidence(e, "cross_block", cross_block_score, config)


def _adjust_confidence(
    e: Event, component_name: str, score: float, config: Config,
) -> Event:
    new_components = dict(e.confidence_components)
    new_components[component_name] = score
    new_confidence = combine_confidence(new_components, config)

    new_extra = dict(e.extra)
    new_extra["cross_block_score"] = score

    return replace(
        e,
        confidence=new_confidence,
        confidence_components=new_components,
        extra=new_extra,
    )
