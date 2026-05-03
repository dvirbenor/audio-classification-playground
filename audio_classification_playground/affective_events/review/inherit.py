"""Time-overlap label inheritance across (recording, config) sessions.

When detection is re-run with a tweaked config, most events in the new
session have a temporal counterpart in the previous session. We carry labels
forward by matching on task, label, source tracks, and temporal overlap.

The matching is one-shot at session creation time (see
``storage.save_session(inherit_from=...)``); each inherited label records
the source session id and the match score so the user can audit it.
"""
from __future__ import annotations

from typing import Iterable


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Overlap divided by the shorter of the two durations (0 if disjoint)."""
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    min_dur = min(a_end - a_start, b_end - b_start)
    return overlap / min_dur if min_dur > 0 else 0.0


def inherit_labels(
    *,
    prev_session: dict,
    new_events: list[dict],
    overlap_threshold: float = 0.5,
) -> dict[str, dict]:
    """Build a labels dict for ``new_events`` by matching to ``prev_session``.

    Each new event is matched against previously-labeled events in
    ``prev_session`` that share the same semantic match key. The match with
    the highest overlap ratio (and at least ``overlap_threshold``) wins.

    Returns a mapping ``event_id -> label_dict`` containing only inherited
    labels; events without a match are omitted (so the user sees them as
    unlabeled, which is the correct semantic).
    """
    prev_labels: dict = prev_session.get("labels") or {}
    if not prev_labels:
        return {}

    prev_events_by_id: dict[str, dict] = {e["event_id"]: e for e in prev_session.get("events", [])}
    prev_session_id = prev_session.get("session_id", "")

    # Pre-bucket previously-labeled events by semantic match key for fast lookup.
    prev_by_key: dict[tuple, list[dict]] = {}
    for prev_eid, prev_label in prev_labels.items():
        prev_e = prev_events_by_id.get(prev_eid)
        if prev_e is None or not prev_label.get("verdict"):
            continue
        prev_by_key.setdefault(_match_key(prev_e, prev_events_by_id), []).append(prev_e)

    inherited: dict[str, dict] = {}
    new_events_by_id = {e["event_id"]: e for e in new_events}
    for new_e in new_events:
        candidates = prev_by_key.get(_match_key(new_e, new_events_by_id), ())
        if not candidates:
            continue
        new_s, new_t = new_e["start_sec"], new_e["end_sec"]
        best_score = 0.0
        best_prev: dict | None = None
        for prev_e in candidates:
            score = _overlap_ratio(new_s, new_t, prev_e["start_sec"], prev_e["end_sec"])
            if score >= overlap_threshold and score > best_score:
                best_score = score
                best_prev = prev_e
        if best_prev is None:
            continue
        prev_label = prev_labels[best_prev["event_id"]]
        inherited[new_e["event_id"]] = {
            "verdict": prev_label.get("verdict", ""),
            "tags": list(prev_label.get("tags") or []),
            "comment": prev_label.get("comment", "") or "",
            "labeler": prev_label.get("labeler", "") or "",
            "labeled_at": prev_label.get("labeled_at", "") or "",
            "inherited_from": prev_session_id,
            "inherited_match_score": float(best_score),
        }
    return inherited


def _match_key(event: dict, events_by_id: dict[str, dict]) -> tuple:
    base = (
        event.get("task", ""),
        event.get("label", ""),
        tuple(event.get("source_track_ids") or ()),
    )
    if event.get("task") == "affect" and event.get("event_type") == "joint":
        return base + (_joint_child_signature(event, events_by_id),)
    return base


def _joint_child_signature(event: dict, events_by_id: dict[str, dict]) -> tuple:
    children = []
    for child_id in event.get("children") or ():
        child = events_by_id.get(child_id)
        if not child:
            continue
        children.append((child.get("label", ""), child.get("direction")))
    return tuple(sorted(children))
