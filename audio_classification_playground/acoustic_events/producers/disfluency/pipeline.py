"""Disfluency event extraction from Vox-Profile speech-flow logits.

This producer consumes already-computed window-pooled logits. It does not run
model inference. The binary fluency head defines candidate regions, while the
multi-label disfluency type head names and explains those regions.

Pure ``Sound Repetition`` is suppressed by default because an audit on
conversational/podcast-like audio found those regions were mostly laughter,
background, or otherwise non-target audio. This is a use-case default, not a
property of the model: set ``DisfluencyConfig.suppressed_types=()`` when sound
repetition itself is a target class.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np

from ...schema import Event, ProducerRun, RegularGridTrack
from .config import (
    DISFLUENCY_TYPE_LABELS,
    FLUENCY_LABELS,
    LABEL_TO_EVENT_LABEL,
    DisfluencyConfig,
)


DEFAULT_PRODUCER_ID = "disfluency.default"
DEFAULT_SOURCE_MODEL = "vox-profile-speech-flow"
TASK = "disfluency"
EVENT_TYPE = "instance"
FLUENCY_TRACK_ID = "disfluency.fluency"
TYPE_TRACK_ID = "disfluency.type"
WINDOW_SEMANTICS = "frame summarizes [i*hop, i*hop + window]"
_AUDIO_DURATION_TOLERANCE_SEC = 1e-6


def produce_disfluency_events(
    *,
    fluency_logits,
    disfluency_type_logits,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None = None,
    config: DisfluencyConfig | None = None,
    producer_id: str = DEFAULT_PRODUCER_ID,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> tuple[ProducerRun, list[RegularGridTrack], list[Event]]:
    """Build a producer run, evidence tracks, and disfluency events."""
    cfg = config or DisfluencyConfig.balanced()
    fluency, type_logits = _validated_logits(
        fluency_logits,
        disfluency_type_logits,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=audio_duration_sec,
    )
    p_disfluent = _softmax(fluency, axis=1)[:, 1]
    type_probs = _sigmoid(type_logits)

    tracks = tracks_from_logits(
        fluency,
        type_logits,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=audio_duration_sec,
        producer_id=producer_id,
    )
    events, summary = _extract_events_with_summary(
        p_disfluent,
        type_probs,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=audio_duration_sec,
        config=cfg,
        producer_id=producer_id,
    )
    run = make_producer_run(
        cfg,
        summary=summary,
        producer_id=producer_id,
        source_model=source_model,
    )
    return run, tracks, events


def tracks_from_logits(
    fluency_logits,
    disfluency_type_logits,
    *,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None = None,
    producer_id: str = DEFAULT_PRODUCER_ID,
) -> list[RegularGridTrack]:
    """Represent activated model outputs as review evidence tracks."""
    fluency, type_logits = _validated_logits(
        fluency_logits,
        disfluency_type_logits,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=audio_duration_sec,
    )
    p_disfluent = _softmax(fluency, axis=1)[:, 1]
    type_probs = _sigmoid(type_logits)

    common_meta = {
        "source_model_output_form": "window_pooled_logits",
        "window_semantics": WINDOW_SEMANTICS,
    }
    return [
        RegularGridTrack(
            track_id=FLUENCY_TRACK_ID,
            producer_id=producer_id,
            task=TASK,
            name="P(disfluent)",
            value_type="probability",
            renderer="probability",
            values=p_disfluent,
            hop_sec=hop_sec,
            window_sec=window_sec,
            meta={**common_meta, "activation": "softmax_class_1"},
        ),
        RegularGridTrack(
            track_id=TYPE_TRACK_ID,
            producer_id=producer_id,
            task=TASK,
            name="disfluency type",
            value_type="probability",
            renderer="multi_probability",
            values=type_probs,
            hop_sec=hop_sec,
            window_sec=window_sec,
            channels=DISFLUENCY_TYPE_LABELS,
            meta={**common_meta, "activation": "sigmoid"},
        ),
    ]


def extract_events(
    fluency_logits,
    disfluency_type_logits,
    *,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None = None,
    config: DisfluencyConfig | None = None,
    producer_id: str = DEFAULT_PRODUCER_ID,
) -> list[Event]:
    """Extract disfluency events from raw logits without building tracks."""
    cfg = config or DisfluencyConfig.balanced()
    fluency, type_logits = _validated_logits(
        fluency_logits,
        disfluency_type_logits,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=audio_duration_sec,
    )
    events, _ = _extract_events_with_summary(
        _softmax(fluency, axis=1)[:, 1],
        _sigmoid(type_logits),
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=audio_duration_sec,
        config=cfg,
        producer_id=producer_id,
    )
    return events


def make_producer_run(
    config: DisfluencyConfig | dict | None = None,
    *,
    summary: dict | None = None,
    producer_id: str = DEFAULT_PRODUCER_ID,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> ProducerRun:
    """Build producer metadata for a disfluency extraction run."""
    cfg = config or DisfluencyConfig.balanced()
    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    outputs = {
        "candidate_region_count": 0,
        "emitted_event_count": 0,
        "suppressed_pure_sound_repetition_count": 0,
        "unspecified_region_count": 0,
        "emitted_unspecified_event_count": 0,
        "label_counts": {},
        "audit_note": (
            "Pure Sound Repetition is suppressed by default for "
            "conversational/podcast-like audio; set suppressed_types=() for "
            "clinical or stuttering-focused use cases."
        ),
    }
    if summary is not None:
        outputs.update(summary)
    return ProducerRun(
        producer_id=producer_id,
        task=TASK,
        source_model=source_model,
        config=cfg_dict,
        config_hash=_config_hash(cfg_dict),
        outputs=outputs,
    )


def _extract_events_with_summary(
    p_disfluent: np.ndarray,
    type_probs: np.ndarray,
    *,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None,
    config: DisfluencyConfig,
    producer_id: str,
) -> tuple[list[Event], dict]:
    regions = _support_regions(p_disfluent, hop_sec, config)
    events: list[Event] = []
    counters = Counter()

    centers = _frame_centers(len(p_disfluent), hop_sec, window_sec)
    suppressed = set(config.suppressed_types)
    min_support_frames = _min_support_frames(config.min_support_sec, hop_sec)

    for region in regions:
        candidate = _candidate_from_region(
            region,
            p_disfluent,
            type_probs,
            centers,
            hop_sec=hop_sec,
            window_sec=window_sec,
            audio_duration_sec=audio_duration_sec,
            suppressed=suppressed,
        )
        active_all = [
            item
            for item in candidate["type_evidence_items"]
            if item["max"] >= config.type_threshold
        ]
        suppressed_active = [
            item for item in active_all if item["name"] in suppressed
        ]
        useful = [
            item for item in active_all if item["name"] not in suppressed
        ]
        candidate["active_types_all"] = active_all
        candidate["active_types"] = useful
        candidate["suppressed_active_types"] = suppressed_active

        active_names = {item["name"] for item in active_all}
        if active_names == {"Sound Repetition"}:
            counters["suppressed_pure_sound_repetition_count"] += 1
        if not useful:
            counters["unspecified_region_count"] += 1
            if not config.emit_unspecified:
                continue
            label = "disfluent"
            counters["emitted_unspecified_event_count"] += 1
        else:
            label = _select_label(useful)

        event = _event_from_candidate(
            candidate,
            label=label,
            event_index=len(events),
            producer_id=producer_id,
            thresholds={
                "seed_threshold": float(config.seed_threshold),
                "shoulder_threshold": float(config.shoulder_threshold),
                "min_support_sec": float(config.min_support_sec),
                "min_support_frames": int(min_support_frames),
                "merge_gap_sec": float(config.merge_gap_sec),
                "type_threshold": float(config.type_threshold),
                "suppressed_types": tuple(config.suppressed_types),
                "emit_unspecified": bool(config.emit_unspecified),
            },
        )
        events.append(event)

    label_counts = Counter(event.label for event in events)
    summary = {
        "candidate_region_count": len(regions),
        "emitted_event_count": len(events),
        "suppressed_pure_sound_repetition_count": counters["suppressed_pure_sound_repetition_count"],
        "unspecified_region_count": counters["unspecified_region_count"],
        "emitted_unspecified_event_count": counters["emitted_unspecified_event_count"],
        "label_counts": dict(sorted(label_counts.items())),
    }
    return events, summary


def _support_regions(
    p_disfluent: np.ndarray,
    hop_sec: float,
    config: DisfluencyConfig,
) -> list[tuple[int, int]]:
    seed_regions = _contiguous_regions(p_disfluent >= config.seed_threshold)
    expanded: list[tuple[int, int]] = []
    for start, end in seed_regions:
        left = start
        right = end
        while left > 0 and p_disfluent[left - 1] >= config.shoulder_threshold:
            left -= 1
        while right < len(p_disfluent) and p_disfluent[right] >= config.shoulder_threshold:
            right += 1
        expanded.append((left, right))

    merged = _merge_regions(expanded, hop_sec=hop_sec, merge_gap_sec=config.merge_gap_sec)
    min_frames = _min_support_frames(config.min_support_sec, hop_sec)
    return [(start, end) for start, end in merged if end - start >= min_frames]


def _candidate_from_region(
    region: tuple[int, int],
    p_disfluent: np.ndarray,
    type_probs: np.ndarray,
    centers: np.ndarray,
    *,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None,
    suppressed: set[str],
) -> dict:
    start, end = region
    local_p = p_disfluent[start:end]
    peak_frame = start + int(np.argmax(local_p))
    peak_p = float(p_disfluent[peak_frame])

    local_types = type_probs[start:end]
    type_at_peak_values = type_probs[peak_frame]
    type_max_values = local_types.max(axis=0)
    type_mean_values = local_types.mean(axis=0)
    type_at_peak = _type_dict(type_at_peak_values)
    type_max = _type_dict(type_max_values)
    type_mean = _type_dict(type_mean_values)

    type_evidence_items = [
        _type_evidence_item(label, type_at_peak, type_max, type_mean)
        for label in DISFLUENCY_TYPE_LABELS
    ]

    center_start = max(0.0, float(centers[start] - hop_sec / 2.0))
    center_end = float(centers[end - 1] + hop_sec / 2.0)
    full_start = float(start * hop_sec)
    full_end = float((end - 1) * hop_sec + window_sec)
    if audio_duration_sec is not None:
        center_end = min(center_end, float(audio_duration_sec))
        full_end = min(full_end, float(audio_duration_sec))

    return {
        "support_start_frame": int(start),
        "support_end_frame": int(end),
        "peak_frame": int(peak_frame),
        "peak_time_sec": float(centers[peak_frame]),
        "peak_p_disfluent": peak_p,
        "mean_p_disfluent": float(local_p.mean()),
        "center_start_sec": center_start,
        "center_end_sec": center_end,
        "full_start_sec": full_start,
        "full_end_sec": full_end,
        "type_at_peak": type_at_peak,
        "type_max": type_max,
        "type_mean": type_mean,
        "type_evidence_items": type_evidence_items,
        "suppressed": suppressed,
    }


def _event_from_candidate(
    candidate: dict,
    *,
    label: str,
    event_index: int,
    producer_id: str,
    thresholds: dict,
) -> Event:
    normalized_label = LABEL_TO_EVENT_LABEL[label]
    event_id = f"{producer_id}.{EVENT_TYPE}.{event_index:06d}"
    start_sec = float(candidate["center_start_sec"])
    end_sec = float(candidate["center_end_sec"])
    return Event(
        event_id=event_id,
        producer_id=producer_id,
        task=TASK,
        event_type=EVENT_TYPE,
        label=normalized_label,
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=float(end_sec - start_sec),
        source_track_ids=(FLUENCY_TRACK_ID, TYPE_TRACK_ID),
        score=float(candidate["peak_p_disfluent"]),
        score_name="probability",
        evidence={
            "peak_time_sec": float(candidate["peak_time_sec"]),
            "peak_p_disfluent": float(candidate["peak_p_disfluent"]),
            "mean_p_disfluent": float(candidate["mean_p_disfluent"]),
            "label_source": "type_at_peak_non_suppressed",
            "active_types": candidate["active_types"],
            "suppressed_active_types": candidate["suppressed_active_types"],
            "type_at_peak": candidate["type_at_peak"],
            "type_max": candidate["type_max"],
            "type_mean": candidate["type_mean"],
        },
        extra={
            "support_start_frame": int(candidate["support_start_frame"]),
            "support_end_frame": int(candidate["support_end_frame"]),
            "peak_frame": int(candidate["peak_frame"]),
            "center_support_bounds": {
                "start_sec": start_sec,
                "end_sec": end_sec,
            },
            "full_receptive_window_bounds": {
                "start_sec": float(candidate["full_start_sec"]),
                "end_sec": float(candidate["full_end_sec"]),
            },
            "model_label_order": {
                "fluency": FLUENCY_LABELS,
                "disfluency_type": DISFLUENCY_TYPE_LABELS,
            },
            "thresholds": thresholds,
            "window_semantics": WINDOW_SEMANTICS,
        },
    )


def _select_label(active_types: Sequence[dict]) -> str:
    best_idx = 0
    best_key = (-math.inf, 0)
    for item in active_types:
        label_order = DISFLUENCY_TYPE_LABELS.index(item["name"])
        key = (float(item["at_peak"]), -label_order)
        if key > best_key:
            best_key = key
            best_idx = label_order
    return DISFLUENCY_TYPE_LABELS[best_idx]


def _type_evidence_item(
    label: str,
    type_at_peak: dict[str, float],
    type_max: dict[str, float],
    type_mean: dict[str, float],
) -> dict:
    return {
        "name": label,
        "at_peak": float(type_at_peak[label]),
        "max": float(type_max[label]),
        "mean": float(type_mean[label]),
    }


def _type_dict(values: np.ndarray) -> dict[str, float]:
    return {
        label: float(values[i])
        for i, label in enumerate(DISFLUENCY_TYPE_LABELS)
    }


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def _merge_regions(
    regions: Sequence[tuple[int, int]],
    *,
    hop_sec: float,
    merge_gap_sec: float,
) -> list[tuple[int, int]]:
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        gap_sec = max(0, start - prev_end) * hop_sec
        if gap_sec <= merge_gap_sec + 1e-12:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _min_support_frames(min_support_sec: float, hop_sec: float) -> int:
    return max(1, int(math.ceil(min_support_sec / hop_sec)))


def _frame_centers(n_frames: int, hop_sec: float, window_sec: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) * hop_sec + window_sec / 2.0


def _validated_logits(
    fluency_logits,
    disfluency_type_logits,
    *,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    fluency = _as_array(fluency_logits)
    types = _as_array(disfluency_type_logits)
    if fluency.ndim != 2 or fluency.shape[1] != len(FLUENCY_LABELS):
        raise ValueError(
            f"fluency_logits must have shape [frames, 2], got {fluency.shape}"
        )
    if types.ndim != 2 or types.shape[1] != len(DISFLUENCY_TYPE_LABELS):
        raise ValueError(
            f"disfluency_type_logits must have shape [frames, 5], got {types.shape}"
        )
    if fluency.shape[0] != types.shape[0]:
        raise ValueError(
            "fluency_logits and disfluency_type_logits must have the same frame count"
        )
    if not np.isfinite(fluency).all() or not np.isfinite(types).all():
        raise ValueError("logits must contain only finite values")
    if hop_sec <= 0.0 or window_sec <= 0.0:
        raise ValueError("hop_sec and window_sec must be positive")
    if audio_duration_sec is not None:
        if audio_duration_sec <= 0.0:
            raise ValueError("audio_duration_sec must be positive when provided")
        if fluency.shape[0] > 0:
            last_window_end = (fluency.shape[0] - 1) * hop_sec + window_sec
            if last_window_end > audio_duration_sec + _AUDIO_DURATION_TOLERANCE_SEC:
                raise ValueError(
                    "logit grid extends beyond audio duration: "
                    f"last_window_end={last_window_end:.6f}, "
                    f"audio_duration_sec={audio_duration_sec:.6f}"
                )
    return fluency, types


def _as_array(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
    return np.asarray(values, dtype=np.float64)


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = [
    "DEFAULT_PRODUCER_ID",
    "DEFAULT_SOURCE_MODEL",
    "EVENT_TYPE",
    "FLUENCY_TRACK_ID",
    "TYPE_TRACK_ID",
    "WINDOW_SEMANTICS",
    "extract_events",
    "make_producer_run",
    "produce_disfluency_events",
    "tracks_from_logits",
]
