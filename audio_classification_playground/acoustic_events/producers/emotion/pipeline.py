"""Categorical emotion event extraction from framewise probabilities."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from typing import Sequence

import numpy as np
import pandas as pd

from ...schema import Event, ProducerRun, RegularGridTrack
from .config import CANONICAL_CHANNELS, Config


DEFAULT_PRODUCER_ID = "emotion.categorical.v1"
DEFAULT_SOURCE_MODEL = "iic/emotion2vec_plus_large"
TRACK_ID = "emotion.categorical.probabilities"

_LABEL_ALIASES = {
    "angry": "anger",
    "anger": "anger",
    "disgusted": "disgust",
    "disgust": "disgust",
    "fearful": "fear",
    "fear": "fear",
    "happy": "happiness",
    "happiness": "happiness",
    "neutral": "neutral",
    "other": "other",
    "sad": "sadness",
    "sadness": "sadness",
    "surprised": "surprise",
    "surprise": "surprise",
    "<unk>": "other",
    "unk": "other",
    "unknown": "other",
}


def run_from_probabilities(
    probabilities,
    labels: Sequence[str],
    *,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None = None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
    config: Config | None = None,
    producer_id: str = DEFAULT_PRODUCER_ID,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> tuple[ProducerRun, list[RegularGridTrack], list[Event]]:
    """Build a producer run, evidence track, and categorical emotion events.

    The probabilities are expected to be already-normalized model scores. Event
    detection uses raw probabilities; only the boolean support mask is closed to
    bridge short framewise holes inside speech segments.
    """
    cfg = config or Config.balanced()
    canonical_probs, canonical_labels = canonicalize_probabilities(
        probabilities, labels, config=cfg
    )
    _validate_timing(canonical_probs.shape[0], hop_sec, window_sec, audio_duration_sec)

    centers = _frame_centers(canonical_probs.shape[0], hop_sec, window_sec)
    speech_mask = _speech_mask(centers, vad_intervals)
    valid_mask = speech_mask if vad_intervals is not None else np.ones_like(speech_mask)
    tracks = tracks_from_probabilities(
        canonical_probs,
        hop_sec=hop_sec,
        window_sec=window_sec,
        producer_id=producer_id,
        source_model=source_model,
    )

    thresholds = _resolved_thresholds(canonical_probs, valid_mask, cfg)
    events = extract_events(
        canonical_probs,
        hop_sec=hop_sec,
        window_sec=window_sec,
        audio_duration_sec=audio_duration_sec,
        vad_intervals=vad_intervals,
        config=cfg,
        producer_id=producer_id,
        thresholds=thresholds,
    )
    run = producer_run(
        cfg,
        probabilities=canonical_probs,
        labels=canonical_labels,
        raw_labels=labels,
        valid_mask=valid_mask,
        thresholds=thresholds,
        events=events,
        vad_intervals=vad_intervals,
        producer_id=producer_id,
        source_model=source_model,
    )
    return run, tracks, events


def canonicalize_probabilities(
    probabilities,
    labels: Sequence[str],
    *,
    config: Config | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Map raw model labels into the producer's stable channel order."""
    cfg = config or Config.balanced()
    probs = _as_array(probabilities)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2-D [frames, classes], got {probs.shape}")
    if probs.shape[1] != len(labels):
        raise ValueError(
            f"probabilities has {probs.shape[1]} columns but {len(labels)} labels"
        )
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must contain only finite values")
    if (probs < -1e-12).any():
        raise ValueError("probabilities must be non-negative")

    canonical = np.zeros((probs.shape[0], len(CANONICAL_CHANNELS)), dtype=np.float64)
    for src_i, raw_label in enumerate(labels):
        label = normalize_label(raw_label)
        if label not in CANONICAL_CHANNELS:
            raise ValueError(f"Unsupported emotion label {raw_label!r} -> {label!r}")
        canonical[:, CANONICAL_CHANNELS.index(label)] += probs[:, src_i]

    row_sums = canonical.sum(axis=1)
    if probs.shape[0] and not np.allclose(
        row_sums, 1.0, atol=cfg.probability_sum_tolerance, rtol=0.0
    ):
        raise ValueError(
            "probabilities must sum to 1.0 per frame after label folding; "
            f"observed min={row_sums.min():.6f}, max={row_sums.max():.6f}"
        )
    canonical = np.clip(canonical, 0.0, 1.0)
    return canonical, CANONICAL_CHANNELS


def normalize_label(label: str) -> str:
    """Normalize emotion2vec labels, including Chinese/English forms."""
    text = str(label).strip()
    english = text.split("/")[-1].strip().lower()
    return _LABEL_ALIASES.get(english, english)


def _as_array(values) -> np.ndarray:
    """Convert numpy-like or torch-like values to a float64 ndarray."""
    if hasattr(values, "detach"):
        values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
    return np.asarray(values, dtype=np.float64)


def tracks_from_probabilities(
    probabilities,
    *,
    hop_sec: float,
    window_sec: float,
    producer_id: str = DEFAULT_PRODUCER_ID,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> list[RegularGridTrack]:
    values = np.asarray(probabilities, dtype=np.float64)
    return [
        RegularGridTrack(
            track_id=TRACK_ID,
            producer_id=producer_id,
            task="emotion",
            name="categorical emotion probabilities",
            value_type="probability",
            renderer="multi_probability",
            values=values,
            hop_sec=hop_sec,
            window_sec=window_sec,
            channels=CANONICAL_CHANNELS,
            meta={
                "source_model": source_model,
                "window_semantics": "frame summarizes [i*hop, i*hop + window]",
                "event_boundary_semantics": "center_support",
            },
        )
    ]


def extract_events(
    probabilities,
    *,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None = None,
    vad_intervals: Sequence[tuple[float, float]] | None = None,
    config: Config | None = None,
    producer_id: str = DEFAULT_PRODUCER_ID,
    thresholds: dict[str, float] | None = None,
) -> list[Event]:
    cfg = config or Config.balanced()
    probs, _ = canonicalize_probabilities(probabilities, CANONICAL_CHANNELS, config=cfg)
    centers = _frame_centers(probs.shape[0], hop_sec, window_sec)
    speech_mask = _speech_mask(centers, vad_intervals)
    valid_mask = speech_mask if vad_intervals is not None else np.ones_like(speech_mask)
    if not valid_mask.any():
        return []

    resolved = thresholds if thresholds is not None else _resolved_thresholds(probs, valid_mask, cfg)
    top1_idx = probs.argmax(axis=1)
    background_idx = [CANONICAL_CHANNELS.index(label) for label in cfg.background_labels]
    background_probs = probs[:, background_idx]
    best_bg_local = background_probs.argmax(axis=1)
    best_bg_idx = np.asarray(background_idx)[best_bg_local]
    best_bg_prob = background_probs[np.arange(probs.shape[0]), best_bg_local]
    segments = _speech_segments(speech_mask if vad_intervals is not None else np.ones_like(speech_mask))

    candidates: list[dict] = []
    for label in cfg.event_labels:
        if label not in resolved:
            continue
        class_i = CANONICAL_CHANNELS.index(label)
        class_prob = probs[:, class_i]
        raw_support = (
            (top1_idx == class_i)
            & (class_prob >= resolved[label])
            & ((class_prob - best_bg_prob) >= cfg.background_margin)
        )
        support = raw_support & speech_mask
        closed_support = _close_support_by_segments(
            support,
            segments,
            hop_sec=hop_sec,
            close_gap_sec=cfg.support_close_gap_sec,
        )
        candidates.extend(
            _candidates_for_label(
                label,
                class_i,
                class_prob,
                raw_support,
                closed_support,
                centers,
                probs,
                best_bg_idx,
                best_bg_prob,
                resolved[label],
                hop_sec,
                window_sec,
                audio_duration_sec,
                cfg,
            )
        )

    candidates.sort(key=lambda c: (c["start_sec"], c["label"]))
    return [
        _event_from_candidate(candidate, n, producer_id=producer_id)
        for n, candidate in enumerate(candidates)
    ]


def producer_run(
    config: Config | dict | None = None,
    *,
    probabilities,
    labels: Sequence[str],
    raw_labels: Sequence[str],
    valid_mask: np.ndarray,
    thresholds: dict[str, float],
    events: Sequence[Event],
    vad_intervals: Sequence[tuple[float, float]] | None,
    producer_id: str = DEFAULT_PRODUCER_ID,
    source_model: str = DEFAULT_SOURCE_MODEL,
) -> ProducerRun:
    cfg = config or Config.balanced()
    cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    probs = np.asarray(probabilities, dtype=np.float64)
    outputs = {
        "class_occupancy": _class_occupancy(probs, valid_mask, thresholds),
        "resolved_thresholds": {k: float(v) for k, v in thresholds.items()},
        "event_summary": _event_summary(events),
        "label_mapping": {
            "raw_labels": [str(label) for label in raw_labels],
            "canonical_labels": list(labels),
        },
        "vad": {
            "provided": vad_intervals is not None,
            "valid_frame_count": int(valid_mask.sum()),
            "total_frame_count": int(valid_mask.size),
            "thresholds_computed_on": "speech_frames" if vad_intervals is not None else "all_frames",
            "non_speech_is_hard_barrier": True,
        },
    }
    return ProducerRun(
        producer_id=producer_id,
        task="emotion",
        source_model=source_model,
        config=cfg_dict,
        config_hash=_config_hash(cfg_dict),
        outputs=outputs,
    )


def to_dataframe(events: Sequence[Event]) -> pd.DataFrame:
    rows = [event.as_dict() for event in events]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["start_sec", "task", "label", "event_id"]
    ).reset_index(drop=True)


def _validate_timing(
    n_frames: int,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None,
) -> None:
    if hop_sec <= 0 or window_sec <= 0:
        raise ValueError("hop_sec and window_sec must be positive")
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if audio_duration_sec is not None and audio_duration_sec < 0:
        raise ValueError("audio_duration_sec must be non-negative")


def _frame_centers(n_frames: int, hop_sec: float, window_sec: float) -> np.ndarray:
    return np.arange(n_frames, dtype=np.float64) * float(hop_sec) + 0.5 * float(window_sec)


def _speech_mask(
    centers: np.ndarray,
    vad_intervals: Sequence[tuple[float, float]] | None,
) -> np.ndarray:
    if vad_intervals is None:
        return np.ones(centers.shape[0], dtype=bool)
    mask = np.zeros(centers.shape[0], dtype=bool)
    for start, end in _normalize_intervals(vad_intervals):
        mask |= (centers >= start) & (centers <= end)
    return mask


def _normalize_intervals(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[list[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals):
        if end <= start:
            raise ValueError(f"VAD interval has non-positive duration: ({start}, {end})")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((s, e) for s, e in merged)


def _speech_segments(speech_mask: np.ndarray) -> list[tuple[int, int]]:
    return _true_runs(speech_mask)


def _resolved_thresholds(
    probs: np.ndarray,
    valid_mask: np.ndarray,
    config: Config,
) -> dict[str, float]:
    if not valid_mask.any():
        return {}
    thresholds: dict[str, float] = {}
    for label in config.event_labels:
        class_i = CANONICAL_CHANNELS.index(label)
        q = float(np.quantile(probs[valid_mask, class_i], config.class_quantile))
        thresholds[label] = float(max(config.absolute_min_probability, q))
    return thresholds


def _class_occupancy(
    probs: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, dict]:
    if not valid_mask.any():
        return {}
    top1_idx = probs.argmax(axis=1)
    occupancy: dict[str, dict] = {}
    for label in CANONICAL_CHANNELS:
        class_i = CANONICAL_CHANNELS.index(label)
        vals = probs[valid_mask, class_i]
        occupancy[label] = {
            "mean_probability": float(vals.mean()),
            "p50": float(np.quantile(vals, 0.50)),
            "p90": float(np.quantile(vals, 0.90)),
            "p95": float(np.quantile(vals, 0.95)),
            "p99": float(np.quantile(vals, 0.99)),
            "max": float(vals.max()),
            "top1_share": float((top1_idx[valid_mask] == class_i).mean()),
            "above_threshold_share": (
                float((vals >= thresholds[label]).mean()) if label in thresholds else 0.0
            ),
        }
    return occupancy


def _event_summary(events: Sequence[Event]) -> dict:
    by_label: dict[str, dict] = {}
    for event in events:
        row = by_label.setdefault(
            event.label,
            {"n_events": 0, "total_event_duration_sec": 0.0},
        )
        row["n_events"] += 1
        row["total_event_duration_sec"] += float(event.duration_sec)
    return {
        "n_events": len(events),
        "total_event_duration_sec": float(sum(e.duration_sec for e in events)),
        "by_label": by_label,
    }


def _close_support_by_segments(
    support: np.ndarray,
    segments: Sequence[tuple[int, int]],
    *,
    hop_sec: float,
    close_gap_sec: float,
) -> np.ndarray:
    closed = np.zeros_like(support, dtype=bool)
    max_gap_frames = int(np.floor(close_gap_sec / hop_sec + 1e-9))
    for start, end in segments:
        segment_support = support[start:end].copy()
        runs = _true_runs(segment_support)
        if not runs:
            continue
        for (prev_start, prev_end), (next_start, next_end) in zip(runs[:-1], runs[1:]):
            gap = next_start - prev_end
            if gap <= max_gap_frames:
                segment_support[prev_end:next_start] = True
        closed[start:end] = segment_support
    return closed


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _candidates_for_label(
    label: str,
    class_i: int,
    class_prob: np.ndarray,
    raw_support: np.ndarray,
    closed_support: np.ndarray,
    centers: np.ndarray,
    probs: np.ndarray,
    best_bg_idx: np.ndarray,
    best_bg_prob: np.ndarray,
    threshold: float,
    hop_sec: float,
    window_sec: float,
    audio_duration_sec: float | None,
    config: Config,
) -> list[dict]:
    candidates: list[dict] = []
    min_frames = max(1, int(np.ceil(config.min_duration_sec / hop_sec - 1e-9)))
    for start, end in _true_runs(closed_support):
        if end - start < min_frames:
            continue
        raw_inside = raw_support[start:end]
        if not raw_inside.any():
            continue
        raw_indices = np.where(raw_inside)[0] + start
        peak_frame = int(raw_indices[np.argmax(class_prob[raw_indices])])
        start_sec = float(centers[start] - 0.5 * hop_sec)
        end_sec = float(centers[end - 1] + 0.5 * hop_sec)
        if audio_duration_sec is not None:
            start_sec = max(0.0, min(start_sec, float(audio_duration_sec)))
            end_sec = max(0.0, min(end_sec, float(audio_duration_sec)))
        else:
            start_sec = max(0.0, start_sec)
        if end_sec < start_sec:
            continue
        top_classes = _top_classes(probs[peak_frame])
        candidates.append({
            "label": label,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": float(end_sec - start_sec),
            "score": float(class_prob[peak_frame]),
            "peak_frame": peak_frame,
            "peak_time_sec": float(centers[peak_frame]),
            "top_classes_at_peak": top_classes,
            "background_label_at_peak": CANONICAL_CHANNELS[int(best_bg_idx[peak_frame])],
            "background_probability_at_peak": float(best_bg_prob[peak_frame]),
            "margin_vs_background_at_peak": float(class_prob[peak_frame] - best_bg_prob[peak_frame]),
            "mean_probability": float(class_prob[raw_indices].mean()),
            "frame_start": int(start),
            "frame_end": int(end),
            "class_threshold": float(threshold),
            "raw_support_frame_count": int(raw_indices.size),
            "closed_support_frame_count": int(end - start),
            "bridge_frame_count": int((end - start) - raw_indices.size),
            "config": config,
            "hop_sec": float(hop_sec),
            "window_sec": float(window_sec),
        })
    return candidates


def _event_from_candidate(candidate: dict, n: int, *, producer_id: str) -> Event:
    cfg = candidate["config"]
    event_id = f"{producer_id}.categorical.{n:06d}"
    return Event(
        event_id=event_id,
        producer_id=producer_id,
        task="emotion",
        event_type="categorical",
        label=candidate["label"],
        start_sec=candidate["start_sec"],
        end_sec=candidate["end_sec"],
        duration_sec=candidate["duration_sec"],
        source_track_ids=(TRACK_ID,),
        score=candidate["score"],
        score_name="probability",
        evidence={
            "peak_time_sec": candidate["peak_time_sec"],
            "top_classes_at_peak": candidate["top_classes_at_peak"],
            "background_label_at_peak": candidate["background_label_at_peak"],
            "background_probability_at_peak": candidate["background_probability_at_peak"],
            "margin_vs_background_at_peak": candidate["margin_vs_background_at_peak"],
            "mean_probability": candidate["mean_probability"],
        },
        extra={
            "frame_start": candidate["frame_start"],
            "frame_end": candidate["frame_end"],
            "peak_frame": candidate["peak_frame"],
            "class_threshold": candidate["class_threshold"],
            "absolute_min_probability": cfg.absolute_min_probability,
            "class_quantile": cfg.class_quantile,
            "background_margin": cfg.background_margin,
            "min_duration_sec": cfg.min_duration_sec,
            "support_close_gap_sec": cfg.support_close_gap_sec,
            "raw_support_frame_count": candidate["raw_support_frame_count"],
            "closed_support_frame_count": candidate["closed_support_frame_count"],
            "bridge_frame_count": candidate["bridge_frame_count"],
            "boundary_mode": "center_support",
            "hop_sec": candidate["hop_sec"],
            "window_sec": candidate["window_sec"],
        },
    )


def _top_classes(row: np.ndarray, n: int = 5) -> dict[str, float]:
    order = np.argsort(row)[::-1][:n]
    return {CANONICAL_CHANNELS[int(i)]: float(row[int(i)]) for i in order}


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


__all__ = [
    "DEFAULT_PRODUCER_ID",
    "DEFAULT_SOURCE_MODEL",
    "TRACK_ID",
    "canonicalize_probabilities",
    "extract_events",
    "normalize_label",
    "producer_run",
    "run_from_probabilities",
    "to_dataframe",
    "tracks_from_probabilities",
]
