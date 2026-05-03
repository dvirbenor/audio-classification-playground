"""Generic acoustic event and prediction-track schema.

The review app consumes this schema rather than any producer-specific event
shape. Producers are free to use very different extraction logic, but their
review artifacts should converge here: events, producer runs, and plottable
evidence tracks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np


SCORE_NAMES = frozenset({
    "peak_z",
    "probability",
    "confidence",
    "prominence_z",
    "logit",
    "margin",
})

GRID_RENDERERS = frozenset({"line", "probability", "multi_probability"})
MARKER_RENDERERS = frozenset({"marker"})


@dataclass(frozen=True)
class ProducerRun:
    """Metadata and producer-scoped outputs for one model/extractor run."""

    producer_id: str
    task: str
    source_model: str
    config: dict = field(default_factory=dict)
    config_hash: str = ""
    outputs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.producer_id:
            raise ValueError("ProducerRun.producer_id must be non-empty")
        if not self.task:
            raise ValueError("ProducerRun.task must be non-empty")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Event:
    """Canonical acoustic event consumed by the review application."""

    event_id: str
    producer_id: str
    task: str
    event_type: str
    label: str

    start_sec: float
    end_sec: float
    duration_sec: float

    source_track_ids: tuple[str, ...]
    score: float
    score_name: str
    direction: str | None = None

    parent_id: str | None = None
    children: tuple[str, ...] = ()
    evidence: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("Event.event_id must be non-empty")
        if not self.producer_id:
            raise ValueError(f"Event {self.event_id!r} has no producer_id")
        if not self.task:
            raise ValueError(f"Event {self.event_id!r} has no task")
        if self.end_sec < self.start_sec:
            raise ValueError(f"Event {self.event_id!r} ends before it starts")
        if self.score_name not in SCORE_NAMES:
            raise ValueError(
                f"Event {self.event_id!r} uses unknown score_name {self.score_name!r}"
            )
        object.__setattr__(self, "source_track_ids", tuple(self.source_track_ids))
        object.__setattr__(self, "children", tuple(self.children))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MarkerItem:
    """Minimum renderable item for sparse marker-style evidence."""

    start_sec: float
    end_sec: float | None
    label: str
    score: float | None = None
    payload: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_sec is not None and self.end_sec < self.start_sec:
            raise ValueError("MarkerItem.end_sec must be >= start_sec")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RegularGridTrack:
    """A regularly-sampled model-output track."""

    track_id: str
    producer_id: str
    task: str
    name: str
    value_type: str
    renderer: Literal["line", "probability", "multi_probability"]
    values: np.ndarray
    hop_sec: float
    window_sec: float
    channels: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)
    kind: Literal["regular"] = "regular"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim not in (1, 2):
            raise ValueError(
                f"RegularGridTrack {self.track_id!r} must be 1-D or 2-D, got {values.shape}"
            )
        if self.renderer not in GRID_RENDERERS:
            raise ValueError(f"Unsupported grid renderer {self.renderer!r}")
        if self.hop_sec <= 0 or self.window_sec <= 0:
            raise ValueError(f"Track {self.track_id!r} has non-positive hop/window")
        channels = tuple(self.channels)
        if self.renderer == "multi_probability" and values.ndim != 2:
            raise ValueError(
                f"Track {self.track_id!r} uses multi_probability but is not 2-D"
            )
        if self.renderer != "multi_probability" and values.ndim != 1:
            raise ValueError(
                f"Track {self.track_id!r} renderer {self.renderer!r} requires 1-D values"
            )
        if values.ndim == 2 and not channels:
            raise ValueError(f"Track {self.track_id!r} requires channel names")
        if values.ndim == 2 and len(channels) != values.shape[1]:
            raise ValueError(
                f"Track {self.track_id!r} has {values.shape[1]} columns but "
                f"{len(channels)} channel names"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "channels", channels)

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    def as_meta(self) -> dict:
        return {
            "kind": self.kind,
            "track_id": self.track_id,
            "producer_id": self.producer_id,
            "task": self.task,
            "name": self.name,
            "value_type": self.value_type,
            "renderer": self.renderer,
            "hop_sec": float(self.hop_sec),
            "window_sec": float(self.window_sec),
            "n_frames": self.n_frames,
            "shape": list(self.values.shape),
            "channels": list(self.channels),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class MarkerTrack:
    """Sparse timestamp/span evidence that is not on a regular hop grid."""

    track_id: str
    producer_id: str
    task: str
    name: str
    renderer: Literal["marker"]
    items: tuple[MarkerItem | dict, ...]
    meta: dict = field(default_factory=dict)
    kind: Literal["marker"] = "marker"

    def __post_init__(self) -> None:
        if self.renderer not in MARKER_RENDERERS:
            raise ValueError(f"Unsupported marker renderer {self.renderer!r}")
        items: list[MarkerItem] = []
        for item in self.items:
            items.append(item if isinstance(item, MarkerItem) else MarkerItem(**item))
        object.__setattr__(self, "items", tuple(items))

    def as_meta(self) -> dict:
        return {
            "kind": self.kind,
            "track_id": self.track_id,
            "producer_id": self.producer_id,
            "task": self.task,
            "name": self.name,
            "renderer": self.renderer,
            "items": [item.as_dict() for item in self.items],
            "meta": dict(self.meta),
        }


PredictionTrack = RegularGridTrack | MarkerTrack


def track_meta(track: PredictionTrack) -> dict:
    return track.as_meta()
