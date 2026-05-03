"""Typed containers for the canonical affect producer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ...schema import Event, ProducerRun, RegularGridTrack

@dataclass(frozen=True)
class Signal:
    """A 1-D continuous prediction sequence on a regular hop grid."""

    name: str
    values: np.ndarray
    hop_sec: float
    window_sec: float

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"Signal '{self.name}' must be 1-D, got shape {values.shape}"
            )
        if self.hop_sec <= 0 or self.window_sec <= 0:
            raise ValueError(f"Signal '{self.name}' has non-positive hop/window")
        object.__setattr__(self, "values", values)

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration_sec(self) -> float:
        return (self.n_frames - 1) * self.hop_sec + self.window_sec

    def frame_centers(self) -> np.ndarray:
        return (
            np.arange(self.n_frames, dtype=np.float64) * self.hop_sec
            + 0.5 * self.window_sec
        )


@dataclass(frozen=True)
class Vad:
    """Voice-activity intervals in seconds, sorted and non-overlapping."""

    intervals: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        merged: list[list[float]] = []
        for s, e in sorted((float(a), float(b)) for a, b in self.intervals):
            if e <= s:
                raise ValueError(f"Vad interval has non-positive duration: ({s}, {e})")
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        object.__setattr__(self, "intervals", tuple((s, e) for s, e in merged))

    @classmethod
    def from_silero(cls, timestamps: Sequence[dict], sample_rate: int) -> "Vad":
        return cls(
            intervals=tuple(
                (d["start"] / sample_rate, d["end"] / sample_rate)
                for d in timestamps
            )
        )

    @classmethod
    def from_intervals_samples(
        cls, intervals_samples: Iterable[tuple[int, int]], sample_rate: int
    ) -> "Vad":
        return cls(intervals=tuple((s / sample_rate, e / sample_rate) for s, e in intervals_samples))

    @classmethod
    def from_frame_probs(
        cls,
        probs: np.ndarray,
        hop_sec: float,
        window_sec: float | None = None,
        threshold: float = 0.5,
    ) -> "Vad":
        speech = probs >= threshold
        if not speech.any():
            return cls(intervals=())
        edges = np.diff(speech.astype(np.int8), prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        width = window_sec if window_sec is not None else hop_sec
        return cls(
            intervals=tuple(
                (s * hop_sec, (e - 1) * hop_sec + width)
                for s, e in zip(starts, ends)
            )
        )

    def merged(self, max_gap_sec: float, min_duration_sec: float = 0.0) -> "Vad":
        if not self.intervals:
            return self
        out: list[list[float]] = [list(self.intervals[0])]
        for s, e in self.intervals[1:]:
            if s - out[-1][1] <= max_gap_sec:
                out[-1][1] = e
            else:
                out.append([s, e])
        return Vad(intervals=tuple((s, e) for s, e in out if e - s >= min_duration_sec))


@dataclass(frozen=True)
class Block:
    block_id: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def center_sec(self) -> float:
        return 0.5 * (self.start_sec + self.end_sec)


__all__ = [
    "Signal",
    "Vad",
    "Block",
    "Event",
    "ProducerRun",
    "RegularGridTrack",
]
