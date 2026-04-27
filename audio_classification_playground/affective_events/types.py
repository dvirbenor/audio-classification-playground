"""Typed containers for the affective-events pipeline.

Inputs are plain numpy arrays plus the metadata required to interpret them
(hop/window for signals, intervals for VAD). Keeping the metadata bundled
prevents threading positional arguments through every function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Signal:
    """A 1-D continuous prediction sequence on a regular hop grid.

    Frame ``i`` summarizes audio in ``[i * hop_sec, i * hop_sec + window_sec]``;
    its center timestamp is ``i * hop_sec + window_sec / 2``.
    """

    name: str
    values: np.ndarray
    hop_sec: float
    window_sec: float

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                f"Signal '{self.name}' must be 1-D, got shape {self.values.shape}"
            )
        if self.hop_sec <= 0 or self.window_sec <= 0:
            raise ValueError(f"Signal '{self.name}' has non-positive hop/window")

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration_sec(self) -> float:
        return (self.n_frames - 1) * self.hop_sec + self.window_sec

    def frame_audio_extent(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-frame ``(starts, ends)`` of the audio each frame summarizes."""
        starts = np.arange(self.n_frames, dtype=np.float64) * self.hop_sec
        ends = starts + self.window_sec
        return starts, ends

    def frame_centers(self) -> np.ndarray:
        return np.arange(self.n_frames, dtype=np.float64) * self.hop_sec + 0.5 * self.window_sec


@dataclass(frozen=True)
class Vad:
    """Voice-activity intervals in seconds. Always sorted and non-overlapping.

    The canonical representation. Use the classmethod constructors for other
    upstream forms (Silero output, per-frame probabilities).
    """

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
        """Convert Silero VAD output (sample indices) to seconds."""
        ivs = tuple((d["start"] / sample_rate, d["end"] / sample_rate) for d in timestamps)
        return cls(intervals=ivs)

    @classmethod
    def from_intervals_samples(
        cls, intervals_samples: Iterable[tuple[int, int]], sample_rate: int
    ) -> "Vad":
        ivs = tuple((s / sample_rate, e / sample_rate) for s, e in intervals_samples)
        return cls(intervals=ivs)

    @classmethod
    def from_frame_probs(
        cls,
        probs: np.ndarray,
        hop_sec: float,
        window_sec: float | None = None,
        threshold: float = 0.5,
    ) -> "Vad":
        """Threshold per-frame VAD probabilities to intervals.

        ``hop_sec`` is the VAD frame hop. If ``window_sec`` is given each speech
        frame contributes ``[i*hop, i*hop + window]``; otherwise frames contribute
        ``[i*hop, (i+1)*hop]``. After construction the intervals are merged.
        """
        speech = probs >= threshold
        if not speech.any():
            return cls(intervals=())
        edges = np.diff(speech.astype(np.int8), prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        w = window_sec if window_sec is not None else hop_sec
        ivs = tuple((s * hop_sec, (e - 1) * hop_sec + w) for s, e in zip(starts, ends))
        return cls(intervals=ivs)

    def merged(self, max_gap_sec: float, min_duration_sec: float = 0.0) -> "Vad":
        """Bridge gaps ``<= max_gap_sec`` and drop intervals shorter than ``min_duration_sec``."""
        if not self.intervals:
            return self
        out: list[list[float]] = [list(self.intervals[0])]
        for s, e in self.intervals[1:]:
            if s - out[-1][1] <= max_gap_sec:
                out[-1][1] = e
            else:
                out.append([s, e])
        kept = tuple((s, e) for s, e in out if (e - s) >= min_duration_sec)
        return Vad(intervals=kept)

    @property
    def starts(self) -> np.ndarray:
        return np.array([s for s, _ in self.intervals], dtype=np.float64)

    @property
    def ends(self) -> np.ndarray:
        return np.array([e for _, e in self.intervals], dtype=np.float64)


@dataclass(frozen=True)
class Block:
    """A merged speech analysis block."""

    block_id: int
    start_sec: float
    end_sec: float
    gap_before_sec: float
    gap_after_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class Event:
    """A single emotional event on one signal (or a fused parent).

    For leaf events ``signal_name`` is one of the input signal names. For
    parents produced by fusion ``signal_name`` is ``"episode:<name>"`` (within
    one signal across multiple blocks) or ``"joint"`` (across signals).
    """

    event_id: str
    signal_name: str
    event_type: str

    start_sec: float
    end_sec: float
    duration_sec: float

    block_ids: tuple[int, ...]

    delta: float
    delta_z: float
    direction: str

    baseline_value: float
    baseline_context_speech_sec: float
    baseline_source: str  # "local" | "global"

    mean_speech_coverage: float
    near_block_start: bool
    near_block_end: bool

    strength: float
    confidence: float
    confidence_components: dict[str, float] = field(default_factory=dict)

    review_audio_start_sec: float = 0.0
    review_audio_end_sec: float = 0.0

    children: tuple[str, ...] = ()
    parent_id: str | None = None

    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)
