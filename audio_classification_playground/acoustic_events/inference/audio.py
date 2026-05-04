"""Shared audio loading and framing utilities for inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import SAMPLE_RATE, decoded_audio_sha256, sanitize_for_filename


@dataclass(frozen=True)
class AudioData:
    path: Path
    recording_id: str
    samples: np.ndarray
    sample_rate: int
    duration_sec: float
    audio_sha256: str


def load_audio(
    audio_path: str | Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    recording_id: str | None = None,
) -> AudioData:
    """Decode audio to mono float32 at the inference sample rate."""
    import librosa

    path = Path(audio_path).resolve()
    samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    rid = sanitize_for_filename(recording_id or path.stem)
    return AudioData(
        path=path,
        recording_id=rid,
        samples=samples,
        sample_rate=int(sample_rate),
        duration_sec=float(len(samples)) / float(sample_rate),
        audio_sha256=decoded_audio_sha256(samples),
    )


def frame_audio(
    samples: np.ndarray,
    *,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> np.ndarray:
    """Return a [frames, window_samples] strided view with tail padding."""
    if window_sec <= 0.0 or hop_sec <= 0.0:
        raise ValueError("window_sec and hop_sec must be positive")
    window_samples = int(round(float(window_sec) * int(sample_rate)))
    hop_samples = int(round(float(hop_sec) * int(sample_rate)))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_sec and hop_sec must produce positive sample counts")

    audio = np.ascontiguousarray(samples, dtype=np.float32)
    if len(audio) < window_samples:
        pad_needed = window_samples - len(audio)
    else:
        remainder = (len(audio) - window_samples) % hop_samples
        pad_needed = 0 if remainder == 0 else hop_samples - remainder
    if pad_needed:
        audio = np.pad(audio, (0, pad_needed), mode="constant")

    n_frames = 1 + (len(audio) - window_samples) // hop_samples
    stride = audio.strides[0]
    return np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, window_samples),
        strides=(hop_samples * stride, stride),
        writeable=False,
    )
