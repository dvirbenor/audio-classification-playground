"""Pre-computed peak-envelope for fast waveform rendering.

The browser cannot render a 75-minute float32 sample stream. We decimate
to a few thousand ``(min, max)`` pairs once per audio file and cache the
result on disk next to the session.

For windowed (high-res) requests, ``soundfile`` is used for efficient
random-access seeking without decoding the entire file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_DEFAULT_N_PEAKS = 8000
_MAX_WINDOW_PEAKS = 4000


def _bin_min_max(y: np.ndarray, n_peaks: int):
    """Reduce a 1-D float array to *n_peaks* (min, max) pairs."""
    n = y.size
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    bin_size = max(1, n // n_peaks)
    n_full = (n // bin_size) * bin_size
    if n_full < n:
        pad = bin_size - (n - n_full)
        y_pad = np.concatenate([y[:n_full + bin_size - pad], np.zeros(pad, dtype=y.dtype)])
        reshaped = y_pad.reshape(-1, bin_size)
    else:
        reshaped = y[:n_full].reshape(-1, bin_size)
    return (
        reshaped.min(axis=1).astype(np.float32),
        reshaped.max(axis=1).astype(np.float32),
    )


def compute_peaks(audio_path: str | Path, n_peaks: int = _DEFAULT_N_PEAKS) -> dict:
    """Decode the audio once and reduce to a min/max envelope.

    Returns a JSON-serializable dict with ``min``, ``max`` (lists of length
    ``n_peaks_actual``), ``sample_rate``, ``duration_sec``, and the bin size.
    """
    import librosa

    audio_path = Path(audio_path)
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    n = int(y.size)
    if n == 0:
        return {
            "min": [], "max": [],
            "sample_rate": int(sr), "duration_sec": 0.0,
            "n_peaks": 0, "bin_size": 0,
        }
    mins, maxs = _bin_min_max(y, n_peaks)
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "sample_rate": int(sr),
        "duration_sec": float(n) / float(sr),
        "n_peaks": int(mins.size),
        "bin_size": max(1, n // n_peaks),
    }


def compute_peaks_window(
    audio_path: str | Path,
    t0_sec: float,
    t1_sec: float,
    n_peaks: int = 2000,
) -> dict:
    """Compute min/max envelope for a time window using soundfile seeking."""
    import soundfile as sf

    n_peaks = min(n_peaks, _MAX_WINDOW_PEAKS)
    audio_path = Path(audio_path)

    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        total_frames = len(f)
        start_sample = max(0, int(t0_sec * sr))
        end_sample = min(int(t1_sec * sr), total_frames)
        if start_sample >= end_sample:
            return {
                "min": [], "max": [],
                "t0_sec": t0_sec, "t1_sec": t1_sec,
                "duration_sec": 0.0,
                "n_peaks": 0, "sample_rate": sr, "bin_sec": 0.0,
            }
        f.seek(start_sample)
        y = f.read(end_sample - start_sample, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

    actual_t0 = start_sample / sr
    actual_t1 = end_sample / sr
    duration = actual_t1 - actual_t0
    mins, maxs = _bin_min_max(y, n_peaks)
    actual_n = int(mins.size)

    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "t0_sec": actual_t0,
        "t1_sec": actual_t1,
        "duration_sec": duration,
        "n_peaks": actual_n,
        "sample_rate": sr,
        "bin_sec": duration / actual_n if actual_n else 0.0,
    }


def cached_peaks(audio_path: str | Path, cache_path: str | Path, n_peaks: int = _DEFAULT_N_PEAKS) -> dict:
    """Compute peaks if the cache is missing/stale; otherwise return the cache."""
    audio_path = Path(audio_path)
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= audio_path.stat().st_mtime:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("n_peaks", 0) > 0:
                return cached
        except (OSError, json.JSONDecodeError):
            pass  # fall through to recompute
    peaks = compute_peaks(audio_path, n_peaks=n_peaks)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(peaks, f)
    return peaks
