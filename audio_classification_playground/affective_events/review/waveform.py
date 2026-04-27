"""Pre-computed peak-envelope for fast waveform rendering.

The browser cannot render a 75-minute float32 sample stream. We decimate
to a few thousand ``(min, max)`` pairs once per audio file and cache the
result on disk next to the session.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


_DEFAULT_N_PEAKS = 8000


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
    bin_size = max(1, n // n_peaks)
    n_full = (n // bin_size) * bin_size
    if n_full < n:
        # Pad the trailing samples into a final bin so we don't lose them.
        pad = bin_size - (n - n_full)
        y_pad = np.concatenate([y[:n_full + bin_size - pad], np.zeros(pad, dtype=y.dtype)])
        reshaped = y_pad.reshape(-1, bin_size)
    else:
        reshaped = y[:n_full].reshape(-1, bin_size)
    mins = reshaped.min(axis=1).astype(np.float32)
    maxs = reshaped.max(axis=1).astype(np.float32)
    return {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "sample_rate": int(sr),
        "duration_sec": float(n) / float(sr),
        "n_peaks": int(mins.size),
        "bin_size": int(bin_size),
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
