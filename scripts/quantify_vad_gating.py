#!/usr/bin/env python3
"""Quantify the compute headroom from VAD-gating inference.

The pipeline runs affect/disfluency/emotion on the *full timeline* (every
sliding window), even though Silero VAD intervals are already computed. If
events only matter on speech, gating inference to windows that overlap speech
would skip the rest. This measures the prize, per archive:

  * speech_fraction      = sum(VAD interval durations) / duration   (naive)
  * window_keep_fraction = fraction of the sliding-window grid that overlaps
                           >=1 speech interval                       (real proxy)

window_keep > speech_fraction because a W-second window with H-second hop
"activates" for any speech in [i*H, i*H + W], dilating each speech region by
~W on the left. Compute saving ~= 1 - window_keep_fraction.

Example:
    uv run python scripts/quantify_vad_gating.py \
        --index benchmark_audio/index.json \
        --json-out optimization_research/baseline_results/vad_gating_potential.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import (
    frame_audio_geometry,
    load_audio,
)
from audio_classification_playground.acoustic_events.inference.models import VadDetector
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    EMOTION_WINDOW_SEC,
    DEFAULT_HOP_SEC,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    DEFAULT_VAD_MIN_SPEECH_SEC,
    DEFAULT_VAD_MIN_SILENCE_SEC,
)

GRIDS = (
    ("affect", AFFECT_WINDOW_SEC),
    ("disfluency", DISFLUENCY_WINDOW_SEC),
    ("emotion", EMOTION_WINDOW_SEC),
)


def window_keep_fraction(
    intervals: list[tuple[float, float]],
    *,
    n_samples: int,
    window_sec: float,
    hop_sec: float,
) -> tuple[int, int]:
    """Return (n_windows_kept, n_windows_total) for VAD-gated inference."""
    n_frames, _, _, _ = frame_audio_geometry(
        n_samples, sample_rate=SAMPLE_RATE, window_sec=window_sec, hop_sec=hop_sec
    )
    if n_frames == 0:
        return 0, 0
    kept = np.zeros(n_frames, dtype=bool)
    for s, e in intervals:
        # window i spans [i*hop, i*hop + window]; overlaps [s, e] iff
        # i*hop < e  and  i*hop + window > s
        i_lo = max(0, int(np.floor((s - window_sec) / hop_sec)) + 1)
        i_hi = min(n_frames - 1, int(np.ceil(e / hop_sec)) - 1)
        if i_hi >= i_lo:
            kept[i_lo : i_hi + 1] = True
    return int(kept.sum()), int(n_frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="benchmark_audio/index.json")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    archives = json.loads(Path(args.index).read_text())
    vad = VadDetector(
        threshold=DEFAULT_VAD_SPEECH_THRESHOLD,
        min_speech_sec=DEFAULT_VAD_MIN_SPEECH_SEC,
        min_silence_sec=DEFAULT_VAD_MIN_SILENCE_SEC,
    )
    print(f"VAD config: threshold={DEFAULT_VAD_SPEECH_THRESHOLD} "
          f"min_speech={DEFAULT_VAD_MIN_SPEECH_SEC}s "
          f"min_silence={DEFAULT_VAD_MIN_SILENCE_SEC}s\n", flush=True)

    results = []
    for i, a in enumerate(archives, 1):
        f = a.get("local_path")
        if not f or not Path(f).is_file():
            continue
        name = Path(f).name.split("__")[-1][:24]
        audio = load_audio(f, sample_rate=SAMPLE_RATE)
        dur = audio.duration_sec
        t = time.perf_counter()
        intervals = vad(audio.samples, audio.sample_rate)
        vad_sec = time.perf_counter() - t

        speech_sec = float(sum(e - s for s, e in intervals))
        speech_frac = speech_sec / dur if dur else 0.0

        grids = {}
        for task, wsec in GRIDS:
            kept, total = window_keep_fraction(
                intervals, n_samples=len(audio.samples),
                window_sec=wsec, hop_sec=DEFAULT_HOP_SEC,
            )
            keep_frac = kept / total if total else 0.0
            grids[task] = {
                "windows_total": total,
                "windows_kept": kept,
                "keep_fraction": round(keep_frac, 4),
                "compute_saving": round(1 - keep_frac, 4),
            }

        rec = {
            "archive_id": a.get("archive_id", name),
            "duration_sec": round(dur, 1),
            "n_speech_intervals": len(intervals),
            "speech_sec": round(speech_sec, 1),
            "speech_fraction": round(speech_frac, 4),
            "vad_compute_sec": round(vad_sec, 2),
            "grids": grids,
        }
        results.append(rec)
        af = grids["affect"]
        print(f"[{i}] {name:24s} dur={dur/60:5.1f}m  intervals={len(intervals):4d}  "
              f"speech={speech_frac*100:5.1f}%  "
              f"affect keep={af['keep_fraction']*100:5.1f}% "
              f"(save {af['compute_saving']*100:4.1f}%)  vad={vad_sec:.1f}s", flush=True)

    if results:
        agg = {
            "n_archives": len(results),
            "mean_speech_fraction": round(np.mean([r["speech_fraction"] for r in results]), 4),
            "mean_affect_keep_fraction": round(
                np.mean([r["grids"]["affect"]["keep_fraction"] for r in results]), 4),
            "mean_affect_compute_saving": round(
                np.mean([r["grids"]["affect"]["compute_saving"] for r in results]), 4),
        }
        print(f"\nAGG: speech~{agg['mean_speech_fraction']*100:.1f}%  "
              f"affect keep~{agg['mean_affect_keep_fraction']*100:.1f}%  "
              f"=> compute saving ~{agg['mean_affect_compute_saving']*100:.1f}%", flush=True)
        report = {"vad_config": {
            "threshold": DEFAULT_VAD_SPEECH_THRESHOLD,
            "min_speech_sec": DEFAULT_VAD_MIN_SPEECH_SEC,
            "min_silence_sec": DEFAULT_VAD_MIN_SILENCE_SEC,
            "hop_sec": DEFAULT_HOP_SEC,
        }, "archives": results, "aggregate": agg}
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
            print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
