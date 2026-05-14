#!/usr/bin/env python3
import argparse, gc, json, time, traceback
from pathlib import Path

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import load_audio, frame_audio
from audio_classification_playground.acoustic_events.inference.models import (
    AffectPredictor,
    DisfluencyPredictor,
    EmotionPredictor,
)
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    EMOTION_WINDOW_SEC,
    DEFAULT_HOP_SEC,
)


def cuda_reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_stats():
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
    }


def ensure_min_windows(samples, sample_rate, window_sec, hop_sec, min_windows):
    if min_windows <= 0:
        return samples
    window = int(round(window_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    needed = window + max(0, min_windows - 1) * hop
    if len(samples) >= needed:
        return samples
    reps = int(np.ceil(needed / max(1, len(samples))))
    return np.tile(samples, reps).astype(np.float32, copy=False)


def make_windows(audio, window_sec, min_windows):
    samples = ensure_min_windows(
        audio.samples, audio.sample_rate, window_sec, DEFAULT_HOP_SEC, min_windows
    )
    return frame_audio(
        samples,
        sample_rate=audio.sample_rate,
        window_sec=window_sec,
        hop_sec=DEFAULT_HOP_SEC,
    )


def run_case(name, predictor_factory, windows):
    print(f"\n=== {name} ===", flush=True)
    cuda_reset()
    case = {"name": name, "windows": int(len(windows)), "status": "ok"}
    try:
        predictor = predictor_factory()
        case["after_load"] = cuda_stats()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _ = predictor(windows)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        case["elapsed_sec"] = time.perf_counter() - start
        case["after_inference"] = cuda_stats()
        print(json.dumps(case, indent=2), flush=True)
    except Exception as exc:
        case["status"] = "error"
        case["error_type"] = type(exc).__name__
        case["error"] = str(exc)
        case["traceback"] = traceback.format_exc(limit=5)
        print(json.dumps(case, indent=2), flush=True)
    finally:
        locals().pop("predictor", None)
        cuda_reset()
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--affect-batch-size", type=int, default=512)
    ap.add_argument("--disfluency-batch-size", type=int, default=512)
    ap.add_argument("--emotion-batch-sizes", type=int, nargs="+", default=[512, 128, 64])
    ap.add_argument("--min-windows", type=int, default=512)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)
    affect_windows = make_windows(audio, AFFECT_WINDOW_SEC, args.min_windows)
    disfluency_windows = make_windows(audio, DISFLUENCY_WINDOW_SEC, args.min_windows)
    emotion_windows = make_windows(audio, EMOTION_WINDOW_SEC, args.min_windows)

    results = []
    results.append(run_case(
        f"affect wavlm batch={args.affect_batch_size}",
        lambda: AffectPredictor("wavlm", device=args.device, batch_size=args.affect_batch_size),
        affect_windows,
    ))
    results.append(run_case(
        f"disfluency wavlm batch={args.disfluency_batch_size}",
        lambda: DisfluencyPredictor("wavlm", device=args.device, batch_size=args.disfluency_batch_size),
        disfluency_windows,
    ))
    for bs in args.emotion_batch_sizes:
        results.append(run_case(
            f"emotion2vec batch={bs}",
            lambda bs=bs: EmotionPredictor(device=args.device, batch_size=bs),
            emotion_windows,
        ))

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
