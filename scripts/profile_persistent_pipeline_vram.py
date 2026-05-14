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


def run_task(name, predictor, windows):
    task = {"task": name, "windows": int(len(windows)), "status": "ok"}
    try:
        before = cuda_stats()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        _ = predictor(windows)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        task["elapsed_sec"] = time.perf_counter() - start
        task["before"] = before
        task["after"] = cuda_stats()
    except Exception as exc:
        task["status"] = "error"
        task["error_type"] = type(exc).__name__
        task["error"] = str(exc)
        task["traceback"] = traceback.format_exc(limit=5)
        task["after_error"] = cuda_stats()
    return task


def run_config(args, audio, affect_windows, disfluency_windows, emotion_windows, cfg):
    affect_bs, disfluency_bs, emotion_bs = cfg
    label = f"{affect_bs}/{disfluency_bs}/{emotion_bs}"
    print(f"\n=== config {label} ===", flush=True)

    cuda_reset()
    result = {
        "config": {
            "affect_batch_size": affect_bs,
            "disfluency_batch_size": disfluency_bs,
            "emotion_batch_size": emotion_bs,
        },
        "status": "ok",
        "tasks": [],
    }

    try:
        load_start = time.perf_counter()
        affect = AffectPredictor("wavlm", device=args.device, batch_size=affect_bs)
        disfluency = DisfluencyPredictor("wavlm", device=args.device, batch_size=disfluency_bs)
        emotion = EmotionPredictor(device=args.device, batch_size=emotion_bs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["model_load_sec"] = time.perf_counter() - load_start
        result["after_model_load"] = cuda_stats()

        result["tasks"].append(run_task("affect", affect, affect_windows))
        if result["tasks"][-1]["status"] != "ok":
            result["status"] = "error"
            return result

        result["tasks"].append(run_task("disfluency", disfluency, disfluency_windows))
        if result["tasks"][-1]["status"] != "ok":
            result["status"] = "error"
            return result

        result["tasks"].append(run_task("emotion", emotion, emotion_windows))
        if result["tasks"][-1]["status"] != "ok":
            result["status"] = "error"
            return result

        result["after_all_tasks"] = cuda_stats()

    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=5)
        result["after_error"] = cuda_stats()
    finally:
        for name in ("affect", "disfluency", "emotion"):
            locals().pop(name, None)
        cuda_reset()

    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_configs(values):
    configs = []
    for value in values:
        parts = value.replace(",", "/").split("/")
        if len(parts) != 3:
            raise ValueError(f"Bad config {value!r}; expected A/D/E like 512/512/64")
        configs.append(tuple(int(x) for x in parts))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-windows", type=int, default=512)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "512/512/64",
            "512/512/128",
            "512/512/256",
            "384/512/128",
            "384/384/128",
            "256/512/128",
            "256/384/128",
        ],
        help="Per-task batch configs as affect/disfluency/emotion",
    )
    ap.add_argument("--json-out")
    args = ap.parse_args()

    configs = parse_configs(args.configs)
    audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)

    affect_windows = make_windows(audio, AFFECT_WINDOW_SEC, args.min_windows)
    disfluency_windows = make_windows(audio, DISFLUENCY_WINDOW_SEC, args.min_windows)
    emotion_windows = make_windows(audio, EMOTION_WINDOW_SEC, args.min_windows)

    print(json.dumps({
        "audio": args.audio,
        "duration_sec": audio.duration_sec,
        "window_counts": {
            "affect": int(len(affect_windows)),
            "disfluency": int(len(disfluency_windows)),
            "emotion": int(len(emotion_windows)),
        },
        "configs": [list(c) for c in configs],
    }, indent=2), flush=True)

    results = [
        run_config(args, audio, affect_windows, disfluency_windows, emotion_windows, cfg)
        for cfg in configs
    ]

    print("\n=== final summary ===")
    print(json.dumps(results, indent=2), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
