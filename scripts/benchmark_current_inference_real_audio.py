#!/usr/bin/env python3
"""Per-task A10G baseline for *current* (eager) inference on real audio.

This is the O1 measurement substrate from INFERENCE_OPTIMIZATION_PLAN.md: it
captures per-task steady-state throughput (windows/s), wall-clock, and peak
VRAM for the production runtime, then projects per-archive latency for each
retrieved file from its real window count.

Models are built through ``ModelSuite`` so the runtime matches production
exactly (eager FP32 WavLM for affect+disfluency, emotion2vec auto->optimized
at batch 64). Each task is timed *separately* -- affect, disfluency, emotion
(GPU) and VAD (CPU) -- never lumped together.

Throughput is measured on one representative file (window compute is
file-independent for a fixed window/batch size) and projected onto all files
via their decoded durations. Use ``--full`` semantics by raising --cap-windows.

Example:
    uv run python scripts/benchmark_current_inference_real_audio.py \
        --index benchmark_audio/index.json --json-out benchmark_audio/baseline.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.models import ModelSuite
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    EMOTION_WINDOW_SEC,
    DEFAULT_HOP_SEC,
)

GPU_TASKS = (
    ("affect", AFFECT_WINDOW_SEC),
    ("disfluency", DISFLUENCY_WINDOW_SEC),
    ("emotion", EMOTION_WINDOW_SEC),
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 2**30


def n_windows(duration_sec: float, window_sec: float, hop_sec: float) -> int:
    """Analytic window count for framing the whole decoded signal."""
    n = math.floor((duration_sec - window_sec) / hop_sec) + 1
    return max(1, n)


def measure_gpu_task(name, predictor, windows, cap, warmup_batches, batch_size):
    warmup_n = min(len(windows), max(1, warmup_batches) * batch_size)
    predictor(windows[:warmup_n])
    _sync()

    timed = windows[:cap] if cap and cap < len(windows) else windows
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    _sync()
    start = time.perf_counter()
    predictor(timed)
    _sync()
    elapsed = time.perf_counter() - start
    n = int(len(timed))
    return {
        "task": name,
        "timed_windows": n,
        "batch_size": int(batch_size),
        "elapsed_sec": round(elapsed, 4),
        "windows_per_sec": round(n / elapsed, 2) if elapsed else None,
        "peak_vram_gib": round(_peak_vram_gib(), 3),
    }


def measure_vad(suite, samples, sample_rate, cap_sec):
    cap_samples = int(cap_sec * sample_rate) if cap_sec else len(samples)
    chunk = samples[:cap_samples]
    audio_sec = len(chunk) / sample_rate
    start = time.perf_counter()
    suite.vad(chunk, sample_rate)
    elapsed = time.perf_counter() - start
    return {
        "task": "vad",
        "device": "cpu",
        "timed_audio_sec": round(audio_sec, 2),
        "elapsed_sec": round(elapsed, 4),
        # realtime factor: seconds of audio processed per second of compute
        "audio_sec_per_compute_sec": round(audio_sec / elapsed, 2) if elapsed else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="benchmark_audio/index.json",
                    help="index.json produced by retrieve_benchmark_audio.py")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap-windows", type=int, default=3072,
                    help="Windows to time per GPU task (steady state). 0 = all.")
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--vad-cap-sec", type=float, default=300.0,
                    help="Seconds of audio to time VAD on. 0 = whole file.")
    ap.add_argument("--throughput-file",
                    help="local_path to measure throughput on; default = shortest.")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    files = json.loads(Path(args.index).read_text())
    files = [f for f in files if f.get("duration_sec")]
    if not files:
        print("no usable files in index", flush=True)
        return 1

    # Pick the throughput-source file (shortest decode by default).
    if args.throughput_file:
        src = next(f for f in files if f["local_path"] == args.throughput_file)
    else:
        src = min(files, key=lambda f: f["duration_sec"])
    print(f"throughput source: {Path(src['local_path']).name} "
          f"({src['duration_sec']}s)\n", flush=True)

    print(f"decoding throughput source @ {SAMPLE_RATE} Hz mono ...", flush=True)
    audio = load_audio(src["local_path"], sample_rate=SAMPLE_RATE)

    # Each task is loaded, measured, and torn down in isolation -- this both
    # matches the production task-fleet topology (one model per GPU pod) and
    # gives clean per-task VRAM/throughput numbers free of cross-task pressure.
    throughput = {}
    bench_batch = {}
    for name, window_sec in GPU_TASKS:
        print(f"\n[{name}] loading model in isolation ...", flush=True)
        load_start = time.perf_counter()
        suite = ModelSuite(
            affect_backbone="wavlm",
            disfluency_backbone="wavlm",
            batch_size=512,
            device=args.device,
            emotion_runtime_mode="auto",
            load_vad=False,
            tasks_to_load=[name],
        )
        predictor = getattr(suite, name)
        _sync()
        model_load_sec = time.perf_counter() - load_start
        bench_batch[name] = predictor.batch_size

        windows = frame_audio(audio.samples, sample_rate=audio.sample_rate,
                              window_sec=window_sec, hop_sec=DEFAULT_HOP_SEC)
        res = measure_gpu_task(name, predictor, windows, args.cap_windows,
                               args.warmup_batches, predictor.batch_size)
        res["window_sec"] = window_sec
        res["model_load_sec"] = round(model_load_sec, 2)
        throughput[name] = res
        print(f"  {name:11s} bs={predictor.batch_size:4d}  {res['timed_windows']:6d} win  "
              f"{res['elapsed_sec']:7.3f}s  {res['windows_per_sec']:8.1f} win/s  "
              f"peak {res['peak_vram_gib']:.2f} GiB  (load {model_load_sec:.1f}s)", flush=True)

        del predictor, suite
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("\n[vad] loading Silero (CPU) ...", flush=True)
    vad_suite = ModelSuite(
        affect_backbone="wavlm", disfluency_backbone="wavlm",
        device=args.device, load_vad=True, tasks_to_load=[],
    )
    vad_res = measure_vad(vad_suite, audio.samples, audio.sample_rate, args.vad_cap_sec)
    print(f"  {'vad (cpu)':11s} {vad_res['timed_audio_sec']:6.0f}s audio  "
          f"{vad_res['elapsed_sec']:7.3f}s  "
          f"{vad_res['audio_sec_per_compute_sec']:.1f}x realtime", flush=True)

    # Project per-archive latency from real window counts + measured throughput.
    print("\nprojected per-archive duration (sequential GPU tasks):", flush=True)
    projections = []
    for f in files:
        dur = f["duration_sec"]
        per_task = {}
        gpu_total = 0.0
        for name, window_sec in GPU_TASKS:
            nw = n_windows(dur, window_sec, DEFAULT_HOP_SEC)
            wps = throughput[name]["windows_per_sec"]
            sec = nw / wps if wps else None
            per_task[name] = {"windows": nw, "proj_sec": round(sec, 2) if sec else None}
            if sec:
                gpu_total += sec
        vad_rtf = vad_res["audio_sec_per_compute_sec"]
        vad_sec = dur / vad_rtf if vad_rtf else None
        proj = {
            "file": Path(f["local_path"]).name,
            "duration_sec": dur,
            "per_task": per_task,
            "vad_proj_sec": round(vad_sec, 2) if vad_sec else None,
            "gpu_total_sec": round(gpu_total, 2),
        }
        projections.append(proj)
        print(f"  {proj['file'][:42]:42s} dur {dur/60:5.1f}m | "
              f"affect {per_task['affect']['proj_sec']:7.1f}s  "
              f"disf {per_task['disfluency']['proj_sec']:7.1f}s  "
              f"emo {per_task['emotion']['proj_sec']:7.1f}s  "
              f"| gpu_total {gpu_total:7.1f}s ({gpu_total/60:.1f}m)", flush=True)

    report = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "runtime": "eager-fp32-wavlm; emotion auto->optimized; tasks isolated (task-fleet topology)",
        "batch_sizes": bench_batch,
        "hop_sec": DEFAULT_HOP_SEC,
        "throughput_source": Path(src["local_path"]).name,
        "throughput": throughput,
        "vad": vad_res,
        "projections": projections,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
