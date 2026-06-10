#!/usr/bin/env python3
"""Compare stock WavLM attention vs the repo-owned SDPA patch.

Validates numerical equivalence (fp32, same weights) and measures throughput at
fp32 / fp16 / fp16+compile for affect + disfluency. The SDPA patch routes WavLM's
gated-bias attention through F.scaled_dot_product_attention (fused/flash kernel).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.models import (
    AffectPredictor,
    DisfluencyPredictor,
)
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
    DEFAULT_HOP_SEC,
)
from audio_classification_playground.acoustic_events.inference.wavlm_sdpa import (
    apply_wavlm_sdpa_patch,
    remove_wavlm_sdpa_patch,
)

PRED = {"affect": (AffectPredictor, AFFECT_WINDOW_SEC, ("arousal", "valence", "dominance")),
        "disfluency": (DisfluencyPredictor, DISFLUENCY_WINDOW_SEC, ("fluency_logits", "disfluency_type_logits"))}


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_and_time(task, windows, *, sdpa, dtype, compile_, device, bs):
    cls = PRED[task][0]
    remove_wavlm_sdpa_patch()
    if sdpa:
        apply_wavlm_sdpa_patch()
    try:
        _free()
        p = cls("wavlm", device=device, batch_size=bs, wavlm_autocast_dtype=dtype,
                wavlm_compile=compile_, wavlm_compile_mode="default")
        p(windows[:bs])  # warmup (triggers compile)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = p(windows)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        peak = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
        del p
        _free()
        return {"win_per_s": len(windows) / elapsed, "elapsed": elapsed, "peak_gib": peak,
                "out": {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}}
    finally:
        remove_wavlm_sdpa_patch()


def maxdiff(a, b):
    return max(float(np.abs(a[k] - b[k]).max()) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--windows", type=int, default=2048)
    ap.add_argument("--tasks", nargs="+", default=["affect", "disfluency"])
    ap.add_argument("--json-out")
    args = ap.parse_args()

    audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)
    report = {"audio": args.audio, "batch_size": args.batch_size, "windows": args.windows,
              "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "tasks": {}}

    configs = [("fp32", None, False), ("fp16", "fp16", False), ("fp16+compile", "fp16", True)]
    for task in args.tasks:
        _, win_sec, _ = PRED[task]
        w = frame_audio(audio.samples, sample_rate=SAMPLE_RATE, window_sec=win_sec, hop_sec=DEFAULT_HOP_SEC)[: args.windows]
        print(f"\n##### {task} ({len(w)} windows) #####", flush=True)
        task_rows = {}
        ref = None  # fp32 stock reference for drift
        for label, dtype, comp in configs:
            stock = build_and_time(task, w, sdpa=False, dtype=dtype, compile_=comp, device=args.device, bs=args.batch_size)
            sdpa = build_and_time(task, w, sdpa=True, dtype=dtype, compile_=comp, device=args.device, bs=args.batch_size)
            if ref is None:
                ref = stock["out"]
            equiv = maxdiff(stock["out"], sdpa["out"])      # SDPA vs stock at same precision
            drift_stock = maxdiff(ref, stock["out"])
            drift_sdpa = maxdiff(ref, sdpa["out"])
            row = {"stock_win_per_s": stock["win_per_s"], "sdpa_win_per_s": sdpa["win_per_s"],
                   "sdpa_speedup": sdpa["win_per_s"] / stock["win_per_s"],
                   "sdpa_vs_stock_maxabs": equiv,
                   "stock_vs_fp32_maxabs": drift_stock, "sdpa_vs_fp32_maxabs": drift_sdpa,
                   "stock_peak_gib": stock["peak_gib"], "sdpa_peak_gib": sdpa["peak_gib"]}
            task_rows[label] = row
            print(f"  {label:14s} stock {stock['win_per_s']:6.1f} -> sdpa {sdpa['win_per_s']:6.1f} win/s "
                  f"({row['sdpa_speedup']:.2f}x)  sdpa~stock maxabs={equiv:.2e}  "
                  f"sdpa-vs-fp32={drift_sdpa:.2e}  peak {stock['peak_gib']:.1f}->{sdpa['peak_gib']:.1f} GiB", flush=True)
        report["tasks"][task] = task_rows

    print("\n" + json.dumps(report, indent=2), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
