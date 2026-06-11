#!/usr/bin/env python3
"""O6 INT8 (torchao W8A8) for WavLM: throughput vs fp16+compile, and event-level A/B vs fp32.

INT8 targets the dominant FFN+projection GEMMs (~97% of WavLM FLOPs at our short
windows). Activations are dynamically quantized (no calibration set needed);
weights are int8. The conv feature extractor, gru gated-bias linear, and the
classifier head are left in higher precision (mixed int8).

Step 1: throughput int8+compile vs fp16+compile.
Step 2: reuse event_level_ab's event builders + differ to check int8 vs fp32 events.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.models import AffectPredictor, DisfluencyPredictor
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC, DISFLUENCY_WINDOW_SEC, DEFAULT_HOP_SEC,
)
from audio_classification_playground.vox_profile.wavlm_inference import compile_wavlm_backbone
import event_level_ab as ab  # reuse affect_events/disfluency_events/diff_events

TASKS = {"affect": (AffectPredictor, AFFECT_WINDOW_SEC),
         "disfluency": (DisfluencyPredictor, DISFLUENCY_WINDOW_SEC)}
INT8_TARGET_SUFFIXES = {"q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense"}


def _target(mod, fqn: str) -> bool:
    return isinstance(mod, nn.Linear) and fqn.split(".")[-1] in INT8_TARGET_SUFFIXES


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build(task, *, int8, dtype, compile_, device, bs):
    """Build a predictor; optionally torchao-int8-quantize the backbone GEMMs + compile."""
    cls = TASKS[task][0]
    p = cls("wavlm", device=device, batch_size=bs, wavlm_autocast_dtype=dtype, wavlm_compile=False)
    n_q = 0
    if int8:
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        n_q = sum(1 for n, m in p._model.named_modules() if _target(m, n))
        quantize_(p._model, Int8DynamicActivationInt8WeightConfig(), filter_fn=_target)
    if compile_:
        # Compile the backbone submodule (as the predictor's own path does) — compiling
        # the whole wrapper trips a dynamo guard on the from_numpy/inference_mode input.
        compile_wavlm_backbone(p._model, mode="default", dynamic=False)
    return p, n_q


def time_pred(p, windows, bs):
    p(windows[:bs])  # warmup (triggers compile)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = p(windows)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    el = time.perf_counter() - t
    peak = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    return out, len(windows) / el, peak


def step1_throughput(audio, device, bs, n_windows):
    print("\n===== STEP 1: throughput (int8 vs fp16, both +compile) =====", flush=True)
    rows = {}
    for task, (_, win_sec) in TASKS.items():
        w = frame_audio(audio.samples, sample_rate=SAMPLE_RATE, window_sec=win_sec, hop_sec=DEFAULT_HOP_SEC)[:n_windows]
        _free()
        p, _ = build(task, int8=False, dtype="fp16", compile_=True, device=device, bs=bs)
        _, fp16_wps, fp16_peak = time_pred(p, w, bs); del p; _free()
        row = {"fp16c_wps": fp16_wps, "fp16_peak_gib": fp16_peak}
        # Fast int8 needs torch.compile; on torch 2.10 / torchao 0.17 that fails to trace.
        try:
            p, nq = build(task, int8=True, dtype=None, compile_=True, device=device, bs=bs)
            _, int8c_wps, int8_peak = time_pred(p, w, bs); del p; _free()
            row.update(int8c_wps=int8c_wps, int8c_speedup=int8c_wps / fp16_wps,
                       int8_peak_gib=int8_peak, n_quantized_linears=nq)
        except Exception as exc:
            row["int8_compile_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            _free()
        # Eager int8 (no compile) — correct but slow fallback; records the degraded speed.
        try:
            p, nq = build(task, int8=True, dtype=None, compile_=False, device=device, bs=bs)
            _, int8e_wps, int8e_peak = time_pred(p, w, bs); del p; _free()
            row.update(int8_eager_wps=int8e_wps, int8_eager_peak_gib=int8e_peak, n_quantized_linears=nq)
        except Exception as exc:
            row["int8_eager_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"; _free()
        rows[task] = row
        msg = f"  {task:11s} fp16+compile {fp16_wps:6.1f} win/s"
        if "int8c_wps" in row:
            msg += f" | int8+compile {row['int8c_wps']:6.1f} ({row['int8c_speedup']:.2f}x)"
        else:
            msg += f" | int8+compile BLOCKED ({row.get('int8_compile_error','')[:60]})"
        if "int8_eager_wps" in row:
            msg += f" | int8 eager {row['int8_eager_wps']:6.1f} win/s"
        print(msg, flush=True)
    return rows


def _arrays(task, p, windows):
    return p(windows)


def step2_ab(audio_paths, device, bs, max_seconds, boundary_tol):
    print("\n===== STEP 2: event-level A/B (int8 vs fp32) =====", flush=True)
    files = []
    for path in audio_paths:
        a = load_audio(path, sample_rate=SAMPLE_RATE)
        samples = a.samples[: int(max_seconds * SAMPLE_RATE)] if max_seconds else a.samples
        dur = len(samples) / SAMPLE_RATE
        rec = {"audio": path, "duration_sec": round(dur, 1)}
        for task, (_, win_sec) in TASKS.items():
            w = frame_audio(samples, sample_rate=SAMPLE_RATE, window_sec=win_sec, hop_sec=DEFAULT_HOP_SEC)
            _free()
            p, _ = build(task, int8=False, dtype=None, compile_=False, device=device, bs=bs)  # fp32 ref
            ref = _arrays(task, p, w); del p; _free()
            p, _ = build(task, int8=True, dtype=None, compile_=False, device=device, bs=bs)  # int8 eager (correctness path that works on torch 2.10)
            cand = _arrays(task, p, w); del p; _free()
            build_ev = ab.affect_events if task == "affect" else ab.disfluency_events
            ev_ref, ev_cand = build_ev(ref, dur), build_ev(cand, dur)
            rec[task] = ab.diff_events(ev_ref, ev_cand, boundary_tol=boundary_tol)
            d = rec[task]
            print(f"  {Path(path).name[:18]:18s} {task:11s} base={d['n_base']:3d} cand={d['n_cand']:3d} "
                  f"drop={d['n_dropped']} add={d['n_added']} label={d['matched_label_agreement']:.3f} "
                  f"exact={d['matched_exact_fraction']:.3f} d_score_max={d['delta_score']['max']:.4f}", flush=True)
        files.append(rec)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True, help="First file used for throughput; all used for A/B.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--windows", type=int, default=2048)
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--boundary-tol", type=float, default=DEFAULT_HOP_SEC)
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    report = {"device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "batch_size": args.batch_size, "torchao": __import__("torchao").__version__}
    audio0 = load_audio(args.audio[0], sample_rate=SAMPLE_RATE)
    report["throughput"] = step1_throughput(audio0, args.device, args.batch_size, args.windows)
    if not args.skip_ab:
        report["ab"] = step2_ab(args.audio, args.device, args.batch_size, args.max_seconds, args.boundary_tol)

    print("\n" + json.dumps(report, indent=2), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DONE_INT8", flush=True)


if __name__ == "__main__":
    main()
