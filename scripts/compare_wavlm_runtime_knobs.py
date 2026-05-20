#!/usr/bin/env python3
"""Compare optional WavLM runtime acceleration knobs against the default path."""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from collections.abc import Mapping

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio, load_audio
from audio_classification_playground.acoustic_events.inference.models import (
    AffectPredictor,
    DisfluencyPredictor,
    ModelSuite,
    configure_torch_matmul,
)
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DEFAULT_HOP_SEC,
    DISFLUENCY_WINDOW_SEC,
)


def cuda_reset() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_stats() -> dict[str, float]:
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


def sync_device(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def set_tf32(enabled: bool) -> tuple[bool, str]:
    original_allow = torch.backends.cuda.matmul.allow_tf32
    original_precision = torch.get_float32_matmul_precision()
    if enabled:
        configure_torch_matmul(allow_tf32=True)
    return original_allow, original_precision


def restore_tf32(state: tuple[bool, str]) -> None:
    original_allow, original_precision = state
    torch.backends.cuda.matmul.allow_tf32 = original_allow
    torch.set_float32_matmul_precision(original_precision)


def ensure_windows(audio, window_sec: float, min_windows: int, max_windows: int | None) -> np.ndarray:
    window = int(round(window_sec * audio.sample_rate))
    hop = int(round(DEFAULT_HOP_SEC * audio.sample_rate))
    needed = window + max(0, min_windows - 1) * hop
    samples = audio.samples
    if len(samples) < needed:
        reps = int(np.ceil(needed / max(1, len(samples))))
        samples = np.tile(samples, reps).astype(np.float32, copy=False)
    windows = frame_audio(
        samples,
        sample_rate=audio.sample_rate,
        window_sec=window_sec,
        hop_sec=DEFAULT_HOP_SEC,
    )
    if max_windows is not None and max_windows > 0:
        windows = windows[:max_windows]
    return windows


class PredictorHandle:
    def __init__(self, predictor, owner=None):
        self.predictor = predictor
        self.owner = owner

    def __call__(self, windows: np.ndarray):
        return self.predictor(windows)


def make_predictor(task: str, args: argparse.Namespace, *, candidate: bool):
    batch_size = args.candidate_batch_size if candidate else args.batch_size
    runtime_kwargs = {}
    if candidate:
        runtime_kwargs.update(
            wavlm_autocast_dtype=args.candidate_autocast_dtype,
            wavlm_compile=args.candidate_compile,
            wavlm_compile_mode=args.candidate_compile_mode,
            wavlm_compile_dynamic=args.candidate_compile_dynamic,
            wavlm_stream_layer_sum=args.candidate_stream_layer_sum,
            allow_tf32=args.candidate_allow_tf32,
        )
    if args.resident_companions == "all":
        emotion_kwargs = {}
        if candidate and args.candidate_allow_tf32:
            # This benchmark toggles TF32 as a candidate experiment. ModelSuite
            # also applies the process-wide flag to emotion2vec, whose preset
            # resolver rejects granular TF32 unless the run is explicitly custom.
            emotion_kwargs["emotion_runtime_mode"] = "custom"
        suite = ModelSuite(
            affect_backbone="wavlm",
            disfluency_backbone="wavlm",
            batch_size=batch_size,
            emotion_batch_size=args.emotion_batch_size,
            device=args.device,
            load_vad=False,
            **emotion_kwargs,
            **runtime_kwargs,
        )
        predictor = suite.affect if task == "affect" else suite.disfluency
        return PredictorHandle(predictor, owner=suite)
    if args.resident_companions == "wavlm":
        predictor_kwargs = {
            key: value
            for key, value in runtime_kwargs.items()
            if key != "allow_tf32"
        }
        affect = AffectPredictor(
            backbone="wavlm",
            device=args.device,
            batch_size=batch_size,
            **predictor_kwargs,
        )
        disfluency = DisfluencyPredictor(
            backbone="wavlm",
            device=args.device,
            batch_size=batch_size,
            **predictor_kwargs,
        )
        predictor = affect if task == "affect" else disfluency
        return PredictorHandle(predictor, owner=(affect, disfluency))

    kwargs = {
        "backbone": "wavlm",
        "device": args.device,
        "batch_size": batch_size,
    }
    if candidate:
        kwargs.update(
            wavlm_autocast_dtype=args.candidate_autocast_dtype,
            wavlm_compile=args.candidate_compile,
            wavlm_compile_mode=args.candidate_compile_mode,
            wavlm_compile_dynamic=args.candidate_compile_dynamic,
            wavlm_stream_layer_sum=args.candidate_stream_layer_sum,
        )
    cls = AffectPredictor if task == "affect" else DisfluencyPredictor
    return PredictorHandle(cls(**kwargs))


def run_predictor(predictor, windows: np.ndarray, *, device: str) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs = predictor(windows)
    sync_device(device)
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "elapsed_sec": elapsed,
        "cuda": cuda_stats(),
        "outputs": outputs,
    }


def strip_outputs(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "outputs"}


def compare_outputs(
    task: str,
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> dict:
    rows = []
    for key in reference:
        ref = np.asarray(reference[key], dtype=np.float32)
        cand = np.asarray(candidate[key], dtype=np.float32)
        diff = np.abs(ref - cand)
        rows.append({
            "name": key,
            "shape": list(ref.shape),
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            "p99_abs_diff": float(np.quantile(diff, 0.99)) if diff.size else 0.0,
            "allclose": bool(np.allclose(ref, cand, atol=atol, rtol=rtol)),
        })

    out = {
        "status": "ok",
        "allclose": all(row["allclose"] for row in rows),
        "atol": atol,
        "rtol": rtol,
        "rows": rows,
    }
    if task == "disfluency" and "fluency_logits" in reference:
        ref_f = np.asarray(reference["fluency_logits"], dtype=np.float32)
        cand_f = np.asarray(candidate["fluency_logits"], dtype=np.float32)
        out["fluency_top1_agreement"] = float(
            np.mean(np.argmax(ref_f, axis=1) == np.argmax(cand_f, axis=1))
        )
    if task == "disfluency" and "disfluency_type_logits" in reference:
        ref_t = np.asarray(reference["disfluency_type_logits"], dtype=np.float32)
        cand_t = np.asarray(candidate["disfluency_type_logits"], dtype=np.float32)
        out["type_logit_sign_agreement"] = float(np.mean((ref_t >= 0.0) == (cand_t >= 0.0)))
    return out


def run_task(task: str, windows: np.ndarray, args: argparse.Namespace) -> dict:
    result = {
        "task": task,
        "window_count": int(len(windows)),
        "baseline_batch_size": args.batch_size,
        "candidate_batch_size": args.candidate_batch_size,
        "resident_suite": args.resident_companions == "all",
        "resident_companions": args.resident_companions,
        "emotion_batch_size": (
            args.emotion_batch_size
            if args.resident_companions == "all"
            else None
        ),
        "candidate": {
            "autocast_dtype": args.candidate_autocast_dtype,
            "compile": args.candidate_compile,
            "compile_mode": args.candidate_compile_mode,
            "compile_dynamic": args.candidate_compile_dynamic,
            "stream_layer_sum": args.candidate_stream_layer_sum,
            "allow_tf32": args.candidate_allow_tf32,
        },
        "status": "ok",
    }
    warmup_window_count = min(max(args.batch_size, args.candidate_batch_size), len(windows))
    warmup_windows = windows[:warmup_window_count]

    try:
        cuda_reset()
        load_started = time.perf_counter()
        baseline = make_predictor(task, args, candidate=False)
        sync_device(args.device)
        result["baseline_model_load_sec"] = time.perf_counter() - load_started

        baseline_warmup = run_predictor(baseline, warmup_windows, device=args.device)
        baseline_run = run_predictor(baseline, windows, device=args.device)
        result["baseline_warmup"] = strip_outputs(baseline_warmup)
        result["baseline"] = strip_outputs(baseline_run)
        baseline_outputs = baseline_run["outputs"]

        del baseline
        cuda_reset()

        tf32_state = set_tf32(args.candidate_allow_tf32)
        try:
            load_started = time.perf_counter()
            candidate = make_predictor(task, args, candidate=True)
            sync_device(args.device)
            result["candidate_model_load_sec"] = time.perf_counter() - load_started

            candidate_warmup = run_predictor(candidate, warmup_windows, device=args.device)
            candidate_run = run_predictor(candidate, windows, device=args.device)
        finally:
            restore_tf32(tf32_state)

        result["candidate_warmup"] = strip_outputs(candidate_warmup)
        result["candidate_run"] = strip_outputs(candidate_run)
        result["comparison"] = compare_outputs(
            task,
            baseline_outputs,
            candidate_run["outputs"],
            atol=args.atol,
            rtol=args.rtol,
        )
        result["speedup"] = (
            result["baseline"]["elapsed_sec"]
            / max(result["candidate_run"]["elapsed_sec"], 1e-12)
        )
        result["baseline_windows_per_sec"] = (
            result["window_count"] / max(result["baseline"]["elapsed_sec"], 1e-12)
        )
        result["candidate_windows_per_sec"] = (
            result["window_count"] / max(result["candidate_run"]["elapsed_sec"], 1e-12)
        )
        result["time_saved_fraction"] = 1.0 - (
            result["candidate_run"]["elapsed_sec"]
            / max(result["baseline"]["elapsed_sec"], 1e-12)
        )

        del candidate
        cuda_reset()
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=8)
        result["cuda"] = cuda_stats()
        cuda_reset()
    return result


def task_window_sec(task: str) -> float:
    return AFFECT_WINDOW_SEC if task == "affect" else DISFLUENCY_WINDOW_SEC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("affect", "disfluency"),
        default=["affect", "disfluency"],
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        help="Candidate batch size. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--resident-suite",
        action="store_true",
        help="Alias for --resident-companions all.",
    )
    parser.add_argument(
        "--resident-companions",
        choices=("none", "wavlm", "all"),
        default="none",
        help=(
            "Load idle companion models during the task run: none, both WavLM "
            "models only, or the full production ModelSuite."
        ),
    )
    parser.add_argument(
        "--emotion-batch-size",
        type=int,
        default=64,
        help="Emotion batch size used when --resident-suite loads the production ModelSuite.",
    )
    parser.add_argument("--min-windows", type=int, default=512)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=1025,
        help="Use a non-multiple of batch size to expose final-batch behavior.",
    )
    parser.add_argument("--candidate-autocast-dtype", choices=("fp16", "bf16"))
    parser.add_argument("--candidate-compile", action="store_true")
    parser.add_argument("--candidate-compile-mode", default="reduce-overhead")
    parser.add_argument("--candidate-compile-dynamic", action="store_true")
    parser.add_argument("--candidate-stream-layer-sum", action="store_true")
    parser.add_argument("--candidate-allow-tf32", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    if args.resident_suite:
        args.resident_companions = "all"
    if args.candidate_batch_size is None:
        args.candidate_batch_size = args.batch_size

    audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)
    report = {
        "audio": args.audio,
        "duration_sec": audio.duration_sec,
        "device": args.device,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(torch.device(args.device))
                if torch.device(args.device).type == "cuda" and torch.cuda.is_available()
                else None
            ),
        },
        "tasks": [],
    }

    for task in args.tasks:
        windows = ensure_windows(
            audio,
            task_window_sec(task),
            min_windows=args.min_windows,
            max_windows=args.max_windows,
        )
        report["tasks"].append(run_task(task, windows, args))

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
