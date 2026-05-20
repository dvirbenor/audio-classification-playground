#!/usr/bin/env python3
"""Export and benchmark the direct emotion2vec scorer with ONNX Runtime.

This is intentionally a sidecar benchmark harness.  It exports the exact
direct scorer used by the fast PyTorch path:

    waveform window -> extract_features -> mean pool -> proj -> masked softmax

The export is static for a fixed ``[batch_size, window_samples]`` input.  Tail
batches are padded and sliced back to their real row count so output comparison
stays aligned with the PyTorch reference.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from funasr import AutoModel

from audio_classification_playground.acoustic_events.inference.audio import (
    frame_audio,
    writable_contiguous_float32,
)
from audio_classification_playground.acoustic_events.inference.emotion2vec import (
    _pad_tensor_batch,
    make_direct_emotion2vec_scorer,
)
from audio_classification_playground.acoustic_events.inference.models import (
    configure_torch_matmul,
)


SAMPLE_RATE = 16_000
WINDOW_SEC = 3.0
HOP_SEC = 0.25


def synthetic_signals(count: int, duration_sec: float) -> list[np.ndarray]:
    rng = np.random.default_rng(12345)
    t = np.arange(int(duration_sec * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    signals = []
    for i in range(count):
        signal = (
            0.05 * np.sin(2 * np.pi * (180 + 17 * i) * t)
            + 0.03 * np.sin(2 * np.pi * (410 + 23 * i) * t)
            + 0.01 * rng.standard_normal(len(t), dtype=np.float32)
        )
        signals.append(np.ascontiguousarray(signal, dtype=np.float32))
    return signals


def load_audio_paths(patterns: list[str]) -> list[np.ndarray]:
    import librosa

    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    signals = []
    for path in paths:
        samples, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        signals.append(np.ascontiguousarray(samples, dtype=np.float32))
    return signals


def trim_to_max_windows(samples: np.ndarray, max_windows: int | None) -> np.ndarray:
    if not max_windows or max_windows <= 0:
        return samples
    window = int(round(WINDOW_SEC * SAMPLE_RATE))
    hop = int(round(HOP_SEC * SAMPLE_RATE))
    max_samples = window + max(0, max_windows - 1) * hop
    return np.ascontiguousarray(samples[:max_samples], dtype=np.float32)


def frame_signals(signals: Sequence[np.ndarray]) -> np.ndarray:
    windows = [
        frame_audio(samples, sample_rate=SAMPLE_RATE, window_sec=WINDOW_SEC, hop_sec=HOP_SEC)
        for samples in signals
    ]
    return np.ascontiguousarray(np.concatenate(windows, axis=0), dtype=np.float32)


def export_onnx(core: torch.nn.Module, onnx_path: Path, *, batch_size: int, opset: int) -> None:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "onnx is not installed. Install onnx in the benchmark environment "
            "before exporting the emotion2vec scorer."
        )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(core.parameters()).device
    dummy = torch.zeros(
        (int(batch_size), int(round(WINDOW_SEC * SAMPLE_RATE))),
        dtype=torch.float32,
        device=device,
    )
    core.eval()
    with torch.inference_mode():
        torch.onnx.export(
            core,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["scores"],
            opset_version=int(opset),
            do_constant_folding=True,
        )


def run_torch_batches(
    scorer,
    windows: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    device = scorer.device
    out = torch.empty(
        (len(windows), len(scorer.selected_labels)),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for start, end, actual_size, batch_np in iter_static_batches(windows, batch_size):
            batch = torch.from_numpy(batch_np).to(device)
            out[start:end] = scorer(batch)[:actual_size]
    return out.detach().cpu().numpy()


def run_ort_batches(session, windows: np.ndarray, *, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    input_name = session.get_inputs()[0].name
    for _, _, actual_size, batch_np in iter_static_batches(windows, batch_size):
        scores = session.run(None, {input_name: batch_np})[0]
        parts.append(np.asarray(scores[:actual_size], dtype=np.float32))
    return np.concatenate(parts, axis=0)


def iter_static_batches(windows: np.ndarray, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        actual = end - start
        batch_np = writable_contiguous_float32(windows[start:end])
        if actual < batch_size:
            pad = np.zeros((batch_size - actual, windows.shape[1]), dtype=np.float32)
            batch_np = np.concatenate([batch_np, pad], axis=0)
        yield start, end, actual, batch_np


def make_session(onnx_path: Path, provider: str, *, cache_dir: Path | None, fallback: bool):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Install onnxruntime-gpu in the benchmark "
            "environment to test CUDA/TensorRT providers."
        ) from exc

    available = set(ort.get_available_providers())
    provider_name = resolve_provider_alias(provider)
    if provider_name not in available:
        raise RuntimeError(
            f"Provider {provider_name!r} is not available. "
            f"Available providers: {sorted(available)}"
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = provider_chain(provider_name, cache_dir=cache_dir, fallback=fallback)
    return ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)


def requested_provider_active(session, provider_name: str) -> bool:
    """Return whether ONNX Runtime actually kept the requested provider active."""
    return provider_name in session.get_providers()


def resolve_provider_alias(provider: str) -> str:
    normalized = provider.strip().lower()
    aliases = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "trt": "TensorrtExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    return aliases.get(normalized, provider)


def provider_chain(provider_name: str, *, cache_dir: Path | None, fallback: bool):
    provider_spec: str | tuple[str, dict[str, str]]
    if provider_name == "TensorrtExecutionProvider":
        options: dict[str, str] = {}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            options.update(
                {
                    "trt_engine_cache_enable": "1",
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_timing_cache_enable": "1",
                    "trt_timing_cache_path": str(cache_dir / "trt_timing.cache"),
                }
            )
        provider_spec = (provider_name, options)
    else:
        provider_spec = provider_name

    providers: list[object] = [provider_spec]
    if fallback:
        if provider_name == "TensorrtExecutionProvider":
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
    return providers


def benchmark(fn, *, warmup_runs: int, timed_runs: int, sync=None):
    for _ in range(max(0, warmup_runs)):
        fn()
    if sync is not None:
        sync()
    timings = []
    last = None
    for _ in range(max(1, timed_runs)):
        start = time.perf_counter()
        last = fn()
        if sync is not None:
            sync()
        timings.append(time.perf_counter() - start)
    return last, timings


def cuda_sync_for_scorer(scorer):
    if getattr(scorer.device, "type", None) != "cuda":
        return None

    def sync() -> None:
        torch.cuda.synchronize(scorer.device)

    return sync


def summarize_scores(
    title: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: Sequence[str],
    reference_seconds: float,
    candidate_seconds: float,
    *,
    diagnostic_rows: int,
) -> None:
    diff = np.abs(reference - candidate)
    reference_top = np.argmax(reference, axis=1)
    candidate_top = np.argmax(candidate, axis=1)
    top1_match = reference_top == candidate_top
    print(f"\n=== {title} ===")
    print(f"shape_reference: {reference.shape}")
    print(f"shape_candidate: {candidate.shape}")
    print(f"reference_seconds: {reference_seconds:.4f}")
    print(f"candidate_seconds: {candidate_seconds:.4f}")
    print(f"speedup: {reference_seconds / max(candidate_seconds, 1e-9):.2f}x")
    print(f"max_abs_diff: {float(diff.max()):.10g}")
    print(f"mean_abs_diff: {float(diff.mean()):.10g}")
    print(f"p99_abs_diff: {float(np.quantile(diff, 0.99)):.10g}")
    print(f"top1_agreement: {float(np.mean(top1_match)):.6f}")
    print(f"top1_flip_count: {int(np.sum(~top1_match))}")
    print_top1_diagnostics(
        reference,
        candidate,
        labels,
        reference_top,
        candidate_top,
        diff,
        max_rows=diagnostic_rows,
    )


def print_top1_diagnostics(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: Sequence[str],
    reference_top: np.ndarray,
    candidate_top: np.ndarray,
    diff: np.ndarray,
    *,
    max_rows: int,
) -> None:
    flip_rows = np.flatnonzero(reference_top != candidate_top)
    if len(flip_rows) == 0:
        return
    reference_margins = top2_margins(reference)
    candidate_margins = top2_margins(candidate)
    print(f"top1_flip_reference_margin_min: {float(reference_margins[flip_rows].min()):.10g}")
    print(f"top1_flip_reference_margin_median: {float(np.median(reference_margins[flip_rows])):.10g}")
    print(f"top1_flip_candidate_margin_median: {float(np.median(candidate_margins[flip_rows])):.10g}")
    print(f"top1_flip_row_max_abs_diff_max: {float(diff[flip_rows].max(axis=1).max()):.10g}")
    for row in flip_rows[:max(0, max_rows)]:
        ref_i = int(reference_top[row])
        cand_i = int(candidate_top[row])
        print(
            "top1_flip: "
            f"row={int(row)} "
            f"time_sec={row * HOP_SEC:.2f} "
            f"reference={labels[ref_i]!r}:{reference[row, ref_i]:.8f} "
            f"candidate={labels[cand_i]!r}:{candidate[row, cand_i]:.8f} "
            f"reference_margin={reference_margins[row]:.8f} "
            f"row_max_abs_diff={diff[row].max():.8f}"
        )


def top2_margins(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] < 2:
        return np.full(scores.shape[0], np.inf, dtype=np.float32)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def print_timings(title: str, timings: Sequence[float]) -> None:
    values = np.asarray(timings, dtype=np.float64)
    print(f"\n=== {title} timings ===")
    print(f"runs: {len(values)}")
    print(f"mean_seconds: {float(values.mean()):.4f}")
    print(f"min_seconds: {float(values.min()):.4f}")
    print(f"max_seconds: {float(values.max()):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="*", help="Optional audio paths or globs.")
    parser.add_argument("--model", default="iic/emotion2vec_plus_large")
    parser.add_argument("--device", default="cuda", help="Example: cuda, cuda:0, cpu.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-windows-per-file", type=int, default=1000)
    parser.add_argument("--synthetic-count", type=int, default=4)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--onnx-path", type=Path)
    parser.add_argument("--reuse-onnx", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help=(
            "ONNX Runtime provider to benchmark. May be repeated. "
            "Aliases: cpu, cuda, tensorrt."
        ),
    )
    parser.add_argument("--trt-cache-dir", type=Path)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--torch-allow-tf32", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--diagnostic-rows", type=int, default=5)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    configure_torch_matmul(allow_tf32=args.torch_allow_tf32)
    signals = load_audio_paths(args.audio) if args.audio else synthetic_signals(
        args.synthetic_count,
        args.duration_sec,
    )
    signals = [trim_to_max_windows(samples, args.max_windows_per_file) for samples in signals]
    windows = frame_signals(signals)

    onnx_path = args.onnx_path
    if onnx_path is None:
        onnx_path = Path("/tmp") / (
            f"emotion2vec_plus_large_b{args.batch_size}_"
            f"w{int(round(WINDOW_SEC * SAMPLE_RATE))}_opset{args.opset}.onnx"
        )

    print(f"windows_shape: {windows.shape}")
    print(f"onnx_path: {onnx_path}")
    print(f"torch_allow_tf32: {args.torch_allow_tf32}")

    auto_model = AutoModel(
        model=args.model,
        device=args.device,
        disable_update=True,
        disable_pbar=True,
    )
    scorer = make_direct_emotion2vec_scorer(
        auto_model,
        sample_rate=SAMPLE_RATE,
        compile_model=False,
    )
    if scorer is None:
        raise RuntimeError("Loaded emotion2vec model does not support the direct scorer path")

    scorer.core.eval()
    dummy = torch.zeros(
        (args.batch_size, int(round(WINDOW_SEC * SAMPLE_RATE))),
        dtype=torch.float32,
        device=scorer.device,
    )
    with torch.inference_mode():
        _ = scorer(_pad_tensor_batch(dummy, args.batch_size))
    sync = cuda_sync_for_scorer(scorer)

    if not args.reuse_onnx or not onnx_path.exists():
        start = time.perf_counter()
        export_onnx(scorer.core, onnx_path, batch_size=args.batch_size, opset=args.opset)
        if sync is not None:
            sync()
        print(f"export_seconds: {time.perf_counter() - start:.4f}")
    else:
        print("export_seconds: reused_existing_onnx")

    reference_scores, torch_timings = benchmark(
        lambda: run_torch_batches(scorer, windows, batch_size=args.batch_size),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        sync=sync,
    )
    torch_seconds = float(np.mean(torch_timings))
    print_timings("PyTorch direct scorer", torch_timings)
    print(f"labels: {list(scorer.selected_labels)}")

    providers = args.provider or ["cuda", "tensorrt"]
    for provider in providers:
        provider_name = resolve_provider_alias(provider)
        try:
            session = make_session(
                onnx_path,
                provider_name,
                cache_dir=args.trt_cache_dir,
                fallback=not args.no_fallback,
            )
        except RuntimeError as exc:
            print(f"\n=== {provider_name} ===")
            print(f"SKIP_PROVIDER: {exc}")
            continue

        print(f"\nprovider_requested: {provider_name}")
        print(f"provider_session_chain: {session.get_providers()}")
        if not requested_provider_active(session, provider_name):
            print(
                "SKIP_PROVIDER_FALLBACK: requested provider is not active; "
                "ONNX Runtime fell back to another provider."
            )
            continue
        candidate_scores, ort_timings = benchmark(
            lambda: run_ort_batches(session, windows, batch_size=args.batch_size),
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        print_timings(provider_name, ort_timings)
        summarize_scores(
            provider_name,
            reference_scores,
            candidate_scores,
            scorer.selected_labels,
            torch_seconds,
            float(np.mean(ort_timings)),
            diagnostic_rows=args.diagnostic_rows,
        )


if __name__ == "__main__":
    main()
