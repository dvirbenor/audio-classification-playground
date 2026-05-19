#!/usr/bin/env python3
"""Compare framed-window vs audio-fed emotion2vec inference paths."""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
from funasr import AutoModel

from audio_classification_playground.acoustic_events.inference.audio import frame_audio
from audio_classification_playground.acoustic_events.inference.emotion2vec import (
    make_direct_emotion2vec_scorer,
    predict_emotion2vec_scores,
    predict_emotion2vec_scores_from_audio,
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

    paths = []
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


def sync_if_cuda(auto_model) -> None:
    device = next(auto_model.model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_tf32(*, enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def precision_state() -> dict[str, object]:
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def run_framed(
    auto_model,
    signals: list[np.ndarray],
    batch_size: int,
    *,
    autocast_dtype: str | None = None,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
):
    all_scores = []
    labels = None
    for samples in signals:
        windows = frame_audio(
            samples,
            sample_rate=SAMPLE_RATE,
            window_sec=WINDOW_SEC,
            hop_sec=HOP_SEC,
        )
        scores, labels_i = predict_emotion2vec_scores(
            auto_model,
            windows,
            sample_rate=SAMPLE_RATE,
            batch_size=batch_size,
            autocast_dtype=autocast_dtype,
            compile_model=compile_model,
            compile_mode=compile_mode,
        )
        if labels is None:
            labels = list(labels_i)
        elif list(labels_i) != labels:
            raise RuntimeError("labels changed between signals")
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0), labels


def run_audio_fed(
    auto_model,
    signals: list[np.ndarray],
    batch_size: int,
    *,
    autocast_dtype: str | None = None,
    direct_scorer=None,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
):
    all_scores = []
    labels = None
    for samples in signals:
        if direct_scorer is not None:
            scores, labels_i = direct_scorer.predict_audio(
                samples,
                sample_rate=SAMPLE_RATE,
                window_sec=WINDOW_SEC,
                hop_sec=HOP_SEC,
                batch_size=batch_size,
                autocast_dtype=autocast_dtype,
            )
        else:
            scores, labels_i = predict_emotion2vec_scores_from_audio(
                auto_model,
                samples,
                sample_rate=SAMPLE_RATE,
                window_sec=WINDOW_SEC,
                hop_sec=HOP_SEC,
                batch_size=batch_size,
                autocast_dtype=autocast_dtype,
                compile_model=compile_model,
                compile_mode=compile_mode,
            )
        if labels is None:
            labels = list(labels_i)
        elif list(labels_i) != labels:
            raise RuntimeError("labels changed between signals")
        all_scores.append(scores)
    return np.concatenate(all_scores, axis=0), labels


def summarize_pair(
    title: str,
    reference_scores,
    candidate_scores,
    reference_labels,
    candidate_labels,
    reference_sec,
    candidate_sec,
    pass_label: str,
    *,
    max_abs_tolerance: float,
    diagnostic_rows: int,
):
    diff = np.abs(reference_scores - candidate_scores)
    reference_top = np.argmax(reference_scores, axis=1)
    candidate_top = np.argmax(candidate_scores, axis=1)
    top1_match = reference_top == candidate_top
    print(f"\n=== {title} ===")
    print(f"labels_equal: {reference_labels == candidate_labels}")
    print(f"shape_reference: {reference_scores.shape}")
    print(f"shape_candidate: {candidate_scores.shape}")
    print(f"reference_seconds: {reference_sec:.4f}")
    print(f"candidate_seconds: {candidate_sec:.4f}")
    print(f"speedup: {reference_sec / max(candidate_sec, 1e-9):.2f}x")
    print(f"max_abs_diff: {float(diff.max()):.10g}")
    print(f"mean_abs_diff: {float(diff.mean()):.10g}")
    print(f"p99_abs_diff: {float(np.quantile(diff, 0.99)):.10g}")
    print(f"top1_agreement: {float(np.mean(top1_match)):.6f}")
    _print_top1_diagnostics(
        reference_scores,
        candidate_scores,
        reference_labels,
        reference_top,
        candidate_top,
        diff,
        max_rows=diagnostic_rows,
    )
    passed = (
        reference_labels == candidate_labels
        and reference_scores.shape == candidate_scores.shape
        and float(diff.max()) <= max_abs_tolerance
        and float(np.mean(top1_match)) == 1.0
    )
    print(f"{pass_label}: {passed}")


def _top2_margins(scores: np.ndarray) -> np.ndarray:
    if scores.shape[1] < 2:
        return np.full(scores.shape[0], np.inf, dtype=np.float32)
    top2 = np.partition(scores, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _print_top1_diagnostics(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    labels: list[str],
    reference_top: np.ndarray,
    candidate_top: np.ndarray,
    diff: np.ndarray,
    *,
    max_rows: int,
) -> None:
    flip_rows = np.flatnonzero(reference_top != candidate_top)
    print(f"top1_flip_count: {int(len(flip_rows))}")
    if len(flip_rows) == 0:
        return

    reference_margins = _top2_margins(reference_scores)
    candidate_margins = _top2_margins(candidate_scores)
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
            f"reference={labels[ref_i]!r}:{reference_scores[row, ref_i]:.8f} "
            f"candidate={labels[cand_i]!r}:{candidate_scores[row, cand_i]:.8f} "
            f"reference_margin={reference_margins[row]:.8f} "
            f"row_max_abs_diff={diff[row].max():.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="*", help="Optional audio paths or globs.")
    parser.add_argument("--model", default="iic/emotion2vec_plus_large")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--synthetic-count", type=int, default=4)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--max-windows-per-file", type=int, default=160)
    parser.add_argument("--candidate-autocast-dtype", choices=("fp16", "bf16"))
    parser.add_argument("--candidate-compile", action="store_true")
    parser.add_argument("--candidate-compile-mode", default="reduce-overhead")
    parser.add_argument("--candidate-allow-tf32", action="store_true")
    parser.add_argument(
        "--candidate-max-abs-tolerance",
        type=float,
        default=1e-3,
        help="Acceptance threshold for optional runtime knobs.",
    )
    parser.add_argument(
        "--diagnostic-rows",
        type=int,
        default=5,
        help="Number of top-1 flip rows to print.",
    )
    args = parser.parse_args()

    auto_kwargs = {
        "model": args.model,
        "batch_size": args.batch_size,
        "disable_update": True,
        "disable_pbar": True,
    }
    if args.device:
        auto_kwargs["device"] = args.device

    print(f"Loading model: {args.model}")
    auto_model = AutoModel(**auto_kwargs)
    print(f"Device: {next(auto_model.model.parameters()).device}")

    signals = (
        load_audio_paths(args.audio)
        if args.audio
        else synthetic_signals(args.synthetic_count, args.duration_sec)
    )
    signals = [trim_to_max_windows(s, args.max_windows_per_file) for s in signals]
    total_windows = sum(
        len(
            frame_audio(
                s,
                sample_rate=SAMPLE_RATE,
                window_sec=WINDOW_SEC,
                hop_sec=HOP_SEC,
            )
        )
        for s in signals
    )
    print(f"Signals: {len(signals)}")
    print(f"Total windows: {total_windows}")
    print(f"Batch size: {args.batch_size}")
    print(
        "Candidate knobs: "
        f"autocast={args.candidate_autocast_dtype} "
        f"compile={args.candidate_compile} "
        f"tf32={args.candidate_allow_tf32}"
    )

    configure_tf32(enabled=False)
    print(f"Reference precision state: {precision_state()}")

    warm = [signals[0]]
    _ = run_audio_fed(auto_model, warm, args.batch_size)
    sync_if_cuda(auto_model)
    _ = run_framed(auto_model, warm, args.batch_size)
    sync_if_cuda(auto_model)

    start = time.perf_counter()
    framed_scores, framed_labels = run_framed(auto_model, signals, args.batch_size)
    sync_if_cuda(auto_model)
    framed_sec = time.perf_counter() - start

    start = time.perf_counter()
    audio_scores, audio_labels = run_audio_fed(auto_model, signals, args.batch_size)
    sync_if_cuda(auto_model)
    audio_sec = time.perf_counter() - start

    summarize_pair(
        "feed path A/B",
        framed_scores,
        audio_scores,
        framed_labels,
        audio_labels,
        framed_sec,
        audio_sec,
        "PASS_FEED_PATH_EQUIVALENCE",
        max_abs_tolerance=1e-6,
        diagnostic_rows=args.diagnostic_rows,
    )

    if (
        args.candidate_autocast_dtype is not None
        or args.candidate_compile
        or args.candidate_allow_tf32
    ):
        configure_tf32(enabled=args.candidate_allow_tf32)
        print(f"Candidate precision state: {precision_state()}")
        if args.candidate_compile:
            print(f"\nCompiling direct scorer with mode={args.candidate_compile_mode!r}")
        candidate_scorer = make_direct_emotion2vec_scorer(
            auto_model,
            sample_rate=SAMPLE_RATE,
            compile_model=args.candidate_compile,
            compile_mode=args.candidate_compile_mode,
        )
        _ = run_audio_fed(
            auto_model,
            warm,
            args.batch_size,
            autocast_dtype=args.candidate_autocast_dtype,
            direct_scorer=candidate_scorer,
            compile_model=args.candidate_compile,
            compile_mode=args.candidate_compile_mode,
        )
        sync_if_cuda(auto_model)
        start = time.perf_counter()
        candidate_scores, candidate_labels = run_audio_fed(
            auto_model,
            signals,
            args.batch_size,
            autocast_dtype=args.candidate_autocast_dtype,
            direct_scorer=candidate_scorer,
            compile_model=args.candidate_compile,
            compile_mode=args.candidate_compile_mode,
        )
        sync_if_cuda(auto_model)
        candidate_sec = time.perf_counter() - start
        summarize_pair(
            "runtime knob A/B",
            audio_scores,
            candidate_scores,
            audio_labels,
            candidate_labels,
            audio_sec,
            candidate_sec,
            "PASS_RUNTIME_KNOB_TOLERANCE",
            max_abs_tolerance=args.candidate_max_abs_tolerance,
            diagnostic_rows=args.diagnostic_rows,
        )


if __name__ == "__main__":
    main()
