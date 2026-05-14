#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from funasr import AutoModel


SAMPLE_RATE = 16_000
WINDOW_SEC = 3.0
HOP_SEC = 0.25


def frame_audio(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    window_samples = int(round(WINDOW_SEC * sample_rate))
    hop_samples = int(round(HOP_SEC * sample_rate))

    audio = np.ascontiguousarray(samples, dtype=np.float32)
    if len(audio) < window_samples:
        pad_needed = window_samples - len(audio)
    else:
        remainder = (len(audio) - window_samples) % hop_samples
        pad_needed = 0 if remainder == 0 else hop_samples - remainder

    if pad_needed:
        audio = np.pad(audio, (0, pad_needed), mode="constant")

    n_frames = 1 + (len(audio) - window_samples) // hop_samples
    stride = audio.strides[0]
    return np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, window_samples),
        strides=(hop_samples * stride, stride),
        writeable=False,
    )


def synthetic_signals(count: int, duration_sec: float) -> list[np.ndarray]:
    rng = np.random.default_rng(12345)
    signals = []
    t = np.arange(int(duration_sec * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE

    for i in range(count):
        base = (
            0.05 * np.sin(2 * np.pi * (180 + 17 * i) * t)
            + 0.03 * np.sin(2 * np.pi * (410 + 23 * i) * t)
            + 0.01 * rng.standard_normal(len(t), dtype=np.float32)
        )
        envelope = np.clip(np.sin(np.pi * t / max(duration_sec, 1e-6)), 0.0, 1.0)
        signals.append(np.ascontiguousarray(base * envelope, dtype=np.float32))

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


def batch_iter(windows: np.ndarray, batch_size: int):
    for start in range(0, len(windows), batch_size):
        yield windows[start : start + batch_size]


def original_generate_scores(
    auto_model,
    windows: np.ndarray,
    *,
    batch_size: int,
    pass_outer_batch_size: bool,
):
    all_scores = []
    labels = None

    for batch_np in batch_iter(windows, batch_size):
        batch = [np.ascontiguousarray(batch_np[i], dtype=np.float32) for i in range(len(batch_np))]
        cfg = {
            "fs": SAMPLE_RATE,
            "granularity": "utterance",
            "extract_embedding": False,
            "disable_pbar": True,
        }
        if pass_outer_batch_size:
            cfg["batch_size"] = len(batch)

        results = auto_model.generate(input=batch, **cfg)

        if labels is None:
            labels = list(results[0]["labels"])
        all_scores.extend(result["scores"] for result in results)

    return np.asarray(all_scores, dtype=np.float32), labels


def direct_batched_scores(auto_model, windows: np.ndarray, *, batch_size: int):
    model = auto_model.model
    kwargs = auto_model.kwargs
    tokenizer = kwargs["tokenizer"]

    if not hasattr(model, "cfg"):
        raise RuntimeError("Underlying FunASR model has no cfg; direct path unsupported")
    if not hasattr(model, "extract_features"):
        raise RuntimeError("Underlying FunASR model has no extract_features; direct path unsupported")
    if getattr(model, "proj", None) is None:
        raise RuntimeError("Underlying FunASR model has no projection head; direct path unsupported")
    if tokenizer is None or not hasattr(tokenizer, "token_list"):
        raise RuntimeError("FunASR tokenizer has no token_list; direct path unsupported")

    labels = list(tokenizer.token_list)
    keep_indices = [idx for idx, label in enumerate(labels) if not str(label).startswith("unuse")]
    selected_labels = [labels[idx] for idx in keep_indices]

    device = next(model.parameters()).device
    unuse_mask = torch.tensor(
        [str(label).startswith("unuse") for label in labels],
        dtype=torch.bool,
        device=device,
    )

    model.eval()
    all_scores = []

    with torch.inference_mode():
        for batch_np in batch_iter(windows, batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(batch_np, dtype=np.float32)).to(device)

            if model.cfg.get("normalize", False):
                batch = F.layer_norm(batch, batch.shape[1:])

            feats = model.extract_features(batch, padding_mask=None)
            x = feats["x"].mean(dim=1)
            logits = model.proj(x)
            logits = logits.masked_fill(unuse_mask.unsqueeze(0), -torch.inf)
            scores = torch.softmax(logits, dim=-1)

            all_scores.append(scores[:, keep_indices].detach().cpu().numpy())

    return np.concatenate(all_scores, axis=0).astype(np.float32, copy=False), selected_labels


def sync_if_cuda(auto_model):
    device = next(auto_model.model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(old_scores, new_scores, old_labels, new_labels, old_sec, new_sec):
    diff = np.abs(old_scores - new_scores)
    old_top = np.argmax(old_scores, axis=1)
    new_top = np.argmax(new_scores, axis=1)

    print()
    print("=== A/B result ===")
    print(f"labels_equal: {old_labels == new_labels}")
    print(f"shape_old: {old_scores.shape}")
    print(f"shape_new: {new_scores.shape}")
    print(f"old_seconds: {old_sec:.4f}")
    print(f"new_seconds: {new_sec:.4f}")
    print(f"speedup: {old_sec / max(new_sec, 1e-9):.2f}x")
    print(f"max_abs_diff: {float(diff.max()):.10g}")
    print(f"mean_abs_diff: {float(diff.mean()):.10g}")
    print(f"p99_abs_diff: {float(np.quantile(diff, 0.99)):.10g}")
    print(f"top1_agreement: {float(np.mean(old_top == new_top)):.6f}")
    print(f"old_row_sum_max_err: {float(np.max(np.abs(old_scores.sum(axis=1) - 1.0))):.10g}")
    print(f"new_row_sum_max_err: {float(np.max(np.abs(new_scores.sum(axis=1) - 1.0))):.10g}")

    worst = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"worst_index: row={worst[0]} class={worst[1]}")
    print(f"worst_label: {old_labels[worst[1]] if old_labels == new_labels else 'labels differ'}")
    print(f"old_worst_value: {old_scores[worst]:.10g}")
    print(f"new_worst_value: {new_scores[worst]:.10g}")

    passed = (
        old_labels == new_labels
        and old_scores.shape == new_scores.shape
        and float(diff.max()) <= 1e-5
        and float(np.mean(old_top == new_top)) == 1.0
    )
    print(f"PASS_NEGLIGIBLE_DIFF: {passed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="*", help="Optional audio paths or globs. If omitted, synthetic signals are used.")
    parser.add_argument("--model", default="iic/emotion2vec_plus_large")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu. Defaults to FunASR choice.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--synthetic-count", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--max-windows", type=int, default=160, help="Caps total windows so old path does not run forever.")
    parser.add_argument(
        "--pass-outer-batch-size",
        action="store_true",
        help="Pass batch_size into AutoModel.generate. Leave off to mimic the original code more closely.",
    )
    args = parser.parse_args()

    auto_kwargs = {
        "model": args.model,
        "disable_update": True,
        "disable_pbar": True,
    }
    if args.device:
        auto_kwargs["device"] = args.device

    print(f"Loading model: {args.model}")
    auto_model = AutoModel(**auto_kwargs)
    device = next(auto_model.model.parameters()).device
    print(f"Device: {device}")

    if args.audio:
        signals = load_audio_paths(args.audio)
        print(f"Loaded real audio files: {len(signals)}")
    else:
        signals = synthetic_signals(args.synthetic_count, args.duration_sec)
        print(f"Generated synthetic signals: {len(signals)}")

    windows = np.concatenate([frame_audio(signal) for signal in signals], axis=0)
    if args.max_windows and len(windows) > args.max_windows:
        windows = windows[: args.max_windows]

    print(f"Total windows: {len(windows)}")
    print(f"Window shape: {windows.shape}")
    print(f"Batch size: {args.batch_size}")

    # Small warmup.
    warm = windows[: min(len(windows), max(1, min(args.batch_size, 4)))]
    _ = direct_batched_scores(auto_model, warm, batch_size=max(1, min(args.batch_size, len(warm))))
    sync_if_cuda(auto_model)
    _ = original_generate_scores(
        auto_model,
        warm,
        batch_size=max(1, min(args.batch_size, len(warm))),
        pass_outer_batch_size=args.pass_outer_batch_size,
    )
    sync_if_cuda(auto_model)

    start = time.perf_counter()
    old_scores, old_labels = original_generate_scores(
        auto_model,
        windows,
        batch_size=args.batch_size,
        pass_outer_batch_size=args.pass_outer_batch_size,
    )
    sync_if_cuda(auto_model)
    old_sec = time.perf_counter() - start

    start = time.perf_counter()
    new_scores, new_labels = direct_batched_scores(auto_model, windows, batch_size=args.batch_size)
    sync_if_cuda(auto_model)
    new_sec = time.perf_counter() - start

    summarize(old_scores, new_scores, old_labels, new_labels, old_sec, new_sec)


if __name__ == "__main__":
    main()
