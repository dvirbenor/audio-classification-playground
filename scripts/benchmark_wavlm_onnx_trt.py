#!/usr/bin/env python3
"""Benchmark WavLM ONNX/TensorRT candidates against the PyTorch reference.

The exported graph accepts WavLM-large *prepared* input values: float32
waveform windows after the same per-window zero-mean/unit-variance
normalization used by the production wrapper.  The report still compares
candidate outputs against the original raw-window PyTorch wrapper so the
runtime can be judged against the current production behavior.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import (
    frame_audio,
    load_audio,
    writable_contiguous_float32,
)
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC,
    DEFAULT_AFFECT_MODELS,
    DEFAULT_DISFLUENCY_MODELS,
    DEFAULT_HOP_SEC,
    DISFLUENCY_WINDOW_SEC,
    _load_affect_wrapper,
    _load_disfluency_wrapper,
)
from audio_classification_playground.vox_profile.wavlm_inference import (
    prepare_wavlm_large_inputs,
)


OUTPUT_NAMES = {
    "affect": ("arousal", "valence", "dominance"),
    "disfluency": ("fluency_logits", "disfluency_type_logits"),
}


class PreparedWavLMWrapper(nn.Module):
    """Traceable WavLM wrapper that assumes normalized input values."""

    def __init__(self, wrapper: nn.Module, task: str) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.task = task

    def forward(self, input_values):
        hidden_states = self.wrapper.backbone_model(
            input_values,
            attention_mask=None,
            output_hidden_states=True,
        ).hidden_states
        features = self.wrapper._weighted_features_from_hidden_states(hidden_states)
        if self.task == "affect":
            return self.wrapper._predict_from_features(features)
        return self.wrapper._predict_from_features(features, return_feature=False)


def sync_device(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


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


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def ensure_windows(audio, *, task: str, batch_size: int, num_batches: int) -> np.ndarray:
    window_sec = AFFECT_WINDOW_SEC if task == "affect" else DISFLUENCY_WINDOW_SEC
    needed_windows = int(batch_size) * int(num_batches)
    window = int(round(window_sec * audio.sample_rate))
    hop = int(round(DEFAULT_HOP_SEC * audio.sample_rate))
    needed_samples = window + max(0, needed_windows - 1) * hop
    samples = audio.samples
    if len(samples) < needed_samples:
        reps = int(np.ceil(needed_samples / max(1, len(samples))))
        samples = np.tile(samples, reps).astype(np.float32, copy=False)
    windows = frame_audio(
        samples,
        sample_rate=audio.sample_rate,
        window_sec=window_sec,
        hop_sec=DEFAULT_HOP_SEC,
    )
    return windows[:needed_windows]


def iter_batches(windows: np.ndarray, batch_size: int):
    for start in range(0, len(windows), batch_size):
        yield windows[start : start + batch_size]


def load_wavlm_wrapper(task: str, device: str):
    if task == "affect":
        wrapper_cls = _load_affect_wrapper("wavlm")
        model_id = DEFAULT_AFFECT_MODELS["wavlm"]
    else:
        wrapper_cls = _load_disfluency_wrapper("wavlm")
        model_id = DEFAULT_DISFLUENCY_MODELS["wavlm"]
    model = wrapper_cls.from_pretrained(model_id).to(device).eval()
    return model_id, model


def prepare_input_batches(wrapper, windows: np.ndarray, batch_size: int) -> tuple[list[np.ndarray], dict]:
    prepared = []
    started = time.perf_counter()
    for batch_np in iter_batches(windows, batch_size):
        batch = torch.from_numpy(writable_contiguous_float32(batch_np))
        signal, attention_mask = prepare_wavlm_large_inputs(
            wrapper.processor,
            batch,
            device="cpu",
        )
        if attention_mask is not None:
            raise RuntimeError("ONNX/TensorRT harness expects fixed-size unmasked windows")
        prepared.append(signal.numpy())
    return prepared, {"elapsed_sec": time.perf_counter() - started}


def tensors_to_numpy(outputs: Sequence[torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().float().cpu().numpy()
        for name, tensor in zip(CURRENT_OUTPUT_NAMES, outputs)
    }


def run_pytorch_original(wrapper, windows: np.ndarray, *, batch_size: int, device: str) -> dict:
    rows: list[list[torch.Tensor]] = [[] for _ in CURRENT_OUTPUT_NAMES]
    reset_cuda_peak()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_np in iter_batches(windows, batch_size):
            batch = torch.from_numpy(writable_contiguous_float32(batch_np))
            outputs = wrapper(batch)
            if not isinstance(outputs, tuple):
                outputs = tuple(outputs)
            for idx, output in enumerate(outputs):
                rows[idx].append(output.detach().float())
    sync_device(device)
    elapsed = time.perf_counter() - started
    arrays = {
        name: torch.cat(parts, dim=0).cpu().numpy()
        for name, parts in zip(CURRENT_OUTPUT_NAMES, rows)
    }
    return {"elapsed_sec": elapsed, "cuda": cuda_stats(), "outputs": arrays}


def run_pytorch_prepared(model: nn.Module, inputs: list[np.ndarray], *, device: str) -> dict:
    rows: list[list[torch.Tensor]] = [[] for _ in CURRENT_OUTPUT_NAMES]
    reset_cuda_peak()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_np in inputs:
            batch = torch.from_numpy(batch_np).to(device)
            outputs = model(batch)
            if not isinstance(outputs, tuple):
                outputs = tuple(outputs)
            for idx, output in enumerate(outputs):
                rows[idx].append(output.detach().float())
    sync_device(device)
    elapsed = time.perf_counter() - started
    arrays = {
        name: torch.cat(parts, dim=0).cpu().numpy()
        for name, parts in zip(CURRENT_OUTPUT_NAMES, rows)
    }
    return {"elapsed_sec": elapsed, "cuda": cuda_stats(), "outputs": arrays}


def export_onnx(
    model: nn.Module,
    dummy_input: np.ndarray,
    onnx_path: Path,
    *,
    device: str,
    opset: int,
    dynamic_batch: bool,
) -> dict:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.from_numpy(dummy_input).to(device)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input_values": {0: "batch"}}
        for name in CURRENT_OUTPUT_NAMES:
            dynamic_axes[name] = {0: "batch"}
    started = time.perf_counter()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input_values"],
        output_names=list(CURRENT_OUTPUT_NAMES),
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    return {
        "path": str(onnx_path),
        "elapsed_sec": time.perf_counter() - started,
        "size_mb": onnx_path.stat().st_size / 1_000_000,
        "opset": opset,
        "dynamic_batch": dynamic_batch,
    }


def run_onnxruntime(onnx_path: Path, inputs: list[np.ndarray], *, device: str) -> dict:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    providers = ort.get_available_providers()
    requested = ["CPUExecutionProvider"]
    if torch.device(device).type == "cuda" and "CUDAExecutionProvider" in providers:
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=requested)
    actual_providers = session.get_providers()
    output_names = [output.name for output in session.get_outputs()]

    rows: list[list[np.ndarray]] = [[] for _ in output_names]
    started = time.perf_counter()
    for batch_np in inputs:
        outputs = session.run(output_names, {"input_values": batch_np})
        for idx, output in enumerate(outputs):
            rows[idx].append(np.asarray(output, dtype=np.float32))
    elapsed = time.perf_counter() - started

    arrays = {
        name: np.concatenate(parts, axis=0)
        for name, parts in zip(output_names, rows)
    }
    return {
        "status": "ok",
        "elapsed_sec": elapsed,
        "providers": actual_providers,
        "outputs": arrays,
    }


def run_trtexec(
    onnx_path: Path,
    *,
    trtexec_path: str | None,
    batch_size: int,
    input_samples: int,
    precision: str,
    workspace_mb: int,
    min_timing_ms: int,
) -> dict:
    executable = trtexec_path or shutil.which("trtexec")
    if not executable:
        return {"status": "unavailable", "error": "trtexec not found"}

    cmd = [
        executable,
        f"--onnx={onnx_path}",
        f"--shapes=input_values:{batch_size}x{input_samples}",
        f"--workspace={workspace_mb}",
        f"--minTiming={min_timing_ms}",
        "--useCudaGraph",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision != "fp32":
        raise ValueError(f"Unsupported TensorRT precision {precision!r}")

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "elapsed_sec": time.perf_counter() - started,
        "returncode": proc.returncode,
        "command": cmd,
        "output_tail": proc.stdout[-6000:],
    }


def compare_outputs(
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
        "allclose": all(row["allclose"] for row in rows),
        "atol": atol,
        "rtol": rtol,
        "rows": rows,
    }
    if "fluency_logits" in reference:
        ref_f = np.asarray(reference["fluency_logits"], dtype=np.float32)
        cand_f = np.asarray(candidate["fluency_logits"], dtype=np.float32)
        out["fluency_top1_agreement"] = float(
            np.mean(np.argmax(ref_f, axis=1) == np.argmax(cand_f, axis=1))
        )
    if "disfluency_type_logits" in reference:
        ref_t = np.asarray(reference["disfluency_type_logits"], dtype=np.float32)
        cand_t = np.asarray(candidate["disfluency_type_logits"], dtype=np.float32)
        out["type_logit_sign_agreement"] = float(
            np.mean((ref_t >= 0.0) == (cand_t >= 0.0))
        )
    return out


def strip_outputs(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "outputs"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("affect", "disfluency"), required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--output-dir", default="/tmp/wavlm_onnx_trt")
    parser.add_argument("--onnx-path")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--skip-onnxruntime", action="store_true")
    parser.add_argument("--run-trtexec", action="store_true")
    parser.add_argument("--trtexec-path")
    parser.add_argument("--trt-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--trt-workspace-mb", type=int, default=8192)
    parser.add_argument("--trt-min-timing-ms", type=int, default=500)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    global CURRENT_OUTPUT_NAMES
    CURRENT_OUTPUT_NAMES = OUTPUT_NAMES[args.task]

    report = {
        "task": args.task,
        "audio": args.audio,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(torch.device(args.device))
                if torch.device(args.device).type == "cuda" and torch.cuda.is_available()
                else None
            ),
        },
    }

    try:
        audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)
        windows = ensure_windows(
            audio,
            task=args.task,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
        )
        report["duration_sec"] = audio.duration_sec
        report["window_count"] = int(len(windows))

        load_started = time.perf_counter()
        model_id, wrapper = load_wavlm_wrapper(args.task, args.device)
        report["model"] = {"id": model_id}
        report["model_load_sec"] = time.perf_counter() - load_started

        prepared_model = PreparedWavLMWrapper(wrapper, args.task).to(args.device).eval()
        prepared_inputs, preprocess = prepare_input_batches(wrapper, windows, args.batch_size)
        report["preprocess"] = preprocess

        original = run_pytorch_original(
            wrapper,
            windows,
            batch_size=args.batch_size,
            device=args.device,
        )
        prepared = run_pytorch_prepared(
            prepared_model,
            prepared_inputs,
            device=args.device,
        )
        report["pytorch_original"] = strip_outputs(original)
        report["pytorch_prepared"] = strip_outputs(prepared)
        report["prepared_vs_original"] = compare_outputs(
            original["outputs"],
            prepared["outputs"],
            atol=args.atol,
            rtol=args.rtol,
        )

        output_dir = Path(args.output_dir)
        onnx_path = Path(args.onnx_path) if args.onnx_path else (
            output_dir / f"{args.task}_wavlm_bs{args.batch_size}.onnx"
        )
        if args.skip_export:
            report["onnx_export"] = {"status": "skipped", "path": str(onnx_path)}
        else:
            report["onnx_export"] = {
                "status": "ok",
                **export_onnx(
                    prepared_model,
                    prepared_inputs[0],
                    onnx_path,
                    device=args.device,
                    opset=args.opset,
                    dynamic_batch=args.dynamic_batch,
                ),
            }

        if not args.skip_onnxruntime:
            ort_result = run_onnxruntime(onnx_path, prepared_inputs, device=args.device)
            if ort_result.get("status") == "ok":
                report["onnxruntime"] = strip_outputs(ort_result)
                report["onnxruntime_vs_original"] = compare_outputs(
                    original["outputs"],
                    ort_result["outputs"],
                    atol=args.atol,
                    rtol=args.rtol,
                )
            else:
                report["onnxruntime"] = ort_result

        if args.run_trtexec:
            report["trtexec"] = run_trtexec(
                onnx_path,
                trtexec_path=args.trtexec_path,
                batch_size=args.batch_size,
                input_samples=prepared_inputs[0].shape[1],
                precision=args.trt_precision,
                workspace_mb=args.trt_workspace_mb,
                min_timing_ms=args.trt_min_timing_ms,
            )

    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=12)
    else:
        report["status"] = "ok"

    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
