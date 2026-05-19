#!/usr/bin/env python3
"""A/B current WavLM preprocessing against the legacy per-window loop.

This script verifies that the optimized WavLM-large input preparation keeps
model outputs unchanged. It runs both forward paths on the same loaded model
instance, so differences are attributable to the preprocessing path only.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import time
import traceback

import numpy as np
import torch
from speechbrain.integrations.huggingface import make_padding_masks

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


def ensure_windows(audio, window_sec: float, min_windows: int) -> np.ndarray:
    window = int(round(window_sec * audio.sample_rate))
    hop = int(round(DEFAULT_HOP_SEC * audio.sample_rate))
    needed = window + max(0, min_windows - 1) * hop
    samples = audio.samples
    if len(samples) < needed:
        reps = int(np.ceil(needed / max(1, len(samples))))
        samples = np.tile(samples, reps).astype(np.float32, copy=False)
    return frame_audio(
        samples,
        sample_rate=audio.sample_rate,
        window_sec=window_sec,
        hop_sec=DEFAULT_HOP_SEC,
    )


def legacy_forward_factory(module, task: str):
    def forward(self, x, length=None, return_feature=False):
        if self.pretrain_model == "wavlm_large":
            with torch.no_grad():
                signal = []
                if length is not None:
                    attention_mask = make_padding_masks(
                        x,
                        wav_len=length / length.max(),
                    ).to(x.device)
                else:
                    attention_mask = make_padding_masks(
                        x,
                        wav_len=torch.ones(len(x)).to(x.device),
                    ).to(x.device)
                for idx in range(len(x)):
                    item = self.processor(
                        x[idx],
                        sampling_rate=16_000,
                        return_tensors="pt",
                        padding=True,
                    )
                    signal.append(item["input_values"][0].to(x.device))
                signal = torch.stack(signal)

        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu()).to(x.device)

        if self.pretrain_model == "wavlm":
            hidden_states = self.backbone_model(
                x,
                output_hidden_states=True,
            ).hidden_states
        else:
            hidden_states = self.backbone_model(
                signal,
                attention_mask=attention_mask,
                output_hidden_states=True,
            ).hidden_states

        if self.use_conv_output:
            stacked_feature = torch.stack(hidden_states, dim=0)
        else:
            stacked_feature = torch.stack(hidden_states, dim=0)[1:]

        _, *origin_shape = stacked_feature.shape
        if self.use_conv_output:
            stacked_feature = stacked_feature.view(
                self.backbone_model.config.num_hidden_layers + 1,
                -1,
            )
        else:
            stacked_feature = stacked_feature.view(
                self.backbone_model.config.num_hidden_layers,
                -1,
            )
        norm_weights = module.F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        features = self.model_seq(features.transpose(1, 2)).transpose(1, 2)
        if length is not None:
            features = torch.stack([
                torch.mean(features[i, 0:length[i], ...], dim=0)
                for i in range(features.shape[0])
            ])
        else:
            features = torch.mean(features, dim=1)

        if task == "affect":
            arousal = self.arousal_layer(features)
            valence = self.valence_layer(features)
            dominance = self.dominance_layer(features)
            if getattr(self, "predict_gender", False):
                return arousal, valence, dominance, self.gender_layer(features)
            if return_feature:
                return features, arousal, valence, dominance
            return arousal, valence, dominance

        return self.fluency_layer(features), self.dysfluency_layer(features)

    return forward


def run_forward(model, cls, forward, batch):
    original = cls.forward
    cls.forward = forward
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "status": "ok",
            "elapsed_sec": elapsed,
            "cuda": cuda_stats(),
            "outputs": [out.detach().cpu().numpy() for out in outputs],
        }
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
            "cuda": cuda_stats(),
        }
    finally:
        cls.forward = original


def compare_outputs(legacy, current) -> dict:
    if legacy["status"] != "ok" or current["status"] != "ok":
        return {"status": "skipped"}
    rows = []
    for idx, (old, new) in enumerate(zip(legacy["outputs"], current["outputs"])):
        rows.append({
            "output": idx,
            "shape": list(new.shape),
            "exact_equal": bool(np.array_equal(old, new)),
            "max_abs_diff": float(np.max(np.abs(old - new))),
        })
    return {
        "status": "ok",
        "all_exact_equal": all(row["exact_equal"] for row in rows),
        "rows": rows,
    }


def task_spec(task: str):
    if task == "affect":
        return (
            importlib.import_module(
                "audio_classification_playground.vox_profile.emotion.wavlm_emotion_dim"
            ),
            DEFAULT_AFFECT_MODELS["wavlm"],
            AFFECT_WINDOW_SEC,
        )
    return (
        importlib.import_module(
            "audio_classification_playground.vox_profile.fluency.wavlm_fluency"
        ),
        DEFAULT_DISFLUENCY_MODELS["wavlm"],
        DISFLUENCY_WINDOW_SEC,
    )


def run_task(args, audio, task: str) -> dict:
    module, model_id, window_sec = task_spec(task)
    cls = module.WavLMWrapper
    windows = ensure_windows(audio, window_sec, args.min_windows)
    batch_np = writable_contiguous_float32(windows[: args.batch_size])
    batch_cpu = torch.from_numpy(batch_np)
    batch_gpu = batch_cpu.to(args.device)

    cuda_reset()
    result = {
        "task": task,
        "model_id": model_id,
        "batch_size": int(len(batch_np)),
        "window_sec": window_sec,
        "status": "ok",
    }
    try:
        load_started = time.perf_counter()
        model = cls.from_pretrained(model_id).to(args.device).eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["model_load_sec"] = time.perf_counter() - load_started
        result["after_model_load_cuda"] = cuda_stats()

        legacy = run_forward(
            model,
            cls,
            legacy_forward_factory(module, task),
            batch_gpu,
        )
        current = run_forward(model, cls, cls.forward, batch_cpu)
        result["legacy"] = {k: v for k, v in legacy.items() if k != "outputs"}
        result["current"] = {k: v for k, v in current.items() if k != "outputs"}
        result["comparison"] = compare_outputs(legacy, current)
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=5)
    finally:
        cuda_reset()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["affect", "disfluency"],
        default=["affect", "disfluency"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-windows", type=int, default=64)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)
    report = {
        "audio": args.audio,
        "duration_sec": audio.duration_sec,
        "device": args.device,
        "tasks": [run_task(args, audio, task) for task in args.tasks],
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
