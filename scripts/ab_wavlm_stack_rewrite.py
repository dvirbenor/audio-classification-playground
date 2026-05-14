#!/usr/bin/env python3
import argparse, gc, json, time, traceback, importlib

import numpy as np
import torch

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import load_audio, frame_audio
from audio_classification_playground.acoustic_events.inference.runners import (
    DEFAULT_AFFECT_MODELS,
    DEFAULT_DISFLUENCY_MODELS,
    AFFECT_WINDOW_SEC,
    DISFLUENCY_WINDOW_SEC,
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
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
    }


def ensure_windows(audio, window_sec, min_windows):
    window = int(round(window_sec * audio.sample_rate))
    hop = int(round(DEFAULT_HOP_SEC * audio.sample_rate))
    needed = window + max(0, min_windows - 1) * hop
    samples = audio.samples
    if len(samples) < needed:
        samples = np.tile(samples, int(np.ceil(needed / max(1, len(samples))))).astype(np.float32)
    return frame_audio(
        samples,
        sample_rate=audio.sample_rate,
        window_sec=window_sec,
        hop_sec=DEFAULT_HOP_SEC,
    )


def patched_forward_factory(module, task):
    torch_mod = module.torch
    F = module.F
    make_padding_masks = module.make_padding_masks

    def weighted_hidden_sum(self, hidden_states):
        selected = hidden_states if self.use_conv_output else hidden_states[1:]
        weights = F.softmax(self.weights, dim=-1)
        out = selected[0] * weights[0]
        for layer, weight in zip(selected[1:], weights[1:]):
            out = out + layer * weight
        return out

    def forward(self, x, length=None, return_feature=False):
        if self.pretrain_model == "wavlm_large":
            with torch_mod.no_grad():
                signal = []
                if length is not None:
                    attention_mask = make_padding_masks(x, wav_len=length / length.max()).to(x.device)
                else:
                    attention_mask = make_padding_masks(
                        x, wav_len=torch_mod.ones(len(x)).to(x.device)
                    ).to(x.device)
                for idx in range(len(x)):
                    item = self.processor(
                        x[idx], sampling_rate=16_000, return_tensors="pt", padding=True
                    )
                    signal.append(item["input_values"][0].to(x.device))
                signal = torch_mod.stack(signal)

        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu()).cuda()

        if self.pretrain_model == "wavlm":
            hidden_states = self.backbone_model(x, output_hidden_states=True).hidden_states
        else:
            hidden_states = self.backbone_model(
                signal, attention_mask=attention_mask, output_hidden_states=True
            ).hidden_states

        features = weighted_hidden_sum(self, hidden_states)
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)

        if length is not None:
            features = torch_mod.stack([
                torch_mod.mean(features[i, 0:length[i], ...], dim=0)
                for i in range(features.shape[0])
            ])
        else:
            features = torch_mod.mean(features, dim=1)

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


def run_model(cls, model_id, windows, batch_size, device):
    cuda_reset()
    result = {"status": "ok"}
    try:
        model = cls.from_pretrained(model_id).to(device).eval()
        batch = torch.from_numpy(np.ascontiguousarray(windows[:batch_size])).to(device)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start = time.perf_counter()
        with torch.inference_mode():
            out = model(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["elapsed_sec"] = time.perf_counter() - start
        result["cuda"] = cuda_stats()
        result["outputs"] = [x.detach().cpu().numpy() for x in out]
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc(limit=5)
    finally:
        cuda_reset()
    return result


def compare_outputs(old, new, atol, rtol):
    if old["status"] != "ok" or new["status"] != "ok":
        return {"status": "skipped"}
    rows = []
    for i, (a, b) in enumerate(zip(old["outputs"], new["outputs"])):
        rows.append({
            "output": i,
            "shape": list(a.shape),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        })
    return {"status": "ok", "rows": rows}


def task_spec(task):
    if task == "affect":
        return (
            importlib.import_module("audio_classification_playground.vox_profile.emotion.wavlm_emotion_dim"),
            DEFAULT_AFFECT_MODELS["wavlm"],
            AFFECT_WINDOW_SEC,
        )
    return (
        importlib.import_module("audio_classification_playground.vox_profile.fluency.wavlm_fluency"),
        DEFAULT_DISFLUENCY_MODELS["wavlm"],
        DISFLUENCY_WINDOW_SEC,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", nargs="+", choices=["affect", "disfluency"], default=["affect", "disfluency"])
    ap.add_argument("--correctness-batch-size", type=int, default=64)
    ap.add_argument("--stress-batch-size", type=int, default=512)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    audio = load_audio(args.audio, sample_rate=SAMPLE_RATE)
    report = {}

    for task in args.tasks:
        module, model_id, window_sec = task_spec(task)
        cls = module.WavLMWrapper
        original_forward = cls.forward
        windows = ensure_windows(audio, window_sec, args.stress_batch_size)

        old_small = run_model(cls, model_id, windows, args.correctness_batch_size, args.device)
        cls.forward = patched_forward_factory(module, task)
        try:
            new_small = run_model(cls, model_id, windows, args.correctness_batch_size, args.device)
            new_stress = run_model(cls, model_id, windows, args.stress_batch_size, args.device)
        finally:
            cls.forward = original_forward

        old_stress = run_model(cls, model_id, windows, args.stress_batch_size, args.device)

        for item in (old_small, new_small, old_stress, new_stress):
            item.pop("outputs", None)

        report[task] = {
            "correctness_batch_size": args.correctness_batch_size,
            "stress_batch_size": args.stress_batch_size,
            "comparison": compare_outputs(
                run_model(cls, model_id, windows, args.correctness_batch_size, args.device),
                (setattr(cls, "forward", patched_forward_factory(module, task)) or run_model(cls, model_id, windows, args.correctness_batch_size, args.device)),
                args.atol,
                args.rtol,
            ),
            "old_stress": old_stress,
            "new_stress": new_stress,
        }
        cls.forward = original_forward

    print(json.dumps(report, indent=2))
    if args.json_out:
        open(args.json_out, "w", encoding="utf-8").write(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
