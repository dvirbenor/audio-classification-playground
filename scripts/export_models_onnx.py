#!/usr/bin/env python3
"""Export WavLM affect + disfluency + emotion2vec models to ONNX for Triton.

WavLM models (affect, disfluency) export via torch.onnx.export.
Emotion2vec uses the Triton Python backend (see triton/emotion/1/model.py) —
no ONNX export needed for it, but this script exports a probe run to verify
output shape so the config.pbtxt class count is correct.

Run on a GPU pod with HF_HOME pointing at the shared EFS model cache:

    source env.shared.sh
    uv run python scripts/export_models_onnx.py \\
        --output /efs/triton-model-repo \\
        [--task affect|disfluency|all] \\
        [--device cuda] \\
        [--opset 17]

Then copy config files and start the Triton server:

    cp -r triton/* /efs/triton-model-repo/
    kubectl apply -f manifests/triton-server-deployment.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16_000
AFFECT_WINDOW_SAMPLES = int(3.5 * SAMPLE_RATE)    # 56000
DISFLUENCY_WINDOW_SAMPLES = int(3.0 * SAMPLE_RATE) # 48000
EMOTION_WINDOW_SAMPLES = int(3.0 * SAMPLE_RATE)    # 48000
VALIDATION_BATCH = 4
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Affect
# ---------------------------------------------------------------------------

def export_affect(output_dir: Path, device: str, opset: int) -> None:
    from audio_classification_playground.vox_profile.emotion.wavlm_emotion_dim import WavLMWrapper

    model_id = "tiantiaf/wavlm-large-msp-podcast-emotion-dim"
    print(f"[affect] loading {model_id} on {device} …")
    model = WavLMWrapper.from_pretrained(model_id).to(device).eval()

    dummy = torch.zeros(VALIDATION_BATCH, AFFECT_WINDOW_SAMPLES, device=device)
    with torch.inference_mode():
        a_pt, v_pt, d_pt = model(dummy)
    print(f"[affect] pytorch shapes: arousal={tuple(a_pt.shape)} valence={tuple(v_pt.shape)} dominance={tuple(d_pt.shape)}")

    out_path = output_dir / "affect" / "1" / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["input_values"],
        output_names=["arousal", "valence", "dominance"],
        dynamic_axes={
            "input_values": {0: "batch_size"},
            "arousal": {0: "batch_size"},
            "valence": {0: "batch_size"},
            "dominance": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[affect] exported → {out_path}")
    _validate(
        out_path,
        inputs={"input_values": dummy.cpu().numpy()},
        expected={
            "arousal": a_pt.cpu().float().numpy(),
            "valence": v_pt.cpu().float().numpy(),
            "dominance": d_pt.cpu().float().numpy(),
        },
    )
    out_path = _convert_fp16(out_path)
    # Re-validate the FP16 model with a relaxed tolerance (weight quantisation adds error)
    _validate(out_path, inputs={"input_values": dummy.cpu().numpy()}, expected={
        "arousal": a_pt.cpu().float().numpy(),
        "valence": v_pt.cpu().float().numpy(),
        "dominance": d_pt.cpu().float().numpy(),
    }, atol=1e-2)


# ---------------------------------------------------------------------------
# Disfluency
# ---------------------------------------------------------------------------

class _DisfluencyWrapper(torch.nn.Module):
    """Pins return_feature=False so the ONNX graph has a fixed output signature."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_feature=False)


def export_disfluency(output_dir: Path, device: str, opset: int) -> None:
    from audio_classification_playground.vox_profile.fluency.wavlm_fluency import WavLMWrapper

    model_id = "tiantiaf/wavlm-large-speech-flow"
    print(f"[disfluency] loading {model_id} on {device} …")
    model = WavLMWrapper.from_pretrained(model_id).to(device).eval()
    wrapper = _DisfluencyWrapper(model)

    dummy = torch.zeros(VALIDATION_BATCH, DISFLUENCY_WINDOW_SAMPLES, device=device)
    with torch.inference_mode():
        f_pt, d_pt = model(dummy, return_feature=False)
    print(f"[disfluency] pytorch shapes: fluency={tuple(f_pt.shape)} disf_type={tuple(d_pt.shape)}")

    out_path = output_dir / "disfluency" / "1" / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["input_values"],
        output_names=["fluency_logits", "disfluency_type_logits"],
        dynamic_axes={
            "input_values": {0: "batch_size"},
            "fluency_logits": {0: "batch_size"},
            "disfluency_type_logits": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"[disfluency] exported → {out_path}")
    _validate(
        out_path,
        inputs={"input_values": dummy.cpu().numpy()},
        expected={
            "fluency_logits": f_pt.cpu().float().numpy(),
            "disfluency_type_logits": d_pt.cpu().float().numpy(),
        },
    )
    out_path = _convert_fp16(out_path)
    _validate(out_path, inputs={"input_values": dummy.cpu().numpy()}, expected={
        "fluency_logits": f_pt.cpu().float().numpy(),
        "disfluency_type_logits": d_pt.cpu().float().numpy(),
    }, atol=1e-2)


# ---------------------------------------------------------------------------
# Emotion — ONNX export attempt with graceful fallback
# ---------------------------------------------------------------------------

class _EmotionExportWrapper(torch.nn.Module):
    """Wraps whatever inner module we find so torch.onnx.export sees a clean graph.

    emotion2vec_plus_large in FunASR is typically structured as:
      auto_model.model               — the top-level nn.Module
        .frontend (optional)         — CNN feature extractor
        .encoder                     — transformer backbone
        .head / .classifier          — linear + softmax

    We try calling model(audio) directly first; if that fails we try the
    frontend → encoder → head pipeline explicitly.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        # FunASR models may return a dict, tuple, or bare tensor.
        if isinstance(out, dict):
            # Common keys: "logit", "logits", "score", "hidden_states"
            for key in ("logit", "logits", "score", "scores"):
                if key in out:
                    return out[key]
            # Fall back to first tensor value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise RuntimeError(f"Cannot extract output tensor from dict keys: {list(out.keys())}")
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def export_emotion(output_dir: Path, device: str, opset: int) -> bool:
    """Attempt to export emotion2vec to ONNX. Returns True on success.

    On failure, prints diagnostic info and instructions for using the Python
    backend fallback (triton/emotion/1/model.py) instead.
    """
    from funasr import AutoModel
    from audio_classification_playground.acoustic_events.inference.emotion2vec import (
        predict_emotion2vec_scores,
    )

    model_id = "iic/emotion2vec_plus_large"
    print(f"[emotion] loading {model_id} on {device} …")
    auto_model = AutoModel(
        model=model_id,
        batch_size=VALIDATION_BATCH,
        device=device,
        disable_update=True,
        disable_pbar=True,
    )

    # --- get FunASR reference outputs ---
    dummy_np = np.random.randn(VALIDATION_BATCH, EMOTION_WINDOW_SAMPLES).astype(np.float32)
    ref_scores, ref_labels = predict_emotion2vec_scores(
        auto_model,
        dummy_np,
        sample_rate=SAMPLE_RATE,
        batch_size=VALIDATION_BATCH,
    )
    n_classes = ref_scores.shape[1]
    print(f"[emotion] FunASR reference: scores={ref_scores.shape}  labels={list(ref_labels)}")

    # --- probe internal structure ---
    print("[emotion] probing FunASR internals …")
    _probe_attrs = ("model", "encoder", "frontend", "backbone", "net")
    for attr in _probe_attrs:
        child = getattr(auto_model, attr, None)
        if child is not None and isinstance(child, torch.nn.Module):
            print(f"  auto_model.{attr}: {type(child).__name__}")
            for sub in ("frontend", "encoder", "head", "classifier", "linear"):
                s = getattr(child, sub, None)
                if s is not None and isinstance(s, torch.nn.Module):
                    print(f"    .{sub}: {type(s).__name__}")

    inner = getattr(auto_model, "model", None)
    if inner is None or not isinstance(inner, torch.nn.Module):
        print("[emotion] ✗ auto_model.model not found or not an nn.Module")
        _print_fallback_instructions()
        return False

    # --- try a direct forward pass ---
    wrapper = _EmotionExportWrapper(inner).eval()
    dummy_t = torch.from_numpy(dummy_np).to(device)
    print("[emotion] trying wrapper.forward(audio) …")
    try:
        with torch.inference_mode():
            out_t = wrapper(dummy_t)
        print(f"[emotion] wrapper output: shape={tuple(out_t.shape)} dtype={out_t.dtype}")
    except Exception as e:
        print(f"[emotion] ✗ forward pass failed: {e}")
        _print_fallback_instructions()
        return False

    # --- check output shape ---
    if out_t.shape[-1] != n_classes:
        print(
            f"[emotion] ✗ wrapper output has {out_t.shape[-1]} classes but FunASR "
            f"returns {n_classes} — preprocessing is outside the nn.Module boundary."
        )
        print("[emotion] This is the common case where FunASR's feature extractor")
        print("  runs before the model. To export it, you'd need to trace the full")
        print("  pipeline including FunASR's frontend — not straightforward.")
        _print_fallback_instructions()
        return False

    # --- cross-check against FunASR reference ---
    out_np = out_t.detach().cpu().float().numpy()
    max_diff = float(np.abs(out_np - ref_scores).max())
    print(f"[emotion] max diff vs FunASR reference: {max_diff:.4f}")
    if max_diff > 0.05:
        print(
            f"[emotion] ✗ wrapper output diverges from FunASR (max_diff={max_diff:.4f} > 0.05). "
            "FunASR applies preprocessing outside the nn.Module."
        )
        _print_fallback_instructions()
        return False
    print("[emotion] ✓ wrapper matches FunASR reference — proceeding with ONNX export")

    # --- export ---
    out_path = output_dir / "emotion" / "1" / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            (dummy_t,),
            str(out_path),
            input_names=["input_values"],
            output_names=["scores"],
            dynamic_axes={
                "input_values": {0: "batch_size"},
                "scores": {0: "batch_size"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    except Exception as e:
        print(f"[emotion] ✗ torch.onnx.export failed: {e}")
        _print_fallback_instructions()
        return False

    print(f"[emotion] exported → {out_path}")
    _validate(
        out_path,
        inputs={"input_values": dummy_np},
        expected={"scores": ref_scores},
        atol=1e-3,
    )
    out_path = _convert_fp16(out_path)
    _validate(out_path, inputs={"input_values": dummy_np},
              expected={"scores": ref_scores}, atol=5e-2)

    print("\n[emotion] ✓ ONNX export succeeded.")
    print("  Switch triton/emotion/config.pbtxt to use the ONNX backend:")
    print('    backend: "onnxruntime"')
    print(f'    output [{{ name: "scores" data_type: TYPE_FP32 dims: [{n_classes}] }}]')
    print("  Remove triton/emotion/1/model.py (Python backend no longer needed).")
    print("  Update TritonEmotionPredictor.output_name from 'probabilities' to 'scores'")
    print("  and apply emotion2vec_scores_to_probabilities() on the client side.")
    return True


def _print_fallback_instructions() -> None:
    print("[emotion] → Using Python backend fallback (already configured):")
    print("           triton/emotion/config.pbtxt  +  triton/emotion/1/model.py")
    print("           No action needed — the Python backend is fully functional.")


# ---------------------------------------------------------------------------
# FP16 conversion
# ---------------------------------------------------------------------------

def _convert_fp16(onnx_path: Path) -> Path:
    """Convert an FP32 ONNX model to FP16 in-place (weights only; I/O stays FP32).

    Uses keep_io_types=True so Triton configs and client code need no changes.
    ONNX Runtime automatically falls back to FP32 for numerically sensitive ops
    (attention softmax, LayerNorm) — unlike TensorRT's indiscriminate cast that
    caused NaNs.

    Requires: onnxmltools  (pip install onnxmltools)
    """
    try:
        import onnx
        from onnxmltools.utils.float16_converter import convert_float_to_float16
    except ImportError:
        print("  [skip] onnxmltools not installed — skipping FP16 conversion")
        print("         Install with: pip install onnxmltools onnx")
        return onnx_path

    model = onnx.load(str(onnx_path))
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    out_path = onnx_path.parent / "model.onnx"  # overwrite: Triton expects model.onnx
    # Keep the original as model_fp32.onnx for reference
    onnx_path.rename(onnx_path.parent / "model_fp32.onnx")
    onnx.save(model_fp16, str(out_path))
    size_fp32 = (onnx_path.parent / "model_fp32.onnx").stat().st_size / 1e9
    size_fp16 = out_path.stat().st_size / 1e9
    print(f"  FP16 converted: {size_fp32:.2f} GB → {size_fp16:.2f} GB  ({out_path})")
    return out_path


# ---------------------------------------------------------------------------
# ONNX Runtime validation
# ---------------------------------------------------------------------------

def _validate(onnx_path: Path, inputs: dict, expected: dict, atol: float = ATOL) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed — skipping validation")
        return

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ort_outputs = sess.run(None, inputs)
    out_names = [o.name for o in sess.get_outputs()]

    all_ok = True
    for name, ort_out in zip(out_names, ort_outputs):
        pt_out = expected[name]
        max_diff = float(np.abs(ort_out - pt_out).max())
        ok = max_diff <= atol
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}: max_diff={max_diff:.2e}")
        if not ok:
            all_ok = False

    if not all_ok:
        raise RuntimeError(
            f"ONNX validation failed for {onnx_path.name} — "
            f"outputs exceed atol={atol:.2e}. "
            "If this is a precision issue try exporting in FP32 (no autocast)."
        )
    print(f"  validation passed ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--output", default="/efs/triton-model-repo",
        help="Triton model repository root (default: /efs/triton-model-repo)",
    )
    ap.add_argument(
        "--task", choices=["affect", "disfluency", "emotion", "all"], default="all",
        help="Which model(s) to export.",
    )
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("affect", "all"):
        export_affect(output_dir, args.device, args.opset)

    if args.task in ("disfluency", "all"):
        export_disfluency(output_dir, args.device, args.opset)

    if args.task in ("emotion", "all"):
        export_emotion(output_dir, args.device, args.opset)

    print("\nNext steps:")
    print(f"  cp -r triton/* {output_dir}/")
    print(f"  kubectl apply -f manifests/triton-server-deployment.yaml")
    print(f"  experiment-launcher launch manifests/acoustic-events-triton-workers.yaml \\")
    print(f"      --template ./manifests/job-template-with-github-ssh.yaml")


if __name__ == "__main__":
    main()
