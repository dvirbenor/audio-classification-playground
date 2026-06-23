#!/usr/bin/env python3
"""Export WavLM affect + disfluency + emotion2vec models to ONNX for Triton.

All three export to ONNX via torch.onnx.export and run on the onnxruntime
backend. Affect/disfluency export the WavLM wrapper directly; emotion exports
DirectEmotion2vecScorer.core (the extract_features inference path), emitting raw
EMOTION2VEC_LABELS scores that the producer canonicalizes client-side.

Run on a GPU pod with HF_HOME pointing at the shared EFS model cache:

    source env.shared.sh
    uv run python scripts/export_models_onnx.py \\
        --output /efs/triton-model-repo \\
        [--task affect|disfluency|all] \\
        [--device cuda] \\
        [--opset 18]

Each task's config.pbtxt is copied from triton/ into the output automatically,
so the output directory is a complete, deploy-ready Triton model directory.
Then start the Triton server:

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
    _validate_batch_shapes(out_path, AFFECT_WINDOW_SAMPLES)


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
    # .eval() the wrapper too: nn.Module defaults to training=True and that flag
    # does not propagate to an already-attached child, so without this the
    # exporter warns it is "exporting a model while it is in training mode".
    wrapper = _DisfluencyWrapper(model).eval()

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
    # disfluency outputs are raw logits (not sigmoid-bounded like affect), so a
    # pure-atol bound on FP32 kernel noise is too tight; allow a small rtol.
    _validate(
        out_path,
        inputs={"input_values": dummy.cpu().numpy()},
        expected={
            "fluency_logits": f_pt.cpu().float().numpy(),
            "disfluency_type_logits": d_pt.cpu().float().numpy(),
        },
        rtol=1e-3,
    )
    out_path = _convert_fp16(out_path)
    _validate(out_path, inputs={"input_values": dummy.cpu().numpy()}, expected={
        "fluency_logits": f_pt.cpu().float().numpy(),
        "disfluency_type_logits": d_pt.cpu().float().numpy(),
    }, atol=1e-2, rtol=1e-3)
    _validate_batch_shapes(out_path, DISFLUENCY_WINDOW_SAMPLES)


# ---------------------------------------------------------------------------
# Emotion — ONNX export attempt with graceful fallback
# ---------------------------------------------------------------------------

def _make_export_alibi(fold):
    """Build a cache-free, batch-symbolic replacement for funasr's get_alibi_bias.

    The stock get_alibi_bias (funasr/models/emotion2vec/base.py) caches a
    ``[heads*batch_size, T, T]`` tensor in a dict and branches on its size. Both
    bake the trace-time batch (``heads*VALIDATION_BATCH`` = 64) into the graph,
    and the dict mutation + data-dependent branch are untraceable by torch.export
    (it raised ConstraintViolationError on batch_size; the legacy tracer baked it).
    This recomputes the identical bias with no cache and no branch, so
    ``batch_size`` stays a symbolic dim under the dynamo exporter — which also
    keeps the attention's ``bsz*num_heads`` reshapes dynamic.

    ``fold`` is the pre-clamped ``alibi_scale.squeeze(0)`` (or None). When set, it
    is multiplied into the bias here so the caller's ``alibi_scale.clamp_min(0)``
    branch can be disabled — that ``clamp_min`` crashes torch.onnx's type-promotion
    pass. Folding it in is exactly what contextualized_features does in the
    ``alibi_scale.size(0) == 1`` path (which then sets ``alibi_scale = None``), so
    the result is numerically identical.
    """
    from funasr.models.emotion2vec.base import get_alibi

    def _alibi(batch_size, time_steps, heads, dtype, device, dims=1, distance="manhattan"):
        base = get_alibi(time_steps, heads, dims=dims, distance=distance).to(
            dtype=dtype, device=device
        )
        out = base.repeat(batch_size, 1, 1).view(batch_size, heads, time_steps, time_steps)
        if fold is not None:
            out = out * fold.to(dtype=out.dtype, device=out.device)
        return out

    return _alibi


def export_emotion(output_dir: Path, device: str, opset: int) -> bool:
    """Export emotion2vec to ONNX as a single pure-ONNX model. Returns True.

    The exportable unit is ``DirectEmotion2vecScorer.core`` — layer-norm +
    ``extract_features`` + mean-pool + ``proj`` + masked softmax + label select —
    which maps raw audio windows directly to native per-class scores. (The
    earlier approach of calling FunASR's ``model.forward`` failed because that is
    the SSL/masking path; ``extract_features`` is the inference path.)

    The ONNX model emits the raw ``EMOTION2VEC_LABELS`` scores, exactly like the
    direct fleet path — the producer's ``emotion2vec_scores_to_probabilities()``
    folds them into CANONICAL_CHANNELS client-side, so no canonicalization is
    baked into the graph.
    """
    from funasr import AutoModel
    from audio_classification_playground.acoustic_events.inference.emotion2vec import (
        EMOTION2VEC_LABELS,
        make_direct_emotion2vec_scorer,
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

    scorer = make_direct_emotion2vec_scorer(auto_model, sample_rate=SAMPLE_RATE)
    if scorer is None:
        print("[emotion] ✗ direct batched scorer unsupported for this FunASR model — "
              "cannot export to ONNX (no Python backend implemented either).")
        return False

    # The client (TritonEmotionPredictor) pairs the ONNX scores with the fixed
    # EMOTION2VEC_LABELS order; fail loudly if the live model no longer matches.
    # Compare by English suffix ("中立/neutral" -> "neutral") since that is the
    # only part normalize_label() — and EMOTION2VEC_LABELS — keeps.
    native_suffixes = tuple(
        str(label).split("/")[-1].strip().lower() for label in scorer.selected_labels
    )
    if native_suffixes != EMOTION2VEC_LABELS:
        raise RuntimeError(
            "emotion2vec label order changed — update EMOTION2VEC_LABELS in "
            "inference/emotion2vec.py.\n"
            f"  expected: {EMOTION2VEC_LABELS}\n"
            f"  got:      {native_suffixes}"
        )
    n_classes = len(EMOTION2VEC_LABELS)
    print(f"[emotion] {n_classes} native classes: {list(EMOTION2VEC_LABELS)}")

    core = scorer.core.eval()
    dummy_np = np.random.randn(VALIDATION_BATCH, EMOTION_WINDOW_SAMPLES).astype(np.float32)
    dummy_t = torch.from_numpy(dummy_np).to(scorer.device)
    # no_grad, NOT inference_mode: the first forward caches an alibi tensor on the
    # encoder; inference-mode tensors then can't be consumed by the export tracer
    # ("Inference tensors cannot be saved for backward").
    with torch.no_grad():
        ref = core(dummy_t).detach().cpu().float().numpy()
    print(f"[emotion] core output: shape={ref.shape}  rows sum to ~{float(ref.sum(axis=1).mean()):.4f}")

    # Make emotion2vec exportable with a dynamic batch. Two model-side blockers:
    #  1. get_alibi_bias() caches a [heads*batch_size, T, T] tensor in a dict and
    #     branches on it — baking the trace batch (heads*VALIDATION_BATCH = 64) and
    #     defeating both exporters (legacy tracer bakes it; torch.export raised
    #     ConstraintViolationError on the cache branch).
    #  2. contextualized_features does `alibi_scale = alibi_scale.clamp_min(0)`,
    #     which crashes torch.onnx's type-promotion decomposition pass.
    # Fix both: install a cache-free, batch-symbolic alibi on every encoder, and
    # fold the (clamped) alibi_scale into it so the clamp_min branch can be
    # disabled by nulling alibi_scale. This mirrors the model's own size(0)==1
    # path exactly, so it is numerically identical (verified by _validate below).
    n_alibi = n_folded = 0
    for m in core.modules():
        if getattr(m, "get_alibi_bias", None) is None:
            continue
        fold = None
        scale = getattr(m, "alibi_scale", None)
        if scale is not None:
            clamped = scale.detach().clamp_min(0)
            if clamped.size(0) == 1:
                fold = clamped.squeeze(0)
                m.alibi_scale = None  # skip the clamp_min branch in forward
                n_folded += 1
            else:
                print(f"[emotion] WARN: alibi_scale.size(0)={clamped.size(0)} != 1 on "
                      f"{type(m).__name__}; clamp_min NOT folded — export may still hit it")
        m.get_alibi_bias = _make_export_alibi(fold)
        n_alibi += 1
    print(f"[emotion] export-safe alibi on {n_alibi} module(s); clamp_min folded on {n_folded}")

    out_path = output_dir / "emotion" / "1" / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from torch.export import Dim

    batch = Dim("batch_size", min=1, max=4096)
    torch.onnx.export(
        core,
        (dummy_t,),
        str(out_path),
        input_names=["input_values"],
        output_names=["scores"],
        dynamic_shapes=({0: batch},),
        opset_version=opset,
        dynamo=True,
    )
    print(f"[emotion] exported → {out_path}")
    _validate(out_path, inputs={"input_values": dummy_np},
              expected={"scores": ref}, atol=1e-3)
    out_path = _convert_fp16(out_path)
    _validate(out_path, inputs={"input_values": dummy_np},
              expected={"scores": ref}, atol=5e-2)
    # Guard against batch-fragile graphs: the dynamic batch axis is worthless if
    # a baked-in reshape only works at the export batch. Exercise several real
    # batch sizes (Triton serves up to 256) before trusting the model.
    _validate_batch_shapes(out_path, EMOTION_WINDOW_SAMPLES)
    print("[emotion] ✓ ONNX export + validation passed.")
    return True


# ---------------------------------------------------------------------------
# FP16 conversion
# ---------------------------------------------------------------------------

def _convert_fp16(onnx_path: Path) -> Path:
    """Convert an FP32 ONNX model to FP16 in-place (weights only; I/O stays FP32).

    Uses keep_io_types=True so Triton configs and client code need no changes.
    Conv ops are kept in FP32 (added to op_block_list): cuDNN's frontend can
    fail the heuristic query for WavLM's wide positional conv in pure FP16
    (HEURISTIC_QUERY_FAILED — a hard error / segfault on some ORT+cuDNN builds),
    and convs are numerically sensitive anyway. The attention/FFN matmuls — the
    bulk of the compute — still run on FP16 tensor cores.

    The float16 converter has moved between packages over time; try the known
    locations in order (ORT ships its own, which is the most relevant for the
    CUDA EP target).
    """
    try:
        import onnx
    except ImportError:
        print("  [skip] onnx not installed — skipping FP16 conversion")
        return onnx_path

    convert_float_to_float16 = None
    for module_path in (
        "onnxruntime.transformers.float16",
        "onnxconverter_common.float16",
        "onnxmltools.utils.float16_converter",
    ):
        try:
            module = __import__(module_path, fromlist=["convert_float_to_float16"])
            convert_float_to_float16 = module.convert_float_to_float16
            break
        except ImportError:
            continue
    if convert_float_to_float16 is None:
        print("  [skip] no float16 converter found — skipping FP16 conversion")
        print("         Install one of: onnxruntime, onnxconverter-common, onnxmltools")
        return onnx_path

    # Full FP32 footprint (proto + external-data sidecar, if any) for the report.
    fp32_bytes = onnx_path.stat().st_size
    ext_data = onnx_path.parent / (onnx_path.name + ".data")
    if ext_data.exists():
        fp32_bytes += ext_data.stat().st_size

    model = onnx.load(str(onnx_path))
    block_list = list(getattr(module, "DEFAULT_OP_BLOCK_LIST", [])) + ["Conv"]
    model_fp16 = convert_float_to_float16(
        model, keep_io_types=True, op_block_list=block_list
    )
    out_path = onnx_path.parent / "model.onnx"  # overwrite: Triton expects model.onnx
    # Keep the original as model_fp32.onnx for reference (its external-data
    # sidecar keeps its original name and stays referenced by model_fp32.onnx).
    onnx_path.rename(onnx_path.parent / "model_fp32.onnx")
    onnx.save(model_fp16, str(out_path))
    size_fp16 = out_path.stat().st_size / 1e9
    print(f"  FP16 converted: {fp32_bytes / 1e9:.2f} GB → {size_fp16:.2f} GB  ({out_path})")
    return out_path


# ---------------------------------------------------------------------------
# ONNX Runtime validation
# ---------------------------------------------------------------------------

def _validate(
    onnx_path: Path, inputs: dict, expected: dict, atol: float = ATOL, rtol: float = 0.0
) -> None:
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
        # allclose semantics: tolerate atol + rtol*|expected|. rtol matters for
        # unbounded logit outputs whose magnitude makes a pure-atol bound on
        # FP32 CUDA-vs-ORT kernel noise unrealistically tight.
        ok = bool(np.allclose(ort_out, pt_out, atol=atol, rtol=rtol))
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {name}: max_diff={max_diff:.2e}")
        if not ok:
            all_ok = False

    if not all_ok:
        raise RuntimeError(
            f"ONNX validation failed for {onnx_path.name} — "
            f"outputs exceed atol={atol:.2e} rtol={rtol:.2e}. "
            "If this is a precision issue try exporting in FP32 (no autocast)."
        )
    print(f"  validation passed ✓")


def _validate_batch_shapes(
    onnx_path: Path, window_samples: int, batches=(1, 3, 7, 64, 256)
) -> None:
    """Run the model at several batch sizes to catch batch-fragile exports.

    ``_validate`` only checks the single ``VALIDATION_BATCH`` shape, so it cannot
    see a graph that hard-codes a trace-time dimension (e.g. an attention reshape
    baked to the export batch by constant folding) — such a graph passes
    validation yet fails on every other batch at serving time, which is exactly
    how the emotion model shipped broken (0 prod inferences). This asserts the
    graph actually executes across batch sizes; numerical correctness is still
    ``_validate``'s job.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed — skipping batch-shape validation")
        return

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    for b in batches:
        x = np.random.randn(b, window_samples).astype(np.float32)
        try:
            sess.run(None, {in_name: x})
        except Exception as e:
            raise RuntimeError(
                f"{onnx_path.name} runs at the export batch but FAILS at batch {b}. "
                f"The graph is batch-fragile — almost always a reshape with a "
                f"trace-time dim baked in by constant folding. Re-export with "
                f"do_constant_folding=False.\n  ORT error: {e}"
            ) from e
        print(f"  [OK  ] batch {b}: ran")
    print("  batch-shape validation passed ✓ (generalises across batch sizes)")


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
    ap.add_argument(
        "--opset", type=int, default=18,
        help="ONNX opset version (default: 18). The dynamo exporter has opset-18 "
             "implementations and keeps 18 even if a lower version is requested, "
             "so 17 only triggers a failed down-convert and a noisy traceback.",
    )
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    if args.task in ("affect", "all"):
        export_affect(output_dir, args.device, args.opset)
        exported.append("affect")

    if args.task in ("disfluency", "all"):
        export_disfluency(output_dir, args.device, args.opset)
        exported.append("disfluency")

    if args.task in ("emotion", "all"):
        if export_emotion(output_dir, args.device, args.opset):
            exported.append("emotion")

    _install_configs(output_dir, exported)

    print("\nNext steps:")
    print(f"  kubectl apply -f manifests/triton-server-deployment.yaml")
    print(f"  experiment-launcher launch manifests/acoustic-events-triton-workers.yaml \\")
    print(f"      --template ./manifests/job-template-with-github-ssh.yaml")


def _install_configs(output_dir: Path, tasks: list[str]) -> None:
    """Copy each exported task's config.pbtxt from the repo's ``triton/`` source
    into ``output_dir/<task>/``, so the output is a complete, deploy-ready Triton
    model directory (weights + configs) with no manual ``cp -r triton/*`` step.
    ``triton/`` stays the single version-controlled source for the configs.
    """
    import shutil

    triton_src = Path(__file__).resolve().parent.parent / "triton"
    for task in tasks:
        src = triton_src / task / "config.pbtxt"
        dst = output_dir / task / "config.pbtxt"
        if not src.exists():
            print(f"  [warn] no config.pbtxt for {task} at {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"  config installed → {dst}")


if __name__ == "__main__":
    main()
