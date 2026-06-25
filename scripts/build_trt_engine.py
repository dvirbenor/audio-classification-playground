#!/usr/bin/env python3
"""Build a mixed-precision TensorRT engine (.plan) from a paralinguistics ONNX.

WHY MIXED PRECISION: blanket fp16 (`trtexec --fp16`) is ~2.4x faster than fp32 but
numerically broken on real audio — the attention **Softmax** overflows fp16 and
produces ~0.42 absolute error in the outputs (would flip every affect label; this
is the "NaN"/reject the optimisation report saw). Validated on an A10G (2026-06-23):
pinning Softmax (and LayerNorm, free) to fp32 restores fp32-level accuracy
(~3e-4) at the SAME 2.36x speed. LayerNorm alone does NOT fix it — Softmax is the
culprit. So we keep everything fp16 except SOFTMAX + NORMALIZATION layers.

TRT engines are arch + TRT-version locked. BUILD THIS INSIDE the same image that
serves it (nvcr.io/nvidia/tritonserver:25.07-py3 → TRT 10.11.0.33) on the TARGET GPU
(Blackwell g7e). It needs only the `tensorrt` module (bundled in that image) —
no polygraphy/torch/onnxruntime. (build_trt_engines.sh instead pip-pins the matching
TRT into an isolated venv, so it can run outside the container.)

PREREQUISITE — FOLD CONSTANTS FIRST. TRT 10.10's ONNX parser rejects the affect
fp16 model ("convMultiInput: input tensor shape misaligns with kernel shape")
because the fp16 converter left the (fp32-kept) Conv weights as a Cast-node output
instead of a static initializer. Fold them once (needs polygraphy+onnxruntime, so
do it on a dev box, not necessarily in the serving container):
    polygraphy surgeon sanitize model.onnx --fold-constants -o model_folded.onnx
then point --onnx at the folded file. (disfluency/emotion fold to 0 nodes — already
clean — but folding them is a harmless no-op.) The cleaner long-term fix is to make
the export keep Conv weights as fp32 initializers so no fold is needed.

VERIFIED on the Blackwell (TRT 10.10, batch 128, real audio, 2026-06-23): affect
1539 / disfluency 1742 / emotion 3003 win/s — ~2.5-3x the ONNX/ORT path — at
fp32-level accuracy. NOTE: blanket fp16 happened to be accurate too on this
GPU+TRT combo (the catastrophic Softmax-fp16 error only showed on A10G+TRT-10.16),
but pinning Softmax+Norm costs ~0 (same win/s) and is guaranteed-safe across
kernel/tactic/GPU changes, so keep it.

Usage (one per model; window-samples = 56000 affect, 48000 disfluency/emotion):
    polygraphy surgeon sanitize model.onnx --fold-constants -o model_folded.onnx
    python scripts/build_trt_engine.py \
        --onnx model_folded.onnx --out /models/affect/1/model.plan \
        --window-samples 56000 --min 1 --opt 128 --max 128

Then serve with platform: "tensorrt_plan" (see triton/<task>/config.pbtxt.trt).
"""
from __future__ import annotations
import argparse, time
import tensorrt as trt

# fp16 is unsafe for these op types (Softmax is the one that matters; Normalization
# is pinned too as a free safety margin — it costs ~0 and slightly tightens error).
PIN_TYPES = {trt.LayerType.SOFTMAX, trt.LayerType.NORMALIZATION}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True, help="output .plan path")
    ap.add_argument("--window-samples", type=int, required=True,
                    help="input length: 56000 (affect) or 48000 (disfluency/emotion)")
    ap.add_argument("--input-name", default="input_values")
    ap.add_argument("--min", type=int, default=1)
    ap.add_argument("--opt", type=int, default=128)
    ap.add_argument("--max", type=int, default=128,
                    help="must equal the Triton config's max_batch_size")
    ap.add_argument("--workspace-mb", type=int, default=24000)
    ap.add_argument("--no-pin", action="store_true",
                    help="DEBUG: blanket fp16 (reproduces the broken ~0.42-error build)")
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  [onnx-parse]", parser.get_error(i))
            return 1

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.FP16)

    n_pinned = 0
    if not args.no_pin:
        # OBEY (not PREFER) so TRT may not silently override our fp32 choices.
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        W = args.window_samples
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in PIN_TYPES:
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                n_pinned += 1

    profile = builder.create_optimization_profile()
    shp = lambda b: (b, args.window_samples)
    profile.set_shape(args.input_name, shp(args.min), shp(args.opt), shp(args.max))
    config.add_optimization_profile(profile)

    print(f"building {args.out}  fp16 + {n_pinned} layers pinned fp32 "
          f"({'BLANKET fp16 — DEBUG' if args.no_pin else 'Softmax+Norm fp32'})  "
          f"profile min/opt/max = {args.min}/{args.opt}/{args.max} x {args.window_samples}")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  BUILD FAILED")
        return 1
    with open(args.out, "wb") as f:
        f.write(serialized)
    # .nbytes (not len()): TRT 10.11's build_serialized_network returns an
    # IHostMemory with no __len__ (10.10 was len()-able). .nbytes works on both.
    print(f"  ok: {serialized.nbytes/1e6:.0f} MB engine in {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
