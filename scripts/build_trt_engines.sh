#!/usr/bin/env bash
# Rebuild the mixed-precision TensorRT engines for affect + disfluency + emotion,
# assemble a Triton tensorrt_plan model repo, and (optionally) publish to S3.
#
# Pipeline per model:  S3 model.onnx -> fold-constants -> build_trt_engine.py
#                      -> <OUT>/<model>/1/model.plan  (+ committed config.pbtxt.trt)
#
# WHAT THIS PRODUCES is ~2.5-3x the ONNX/ORT throughput (Blackwell, batch 128:
# affect 1539 / disfluency 1742 / emotion 3003 win/s). The recipe (pin Softmax +
# Norm to fp32, fp16 everything else) lives in scripts/build_trt_engine.py.
#
# *** REQUIREMENTS — engines are arch + TRT-version locked ***
#   - Run on a Blackwell sm_120 GPU (RTX PRO 6000 / g7e).
#   - TRT is pinned to 10.10.0.31 to match the serving image (tritonserver:25.05).
#     If you change the Triton image, change TRT_VERSION and rebuild, or the
#     engines won't deserialize.
#   - Needs `uv` + network (sets up an isolated venv with the pinned toolchain).
#
# Usage:
#   bash scripts/build_trt_engines.sh                 # build to $OUT (default /efs/triton-trt-repo)
#   PUBLISH=1 bash scripts/build_trt_engines.sh        # ...and `aws s3 sync` to paralinguistics-trt
#   OUT=/tmp/repo MAXB=128 bash scripts/build_trt_engines.sh
#   SKIP_VENV=1 bash scripts/build_trt_engines.sh      # use current env (must have tensorrt 10.10 + polygraphy)
set -euo pipefail

S3_ONNX="s3://riverside-build-assets/paralinguistics"
S3_TRT="s3://riverside-build-assets/paralinguistics-trt"
TRT_VERSION="${TRT_VERSION:-10.10.0.31}"     # must match the serving Triton image's TRT
OUT="${OUT:-/efs/triton-trt-repo}"
MAXB="${MAXB:-128}"                          # also set max_batch_size in triton/<m>/config.pbtxt.trt
WORK="${WORK:-$(mktemp -d)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# model:window-samples  (affect 3.5s, disfluency/emotion 3.0s @ 16 kHz)
MODELS=( "affect:56000" "disfluency:48000" "emotion:48000" )

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1

# --- toolchain: pinned TRT 10.10 + fold deps (polygraphy/onnxruntime/onnx) ---
if [ "${SKIP_VENV:-0}" != "1" ]; then
  VENV="$WORK/trt-venv"
  uv venv "$VENV" --python 3.12
  UV_LINK_MODE=copy uv pip install --python "$VENV/bin/python" \
      "tensorrt-cu12==${TRT_VERSION}" polygraphy onnxruntime onnx numpy
  PY="$VENV/bin/python"
  POLYGRAPHY="$VENV/bin/polygraphy"
else
  PY="$(command -v python3)"; POLYGRAPHY="$(command -v polygraphy)"
fi
echo "TRT: $("$PY" -c 'import tensorrt as t; print(t.__version__)')  (serving image must match)"

for entry in "${MODELS[@]}"; do
  m="${entry%%:*}"; win="${entry##*:}"
  echo "================  $m  (window $win, max_batch $MAXB)  ================"
  aws s3 cp "$S3_ONNX/$m/1/model.onnx" "$WORK/$m.onnx"
  # Fold weight-Cast nodes into static fp32 initializers — TRT 10.10's parser
  # rejects the affect conv otherwise (no-op for disfluency/emotion).
  "$POLYGRAPHY" surgeon sanitize "$WORK/$m.onnx" --fold-constants -o "$WORK/${m}_folded.onnx"
  mkdir -p "$OUT/$m/1"
  "$PY" "$REPO_ROOT/scripts/build_trt_engine.py" \
      --onnx "$WORK/${m}_folded.onnx" --out "$OUT/$m/1/model.plan" \
      --window-samples "$win" --min 1 --opt "$MAXB" --max "$MAXB"
  cp "$REPO_ROOT/triton/$m/config.pbtxt.trt" "$OUT/$m/config.pbtxt"
done

echo "TRT model repo ready: $OUT"
ls -la "$OUT"/*/1/model.plan
if [ "${PUBLISH:-0}" = "1" ]; then
  echo "publishing to $S3_TRT ..."
  aws s3 sync "$OUT" "$S3_TRT"
  echo "published. (verify: aws s3 ls $S3_TRT/affect/1/)"
fi
