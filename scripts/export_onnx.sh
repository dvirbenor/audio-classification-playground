#!/usr/bin/env bash
#
# export_onnx.sh — export the acoustic-event models to ONNX for the Triton repo.
#
# WHY THIS EXISTS
#   `python scripts/export_models_onnx.py` needs two pieces of environment set up
#   correctly or it silently produces the wrong thing:
#     1. env.shared.sh — so model weights resolve from the shared EFS cache
#        (HF_HOME / MODELSCOPE_CACHE / TORCH_HOME) instead of re-downloading.
#     2. LD_LIBRARY_PATH — onnxruntime-gpu loads CUDA/cuDNN from the torch-bundled
#        nvidia/* wheels, which are NOT on the default loader path. Without them
#        `import onnxruntime` fails to find libcudart.so.* and the exporter
#        SILENTLY SKIPS ORT validation and the FP16 conversion — leaving FP32
#        models on disk while the Triton configs expect FP16.
#   This wrapper bakes both in so the export is reproducible.
#
# USAGE
#   scripts/export_onnx.sh                          # all tasks -> ./triton-model-repo
#   scripts/export_onnx.sh --task affect            # one task
#   scripts/export_onnx.sh --output /efs/triton-model-repo
#   Any flags are forwarded verbatim to export_models_onnx.py
#   (--task affect|disfluency|emotion|all, --device cuda|cpu, --opset N).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 1. Resolve weights from the shared EFS model cache (best-effort; don't let a
#    non-zero line in env.shared.sh abort us under `set -e`).
if [[ -f env.shared.sh ]]; then
  set +e
  # shellcheck disable=SC1091
  source env.shared.sh
  set -e
fi

# 2. Put the torch-bundled NVIDIA libs on the loader path for onnxruntime-gpu.
#    Glob python3.* so this works regardless of the venv's minor version.
VENV="${VIRTUAL_ENV:-$REPO_ROOT/.venv}"
nvidia_libs="$(find "$VENV"/lib/python3.*/site-packages/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')"
if [[ -n "$nvidia_libs" ]]; then
  export LD_LIBRARY_PATH="${nvidia_libs}${LD_LIBRARY_PATH:-}"
fi

# Unbuffered so progress streams when this is piped to a log.
export PYTHONUNBUFFERED=1

# 3. Default the output dir to the in-repo triton-model-repo unless the caller
#    passed --output. Forward everything else through.
DEFAULT_OUTPUT="$REPO_ROOT/triton-model-repo"
pass_args=("$@")
want_output=true
for a in "$@"; do
  case "$a" in --output|--output=*) want_output=false ;; esac
done
$want_output && pass_args+=(--output "$DEFAULT_OUTPUT")

echo "[export_onnx] repo:   $REPO_ROOT"
echo "[export_onnx] cuda libs: ${nvidia_libs:-<none found — ORT may skip validation/FP16>}"
echo "[export_onnx] run:    uv run python scripts/export_models_onnx.py ${pass_args[*]}"
exec uv run python scripts/export_models_onnx.py "${pass_args[@]}"
