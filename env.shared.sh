# Shared model cache on EFS — source this file on every pod so all
# frameworks resolve weights from the same pre-populated directory.
#
#   source env.shared.sh
#
# To seed the cache once (from a pod that already has the models):
#
#   CACHE_ROOT=/efs/dvir/data/.shared-model-cache
#   mkdir -p "$CACHE_ROOT"
#   cp -a /workspace/.persistent/.cache/huggingface "$CACHE_ROOT/huggingface"
#   cp -a /workspace/.cache/modelscope              "$CACHE_ROOT/modelscope"
#   cp -a /workspace/.cache/torch                   "$CACHE_ROOT/torch"

export HF_HOME=/efs/dvir/data/.shared-model-cache/huggingface
export MODELSCOPE_CACHE=/efs/dvir/data/.shared-model-cache/modelscope
export TORCH_HOME=/efs/dvir/data/.shared-model-cache/torch
