#!/usr/bin/env bash
#
# run_mps_colocated.sh — run the GPU inference task-fleets co-located on ONE GPU,
# sharing it via CUDA MPS so their kernels overlap (one model's GEMM fills
# another's memory-bound phase).
#
# This is the per-pod entrypoint for the single-Blackwell deployment AND the
# thing you run by hand on a dev GPU pod to validate the setup before a fleet
# rollout. The k8s manifest does clone/`uv sync`/`source env.shared.sh`, then
# calls this script; a manual test does the same. Same behaviour either way.
#
# It launches the affect, disfluency and emotion task-group workers as separate
# processes (each `orchestration run --task-group <task>`), all claiming their
# own per-task lock namespace under the same --output tree — exactly the
# supported task-fleet mode, just co-located. VAD stays a separate CPU fleet
# that must lead; the GPU workers read its `vad/` artifacts from disk and fall
# back to full-timeline compute when one isn't ready (never block on it).
#
# Configuration is via environment variables (set by the manifest, or exported
# before a manual run):
#
#   PARQUET                (required) path to all_archives.parquet
#   OUTPUT                 (required) EFS output base directory
#   AUDIO_CACHE            (optional) shared decoded-audio cache dir
#   CACHE_BYTES            (optional) cache cap in bytes (required if AUDIO_CACHE set)
#   TASKS                  (default "affect disfluency emotion") GPU tasks to co-locate
#   AFFECT_BACKBONE        (default "wavlm")
#   DISFLUENCY_BACKBONE    (default "wavlm")
#   WAVLM_RUNTIME_PRESET   (default "compiled_static")
#   WAVLM_STATIC_BATCH_SIZE(optional) override the compiled_static batch dim (default 256)
#   EMOTION_RUNTIME_MODE   (default "optimized")
#   VAD_GATING             (default "1"; set "0" to disable --vad-gating)
#   REQUIRE_VAD            (default "0"; set "1" to add --require-precomputed-vad so
#                          GPU workers SKIP archives without a vad/ artifact instead
#                          of falling back to full-timeline — enforces real gating)
#   MAX_RETRIES            (default "3")
#   DEVICE                 (default "cuda")
#   ENABLE_MPS             (default "1"; set "0" to skip the MPS daemon, e.g. a 1-task test)
#   CUDA_MPS_PIPE_DIRECTORY(default "/tmp/nvidia-mps")
#   CUDA_MPS_LOG_DIRECTORY (default "/tmp/nvidia-log")
#   <TASK>_EXTRA_ARGS      (optional) extra flags per task, e.g.
#                          EMOTION_EXTRA_ARGS="--prefetch-workers 8 --prefetch-lookahead 16"
#
set -uo pipefail

log() { echo "[$(date -Iseconds)] [mps-launcher] $*"; }
die() { log "FATAL: $*"; exit 1; }

: "${PARQUET:?PARQUET is required}"
: "${OUTPUT:?OUTPUT is required}"

TASKS="${TASKS:-affect disfluency emotion}"
AFFECT_BACKBONE="${AFFECT_BACKBONE:-wavlm}"
DISFLUENCY_BACKBONE="${DISFLUENCY_BACKBONE:-wavlm}"
WAVLM_RUNTIME_PRESET="${WAVLM_RUNTIME_PRESET:-compiled_static}"
EMOTION_RUNTIME_MODE="${EMOTION_RUNTIME_MODE:-optimized}"
VAD_GATING="${VAD_GATING:-1}"
REQUIRE_VAD="${REQUIRE_VAD:-0}"
MAX_RETRIES="${MAX_RETRIES:-3}"
DEVICE="${DEVICE:-cuda}"
ENABLE_MPS="${ENABLE_MPS:-1}"

MODULE="audio_classification_playground.acoustic_events.orchestration"

# --- CUDA MPS daemon ------------------------------------------------------
# Do NOT set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: leaving the SM partition
# unbounded lets the MPS scheduler freely interleave the co-located clients,
# which is exactly what fills the tensor-core gaps.
mps_started=0
start_mps() {
  [ "$ENABLE_MPS" = "1" ] || { log "ENABLE_MPS=0 — skipping MPS daemon"; return; }
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  command -v nvidia-cuda-mps-control >/dev/null 2>&1 \
    || die "nvidia-cuda-mps-control not found; install CUDA MPS or set ENABLE_MPS=0"
  log "Starting CUDA MPS control daemon (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
  nvidia-cuda-mps-control -d || die "failed to start MPS control daemon"
  # Verify the daemon is reachable — if it isn't, clients silently fall back to
  # time-sliced (non-overlapping) execution, which defeats the whole point.
  if ! echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control >/dev/null 2>&1; then
    die "MPS control daemon not reachable after start; aborting (would silently time-slice)"
  fi
  mps_started=1
  log "CUDA MPS active"
}

stop_mps() {
  [ "$mps_started" = "1" ] || return 0
  log "Stopping CUDA MPS control daemon"
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  mps_started=0
}

# --- build a worker command for one task ----------------------------------
# Populates the global WORKER_ARGV array (avoids bash-4 `mapfile`/process subst).
build_worker_argv() {
  local task="$1"
  WORKER_ARGV=(
    python -m "$MODULE" run
      --parquet "$PARQUET"
      --output "$OUTPUT"
      --task-group "$task"
      --affect-backbone "$AFFECT_BACKBONE"
      --disfluency-backbone "$DISFLUENCY_BACKBONE"
      --completion-policy exists
      --device "$DEVICE"
      --max-retries "$MAX_RETRIES"
  )
  # WavLM-backed tasks (affect/disfluency) take the runtime preset + optional batch override.
  if [ "$task" = "affect" ] || [ "$task" = "disfluency" ]; then
    WORKER_ARGV+=(--wavlm-runtime-preset "$WAVLM_RUNTIME_PRESET")
    if [ -n "${WAVLM_STATIC_BATCH_SIZE:-}" ]; then
      WORKER_ARGV+=(--wavlm-static-batch-size "$WAVLM_STATIC_BATCH_SIZE")
    fi
  fi
  if [ "$task" = "emotion" ]; then
    WORKER_ARGV+=(--emotion-runtime-mode "$EMOTION_RUNTIME_MODE")
  fi
  if [ "$VAD_GATING" = "1" ]; then
    WORKER_ARGV+=(--vad-gating)
  fi
  if [ "$REQUIRE_VAD" = "1" ]; then
    WORKER_ARGV+=(--require-precomputed-vad)
  fi
  if [ -n "${AUDIO_CACHE:-}" ]; then
    : "${CACHE_BYTES:?CACHE_BYTES is required when AUDIO_CACHE is set}"
    WORKER_ARGV+=(--audio-cache-dir "$AUDIO_CACHE" --max-cache-bytes "$CACHE_BYTES")
  fi
  # Per-task extra flags, e.g. EMOTION_EXTRA_ARGS="--prefetch-workers 8".
  local extra_var extra
  extra_var="$(echo "$task" | tr '[:lower:]-' '[:upper:]_')_EXTRA_ARGS"
  extra="${!extra_var:-}"
  # shellcheck disable=SC2206
  [ -n "$extra" ] && WORKER_ARGV+=($extra)
}

# --- launch + supervise ----------------------------------------------------
pids=()
shutting_down=0

term_handler() {
  shutting_down=1
  log "signal received — forwarding SIGTERM to workers (graceful drain)"
  [ "${#pids[@]}" -gt 0 ] && for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap term_handler TERM INT

start_mps

for task in $TASKS; do
  build_worker_argv "$task"
  log "launching worker: ${WORKER_ARGV[*]}"
  "${WORKER_ARGV[@]}" &
  pids+=("$!")
  log "  -> $task pid $!"
done

log "co-located workers running (pids: ${pids[*]})"

# Supervise by polling. If a worker dies non-zero during normal operation, tear
# the rest down and fail so k8s restarts the pod. During a graceful shutdown
# (shutting_down=1) a non-zero exit is expected and not treated as failure.
# Polling (vs `wait -n`) keeps this portable to bash < 4.3.
exit_code=0
while [ "${#pids[@]}" -gt 0 ]; do
  sleep 2
  alive=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    else
      wait "$pid"; rc=$?
      if [ "$rc" -ne 0 ] && [ "$shutting_down" != "1" ]; then
        log "worker pid $pid exited rc=$rc — tearing down the co-located fleet"
        exit_code="$rc"
        term_handler
      else
        log "worker pid $pid exited rc=$rc"
      fi
    fi
  done
  pids=( ${alive[@]+"${alive[@]}"} )
done

stop_mps
log "all co-located workers exited (exit_code=$exit_code)"
exit "$exit_code"
