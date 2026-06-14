#!/usr/bin/env bash
#
# compare_static_batch_throughput.sh — A/B the co-located MPS fleet at two WavLM
# static batch sizes (default 256 vs 512) over a FIXED sample, to measure the
# overall speed gain on a big GPU (Blackwell).
#
# It runs scripts/run_mps_colocated.sh once per batch size, each writing to its
# own scratch --output dir (so neither run skips the other's work), times the
# wall-clock, and prints `orchestration timings` for each. The worker timings
# record the ACTUAL batch size (wavlm_batch_size), so the per-task numbers are
# unambiguous.
#
# Run this on a Blackwell GPU pod, NOT against the production output tree:
#   source env.shared.sh
#   export PARQUET=/path/to/SMALL_subset.parquet     # bounded! a few dozen archives
#   export OUTPUT_BASE=/efs/.../scratch/batch-ab      # scratch, throwaway
#   bash scripts/compare_static_batch_throughput.sh
#
# Config (env):
#   PARQUET       (required) a SMALL parquet subset — the full corpus never finishes
#   OUTPUT_BASE   (required) scratch base dir; per-size outputs go in <base>/bs<N>
#   BATCH_SIZES   (default "256 512")
#   AUDIO_CACHE   (optional) shared decoded-audio cache (speeds repeated decode)
#   CACHE_BYTES   (optional) required if AUDIO_CACHE set
#   Any run_mps_colocated.sh var is forwarded (TASKS, VAD_GATING, ENABLE_MPS, ...).
#
# Notes:
#  - Both runs do identical work: with fresh output dirs and no pre-computed VAD,
#    VAD-gating falls back to full-timeline for both, so the comparison is fair.
#    For a gating-representative bench, pre-populate the `vad/` artifacts in both
#    dirs first (run the vad task-group into each).
#  - Wall-clock includes one-time model-load + torch.compile warmup (which differs
#    per batch size). The `orchestration timings` per-archive inference numbers
#    EXCLUDE warmup — prefer those for steady-state throughput; use wall-clock for
#    end-to-end. Let it process enough archives that warmup is amortized.
#
set -uo pipefail

log() { echo "[$(date -Iseconds)] [batch-ab] $*"; }

: "${PARQUET:?PARQUET is required (point at a SMALL subset)}"
: "${OUTPUT_BASE:?OUTPUT_BASE is required (scratch dir)}"
BATCH_SIZES="${BATCH_SIZES:-256 512}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$HERE/run_mps_colocated.sh"
MODULE="audio_classification_playground.acoustic_events.orchestration"

declare -a summary_lines=()

for bs in $BATCH_SIZES; do
  out="$OUTPUT_BASE/bs$bs"
  mkdir -p "$out"
  log "=== batch size $bs -> $out ==="

  start=$(date +%s)
  PARQUET="$PARQUET" \
  OUTPUT="$out" \
  WAVLM_STATIC_BATCH_SIZE="$bs" \
  ENABLE_MPS="${ENABLE_MPS:-1}" \
  bash "$LAUNCHER"
  rc=$?
  end=$(date +%s)
  wall=$((end - start))

  if [ "$rc" -ne 0 ]; then
    log "WARNING: batch size $bs run exited rc=$rc (see logs); recording wall-clock anyway"
  fi
  log "batch size $bs wall-clock: ${wall}s"

  log "--- timings (batch $bs) ---"
  python -m "$MODULE" timings --output "$out" || true

  summary_lines+=("batch=$bs  wall_clock=${wall}s  exit=$rc  output=$out")
done

echo
log "================ overall comparison ================"
for line in "${summary_lines[@]}"; do
  log "$line"
done
log "Compare the per-task inference_sec distributions above (warmup-excluded) and"
log "the wall-clock totals here. Higher batch should lower per-window cost until"
log "the SM array saturates; with MPS overlap the knee may be earlier than alone."
