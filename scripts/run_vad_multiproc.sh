#!/usr/bin/env bash
#
# run_vad_multiproc.sh — run N CPU VAD worker PROCESSES on one pod so VAD actually
# uses all the cores you're paying for.
#
# WHY: Silero VAD is GIL-bound — get_speech_timestamps is a Python loop over ~30 ms
# windows (~110k tiny model calls for a 60-min archive), and --vad-prefetch-workers is
# a ThreadPoolExecutor, so a single worker process pins to ~ONE core no matter how many
# VAD threads it gets (the threads serialize on the GIL). Separate PROCESSES have
# separate GILs, so N processes use N cores. This mirrors run_mps_colocated.sh
# (multi-proc on one node) but CPU-only and without MPS. Workers coordinate via
# per-archive locks + --completion-policy exists, so the N procs never double-process.
#
# HYPERTHREADING: VAD is CPU-bound, so one process per PHYSICAL core beats one per
# logical thread — SMT siblings contend for the same execution units. On SMT x86
# (the c6a/c7a/c6i/c7i instances this backfill targets) physical cores = nproc/2, which
# is the default here. But `nproc` is unreliable under a cgroup CPU quota (it may report
# the node, not the pod's allocation), so PIN VAD_PROCS explicitly in the manifest
# (16-vCPU 4xlarge => VAD_PROCS=8). Non-SMT hosts (Graviton) want VAD_PROCS=nproc.
#
# Config (env):
#   PARQUET               (required) all_archives.parquet
#   OUTPUT                (required) EFS output base
#   VAD_PROCS             (default: nproc/2) number of VAD worker processes
#   AUDIO_CACHE/CACHE_BYTES (optional) shared decoded-audio cache (CACHE_BYTES required if set)
#   PREFETCH_WORKERS      (default 2) decode threads PER proc (decode releases the GIL)
#   PREFETCH_LOOKAHEAD    (default 4) per proc
#   VAD_PREFETCH_WORKERS  (default 1) VAD threads per proc (>1 only GIL-contends within a proc)
#   AFFECT_BACKBONE/DISFLUENCY_BACKBONE (default wavlm) required by the `run` CLI
#   COMPLETION_POLICY     (default exists)
#   MAX_RETRIES           (default 3)
#   DEVICE                (default cpu)
#
# Prints a machine-parseable result line for the backfill loop:
#   VAD_MULTIPROC total_processed=<N> clean=<0|1>
# clean=1 iff every proc exited 0 with a "Worker finished" line; the wrapping loop
# should retry when clean=0 and stop when total_processed=0.
#
set -uo pipefail

log() { echo "[$(date -Iseconds)] [vad-multiproc] $*"; }

: "${PARQUET:?PARQUET is required}"
: "${OUTPUT:?OUTPUT is required}"

_default_procs() {
  local n
  n="$(nproc 2>/dev/null || echo 2)"
  local half=$(( n / 2 ))
  [ "$half" -ge 1 ] && echo "$half" || echo 1
}
VAD_PROCS="${VAD_PROCS:-$(_default_procs)}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"
PREFETCH_LOOKAHEAD="${PREFETCH_LOOKAHEAD:-4}"
VAD_PREFETCH_WORKERS="${VAD_PREFETCH_WORKERS:-1}"
AFFECT_BACKBONE="${AFFECT_BACKBONE:-wavlm}"
DISFLUENCY_BACKBONE="${DISFLUENCY_BACKBONE:-wavlm}"
COMPLETION_POLICY="${COMPLETION_POLICY:-exists}"
MAX_RETRIES="${MAX_RETRIES:-3}"
DEVICE="${DEVICE:-cpu}"

MODULE="audio_classification_playground.acoustic_events.orchestration"
WORK_DIR="$(mktemp -d)"

# Populates the global WORKER_ARGV array (avoids bash-4 mapfile/process subst).
build_worker_argv() {
  WORKER_ARGV=(
    python -m "$MODULE" run
      --parquet "$PARQUET" --output "$OUTPUT"
      --task-group vad --completion-policy "$COMPLETION_POLICY"
      --affect-backbone "$AFFECT_BACKBONE" --disfluency-backbone "$DISFLUENCY_BACKBONE"
      --device "$DEVICE"
      --prefetch-workers "$PREFETCH_WORKERS"
      --prefetch-lookahead "$PREFETCH_LOOKAHEAD"
      --vad-prefetch-workers "$VAD_PREFETCH_WORKERS"
      --max-retries "$MAX_RETRIES"
  )
  if [ -n "${AUDIO_CACHE:-}" ]; then
    : "${CACHE_BYTES:?CACHE_BYTES is required when AUDIO_CACHE is set}"
    WORKER_ARGV+=(--audio-cache-dir "$AUDIO_CACHE" --max-cache-bytes "$CACHE_BYTES")
  fi
}

pids=()
shutting_down=0
term_handler() {
  shutting_down=1
  log "signal received — forwarding SIGTERM to ${#pids[@]} workers (graceful drain)"
  [ "${#pids[@]}" -gt 0 ] && for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap term_handler TERM INT

log "launching $VAD_PROCS VAD worker processes "\
"(prefetch=$PREFETCH_WORKERS lookahead=$PREFETCH_LOOKAHEAD vad_prefetch=$VAD_PREFETCH_WORKERS)"
build_worker_argv
for i in $(seq 1 "$VAD_PROCS"); do
  "${WORKER_ARGV[@]}" > "$WORK_DIR/proc-$i.log" 2>&1 &
  pids+=("$!")
  log "  -> proc $i pid $!"
done

# Supervise by polling (bash<4.3 compatible). Unlike the MPS launcher we do NOT tear
# down siblings when one VAD proc exits: a proc exiting 0 just means it found no more
# claimable work (peers may still be busy). Only a NON-zero exit during normal
# operation is a failure we record (so the backfill loop / k8s retries).
exit_code=0
ticks=0
while [ "${#pids[@]}" -gt 0 ]; do
  sleep 2
  ticks=$((ticks + 1))
  alive=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    else
      wait "$pid"; rc=$?
      if [ "$rc" -ne 0 ] && [ "$shutting_down" != "1" ]; then
        log "worker pid $pid exited rc=$rc"
        exit_code="$rc"
      else
        log "worker pid $pid exited rc=$rc"
      fi
    fi
  done
  pids=( ${alive[@]+"${alive[@]}"} )
  # Liveness heartbeat ~every 60s (real throughput: use `orchestration progress --fast`).
  if [ $((ticks % 30)) -eq 0 ] && [ "${#pids[@]}" -gt 0 ]; then
    log "${#pids[@]}/$VAD_PROCS procs alive"
  fi
done

# Aggregate processed counts + clean-finish detection across procs.
total=0
finishes=0
for i in $(seq 1 "$VAD_PROCS"); do
  f="$WORK_DIR/proc-$i.log"
  [ -f "$f" ] || continue
  line="$(grep -oE 'Worker finished: processed=[0-9]+' "$f" | tail -1 || true)"
  if [ -n "$line" ]; then
    finishes=$((finishes + 1))
    n="$(printf '%s' "$line" | grep -oE '[0-9]+' || echo 0)"
    total=$((total + n))
  fi
done
clean=0
[ "$exit_code" -eq 0 ] && [ "$finishes" -eq "$VAD_PROCS" ] && clean=1

log "all workers exited (exit_code=$exit_code, clean_finishes=$finishes/$VAD_PROCS)"
echo "VAD_MULTIPROC total_processed=$total clean=$clean"
rm -rf "$WORK_DIR"
exit "$exit_code"
