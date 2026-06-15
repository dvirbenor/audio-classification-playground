#!/usr/bin/env bash
#
# benchmark_mps_vs_dedicated.sh — controlled, gated, sustained A/B of MPS-colocated
# vs dedicated (serial) GPU inference on ONE GPU, with nvidia-smi capture.
#
# WHY THIS EXISTS
#   The production cross-pool number (MPS pod vs the per-task fleet) is confounded:
#   the two pools claim DIFFERENT archives, run for DIFFERENT durations, and (the
#   real trap) the controlled validation in MPS_COLOCATION_TESTING.md ran only 3
#   archives on a warmed cache, so it never exposed sustained-load effects. This
#   bench fixes all three:
#     1. SAME GPU, SAME fixed archive subset for every arm (apples-to-apples).
#     2. GATING ENFORCED — a VAD pre-pass populates vad/ for the subset and the GPU
#        workers run with --require-precomputed-vad, so EVERY archive is actually
#        VAD-gated (computing full-timeline on non-gated data is meaningless; this
#        also measures the gating speedup honestly).
#     3. SUSTAINED — run enough archives to amortize compile warmup AND reach thermal
#        steady state, with nvidia-smi dmon capturing SM%, clocks, power, temp so you
#        can tell GPU THROTTLING (clocks sag under sustained 3-way load) from PREFETCH
#        STARVATION (GPU idle waiting on I/O; visible as low SM% at full clocks).
#
# ARMS
#   dedicated : each task run ALONE, back-to-back, on the full GPU (ENABLE_MPS=0).
#               Sum of wall-clocks = one GPU doing all 3 tasks serially.
#               max() of the three = the 3-dedicated-GPU fleet's wall (GPUs in parallel).
#   mps       : affect+disfluency+emotion co-located on the one GPU via CUDA MPS
#               (scripts/run_mps_colocated.sh, ENABLE_MPS=1).
#   The headline is mps_wall vs dedicated_serial_wall (the ~1.44x question) and the
#   per-GPU comparison vs the 3-GPU fleet.
#
# RUN ON A GPU POD, against a SCRATCH tree (never the prod output):
#   source env.shared.sh && source .venv/bin/activate   # bare `python` must be on PATH
#   export PARQUET=/efs/.../scratch/subset.parquet       # a few dozen archives, bounded
#   export OUTPUT_BASE=/efs/.../scratch/mps-vs-dedicated  # throwaway
#   export AUDIO_CACHE=/efs/.../scratch/audio_cache CACHE_BYTES=200000000000
#   bash scripts/benchmark_mps_vs_dedicated.sh
#
# CONFIG (env)
#   PARQUET                (required) SMALL parquet subset
#   OUTPUT_BASE            (required) scratch base; per-arm outputs go in <base>/<arm>
#   AUDIO_CACHE/CACHE_BYTES(optional) shared decoded-audio cache; if set, warmed once
#   WARM_CACHE             (default 1 when AUDIO_CACHE set, else 0) pre-warm => pure GPU
#   ARMS                   (default "dedicated mps")
#   TASKS                  (default "affect disfluency emotion")
#   WAVLM_STATIC_BATCH_SIZE(default 512)  WAVLM_RUNTIME_PRESET (default compiled_static)
#   EMOTION_RUNTIME_MODE   (default optimized)
#   AFFECT_BACKBONE/DISFLUENCY_BACKBONE (default wavlm)
#   SMI_INTERVAL           (default 1) nvidia-smi dmon sample seconds
#   CLEAN                  (default 1) rm -rf each arm dir before running
#   MAX_RETRIES            (default 3)
#
set -uo pipefail

log() { echo "[$(date -Iseconds)] [bench] $*"; }
die() { log "FATAL: $*"; exit 1; }

: "${PARQUET:?PARQUET is required (point at a SMALL subset)}"
: "${OUTPUT_BASE:?OUTPUT_BASE is required (scratch dir)}"

ARMS="${ARMS:-dedicated mps}"
TASKS="${TASKS:-affect disfluency emotion}"
WAVLM_STATIC_BATCH_SIZE="${WAVLM_STATIC_BATCH_SIZE:-512}"
WAVLM_RUNTIME_PRESET="${WAVLM_RUNTIME_PRESET:-compiled_static}"
EMOTION_RUNTIME_MODE="${EMOTION_RUNTIME_MODE:-optimized}"
AFFECT_BACKBONE="${AFFECT_BACKBONE:-wavlm}"
DISFLUENCY_BACKBONE="${DISFLUENCY_BACKBONE:-wavlm}"
SMI_INTERVAL="${SMI_INTERVAL:-1}"
CLEAN="${CLEAN:-1}"
MAX_RETRIES="${MAX_RETRIES:-3}"
if [ -n "${AUDIO_CACHE:-}" ]; then WARM_CACHE="${WARM_CACHE:-1}"; else WARM_CACHE="${WARM_CACHE:-0}"; fi

MODULE="audio_classification_playground.acoustic_events.orchestration"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$OUTPUT_BASE/_bench_results"
mkdir -p "$RESULTS"

have_smi=0
command -v nvidia-smi >/dev/null 2>&1 && have_smi=1 || log "WARNING: nvidia-smi not found — skipping GPU telemetry capture"

cache_args=()
if [ -n "${AUDIO_CACHE:-}" ]; then
  : "${CACHE_BYTES:?CACHE_BYTES is required when AUDIO_CACHE is set}"
  cache_args=(--audio-cache-dir "$AUDIO_CACHE" --max-cache-bytes "$CACHE_BYTES")
fi

# --- nvidia-smi dmon capture (one device assumed; the pod has nvidia.com/gpu: 1) ---
SMI_PID=""
start_smi() {  # $1 = label
  [ "$have_smi" = "1" ] || return 0
  local f="$RESULTS/smi_$1.log"
  : > "$f"
  # pucm = power+temp, utilization (sm/mem), proc+mem clocks. -o DT prefixes date/time.
  nvidia-smi dmon -s pucm -d "$SMI_INTERVAL" -o DT >> "$f" 2>/dev/null &
  SMI_PID=$!
}
stop_smi() {
  [ -n "$SMI_PID" ] || return 0
  kill "$SMI_PID" 2>/dev/null || true
  wait "$SMI_PID" 2>/dev/null || true
  SMI_PID=""
}
cleanup() { stop_smi; }
trap cleanup EXIT INT TERM

# Summarize an nvidia-smi dmon log: mean/max SM%, mean/min SM-clock (throttle tell),
# mean/max power and temp. Robust to column order (parses the names header).
smi_summary() {  # $1 = label
  [ "$have_smi" = "1" ] || { echo "  (no telemetry)"; return; }
  local f="$RESULTS/smi_$1.log"
  [ -s "$f" ] || { echo "  (no telemetry samples)"; return; }
  python - "$f" <<'PY'
import sys
rows=[]; names=None
for line in open(sys.argv[1]):
    s=line.strip()
    if not s: continue
    if s.startswith("#"):
        toks=s.lstrip("#").split()
        if "sm" in toks: names=toks           # the names header (has 'sm','pclk',...)
        continue
    if names is None: continue
    parts=s.split()
    if len(parts)!=len(names): continue
    rows.append(dict(zip(names,parts)))
def col(key):
    out=[]
    for r in rows:
        v=r.get(key)
        try: out.append(float(v))
        except (TypeError,ValueError): pass
    return out
def stat(key,fn,lbl):
    xs=col(key)
    return f"{lbl}={fn(xs):.0f}" if xs else f"{lbl}=NA"
n=len(rows)
sm=col("sm"); pclk=col("pclk")
parts=[f"n={n}"]
if sm:   parts+= [f"SM%%(mean/max)={sum(sm)/len(sm):.0f}/{max(sm):.0f}"]
if pclk: parts+= [f"SMclk(mean/min/max)={sum(pclk)/len(pclk):.0f}/{min(pclk):.0f}/{max(pclk):.0f}MHz"]
print("  " + "  ".join(parts + [stat("pwr",max,"pwrMax(W)"), stat("gtemp",max,"tempMax(C)")]))
print("  (SMclk sagging below its max under load => THROTTLING; low SM%% at max clock => STARVATION)")
PY
}

run_vad_prepass() {  # $1 = out dir — populate vad/ for the subset (CPU), so gating can be enforced
  local out="$1"
  log "VAD pre-pass -> $out (CPU; enables --require-precomputed-vad)"
  python -m "$MODULE" run \
    --parquet "$PARQUET" --output "$out" \
    --task-group vad --completion-policy exists \
    --affect-backbone "$AFFECT_BACKBONE" --disfluency-backbone "$DISFLUENCY_BACKBONE" \
    --device cpu --prefetch-workers 6 --prefetch-lookahead 12 --vad-prefetch-workers 1 \
    --max-retries "$MAX_RETRIES" "${cache_args[@]}" \
    || die "VAD pre-pass failed for $out"
}

run_one_gpu_task() {  # $1=task $2=out  — dedicated: one task alone on the full GPU
  local task="$1" out="$2"
  local argv=( python -m "$MODULE" run
    --parquet "$PARQUET" --output "$out"
    --task-group "$task" --completion-policy exists
    --affect-backbone "$AFFECT_BACKBONE" --disfluency-backbone "$DISFLUENCY_BACKBONE"
    --device cuda --max-retries "$MAX_RETRIES"
    --vad-gating --require-precomputed-vad )
  if [ "$task" = "affect" ] || [ "$task" = "disfluency" ]; then
    argv+=( --wavlm-runtime-preset "$WAVLM_RUNTIME_PRESET" --wavlm-static-batch-size "$WAVLM_STATIC_BATCH_SIZE" )
  fi
  [ "$task" = "emotion" ] && argv+=( --emotion-runtime-mode "$EMOTION_RUNTIME_MODE" )
  argv+=( "${cache_args[@]}" )
  "${argv[@]}"
}

# --- optional one-time cache warm (=> pure GPU time, isolates MPS effect) ----------
if [ "$WARM_CACHE" = "1" ]; then
  log "Warming audio cache (pure-GPU mode) ..."
  python -m "$MODULE" warm-cache --parquet "$PARQUET" --output "$OUTPUT_BASE" \
    "${cache_args[@]}" --warm-workers 8 --s3-max-pool-connections 64 --max-retries "$MAX_RETRIES" \
    || log "WARNING: warm-cache returned non-zero; continuing"
fi

declare -a SUMMARY=()
declare -A WALL=()

for arm in $ARMS; do
  out="$OUTPUT_BASE/$arm"
  [ "$CLEAN" = "1" ] && rm -rf "$out"
  mkdir -p "$out"
  log "================ ARM: $arm  ->  $out ================"

  run_vad_prepass "$out"

  start_smi "$arm"
  arm_start=$(date +%s)

  if [ "$arm" = "dedicated" ]; then
    for task in $TASKS; do
      log "--- dedicated: $task (alone on full GPU) ---"
      t0=$(date +%s)
      run_one_gpu_task "$task" "$out" || log "WARNING: dedicated $task exited non-zero"
      t1=$(date +%s)
      WALL["dedicated/$task"]=$((t1 - t0))
      log "    dedicated $task wall=${WALL["dedicated/$task"]}s"
    done
  elif [ "$arm" = "mps" ]; then
    log "--- mps: $TASKS co-located via CUDA MPS, gating enforced ---"
    PARQUET="$PARQUET" OUTPUT="$out" \
    TASKS="$TASKS" \
    WAVLM_RUNTIME_PRESET="$WAVLM_RUNTIME_PRESET" \
    WAVLM_STATIC_BATCH_SIZE="$WAVLM_STATIC_BATCH_SIZE" \
    EMOTION_RUNTIME_MODE="$EMOTION_RUNTIME_MODE" \
    AFFECT_BACKBONE="$AFFECT_BACKBONE" DISFLUENCY_BACKBONE="$DISFLUENCY_BACKBONE" \
    VAD_GATING="1" REQUIRE_VAD="1" ENABLE_MPS="1" MAX_RETRIES="$MAX_RETRIES" \
    ${AUDIO_CACHE:+AUDIO_CACHE="$AUDIO_CACHE" CACHE_BYTES="$CACHE_BYTES"} \
    bash "$HERE/run_mps_colocated.sh" || log "WARNING: mps arm exited non-zero"
  else
    log "WARNING: unknown arm '$arm' — skipping"; continue
  fi

  arm_end=$(date +%s)
  stop_smi
  WALL["$arm"]=$((arm_end - arm_start))

  log "--- timings ($arm) [per-archive inference_sec, warmup-excluded] ---"
  python -m "$MODULE" timings --output "$out" || true
  log "--- GPU telemetry ($arm) ---"
  smi_summary "$arm"
  SUMMARY+=("arm=$arm wall=${WALL["$arm"]}s output=$out smi=$RESULTS/smi_$arm.log")
done

echo
log "================ SUMMARY ================"
for line in "${SUMMARY[@]}"; do log "$line"; done

# Headline ratios (only when both arms ran).
ded_serial=0; have_ded=0
for task in $TASKS; do
  v="${WALL["dedicated/$task"]:-}"
  [ -n "$v" ] && { ded_serial=$((ded_serial + v)); have_ded=1; }
done
mps_w="${WALL["mps"]:-}"
if [ "$have_ded" = "1" ] && [ -n "$mps_w" ] && [ "$mps_w" -gt 0 ]; then
  log "Dedicated SERIAL (1 GPU, all tasks back-to-back): ${ded_serial}s"
  log "MPS co-located    (1 GPU, all tasks concurrent):   ${mps_w}s"
  ratio="$(python - "$ded_serial" "$mps_w" <<'PY'
import sys
ded=float(sys.argv[1]); mps=float(sys.argv[2])
print(f"{ded/mps:.2f}x ({'GAIN' if ded/mps>=1 else 'LOSS'} per GPU)")
PY
)"
  log "=> MPS speedup vs serial-on-1-GPU: $ratio"
  log "NOTE: vs a 3-dedicated-GPU fleet, compare MPS wall (${mps_w}s on 1 GPU) to the"
  log "      SLOWEST single task above (fleet wall on 3 GPUs); per-GPU = archives / (wall x GPUs)."
fi
log "Raw nvidia-smi dmon logs in $RESULTS/ — inspect SM-clock over time for throttling."
log "Done."
