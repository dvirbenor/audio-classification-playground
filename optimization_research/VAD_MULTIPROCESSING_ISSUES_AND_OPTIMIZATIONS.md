# VAD Multiprocessing: Issues & Optimizations

How the CPU VAD fleet went from ~53 to ~343 archives/h **per pod** (~6.5×), why the
naive knobs didn't work, and the supporting fixes (real-throughput status, gating
enforcement). Investigated 2026-06-15.

---

## TL;DR — throughput rate increase

| Configuration | Per-pod | Per-worker-process | Aggregate |
|---|---|---|---|
| Old `vad-cpu` fleet (`--vad-prefetch-workers 1`) | ~30–80/h | ~30–80/h | ~100/h (3 active) |
| Backfill, single process, `--vad-prefetch-workers 8` | **~53/h** | ~53/h | (1 proc, 1 core) |
| **Backfill, `run_vad_multiproc.sh` `VAD_PROCS=8`** | **~343/h** | ~43/h | **~6,516/h (19 pods)** |

**~6.5× per pod** at the same instance/cost. The ~474k-archive VAD backlog drops from
~64 days (single-process at current pod count) to **~3 days**. Measured from real
completion timestamps (`ts`), not the dashboard `Pace`.

---

## The problem

VAD coverage was only **115,657 / 589,648 archives (~19.6%)** (authoritative
`orchestration progress --fast` scan, 2026-06-15; ~474k backlog, 1,615 permanent audio
errors unreachable). Because the GPU tasks had run *ahead* of VAD, **0 / 203,375 GPU
completions were VAD-gated** — the entire fleet computed full-timeline, leaving the
33–56% gating speedup unrealized.

The first backfill attempt ran one `orchestration run --task-group vad
--vad-prefetch-workers 8` per pod on 16‑vCPU 4xlarges, but **htop showed a single core
busy** and throughput was ~53 arc/h/pod — no better than one core, despite 8 "VAD
workers" and 13 CPUs requested.

## Root cause: Silero VAD is GIL-bound, and the knob was threads

1. **Silero is a Python loop over ~30 ms windows.** `_detect_silero_vad` →
   `get_speech_timestamps` ([runners.py](../audio_classification_playground/acoustic_events/inference/runners.py))
   iterates ~110k tiny per-window model calls for a 60‑min archive. The Python loop
   holds the GIL between calls; each op is too small for intra-op threading.
2. **`--vad-prefetch-workers` is a `ThreadPoolExecutor`** ([prefetch.py](../audio_classification_playground/acoustic_events/orchestration/prefetch.py)).
   Threads share one GIL, so N VAD threads **serialize onto one core**. Raising the
   value does nothing — confirmed: `--vad-prefetch-workers 8` still used one core.

This corrects the earlier hypothesis (in the VAD-gating notes) that more
`--vad-prefetch-workers` would give ~8×. It can't — it's the wrong axis.

## The fix: one process per physical core

`scripts/run_vad_multiproc.sh` launches **N independent VAD worker processes** on one
pod (separate interpreters → separate GILs → separate cores), each claiming distinct
archives via the existing per-archive locks + `--completion-policy exists`. Mirrors
`run_mps_colocated.sh`, but CPU-only (no MPS). It supervises, forwards SIGTERM for
graceful drain, and emits `VAD_MULTIPROC total_processed=<N> clean=<0|1>` for the
backfill loop.

**Sizing — one process per *physical* core.** VAD is CPU-bound, so SMT siblings just
contend for execution units; running `nproc` processes is worse than `nproc/2`. On the
SMT x86 instances (c6a/c7a/c6i/c7i 4xlarge = 16 vCPU = 8 physical cores) that is
**`VAD_PROCS=8`**. Non-SMT (Graviton) would use `nproc`. `nproc` is unreliable under a
cgroup CPU quota, so `VAD_PROCS` is pinned in the manifest rather than auto-detected.

Per-process config: `--vad-prefetch-workers 1` (more only GIL-contends within a proc),
`--prefetch-workers 2` (decode releases the GIL, so a little decode concurrency per proc
overlaps download/decode with VAD).

Deployed in `manifests/acoustic-events-vad-backfill.yaml` (loops the launcher until a
pass processes zero archives, retrying on a non-clean pass).

## Results & interpretation

- **~53 → ~343 arc/h per pod (~6.5×)**; ~6,516 arc/h across 19 pods (real `ts`-delta,
  3‑min steady-state sample; still climbing as the per-proc model-load warmup amortizes).
- The 6.5× (vs the 8× ceiling) is **mild network/decode contention, not a hard I/O
  wall** — if S3 download were the bottleneck, 8 processes would plateau near 1×.
  Getting 6.5× proves the cores were the constraint (GIL) and the fix addressed it.
- Verify the launcher engaged: **8 timing files per pod hostname** (8 UUIDs) /
  `[vad-multiproc] launching 8` in the job log / 8 `python` PIDs in htop.

## Supporting fixes

### 1. `status` Pace was latency-derived nonsense → now real throughput
`heartbeat.py` computed `Pace = 3600 / mean(total_sec)`. For VAD, `total_sec` records
only the main-loop slice (~0.05 s), so Pace showed **~50,000–72,000 arc/h** while the
real rate was tens/h. It also *under*-reported MPS. Now Pace is derived from real
completion-timestamp deltas across the recent tail, and a host's concurrently-active
processes are **summed** (so the MPS pod shows ~392/h = its 3 procs, and a multiproc VAD
pod shows the pooled rate). Caveat: ts-based Pace is noisy for a just-started worker with
few records (can read ~500/h while real done/elapsed is ~50/h) — for ground truth use
`scripts/mps_vs_fleet_throughput.py --window 10m`.

### 2. `status --active-within` declutters dead workers
The dashboard listed ~111 workers, most dead (15‑day-old runs, stopped pods), and the
fleet pace (~13,124/h) summed their stale historical rates. `--active-within 30m` drops
idle workers from **both the rows and the totals** (→ 40 live workers, ~8,828/h
meaningful aggregate).

### 3. Gating enforcement: `--require-precomputed-vad`
New worker flag: a GPU worker **skips (never claims)** an archive that lacks a `vad/`
artifact, instead of silently falling back to full-timeline. Surfaced in worker logs as
`skipped=… (no_vad=…)`; no-op for task-groups that produce vad themselves. Exposed via
`REQUIRE_VAD=1` in `run_mps_colocated.sh`. Enabled on the GPU fleet (MPS manifest +
the dedicated `cache-workers-optimized` affect/disfluency/emotion tasks) now that the
backfill (~6.5k/h) far outpaces GPU consumption (~1.5k/h), so gated work is always
available and `completion-policy=exists` means already-done archives aren't reprocessed.
Track realization via the **VAD-gating coverage** section of
`scripts/mps_vs_fleet_throughput.py` (was `0% gated`; should climb).

## How to monitor

```bash
# real per-worker VAD rate (ignore the noisy Pace for fresh workers)
uv run python scripts/mps_vs_fleet_throughput.py --window 10m     # Other (CPU/VAD) section

# clean dashboard (live workers only) + correct fleet pace
uv run python -m audio_classification_playground.acoustic_events.orchestration \
  status --output <OUTPUT> --active-within 30m

# authoritative VAD coverage (walks the tree)
uv run python -m audio_classification_playground.acoustic_events.orchestration \
  progress --output <OUTPUT> --fast
```

## Open items / further levers

- **Network ceiling.** If pods are ever scaled until 8 procs saturate the pod NIC
  (procs sit in `D`/I/O-wait at low CPU), add an audio cache to the backfill
  (`AUDIO_CACHE`/`CACHE_BYTES`, already supported by the launcher) or raise per-proc
  `PREFETCH_WORKERS`.
- **ONNX Silero** (`onnx=True`) runs via onnxruntime, which releases the GIL — could let
  in-process threads parallelize and/or speed each archive. Multiprocess was the surer
  win and needed no inference-path change, but ONNX is worth A/B-ing for per-core speed.
- **Backfill vs leading fleet.** A one-shot backfill doesn't help the GPU fleet reuse
  decode (the ~500 GB cache is a rolling ~2k-archive window). For steady state, run VAD
  as a *leading* fleet sized to outpace GPU consumption so decode is paid once and gating
  is realized inline.

## Related

- `optimization_research/VAD_GATING_IMPLEMENTATION_PLAN.md` — the gating mechanism.
- `scripts/run_vad_multiproc.sh`, `scripts/run_mps_colocated.sh` — the launchers.
- `scripts/mps_vs_fleet_throughput.py` — real-throughput + gating-coverage tooling.
