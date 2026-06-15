# VAD Multiprocessing: Issues & Optimizations

How the CPU VAD fleet went from ~53 to ~660 archives/h **per pod** (~12×), why the
naive knobs didn't work, and the supporting fixes (real-throughput status, gating
enforcement). Investigated 2026-06-15. **Three** stages, each fixing what the prior one
exposed:
1. **One process per physical core** (~53→~343/h) — Silero VAD is GIL-bound, so processes
   not threads. ([The fix](#the-fix-one-process-per-physical-core))
2. **Pin per-process math threads + leaner decode** (~343→~660/h, fixed a 20-pod
   OOM/oversubscription failure) — torch fanned intra-op threads across all cores.
   ([Scale-up to 20 pods](#scale-up-to-20-pods-oversubscription--oom-and-the-real-per-process-fix-2026-06-15))
3. **Per-pass undone work-list** — the ~660/h was measured on an empty scratch dir; against
   the partially-done prod tree the workers stat-walked the done-prefix at ~2% CPU. Feed
   them only vad-absent archives. ([The skip-scan trap](#the-skip-scan-trap-re-scanning-a-partially-done-corpus-2026-06-15-cont))

**Current production state:** 3 pods, compute-bound (~100% CPU/proc), ~1.8–2k arc/h
aggregate; scaling pods up now scales ~linearly. The TL;DR table below is the *stage-1*
snapshot — later stages supersede its per-pod and aggregate figures.

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
`--prefetch-workers 1` + `--prefetch-lookahead 2` (decode releases the GIL; one bursty
decode thread overlaps download/decode with VAD without pinning a second core or holding
OOM-causing buffers). **Plus each worker is pinned to one math thread** (`OMP/MKL/
OPENBLAS/NUMEXPR=1`) — see the scale-up section below for why this is mandatory, not
optional. *(The earlier `--prefetch-workers 2` / no-pinning config OOM'd and oversubscribed
at 20 pods.)*

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

## Scale-up to 20 pods: oversubscription + OOM, and the real per-process fix (2026-06-15)

The single-pod multiproc result above (~343/h) did **not** survive scaling to a 20-pod
Job. The launch **failed**: Job condition `Failed` / `BackoffLimitExceeded`
(`completions=20`, `succeeded=0`, **`failed=29`**, `backoffLimit=10`), and Kubernetes
terminated every still-running pod once failures passed the limit ("pods terminating
despite still running"). Per-pod throughput had also collapsed to **~70–110 arc/h** — a
~5× regression vs the single-pod number.

**Diagnosis (htop on a live pod):** `VAD_PROCS=8`, but **~16 CPU-hot threads on 8
physical cores**, load avg ~9.4. The "extra" rows paired with the 8 workers by
*byte-identical* RSS (e.g. 2317M↔2317M, 3232M↔3232M) — two real processes would have
COW-diverged after an hour, so these were **threads of the 8 procs**, not extra procs.

**Two root causes the original multiproc fix missed** ("separate GILs → separate cores"
was necessary but *not sufficient*):

1. **No math-thread pinning.** Silero VAD is Silero-via-`torch.hub`
   ([runners.py](../audio_classification_playground/acoustic_events/inference/runners.py));
   torch (and OpenMP/MKL/OpenBLAS) default their intra-op pools to **all cores**, and
   those C++ ops run *outside* the GIL. So each of the 8 worker processes fanned threads
   across all 8 physical cores → **N×N contention**, thrashing instead of 8 clean cores.
   The GIL argument explains why *Python-thread* VAD workers don't scale; it does not stop
   torch from over-threading each process.
2. **Over-aggressive decode prefetch.** `PREFETCH_WORKERS=2 × PREFETCH_LOOKAHEAD=4` kept a
   second GIL-releasing **decode** core busy per process *and* held several ~250 MB decoded
   archives resident. Across 8 procs that pushed peak RSS **past the 24 Gi pod limit →
   OOMKill → non-zero exit → `backoffLimit` → whole Job Failed.** So the throughput
   regression and the terminations were the *same* root cause (too many threads + too much
   resident memory per pod).

**Fixes:**

- **Pin each worker to one compute thread** — `scripts/run_vad_multiproc.sh` now exports
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1` *before* the
  python workers start (torch reads `OMP_NUM_THREADS` at import to set its default intra-op
  count). Now 8 procs = 8 single-threaded VAD cores.
- **Leaner decode** — manifest `PREFETCH_WORKERS 2→1`, `PREFETCH_LOOKAHEAD 4→2`. Decode
  (~12 s) is far cheaper than VAD (~70 s), so one bursty decode thread per proc keeps
  archives ready without pinning a second core or holding OOM-causing buffers.

**Single-pod validation (real `c*.4xlarge`, 16 vCPU / 8 physical, 24 Gi limit):**

| Metric | Broken 20-pod run | After fix |
|---|---|---|
| CPU-hot threads (>20%) | ~16 on 8 cores (thrash) | **8** (one VAD/core) |
| Total RSS | > 24 Gi → **OOMKilled** | **18.9 GB** (~5.5 GB headroom) |
| Thread pin | none (torch fanned out) | `OMP/MKL/OPENBLAS=1` confirmed in `/proc/<pid>/environ` |
| Throughput | ~80 arc/h/pod | **~660 arc/h/pod** (22 archives / 120 s steady-state) |

So the corrected config roughly **doubled** per-pod throughput vs the pre-pinning ~343/h
*and* fixed the scale-up OOM. ~~`~660/h × 20 pods ≈ ~13k/h → backlog in ~1.5–2 days`~~ —
**this linear projection did NOT hold against the real corpus** (see next section): that
~660/h was measured against an *empty scratch* output, which masked a separate skip-scan
bottleneck that dominates against the partially-done prod tree.

> **Lesson:** when sizing CPU-bound multi-process work, count *all* threads each process can
> spawn (torch/BLAS intra-op pools), not just the obvious worker thread — and pin them. And
> validate memory at the per-pod cgroup limit before scaling, since one OOM per pod ×
> `backoffLimit` fails the entire Job.

## The skip-scan trap: re-scanning a partially-done corpus (2026-06-15, cont.)

With the thread-pin + lean-decode fix in, the backfill **still stalled at ~2% CPU/proc**
when relaunched against the **prod output tree** — but with *no* OOM this time. Reducing
20 → 3 pods did **not** help (still ~2%), which **ruled out peer-EFS contention** (3 pods =
24 procs would have recovered if contention were the cause).

**Root cause — the skip-scan.** The workers run `--completion-policy exists` over
`all_archives.parquet` in deterministic **session-sorted** order. The leading `vad-cpu`
fleet had already completed the early sessions, so every proc loads all ~589k entities and
**stat-walks the ~100k+ already-done prefix on EFS** (one `is_task_artifact_complete_for_archive`
per archive) before reaching fresh work — EFS-metadata-bound, ~zero compute. Confirmed in
the run log: `Loaded 589648 unique entities`, then **silent at 2% CPU**. The earlier
~660/h validation hid this entirely because it used an **empty scratch output** → every
archive was fresh, no done-prefix to walk. *(Lesson: validate against a representative,
partially-done output tree, not an empty scratch dir.)*

**Rejected fix — random shuffle seed alone.** Shuffling each proc's scan order helps *only*
while the corpus is mostly undone (at 80% undone a random draw hits fresh work in ~1 try).
It **degrades as coverage rises** — at 90% done, 90% of draws hit already-done archives, so
the skip-scan returns inverted — and it makes "0 processed ⇒ done" termination unreliable.
It doesn't converge.

**The fix — a per-pass UNDONE work-list.** `build_subset_parquet.py` gained
`--vad-absent-under <output> --all` (+ a parallel 64-thread EFS scan): it emits **only the
vad-absent archives**, using the *same* `is_task_artifact_complete_for_archive` primitive
the worker's `exists` policy uses, so filter and workers agree exactly (nothing wrongly
dropped). The backfill loop now **rebuilds this work-list each pass** and feeds the workers
*only undone archives* → a ~**100% claim hit rate at any coverage level** (20% or 95% done),
and it **terminates** cleanly when the list is empty (robust, vs the flaky `processed=0`
heuristic). The per-proc shuffle `--seed` is kept purely as **anti-herd** over the
now-all-undone list (so N procs don't collide on its first rows). The per-pass scan cost
(~one parallel tree walk) is paid once by one process, vs 24 procs continuously re-statting.

**Measurement gotcha — startup ramp vs steady state.** At multi-pod scale the **first
~6–10 min is a slow startup**: ~24 procs contend on the shared EFS model-cache + first-S3
fetches, so they idle at ~2% CPU — measuring *then* reads "broken." A `/proc/<pid>` snapshot
(one thread already `R`-state at ~66% CPU, no disk-read movement) flagged it as startup, and
a delayed re-measure confirmed steady state: **8/8 procs at 100–115% CPU** (VAD thread +
decode core), RSS ~10 GB, compute-bound. **Always distinguish ramp from steady state before
concluding.**

**Outcome & current config.** Running **3 pods** (deliberately small — "for safety"),
confirmed compute-bound ≈ **~1.8–2k arc/h aggregate → ~456k backlog in ~9–10 days**. Since
it is now genuinely compute-bound (NOT EFS-walled), **scaling pods up scales throughput
~linearly** — the earlier 20-pod collapse was the *pre-fix* code (skip-scan + OOM), not a
hard EFS ceiling. Bump the template's `parallelism`/`completions` to trade the ~10-day ETA
down (watch only the one-time startup model-cache contention; steady-state stays per-pod
compute-bound). Deployed in `scripts/build_subset_parquet.py` +
`manifests/acoustic-events-vad-backfill.yaml` (commit `d3c9a13`).

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
