# Triton Backfill Debugging Findings

**Date:** 2026-06-25  
**Job:** `arno-triton-backfill-models-gated-001-260625-*`  
**Fleet:** 60 CPU worker pods + 12 Triton TRT GPU pods (g7e.2xlarge, Blackwell sm_120)

---

## Summary

Starting symptom: 12 Triton GPU pods running, `nvidia-smi` showing 0% GPU utilization, workers
apparently stuck. Through the session we peeled back four separate bottlenecks, all of which had
to be fixed before any throughput was observed. The last and most destructive was a 38-minute
blocking rglob that paralyzed all 1,440 prefetch threads fleet-wide.

**Before all fixes:** 43 archives in 33 min = 1.3/min total.  
**After all fixes:** 377 archives in ~8 min = ~47/min and climbing.

---

## Bug 1 — `_fill_claimed_queue` doing O(N²) EFS reads at startup

**File:** `orchestration/worker.py` + `orchestration/progress.py`

Each call to `_group_complete(sid, aid)` read 9 `manifest.json` files from EFS (one per task per
archive). With ~3,600 unclaimed archives per worker and 60 workers, startup triggered ~2 million
EFS metadata reads before any work began. Workers spent 25+ minutes in `_fill_claimed_queue`.

**Fix:** Added `load_completed_archive_set()` in `progress.py`. At startup, each worker reads
the per-worker JSONL completion logs once (`_meta/progress_completions/*.jsonl`) into an in-memory
`set[tuple[session_id, archive_id]]`. `_fill_claimed_queue` skips any archive in this set with
a single `in` check instead of reading 9 EFS files. 32,111 completed archives are now skipped
in microseconds. Workers reach entity ordering in under 2 minutes.

---

## Bug 2 — `count_inference_attempts_for` scanning all error files on every claim

**File:** `orchestration/errors.py`

`count_inference_attempts_for` used `rglob` over the entire
`_meta/inference_errors/<task_group>/` tree on every claimed archive to count retry attempts.
With 5,597 error files, each claim triggered a full directory walk.

**Fix:** Changed `append_inference_error` to write errors into a per-archive subdir:
`_meta/inference_errors/<task_group>/<session_id>/<archive_id>/<uuid>.json`.
`count_inference_attempts_for` now does a targeted `glob("*.json")` on the per-archive dir —
O(1–3) reads instead of O(5,597). A legacy `rglob` fallback handles pre-migration files.

A one-time migration script (`scripts/migrate_inference_errors_to_per_archive.py`) was run on the
pod to move 5,597 existing flat error files into the new layout.

---

## Bug 3 — TensorRT serialization version mismatch (v239 vs v240)

**File:** `manifests/triton-trt-deployment.yaml`

The Triton deployment had been bumped to `tritonserver:25.07-py3` (TRT 10.11.0.33, serialization
version 240) but `--model-repository` still pointed to `s3://riverside-build-assets/paralinguistics-trt`
which contained engines built with TRT 10.9 (version 239). All Triton pods crashed on startup with:

```
Version tag does not match. Current Version: 240, Serialized Engine Version: 239
```

**Fix:** Changed `--model-repository` to `s3://riverside-build-assets/paralinguistics-trt/GB202`
which contains the TRT 10.11.0.33 engines built for Blackwell sm_120. The hard constraint is now
documented in the manifest header: image and model repo are locked together — bumping one without
the other destroys the fleet.

---

## Bug 4 — Audio cache rglob blocking all 1,440 prefetch threads for 38 minutes

**File:** `orchestration/audio_cache.py`  
**Root cause of 0% GPU utilization.**

### What happened

`_current_cache_bytes()` called `_ready_object_bytes()` which ran `rglob("*.npy")` over the
536 GB EFS audio cache. On a full cache (~200k+ files), this scan took **30–38 minutes**.

The method held `_bytes_cache_lock` (an in-process threading lock) for the entire duration of
the rglob. With 24 prefetch threads per pod, all 24 queued behind this lock waiting for thread 1
to finish the scan. The TTL was 60 seconds, so the lock was re-acquired and the rglob re-run
constantly. With 60 pods, this was **1,440 threads perpetually blocked**.

`_try_reserve()` had the same bug independently: it called `_ready_object_bytes()` directly while
holding a **shared EFS capacity lock file**, serialising all 1,440 threads across all pods on a
single EFS file lock with a 38-minute hold time.

### What it looked like

Timing files showed archives with:
- `prefetch_submit_to_ready_sec: 2371s` (38 min queue wait)
- `download_sec: 33s`, `decode_sec: 23s` (actual work: 56s)
- `cache_fallback_reason: capacity` on every single archive (cache full, no hits)

The prefetch thread was blocked inside `get_decoded_audio` → `_current_cache_bytes()` for 38
minutes before the download even started. GPU was starved; Triton had nothing to process.

### Fix

Three changes to `SharedAudioCache.__init__` and two methods:

**1. Pessimistic initialization + immediate background scan**

```python
# Before (optimistic start, blocks on first rglob)
self._bytes_cached: int = 0
self._bytes_cache_ts: float = -1e9

# After (pessimistic start, background scan corrects downward if needed)
self._bytes_cached: int = int(max_cache_bytes)
self._bytes_cache_ts: float = time.monotonic()
self._bytes_scan_running: bool = True
threading.Thread(target=self._run_bytes_scan, daemon=True, ...).start()
```

Callers return `max_cache_bytes` instantly. The background scan runs the rglob once and updates
`_bytes_cached` to the actual value. Pre-warmed cache hits (`_load_ready_object`) are unaffected
— they bypass `_current_cache_bytes` entirely.

**2. "Cache full" short-circuit — never re-scan once full**

```python
def _current_cache_bytes(self) -> int:
    with self._bytes_cache_lock:
        if self._bytes_cached >= self.max_cache_bytes:
            return self._bytes_cached  # cache only grows; once full, stays full
        ...
```

**3. Non-blocking TTL refresh via background thread**

When the TTL expires (and the cache is not full), a daemon thread runs the next rglob instead of
blocking the caller. Callers always return the last known value immediately.

**4. `_try_reserve` fast-path before EFS lock**

```python
def _try_reserve(self, object_key, decoded_bytes):
    if self._current_cache_bytes() + decoded_bytes > self.max_cache_bytes:
        return False  # bail before acquiring the shared EFS capacity lock
    ...
```

When the cache is full (which it is here), `_try_reserve` returns `False` in microseconds without
ever touching the EFS lock file.

### Result

| Metric | Before | After |
|---|---|---|
| `prefetch_submit_to_ready_sec` avg | **2,300s** (38 min) | **51s** |
| Archives completed (first 8 min) | 43 in 33 min | 377 in ~8 min |
| Throughput | 1.3/min total | ~47/min (climbing) |
| Triton total exec count (4 min) | 2,024 entire run | 8,693 |
| Pods with >0 Triton execs | 6 of 12 | **12 of 12** |

---

## Fleet Configuration (working state)

```
Workers:       60 CPU pods (job-template-triton-workers-parallel.yaml)
               --prefetch-workers 24 --prefetch-lookahead 32
               --audio-cache-dir /efs/dvir/.../acoustic-understanding/_audio_cache
               --max-cache-bytes 536870912000 (500 GB)

GPU fleet:     12 × triton-trt pods on g7e.2xlarge (Blackwell RTX PRO 6000, sm_120)
               tritonserver:25.07-py3  (TRT 10.11.0.33)
               s3://riverside-build-assets/paralinguistics-trt/GB202/
               models: affect / disfluency / emotion  (max_batch=128)

Connection:    ClusterIP service triton-trt.nlp-audio-understanding:8001 (gRPC)
               connection-level load balancing across replicas
```

## Steady-state timings (post-fix, full cache / S3 fallback)

```
download_sec:                avg 15.5s   max 66.6s
prefetch_submit_to_ready:    avg 51.0s   max 79.0s
inference_sec (3 models):    avg 33.5s   max 229s
cache_hit rate:              0% (cache full, remaining archives not pre-warmed)
```

Note: the audio cache is 100% full and remaining archives were not pre-warmed. Every archive
takes the S3 download path. If the cache were warm, download would drop to ~2–5s (EFS read of
cached decoded f32).

---

## Lessons

1. **rglob over a large EFS directory is dangerous in hot paths.** Any cache size check, byte
   count, or directory scan over a hundreds-of-GB EFS tree must run in a background thread.
   Never hold a lock while doing it.

2. **Pessimistic initialization > optimistic when a cache is likely full.** Starting at 0
   causes every thread to attempt a cache write on the first miss, triggering `_try_reserve`
   and the EFS capacity lock chain. Starting at `max_cache_bytes` routes all misses to the
   cheap S3 fallback immediately.

3. **EFS lock files are global — one thread holding one for 38 minutes affects all 60 pods.**
   `_try_reserve` used the EFS capacity lock as a synchronization primitive around a slow rglob.
   Any cross-pod lock must be held for milliseconds, not minutes.

4. **TRT engines are architecture + version locked.** Image bump without model-repo bump =
   silent serialization version mismatch. The manifest header now documents this constraint
   explicitly. `25.07-py3` ↔ TRT 10.11.0.33 (serialization v240) ↔ `GB202/` S3 path.

5. **Completion tracking via JSONL logs is much cheaper than reading artifact manifests.**
   Reading 9 EFS files per archive × 3,600 archives × 60 workers at startup = 2M EFS reads.
   Reading 60 JSONL files once and building an in-memory set = 60 EFS reads.
