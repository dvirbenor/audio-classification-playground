# Batch Acoustic Inference Orchestration

Run affect, disfluency, emotion, and VAD inference across ~600k audio
archives in parallel on multiple Kubernetes pods with shared EFS storage.

## Quick Start

```bash
# Existing all-in-one worker mode (default)
python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --affect-backbone wavlm \
    --disfluency-backbone wavlm \
    --device cuda

# Task-fleet mode: run these as separate worker fleets.
# Do not mix task-fleet workers with all-in-one workers on the same output tree.
python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --task-group affect \
    --affect-backbone wavlm \
    --disfluency-backbone wavlm \
    --affect-batch-size 256 \
    --device cuda

python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --task-group disfluency \
    --affect-backbone wavlm \
    --disfluency-backbone wavlm \
    --disfluency-batch-size 384 \
    --device cuda

python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --task-group emotion \
    --affect-backbone wavlm \
    --disfluency-backbone wavlm \
    --emotion-batch-size 128 \
    --prefetch-workers 14 \
    --prefetch-lookahead 28 \
    --device cuda

python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --task-group vad \
    --affect-backbone wavlm \
    --disfluency-backbone wavlm \
    --prefetch-workers 12 \
    --prefetch-lookahead 24 \
    --vad-prefetch-workers 1 \
    --device cpu

# Optional shared decoded-audio cache: start one warmer, then launch workers
# with the same cache directory and cap.
python -m audio_classification_playground.acoustic_events.orchestration warm-cache \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --audio-cache-dir /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference/_meta/audio_cache \
    --max-cache-bytes 1099511627776

python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --task-group affect \
    --affect-backbone wavlm \
    --disfluency-backbone wavlm \
    --device cuda \
    --audio-cache-dir /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference/_meta/audio_cache \
    --max-cache-bytes 1099511627776

# Quick pulse check — no parquet needed, finishes in seconds
python -m audio_classification_playground.acoustic_events.orchestration progress \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --fast

# Full progress with totals (requires the manifest parquet)
python -m audio_classification_playground.acoustic_events.orchestration progress \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference

# Grouped error summary — counts + archives per error type
python -m audio_classification_playground.acoustic_events.orchestration errors \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --kind audio --group

# Flat error listing (one line per record)
python -m audio_classification_playground.acoustic_events.orchestration errors \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --kind audio --summary

# Inference errors (same flags: --group or --summary)
python -m audio_classification_playground.acoustic_events.orchestration errors \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --kind inference --group

# Timing distribution summary (split by VAD mode by default)
python -m audio_classification_playground.acoustic_events.orchestration timings \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference

# Timing CSV export for notebooks
python -m audio_classification_playground.acoustic_events.orchestration timings \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --csv --no-split-by-vad-mode > timings.csv

# Per-worker breakdown
python -m audio_classification_playground.acoustic_events.orchestration timings \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --by-worker

# Fleet heartbeat — per-worker lock/pace dashboard (no parquet needed)
python -m audio_classification_playground.acoustic_events.orchestration status \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference

# Reclaim stale locks from crashed pods
python -m audio_classification_playground.acoustic_events.orchestration reclaim-stale \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --older-than 60
```

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                      Worker Pod                        │
│                                                        │
│  ┌─────────────┐   ┌──────────────────────────────┐   │
│  │  Prefetcher  │   │        Main Loop              │   │
│  │  (threads)   │──▶│  claim → prefetch → infer    │   │
│  │  S3 download │   │  → write artifact → release  │   │
│  │  + decode    │   └──────────────────────────────┘   │
│  │  + CPU VAD   │                                      │
│  └─────────────┘                                       │
│                    ┌──────────────────────────────┐    │
│                    │  GPU-Resident Models          │    │
│                    │  affect / disfluency /        │    │
│                    │  emotion                      │    │
│                    └──────────────────────────────┘    │
└────────────────────────────────────────────────────────┘
          │                         │
          ▼                         ▼
    ┌───────────┐          ┌──────────────────┐
    │  S3       │          │  EFS             │
    │  (audio)  │          │  (artifacts,     │
    └───────────┘          │   locks, errors) │
                           └──────────────────┘
```

### Output Layout

```
<output_base>/
  <session_id>/
    <archive_id>/
      vad/
        manifest.json
        predictions.npz
      affect/
        manifest.json
        predictions.npz
      disfluency/
        manifest.json
        predictions.npz
      emotion/
        manifest.json
        predictions.npz
  _meta/
    locks/
      <session_id>__<archive_id>.lock
      affect/
        <session_id>__<archive_id>.lock
      disfluency/
        <session_id>__<archive_id>.lock
      emotion/
        <session_id>__<archive_id>.lock
      vad/
        <session_id>__<archive_id>.lock
    audio_errors/
      <session_id>__<archive_id>__<error_type>.json
    inference_errors/
      <uuid>.json
      affect/
        <uuid>.json
      disfluency/
        <uuid>.json
      emotion/
        <uuid>.json
      vad/
        <uuid>.json
      emotion-vad/
        <uuid>.json
    timings/
      <worker_id>.jsonl
    progress_complete.txt
```

### Processing Flow

1. **Manifest loading**: Reads the parquet file and deduplicates by
   `(session_id, archive_id)`.
2. **Pre-filtering**: Loads permanent audio errors and inference attempt
   counts to skip known-bad archives.
3. **Model loading**: In `--task-group all`, affect, disfluency, and emotion
   predictors are loaded once. In task-fleet mode, each worker loads only the
   resident model(s) needed by its task group.
4. **Session-grouped iteration**: By default, entities are sorted by
   `(date, session_id)` so all archives of a session are processed
   contiguously and older sessions complete first. Pass `--seed N` to
   restore the legacy shuffled ordering.
5. **Claimed prefetch**: completion check → atomic lock claim →
   authoritative retry count check → submit download/decode and, when the
   task group writes VAD, optional VAD prefetch.
6. **Per-archive inference**: prefetch get → filtered `run_all_inference`
   for missing tasks in the task group → release lock.

### Coordination

- **Locks**: Atomic `O_CREAT | O_EXCL` files on EFS. All-in-one workers use
  flat archive locks. Task fleets use per-task locks under
  `_meta/locks/<task>/`. Multi-task groups such as `emotion-vad` claim every
  missing task lock they may write, so they cannot race with split `emotion`
  or `vad` workers. Stale locks are reclaimed recursively via the
  `reclaim-stale` CLI.
- **Mixed-mode safety**: Workers refuse to start when all-in-one locks and
  task-fleet locks coexist on the same output tree. Use separate output trees
  for control/candidate comparisons.
- **Error logs**: Audio errors are deduplicated by archive/error type because
  multiple task fleets can encounter the same bad source audio. Inference
  errors remain per-attempt and are task-scoped in fleet mode.
- **Progress**: Derived from task artifact existence. A task is complete when
  `manifest.json` and `predictions.npz` exist and the manifest status is
  `complete`. The `--parquet` scanner checks only entities listed in the
  manifest using parallel I/O (64 threads), and caches fully-complete
  archives in `_meta/progress_complete.txt` so subsequent calls skip them.

### Error Handling

| Error type | Category | Behaviour |
|---|---|---|
| `no_matching_file` | Audio, permanent | Skipped forever |
| `glacier_storage_class` | Audio, transient | Retried on next encounter (see [Glacier Restore](#glacier-restore)) |
| `download_failed` | Audio, transient | Retried on next encounter |
| Inference exception | Inference | Retried up to `--max-retries` (default 3) |
| Deterministic error (e.g. corrupt file) | Inference | Marked permanent on first hit |
| `torch.cuda.OutOfMemoryError` | Inference | GPU cleaned, retried |
| `ShutdownRequested` (SIGTERM) | Graceful | Lock released, no error logged |

### SIGTERM / Graceful Shutdown

The worker installs a SIGTERM handler immediately on startup. When SIGTERM
arrives:

1. A `threading.Event` is set.
2. Between tasks, `shutdown_check()` raises `ShutdownRequested`.
3. The worker releases the current lock and exits cleanly.

**Important**: Set `terminationGracePeriodSeconds` in your K8s pod spec to
exceed the maximum expected per-archive inference time (typically 30–120s).
Stale-lock reclaim remains necessary for hard kills.

Async VAD prefetch means hard-killed pods can leave up to
`--prefetch-lookahead` stale locks, not just the currently inferred archive.
Keep `--prefetch-lookahead` conservative unless stale reclaim runs frequently.

### Task Groups and Completion Policy

`--task-group all` is the default and preserves the original all-in-one
worker behavior. Task-fleet mode lets you run one resident model per GPU
worker fleet, and move VAD extraction onto CPU-only workers when desired:

| Task group | Tasks written | Models loaded | Lock path | Prefetch defaults |
|---|---|---|---|---|
| `all` | `vad`, `affect`, `disfluency`, `emotion` | affect, disfluency, emotion | `_meta/locks/*.lock` | `4/4/1` |
| `affect` | `affect` | affect | `_meta/locks/affect/*.lock` | `lookahead=8, workers=8, vad=0` |
| `disfluency` | `disfluency` | disfluency | `_meta/locks/disfluency/*.lock` | `lookahead=8, workers=8, vad=0` |
| `emotion` | `emotion` | emotion | `_meta/locks/emotion/*.lock` | `lookahead=28, workers=14, vad=0` |
| `vad` | `vad` | none; CPU VAD is owned by prefetch | `_meta/locks/vad/*.lock` | `lookahead=24, workers=12, vad=1` |
| `emotion-vad` | `vad`, `emotion` | emotion + CPU VAD | `_meta/locks/vad/*.lock` and `_meta/locks/emotion/*.lock` for missing tasks | `lookahead=12, workers=8, vad=1` |

The default completion policy is `--completion-policy exists`. This protects
already-generated production work: if a task artifact is complete on disk, it
is skipped regardless of config hash, runtime preset, batch size, or task
group. Config hashes are still written for audit/debug.

Use strict config-aware reuse only for controlled experiments:

```bash
--completion-policy config
```

Use `--force-recompute` only when you intentionally want to ignore existing
task artifacts for the selected task group. In task-fleet mode, partial
archives are natural: for example, `emotion-vad` claims only the missing
`vad`/`emotion` task locks and reuses whichever artifact is already complete.

Do not run `--task-group all` workers and task-fleet workers on the same
output tree at the same time. The worker has a startup guard for active mixed
locks, but operationally the clean pattern is one output tree per mode.

### Batch Sizes

`--batch-size` remains the legacy default for affect and disfluency.
Emotion2vec uses `--emotion-batch-size 64` by default in the worker. Explicit
task flags always win:

```bash
--batch-size 512 \
--affect-batch-size 384 \
--disfluency-batch-size 512 \
--emotion-batch-size 64
```

Tune these as rollout settings after profiling on the target GPU. Do not
increase emotion back to 512 without a target-GPU VRAM check.

If you want to preserve a previous production run's explicit settings, keep
the task-relevant flags on each fleet command. For example:

```bash
# affect fleet
--affect-batch-size 256

# disfluency fleet
--disfluency-batch-size 384

# emotion fleet
--emotion-batch-size 128
```

If the WavLM runtime auto-selects `compiled_static`, WavLM task batches are
forced to the compiled static batch size. Use `--wavlm-runtime-preset
fast_exact` for experiments that intentionally keep non-static batch sizes.

The persistent worker uses an audio-fed emotion2vec path: decoded audio is
copied to the model device once, and the same overlapping 3s / 0.25s windows
are formed there before the normal `extract_features -> mean -> proj -> softmax`
classification path.

### Emotion2vec Runtime

`--emotion-runtime-mode auto` is the default. On CUDA it resolves to the
optimized preset:

- resident direct scorer, loaded once per worker
- fixed `[64, 48000]` batches, including padded tail/short-audio batches
- `torch.compile(mode="default")`, warmed once with a zero batch
- TF32 enabled only while constructing/warming/running emotion2vec, then
  restored so affect/disfluency do not inherit it
- no BF16/FP16 autocast by default

On non-CUDA devices, `auto` resolves to `fp32-eager`. Use
`--emotion-runtime-mode fp32-eager` to reproduce old FP32 artifacts, or
`--emotion-runtime-mode custom` with `--emotion-compile`,
`--emotion-compile-mode`, `--emotion-autocast-dtype`, or `--allow-tf32` for
experiments. Presets intentionally reject those granular knobs so the recorded
`inference_config` cannot disagree with the runtime.

The optimized preset is a new emotion config identity (`torch_compile=True`,
`torch_compile_mode=default`, `torch_allow_tf32=True`), so existing FP32-eager
emotion artifacts are intentionally reprocessed when the worker default is
deployed. Flat-layout completion still ignores batch-size-only differences.

To verify equivalence and speed on a target GPU:

```bash
uv run python scripts/compare_emotion2vec_feed_path.py \
    --device cuda --batch-size 64 --max-windows-per-file 160 /path/to/audio.wav
```

The script compares the old framed-window feed to the audio-fed feed and
prints label equality, absolute-difference stats, top-1 agreement, and speedup.
It can also test optional runtime knobs against the FP32 eager audio-fed path:

```bash
uv run python scripts/compare_emotion2vec_feed_path.py \
    --device cuda --batch-size 64 \
    --candidate-compile \
    --candidate-allow-tf32 \
    /path/to/audio.wav
```

The reference optimized operating point measured about `2.2x-2.4x` faster than
FP32 eager on the 6039-window reference archive, with event-core parity. Typical
probability drift was around `mean_abs_diff ~7e-05` and `p99_abs_diff ~0.001`;
top-1 flips, when present, were marginal frames with tiny reference margins.
Rollout should gate on same-job relative speedup and event parity, and treat
probability drift thresholds as review alerts rather than hard failures.

```bash
uv run python -m audio_classification_playground.acoustic_events.inference run emotion \
    --audio /path/to/audio.wav \
    --out /tmp/e2v-optimized \
    --device cuda \
    --emotion-runtime-mode optimized
```

Avoid `--emotion-compile-mode reduce-overhead` for this FunASR model:
it uses CUDA Graph replay and can fail with overwritten internal tensors across
repeated batch calls. Use BF16/FP16 only after separate event-level validation.

To explore ONNX Runtime / TensorRT without changing production inference, use
the sidecar benchmark harness. In the current Python 3.10 GPU image, pin
ONNX Runtime GPU below 1.24 because newer wheels are Python 3.11+ only:

```bash
uv pip install onnx 'onnxruntime-gpu==1.23.2'
```

Then run:

```bash
uv run python scripts/benchmark_emotion2vec_onnx_tensorrt.py \
    --device cuda \
    --batch-size 64 \
    --max-windows-per-file 10000 \
    --opset 18 \
    --provider cuda \
    --provider tensorrt \
    --trt-cache-dir /workspace/tmp_data/e2v-trt-cache \
    /path/to/audio.wav
```

The harness exports the exact direct scorer as a static
`[batch_size, 48000]` ONNX graph, benchmarks each provider, and prints the same
probability drift / top-1 diagnostics against the PyTorch direct scorer. Keep
using the normal artifact comparison scripts before promoting any exported
runtime into production.

### Completion Semantics

The production orchestrator defaults to artifact-existence completion:

```
manifest.json exists
+ predictions.npz exists
+ manifest.status == "complete"
= task complete
```

This is intentionally broader than config-aware reuse so an orchestration
refactor, batch-size change, runtime preset change, or task-fleet split does
not sacrifice already-produced raw predictions. Config hashes remain in
manifests for audit/debug and can be enforced with `--completion-policy
config` when running a controlled comparison.

### Data Provenance

Each `manifest.json` records:

- `audio.path`: the canonical S3 URI (e.g. `s3://riverside-pro-main/...`)
- `audio.source_key`: the bare S3 key
- `audio.sha256`: SHA-256 of the decoded mono 16 kHz float32 samples
- `inference_config`: full config dict
- `inference_config_hash`: 16-char hex digest of the config

## Monitoring Progress and Errors

### Quick pulse check (`--fast`)

For a fast, parquet-free overview of what's on disk:

```bash
python -m audio_classification_playground.acoustic_events.orchestration progress \
    --output /efs/.../models-inference --fast
```

Output:

```
Complete (all 4 tasks):  14
Partially done:           1
Lock files:               1

Per-task artifact count:
  affect       14
  disfluency   14
  emotion      14
  vad          15

Audio errors:      17 records, 15 permanent archives
Inference errors:  93 records across 14 archives

(--fast mode: totals/remaining unavailable without --parquet)
```

This walks the output tree using `find(1)` for speed on network filesystems
(falls back to `os.scandir` if `find` is unavailable).  No manifest loading,
no per-entity probing.

### Full progress (with `--parquet`)

When you need totals and remaining counts, pass the manifest:

```bash
python -m audio_classification_playground.acoustic_events.orchestration progress \
    --parquet /efs/.../all_archives.parquet \
    --output  /efs/.../models-inference
```

This checks only the entities from the parquet using 64 parallel threads
against EFS.  Fully-complete archives are cached in
`_meta/progress_complete.txt` and skipped on subsequent runs, so repeated
calls get progressively faster as work completes.  Use `--no-cache` to
force a full re-scan if needed.

### Grouped error summary (`--group`)

```bash
python -m audio_classification_playground.acoustic_events.orchestration errors \
    --output /efs/.../models-inference --kind audio --group
```

Output:

```
Audio errors by type:

  no_matching_file             12 records,   10 archives (permanent)
    e.g. abc123/def456 — No audio file found in ...
  glacier_storage_class          3 records,    3 archives (transient)
    e.g. xyz789/ghi012 — Object is in GLACIER...
  download_failed                2 records,    2 archives (transient)
    e.g. ...

Total: 17 records, 15 unique archives
```

Groups are sorted by record count descending, then error type ascending.
Use `--kind inference` for inference errors (no permanent/transient label,
since retries are the meaningful signal there).

The original flat listing (`--summary` without `--group`) remains available.

### Timing distributions (`timings`)

Each worker appends a JSONL line per completed archive to
`_meta/timings/<worker_id>.jsonl`.  The `timings` subcommand reads all
worker files and computes distributional statistics (percentiles, mean,
std) for each timing field.

```bash
python -m audio_classification_playground.acoustic_events.orchestration timings \
    --output /efs/.../models-inference
```

By default, records are split into three populations by VAD execution
mode (derived from stored booleans — no enum stored in the record):

| VAD mode | Condition | `vad_sec` meaning |
|---|---|---|
| `prefetched` | `precomputed_vad=true` | Artifact write using pre-fetched intervals |
| `cached` | `precomputed_vad=false, vad_reused=true` | Cache load of existing artifact |
| `inline` | `precomputed_vad=false, vad_reused=false` | Full inline Silero VAD run |

Disable splitting with `--no-split-by-vad-mode` for a single combined view.

**Flags:**

| Flag | Description |
|---|---|
| `--csv` | Output raw records as CSV (includes derived `vad_mode` column) |
| `--fields F1,F2,...` | Only display specific timing fields |
| `--min-audio-sec N` | Filter to archives with `audio_duration_sec >= N` |
| `--max-audio-sec N` | Filter to archives with `audio_duration_sec <= N` |
| `--by-worker` | Per-worker breakdown followed by aggregate |
| `--worker SUBSTR` | Filter to a single worker (substring match) |
| `--no-split-by-vad-mode` | Disable VAD mode splitting |

**Record schema** (one JSON object per JSONL line):

```json
{
    "worker_id": "pod-abc12_3f8a",
    "task_group": "affect",
    "tasks_run": ["affect"],
    "session_id": "abc",
    "archive_id": "def",
    "ts": "2026-05-15T01:30:00Z",
    "s3_key": "accounts/studio/takes/audio.wav",
    "audio_source_extension": ".wav",
    "audio_object_size_bytes": 123456789,
    "audio_storage_class": "STANDARD",
    "audio_duration_sec": 183.4,
    "prefetch_scheduler_wait_sec": 0.10,
    "prefetch_get_wait_sec": 0.02,
    "prefetch_wait_sec": 0.12,
    "decode_queue_wait_sec": 0.01,
    "download_decode_sec": 1.4,
    "vad_queue_wait_sec": 0.05,
    "vad_precompute_sec": 0.8,
    "prefetch_submit_to_ready_sec": 2.31,
    "prefetch_ready_age_sec": 0.42,
    "precomputed_vad": true,
    "vad_reused": false,
    "affect_reused": false,
    "disfluency_reused": false,
    "emotion_reused": false,
    "vad_sec": 0.02,
    "affect_sec": 1.73,
    "disfluency_sec": 2.01,
    "emotion_sec": 0.94,
    "inference_sec": 4.70,
    "total_sec": 6.22
}
```

`task_group` identifies the worker mode. `tasks_run` lists the missing tasks
that were actually executed for that archive; in a partial `emotion-vad`
retry it may contain only `["emotion"]` or only `["vad"]`.
`prefetch_scheduler_wait_sec` is the canonical GPU-starvation signal: it
measures foreground time spent waiting for any queued prefetch to become
ready. `prefetch_get_wait_sec` should be near-zero under the default
ready-first scheduler; in `ACP_PREFETCH_SCHEDULER=fifo` rollback mode it
carries the old head-of-queue blocking wait. `prefetch_wait_sec` is kept as
a compatibility aggregate (`scheduler_wait + get_wait`). `decode_queue_wait_sec`
and `vad_queue_wait_sec` expose executor queueing, while
`prefetch_ready_age_sec` highlights over-prefetching or too-deep lookahead.
`inference_sec` is wall-clock time around the filtered `run_all_inference`
call. Per-task `*_sec` fields come from `InferenceRunResult.task_elapsed_sec`.
`*_reused` booleans come from `InferenceRunResult.reused`.

**Multi-pod design:** each pod writes its own JSONL file (append-only,
no locks). The CLI globs all files at analysis time. `worker_id` is
embedded in every record, so analysis is correct even if files are
concatenated or moved.

### Fleet heartbeat (`status`)

A compact, at-a-glance dashboard showing which pods are alive, how many
archives each has completed, and the current processing pace.  No parquet
or manifest required — reads lock and timing metadata under `_meta/` on EFS,
so it completes quickly even with hundreds of thousands of archives.

```bash
python -m audio_classification_playground.acoustic_events.orchestration status \
    --output /efs/.../models-inference
```

Sample output:

```
Fleet heartbeat                              2026-05-17 09:30:00 UTC
================================================================

Worker          Group       Locks  Done  Last activity  Pace (arc/h)
--------------------------------------------------------------------
pod-gpu-abc123  affect          8   512  12s ago              ~68.2
pod-gpu-def456  disfluency      8   300  3s ago               ~74.4
pod-gpu-ghi789  emotion-vad    12   301  47s ago              ~92.0
--------------------------------------------------------------------
Fleet (3 workers)             28  1,113                      ~234.6

Errors: 7 audio, 2 inference
```

Add `--summary` for completed/partial counts (triggers a tree walk, slower):

```bash
python -m audio_classification_playground.acoustic_events.orchestration status \
    --output /efs/.../models-inference --summary
```

```
...same table...

Completed: 14,203  |  Partial: 42  |  Errors: 7 audio, 2 inference
```

| Column | Source | What it tells you |
|---|---|---|
| Group | lock metadata or timing records | Which task fleet the pod belongs to |
| Locks | `_meta/locks/**/*.lock` ownership | Is the pod alive? (0 = dead or finished) |
| Done | `_meta/timings/<worker_id>.jsonl` line count | Is it making progress? |
| Last activity | Latest lock claim time or timing record timestamp | Is it stuck? (>5 min = suspicious) |
| Pace | Mean `total_sec` of last N records, extrapolated to archives/hour | How fast is it going? |

**Flags:**

| Flag | Description |
|---|---|
| `--tail N` | Number of recent timing records per worker for pace calculation (default: 20) |
| `--summary` | Include completed/partial counts from disk (walks the output tree) |

For continuous monitoring, wrap with `watch`:

```bash
watch -n 15 python -m audio_classification_playground.acoustic_events.orchestration status \
    --output /efs/.../models-inference
```

Or create a shell alias for quick access:

```bash
alias hb='python -m audio_classification_playground.acoustic_events.orchestration status --output /efs/.../models-inference'
```

## Glacier Restore

Archives stored in `GLACIER` or `DEEP_ARCHIVE` produce transient
`glacier_storage_class` audio errors.  Each error JSON includes a
structured `s3_key` field with the resolved S3 key, making it directly
consumable by restore tooling.

Workers retry Glacier archives on every pass (no retry budget for audio
errors).  Once the S3 object is restored, the next attempt succeeds
automatically.

### Extracting S3 keys for a restore request

```bash
uv run python -c "
import json, pathlib
d = pathlib.Path('/efs/.../models-inference/_meta/audio_errors')
keys = set()
for f in d.rglob('*.json'):
    try:
        data = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if data.get('error_type') == 'glacier_storage_class':
        k = data.get('s3_key', '')
        if k:
            keys.add(k)
for k in sorted(keys):
    print(k)
" > glacier_keys.txt
```

Inference errors may be nested by task group in task-fleet mode. The
`errors`, `status`, and summary helpers scan recursively, so operators do not
need extra flags to see fleet-mode failures.

### Rollout with Glacier reclassification

If upgrading from a version where `glacier_storage_class` was permanent,
old error JSONs with `"is_permanent": true` will suppress retries until
removed.  Workers cache `permanent_errors` in memory at startup, so the
cleanup sequence matters:

1. **Extract keys** from existing error files (see above).
2. **Submit Glacier restore** request.
3. **Stop all workers** — ensure no old-code pods remain.
4. **Deploy new code** — workers remain stopped.
5. **Delete old permanent Glacier error files:**

```bash
uv run python -c "
import json, pathlib
d = pathlib.Path('/efs/.../models-inference/_meta/audio_errors')
removed = 0
for f in sorted(d.rglob('*.json')):
    try:
        data = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if data.get('error_type') == 'glacier_storage_class' and data.get('is_permanent'):
        f.unlink()
        removed += 1
print(f'Removed {removed} old permanent glacier error files')
"
```

6. **Start fresh new-code workers.**

## Deployment

### Kubernetes Pod Spec (minimal)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: acoustic-worker
spec:
  terminationGracePeriodSeconds: 120
  containers:
  - name: worker
    image: <your-image>
    command:
    - python
    - -m
    - audio_classification_playground.acoustic_events.orchestration
    - run
    - --parquet
    - /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet
    - --output
    - /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference
    - --affect-backbone
    - wavlm
    - --disfluency-backbone
    - wavlm
    - --task-group
    - affect
    - --affect-batch-size
    - "256"
    resources:
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
    - name: efs
      mountPath: /efs
  volumes:
  - name: efs
    persistentVolumeClaim:
      claimName: efs-pvc
```

### Recommended Pilot

Before the full 600k run:

```bash
# All-in-one smoke on 100 archives
python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /path/to/pilot_100.parquet \
    --output  /efs/pilot-test \
    --affect-backbone wavlm \
    --disfluency-backbone whisper \
    --batch-size 512 \
    --emotion-batch-size 64

# Verify
python -m audio_classification_playground.acoustic_events.orchestration progress \
    --parquet /path/to/pilot_100.parquet \
    --output  /efs/pilot-test
```

For task-fleet smoke, use a separate output tree and launch the four task
groups (`affect`, `disfluency`, `emotion`, `vad`) with the same pilot parquet.
Then verify progress, status, errors, timings, and stale-lock reclaim against
that output tree before reusing the production output.

### Multi-Pod Parallelism

For all-in-one mode, simply launch N pods with the same arguments. Each pod:

- Sorts archives by (date, session_id) for session-contiguous processing
- Claims archives via atomic lock files
- Skips already-complete or locked archives
- Reports progress to the same shared EFS directory

For task-fleet mode, launch separate deployments/jobs with different
`--task-group` values. Task-scoped locks allow affect, disfluency, and
emotion/VAD fleets to work on the same archive concurrently without duplicate
writes within a task. The legacy combined `emotion-vad` group now uses the
same per-task `emotion` and `vad` locks, so it is safe with split fleets after
old `emotion-vad` pods have stopped. Do not run all-in-one and task-fleet
workers on the same output tree at the same time.

### Async VAD Prefetch

In all-in-one mode, `--vad-prefetch-workers` defaults to `1`, making CPU VAD
the default prefetch path. In affect/disfluency/emotion task fleets, the
default is `0` because those workers do not write or consume VAD. In `vad`
and `emotion-vad`, the default is `1`.

All-in-one starting settings:

```bash
--prefetch-lookahead 4 --prefetch-workers 4 --vad-prefetch-workers 1
```

Task-fleet defaults are already deeper because each single-task worker
consumes prefetched audio faster:

```text
affect:       --prefetch-lookahead 8  --prefetch-workers 8  --vad-prefetch-workers 0
disfluency:   --prefetch-lookahead 8  --prefetch-workers 8  --vad-prefetch-workers 0
emotion:      --prefetch-lookahead 28 --prefetch-workers 14 --vad-prefetch-workers 0
vad:          --prefetch-lookahead 24 --prefetch-workers 12 --vad-prefetch-workers 1
emotion-vad:  --prefetch-lookahead 12 --prefetch-workers 8  --vad-prefetch-workers 1
```

Workers use ready-first scheduling by default: if the FIFO head is still
downloading/decoding but a later queued archive is ready, the worker consumes
the earliest ready queued archive. Set `ACP_PREFETCH_SCHEDULER=fifo` before
starting a pod to restore legacy head-of-queue behavior for rollback.

Tune by watching `prefetch_scheduler_wait_sec` in timing records. Larger
`--prefetch-lookahead` values increase both locks held per pod and decoded
audio memory in flight; long archives at 16 kHz float32 can make that memory
noticeable.

### EFS Metadata Budget

The `_meta/` directory will contain:

- Up to ~600k lock files (small, ephemeral)
- Audio error JSON files deduped by archive/error type
- Inference error JSON files per failed attempt, task-scoped in fleet mode
- Timing JSONL files (one per worker process, ~200 bytes per record)

EFS handles this volume well, but monitor the metadata cost if running
hundreds of thousands of archives.

## Downstream Consumption

Once inference is complete, the flat artifact layout is directly consumable
by the existing producer and composition pipelines:

```python
from audio_classification_playground.acoustic_events.inference import (
    load_prediction_artifact,
)

artifact = load_prediction_artifact(
    "/efs/.../models-inference/<session_id>/<archive_id>/affect/"
)
```
