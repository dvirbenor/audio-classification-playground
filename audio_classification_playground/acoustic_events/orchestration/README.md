# Batch Acoustic Inference Orchestration

Run affect, disfluency, emotion, and VAD inference across ~600k audio
archives in parallel on multiple Kubernetes pods with shared EFS storage.

## Quick Start

```bash
# Single-pod run
python -m audio_classification_playground.acoustic_events.orchestration run \
    --parquet /efs/dvir/data/magic-clips-research/dataset-reference/all_archives.parquet \
    --output  /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --affect-backbone wavlm \
    --disfluency-backbone whisper \
    --batch-size 512 \
    --emotion-batch-size 64 \
    --prefetch-lookahead 4 \
    --prefetch-workers 4 \
    --vad-prefetch-workers 1

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
    audio_errors/
      <uuid>.json
    inference_errors/
      <uuid>.json
```

### Processing Flow

1. **Manifest loading**: Reads the parquet file and deduplicates by
   `(session_id, archive_id)`.
2. **Pre-filtering**: Loads permanent audio errors and inference attempt
   counts to skip known-bad archives.
3. **Model loading**: Affect, disfluency, and emotion predictors are loaded
   once. VAD is handled by background CPU workers by default.
4. **Shuffled iteration**: Entities are shuffled (for balanced load across
   pods) and iterated.
5. **Claimed prefetch**: config-aware completion check → atomic lock claim →
   authoritative retry count check → submit download/decode/VAD prefetch.
6. **Per-archive inference**: prefetch get → `run_all_inference` with
   precomputed VAD intervals when available → release lock.

### Coordination

- **Locks**: Atomic `O_CREAT | O_EXCL` files on EFS. Released on both
  success and failure. Stale locks reclaimed via the `reclaim-stale` CLI.
  In async VAD mode, each pod may hold up to `--prefetch-lookahead` locks
  while prefetch and inference overlap.
- **Error logs**: Individual JSON files per error event — no `flock`,
  no contention.
- **Progress**: Derived from the directory structure: if
  `manifest.json + predictions.npz` exist for all four tasks in an archive
  directory, that archive is complete.  The progress scanner walks only
  directories that exist on disk (via `os.scandir`) rather than probing
  every expected path, keeping cost proportional to work done.

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

### Batch Sizes

`--batch-size` remains the legacy default for affect and disfluency.
Emotion2vec uses `--emotion-batch-size 64` by default in the worker because
the fast batched path has much higher VRAM pressure than the old FunASR
generate path. Explicit task flags always win:

```bash
--batch-size 512 \
--affect-batch-size 384 \
--disfluency-batch-size 512 \
--emotion-batch-size 64
```

Tune these as rollout settings after profiling on the target GPU. Do not
increase emotion back to 512 without a target-GPU VRAM check.

### Config-Aware Completion

With the flat output layout (`session/archive/task/`), the config hash is
no longer embedded in the directory path. The worker reads each task's
`manifest.json` and validates it against the current model/input config.
Batch size is treated as runtime-only for this flat layout, so existing
complete artifacts are reused when only batch size changed. Stale artifacts
from a previous backbone, window, hop, sample rate, model, or transform are
automatically re-run.

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

This walks the output tree with `os.scandir` and the `_meta/` directories.
No manifest loading, no per-entity probing.  Finishes in seconds even on
EFS with hundreds of thousands of archives.

### Full progress (with `--parquet`)

When you need totals and remaining counts, pass the manifest:

```bash
python -m audio_classification_playground.acoustic_events.orchestration progress \
    --parquet /efs/.../all_archives.parquet \
    --output  /efs/.../models-inference
```

This uses the same walk-based scanner internally, so it is dramatically
faster than the original per-entity probe approach.

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
for f in d.glob('*.json'):
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
for f in sorted(d.glob('*.json')):
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
    - whisper
    - --batch-size
    - "512"
    - --emotion-batch-size
    - "64"
    - --prefetch-lookahead
    - "4"
    - --prefetch-workers
    - "4"
    - --vad-prefetch-workers
    - "1"
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
# Run on 100 archives to verify everything works
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

### Multi-Pod Parallelism

Simply launch N pods with the same arguments. Each pod:

- Shuffles with a different random seed (default: unseeded, i.e. random)
- Claims archives via atomic lock files
- Skips already-complete or locked archives
- Reports progress to the same shared EFS directory

### Async VAD Prefetch

`--vad-prefetch-workers` defaults to `1`, making CPU VAD the default
prefetch path. Use `--vad-prefetch-workers 0` only as an emergency fallback
to run VAD synchronously inside `run_all_inference`.

Recommended starting settings:

```bash
--prefetch-lookahead 4 --prefetch-workers 4 --vad-prefetch-workers 1
```

Increase `--vad-prefetch-workers` to `2` only if timing logs show the GPU is
waiting for VAD-ready prefetch results. Larger `--prefetch-lookahead` values
increase both locks held per pod and decoded-audio memory in flight; long
archives at 16 kHz float32 can make that memory noticeable.

### EFS Metadata Budget

The `_meta/` directory will contain:

- Up to ~600k lock files (small, ephemeral)
- Error JSON files (one per error event, typically <1 KB each)

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
