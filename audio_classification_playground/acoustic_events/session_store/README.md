# Session Event Store

Aggregates per-archive event packages into date-partitioned, session-level
parquet files for downstream pipeline consumption.

## What it does

The inference pipeline produces events per **archive** (one speaker track).
This module collects them at the **session** level — writing a row only when
all archives of a session have completed — and stores them in compact parquet
files partitioned by date.

Each row contains thinned events (only semantically meaningful fields) for
the complete session, ready for transcript decoration or analytics.

## Prerequisites

Before running `populate`, ensure:

1. **Inference** has produced prediction artifacts for the target archives.
2. **Event packages** have been built from those artifacts (the `event_packages run` worker).
3. **Completion shards** exist — these are written automatically by the event package worker.

If packages were regenerated manually (outside the normal worker flow), run
`reconcile-index` first to sync completion rows with the actual package state.

## Usage

### Populate the store

```bash
uv run python -m audio_classification_playground.acoustic_events.session_store populate \
  --manifest /efs/manifests/source-archives.parquet \
  --events-output /efs/event-packages \
  --store-output /efs/session-events/
```

This is idempotent — safe to run repeatedly. Only new or changed sessions
trigger writes.

### Options

| Flag | Purpose |
|------|---------|
| `--dates 2025-02-19 2025-02-20` | Process only these date partitions (efficiency knob). Omit for full consistency across all dates. |
| `--force` | Re-read and rewrite all sessions, ignoring fingerprint comparisons. Use after bulk reprocessing. |
| `--verbose` | Per-session DEBUG logging to stderr. Useful for diagnosing why a session was skipped. |

### Typical workflows

**Incremental refresh** (run periodically as new archives complete):

```bash
uv run python -m audio_classification_playground.acoustic_events.session_store populate \
  --manifest /efs/manifests/source-archives.parquet \
  --events-output /efs/event-packages \
  --store-output /efs/session-events/
```

**Targeted date range** (you know which dates have new data):

```bash
uv run python -m audio_classification_playground.acoustic_events.session_store populate \
  --manifest /efs/manifests/source-archives.parquet \
  --events-output /efs/event-packages \
  --store-output /efs/session-events/ \
  --dates 2025-05-20 2025-05-21 2025-05-22
```

**Full rebuild** (after config changes or bulk re-eventification):

```bash
uv run python -m audio_classification_playground.acoustic_events.session_store populate \
  --manifest /efs/manifests/source-archives.parquet \
  --events-output /efs/event-packages \
  --store-output /efs/session-events/ \
  --force
```

## Output format

```
session-events/
  2025-02-19.parquet
  2025-02-20.parquet
  ...
```

Each parquet file has columns:

| Column | Type | Content |
|--------|------|---------|
| `session_id` | string | Session identifier |
| `date` | string | Partition date |
| `archive_count` | int | Archives in this session |
| `event_count` | int | Total thinned events |
| `session_fingerprint` | string | For staleness detection |
| `data` | string | JSON with full session events |

### Reading the data

```python
import json
import pyarrow.parquet as pq

table = pq.read_table("session-events/2025-02-19.parquet")
for i in range(table.num_rows):
    record = json.loads(table.column("data")[i].as_py())
    for archive in record["event_items"]:
        for event in archive["events_data"]:
            print(event["start_sec"], event["task"], event["label"])
```

Or with typed access:

```python
from audio_classification_playground.acoustic_events.session_store import SessionEventsRecord

record = SessionEventsRecord.model_validate(json.loads(data_str))
```

## How it decides what to write

1. Loads the manifest to know which archives belong to each session.
2. Reads the completion index to know which archives are done.
3. A session is **ready** only when all its manifest archives appear as complete.
4. Compares fingerprints against existing parquet rows — only new or changed sessions trigger EFS reads and a partition rewrite.

Sessions that are incomplete, or whose packages fail to load, are skipped
and reported in the summary output.
