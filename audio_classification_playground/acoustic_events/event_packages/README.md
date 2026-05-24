# Atomic Event Packages

This module builds decisive per-archive event collections from completed
inference artifacts. It is intentionally separate from review packages:
there is no human-labeling state, no `labels.json`, and no track arrays. The
output is meant to be compact input for downstream transcript decoration and
annotation.

## Output Layout

Each archive gets one package directory:

```text
<events-output>/<session_id>/<archive_id>/
  events.jsonl
  package.json
```

`events.jsonl` contains only event-local rows. Archive-level identity and
provenance (`session_id`, `archive_id`, `date`, source artifact fingerprints,
audio metadata, producer configs, and counts) live in `package.json`.

Version 1 emits atomic events only:

- affect leaf deviation events, labeled as `axis` + `direction`, for example
  `arousal+`
- disfluency events, including multi-label `labels` when multiple disfluency
  types are active
- emotion events

VAD is used as producer context where needed, but VAD does not produce event
rows.

## Typical Order

1. Run the inference orchestration until archives begin producing completed
   `vad`, `affect`, `disfluency`, and `emotion` artifacts.
2. Start one or more CPU event-package workers against the same source archive
   manifest used by inference orchestration.
3. Inspect progress from the completion index and completion shards.
4. Compact the completion index after large batches or at the end of a run.
5. Use reconcile and stale-lock cleanup only as operational maintenance tools.

## Commands

Build one archive package, useful for debugging:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages eventify \
  --inference-archive /efs/inference/<session_id>/<archive_id> \
  --events-archive /efs/event-packages/<session_id>/<archive_id> \
  --date 2026-05-24
```

Run a one-pass worker:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages run \
  --parquet /efs/manifests/source-archives.parquet \
  --inference-output /efs/inference \
  --events-output /efs/event-packages
```

By default, `run` prints one JSON summary at the end of each pass. Add
`--verbose` to log pass starts, pass completions, and periodic in-pass counters
to stderr while keeping the JSON summary on stdout:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages run \
  --parquet /efs/manifests/source-archives.parquet \
  --inference-output /efs/inference \
  --events-output /efs/event-packages \
  --verbose \
  --log-every 1000
```

`--log-every` counts in-shard archives, not total manifest rows. Use
`--log-every 1` only for small debugging runs; at production scale it will emit
one progress line per in-shard archive.

Run continuously while inference artifacts are still arriving:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages run \
  --parquet /efs/manifests/source-archives.parquet \
  --inference-output /efs/inference \
  --events-output /efs/event-packages \
  --watch \
  --poll-interval-sec 300
```

`--watch` means the worker does not exit after one manifest scan. It runs a
pass, prints a summary, sleeps for `--poll-interval-sec`, and scans again. This
is useful while inference workers are still completing artifacts. Archives that
are not ready in one pass are skipped quickly and revisited on the next pass.

Use multiple workers by assigning each process a different shard:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages run \
  --parquet /efs/manifests/source-archives.parquet \
  --inference-output /efs/inference \
  --events-output /efs/event-packages \
  --watch \
  --num-shards 16 \
  --shard-index 0
```

Launch the same command for shard indexes `0` through `15`. A single worker is
sequential inside one Python process. Parallelism comes from running multiple
processes or pods with non-overlapping shard indexes.

`--num-shards` is the total number of deterministic work partitions. It does
not spawn threads or processes by itself. `--shard-index` is the zero-based
partition handled by this worker. Each archive is assigned by hashing
`(session_id, archive_id)` and taking the result modulo `num_shards`; the worker
processes only archives whose assigned shard equals its `shard-index`.

For example, with `--num-shards 16`, valid shard indexes are `0` through `15`.
All workers in that fleet must use the same `num-shards` value, and each active
worker should use a different `shard-index`.

Report progress:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages progress \
  --parquet /efs/manifests/source-archives.parquet \
  --events-output /efs/event-packages
```

Compact completion shards into `_index/completed.parquet`:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages compact-index \
  --events-output /efs/event-packages
```

Check or repair index rows from package truth:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages reconcile-index \
  --parquet /efs/manifests/source-archives.parquet \
  --events-output /efs/event-packages \
  --dry-run
```

`reconcile-index` treats completed package directories as the source of truth
and compares them with the completion index plus any uncompacted completion
shards.

With `--dry-run`, it reports:

- `missing_index_rows`: package exists and is complete, but no completion row
  exists
- `indexed_missing_packages`: the index says the archive completed, but the
  package directory is missing or incomplete
- `fingerprint_mismatches`: the index row exists, but its fingerprints no
  longer match the current package manifest
- `foreign_index_rows`: indexed rows that do not correspond to an archive in
  the provided source manifest

Without `--dry-run`, it appends repair rows for `missing_index_rows` and
`fingerprint_mismatches`. It does not rewrite event packages and does not delete
stale or foreign index rows; those remain visible for operator review.

## Rerun Semantics

Default runs are idempotent. If a package is already complete for the current
event configuration, the worker skips it. When the completion index already
contains that archive, the worker does not read or rewrite the package.

If a package exists but its completion row is missing, the worker can heal the
index by appending a completion row. `compact-index` deduplicates rows by
`(session_id, archive_id)`.

`--validate-inputs` makes the skip check stricter by comparing the current
source artifact provenance to the package input fingerprint. This costs more
EFS reads and is best used for audits or suspected source changes.

`--force` recomputes packages and replaces `events.jsonl` and `package.json`
with same-directory atomic file replacements.

Archives whose four required artifacts are not complete are marked `not_ready`
for that pass and the worker moves on. In `--watch` mode they are revisited on
later polling cycles.

## Operational Notes

Workers use event-package locks under `<events-output>/_meta/locks/`, separate
from inference locks. Stale locks are reclaimed during worker passes and can
also be cleaned explicitly:

```bash
uv run python -m audio_classification_playground.acoustic_events.event_packages reclaim-stale-locks \
  --events-output /efs/event-packages \
  --older-than 60
```

Packaging failures are recorded under
`<events-output>/_meta/packaging_errors/`. After the configured max attempts,
an archive is reported as `failed_exhausted` until it is retried with
`--retry-failed` or the error records are handled operationally.
