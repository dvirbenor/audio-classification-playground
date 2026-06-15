# Glacier restore scripts

Utilities for bringing source audio back from S3 Glacier / Deep Archive so the
inference fleet can read it again. Source audio lives in `s3://riverside-pro-main`
(bucket-relative keys); archived objects make `audio_resolver` raise
`glacier_storage_class` until a readable copy is restored.

**Workflow:** restore → wait → check status → re-run the inference job.

A `glacier_storage_class` error is *transient* and does **not** burn the inference
retry budget, so once restores complete you just re-launch the same job over the
same manifest — completed archives are skipped, restored ones flow through. No
code or manifest change needed. Mind the restore window (`--days`): process the
archives before the readable copies expire.

Both scripts read a CSV with an `s3_key` column (the layout of
`wav_glacier_keys.csv` / `wav_full_keys.csv`). If the CSV also has an
`is_glacier` column, only rows where it is truthy are used. Keys are
deduplicated, order preserved.

---

## `restore_glacier.py`

Initiates asynchronous `restore_object` requests for each key, and (with
`--status`) polls readiness. Idempotent: already-restored / in-progress objects
count as success, and a state file records requested keys so reruns skip them.

```bash
# Dry run — show what would be restored
uv run python scripts/restore_glacier.py --csv wav_glacier_keys.csv --dry-run

# Fire restores (fastest tier per object)
uv run python scripts/restore_glacier.py --csv wav_glacier_keys.csv

# Poll readiness (optionally on a random sample for a quick estimate)
uv run python scripts/restore_glacier.py --csv wav_glacier_keys.csv --status --sample 500
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv PATH` | `wav_glacier_keys.csv` | CSV with an `s3_key` column. |
| `--bucket NAME` | `riverside-pro-main` | Source S3 bucket. |
| `--days N` | `7` | Days the restored copy stays readable. |
| `--tier T` | `fastest` | `fastest` heads each object and picks Expedited for `GLACIER` (1–5 min) / Standard for `DEEP_ARCHIVE` (≤12 h). Or force `Expedited` / `Standard` / `Bulk`. Expedited is **not** supported for Deep Archive. |
| `--workers N` | `32` | Concurrent S3 requests. |
| `--limit N` | none | Only process the first N keys. |
| `--sample N` | none | Randomly sample N keys instead of all (status estimate). |
| `--state PATH` | `restore_glacier_state.json` | JSON checkpoint of requested keys, skipped on rerun. Pass `''` to disable. |
| `--status` | off | Poll readiness instead of issuing restores. |
| `--dry-run` | off | List what would be done and exit. |

**Status / readiness values** (from the `Restore` response header, the same
signal `audio_resolver` gates on):

- `ready` — restore complete, readable now (`ongoing-request="false"`)
- `in_progress` — requested, copy not ready yet (`ongoing-request="true"`)
- `not_requested` — still archived, no restore initiated
- `not_archived` — already in a live class; nothing to restore

> Note: Deep Archive's fastest tier is Standard (≤12 h) — there is no knob to beat
> it. Expedited (1–5 min) applies only to Glacier Flexible Retrieval.

---

## `check_glacier_restore.py`

Standalone status verifier. Same readiness check as `restore_glacier.py --status`,
but can dump the actual key lists per status to files via `--out-prefix` (handy
for feeding the still-pending keys back into another restore pass).

```bash
uv run python scripts/check_glacier_restore.py --csv wav_glacier_keys.csv
uv run python scripts/check_glacier_restore.py --csv wav_glacier_keys.csv --out-prefix restore_status
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv PATH` | `wav_glacier_keys.csv` | CSV with an `s3_key` column. |
| `--bucket NAME` | `riverside-pro-main` | Source S3 bucket. |
| `--workers N` | `32` | Concurrent `head_object` requests. |
| `--limit N` | none | Only check the first N keys. |
| `--out-prefix P` | none | Write `<P>.<status>.txt` files listing keys per status. |

Prints per-status counts plus a `ready/total` percentage. Exit code is `0` only
when nothing is still pending (all `ready` or `not_archived`), else `1` — usable
in a polling loop.
