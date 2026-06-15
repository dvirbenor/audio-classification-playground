# MPS-vs-fleet throughput measurement

`mps_vs_fleet_throughput.py` measures real, wall-clock throughput of the inference
fleet and answers one question: **does an MPS-colocated pod (1 GPU running
affect+disfluency+emotion as 3 processes sharing CUDA MPS) do more work per GPU
than the per-task fleet (the same 3 tasks on 3 dedicated GPUs)?**

## Why not the `orchestration status` dashboard

The dashboard's `Pace` column is a *latency-derived* estimate
(`3600 / mean(total_sec)` over a single timing file). It under-reports a
co-located MPS pod ~3× and actively misleads, because MPS trades per-archive
latency for aggregate throughput. The trustworthy signal is **completions over
real elapsed time**, which is what this script computes.

## How it works

Either way the unit is **completions over real elapsed time** (archives/hour),
reported per worker, per task, and per GPU. There are two modes.

### History mode (default)

Every record in `<output>/_meta/timings/*.jsonl` already stores a per-completion
wall-clock timestamp (`ts`, e.g. `2026-06-15T07:30:37Z`). History mode reads
those timestamps and buckets completions into `--window`-sized bins. Because the
rate is reconstructed from data that **already exists**, it:

- needs **no waiting** — it's a single pass over the files;
- works **retrospectively**, even after the queue drains and the counters stop
  moving (live sampling would just report 0 then);
- emits a **throughput-over-time curve** (one row per window) so you can see
  ramp-up and steady state, not just a single delta;
- prints the MPS-vs-fleet headline over the **most recent `--window` of data**,
  anchored to the latest completion timestamp. Use a longer `--window` (e.g.
  `1h`) for a more stable headline; a short one for finer curve resolution.

Records with no parseable `ts` are **counted and warned about**, never silently
dropped. Only workers that completed something in the headline window count
toward the per-GPU denominator, so stale timing files left by dead pods don't
inflate the GPU count.

Caveats: `ts` is whole-second resolution (fine for per-hour rates), and each pod
stamps with its own clock — so cross-pod windows are only as good as NTP
agreement (negligible at ≥1-minute windows; use `--live` if you need
sub-minute precision or byte-for-byte parity with the dashboard's `Done`).

### Live mode (`--live`)

Takes two (or more) snapshots `--interval` seconds apart of the line counts in
`*.jsonl` — the same counter behind the dashboard's `Done` column, which sums
correctly across a worker's processes — and reports real deltas / real elapsed.
Use it when you want the rate **right now** or dashboard parity.

- **Done counts are summed per hostname**, so a worker that restarts (new UUID
  timing file) mid-window is still counted.
- A pod that just started, or is glacier-stalled with 0 completions, is surfaced
  in a **Stalled** section rather than silently dropped.
- Workers idle longer than `--active-within` (dead pods that left stale
  locks/timing files on the shared tree) are ignored.
- `--samples > 2` adds a per-interval stability table.

### Common to both

- **MPS pods** are identified by a hostname substring (`--mps-match`, default
  `mps`); everything else with a GPU task is the fleet.
- VAD is CPU-only and excluded from the GPU comparison (reported separately under
  "Other").

Imports `load_recent_timings` / `parse_active_locks` from
`orchestration.heartbeat`, so it **must run from the repo root via `uv run`**.

## Usage

Default (history) — instant, no waiting, works after a run finishes:

```bash
uv run python scripts/mps_vs_fleet_throughput.py \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --window 5m
```

Stable headline over the last hour, with a 6-bin curve and JSON dump:

```bash
uv run python scripts/mps_vs_fleet_throughput.py --window 1h --max-bins 6 --json mps_vs_fleet.json
```

Live sampling (watch the counters advance) with a per-interval stability series:

```bash
uv run python scripts/mps_vs_fleet_throughput.py --live --interval 300 --samples 4
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output PATH` | `/efs/.../models-inference` | Inference output base (must contain `_meta/timings/`). |
| `--live` | off | Use live line-count sampling instead of history mode. |
| `--mps-match S` | `mps` | Hostname substring identifying MPS-colocated pods. |
| `--json PATH` | none | Also dump the result (incl. the curve) as JSON to this path. |
| **history mode** | | |
| `--window D` | `5m` | Analysis/bin window: `300`, `90s`, `5m`, `1h`. Sets both the curve bin size and the headline window. |
| `--max-bins N` | `12` | Max windows shown in the throughput-over-time curve. |
| **live mode** | | |
| `--interval N` | `300` | Seconds between snapshots. |
| `--samples N` | `2` | Number of snapshots; the headline uses first vs last. `>2` adds a per-interval stability table. Must be ≥ 2. |
| `--active-within N` | `1800` | Ignore workers idle longer than N seconds (dead pods with stale locks). `0` disables the filter. |

## Output sections

- **Throughput over time** (history only) — arc/h per `--window` bin for MPS vs
  fleet; shows ramp-up and steady state.
- **MPS-colocated pod(s)** — one row per task process, with a per-pod subtotal.
- **Per-task fleet** — dedicated-GPU workers.
- **Per-task: MPS vs fleet** — archives/hour per task and the MPS/fleet ratio.
- **Aggregate / per-GPU efficiency** — the headline: arc/h per GPU for MPS vs
  fleet, and the `MPS/fleet` per-GPU multiplier (>1 = MPS does more work per GPU).
- **Other (CPU/VAD)** — informational.
- **Stalled** (live only) — hosts holding locks but with 0 completions in the
  window (model load on startup, or glacier-cluster starvation — see
  `_meta/audio_errors`).
- **Warnings** — e.g. records with no parseable `ts` (history), or a timing file
  whose line count went backwards (live, clamped to 0).
