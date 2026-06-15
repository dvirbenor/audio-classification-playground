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

It takes two (or more) snapshots `--interval` seconds apart of the line counts in
`<output>/_meta/timings/*.jsonl` — the same counter behind the dashboard's `Done`
column, which sums correctly across a worker's processes — and reports
archives/hour per worker, per task, and per GPU.

- **MPS pods** are identified by a hostname substring (`--mps-match`, default
  `mps`); everything else with a GPU task is the fleet.
- **Done counts are summed per hostname** from real deltas / real elapsed, so a
  worker that restarts (new UUID timing file) mid-window is still counted.
- A pod that just started, or is glacier-stalled with 0 completions, is surfaced
  in a **Stalled** section rather than silently dropped.
- VAD is CPU-only and excluded from the GPU comparison (reported separately under
  "Other").
- Workers idle longer than `--active-within` (dead pods that left stale
  locks/timing files on the shared tree) are ignored.

Imports `load_recent_timings` / `parse_active_locks` from
`orchestration.heartbeat`, so it **must run from the repo root via `uv run`**.

## Usage

```bash
uv run python scripts/mps_vs_fleet_throughput.py \
    --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
    --interval 300
```

Longer, more stable read with a per-interval stability series and JSON dump:

```bash
uv run python scripts/mps_vs_fleet_throughput.py --interval 300 --samples 4 --json mps_vs_fleet.json
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output PATH` | `/efs/.../models-inference` | Inference output base (must contain `_meta/timings/`). |
| `--interval N` | `300` | Seconds between snapshots. |
| `--samples N` | `2` | Number of snapshots; the headline uses first vs last. `>2` adds a per-interval stability table. Must be ≥ 2. |
| `--mps-match S` | `mps` | Hostname substring identifying MPS-colocated pods. |
| `--active-within N` | `1800` | Ignore workers idle longer than N seconds (dead pods with stale locks). `0` disables the filter. |
| `--json PATH` | none | Also dump the result as JSON to this path. |

## Output sections

- **MPS-colocated pod(s)** — one row per task process, with a per-pod subtotal.
- **Per-task fleet** — dedicated-GPU workers.
- **Per-task: MPS vs fleet** — archives/hour per task and the MPS/fleet ratio.
- **Aggregate / per-GPU efficiency** — the headline: arc/h per GPU for MPS vs
  fleet, and the `MPS/fleet` per-GPU multiplier (>1 = MPS does more work per GPU).
- **Other (CPU/VAD)** — informational.
- **Stalled** — hosts holding locks but with 0 completions in the window (model
  load on startup, or glacier-cluster starvation — see `_meta/audio_errors`).
- **Warnings** — e.g. a timing file whose line count went backwards (clamped to 0).
