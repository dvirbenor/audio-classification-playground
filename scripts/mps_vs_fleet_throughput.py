#!/usr/bin/env python3
"""MPS-colocated vs per-task fleet throughput, measured over wall-clock.

The ``orchestration status`` dashboard's ``Pace`` column is a latency-derived
estimate (``3600 / mean(total_sec)`` over a *single* timing file), so it
under-reports a co-located MPS pod ~3x and actively misleads when MPS trades
per-archive latency for aggregate throughput. The trustworthy signal is
completions over real elapsed time.

This samples ``_meta/timings/*.jsonl`` line counts twice (``--interval`` apart)
-- exactly the counter behind the dashboard's ``Done`` column, which sums
correctly across a worker's processes -- and reports archives/hour per worker,
per task, and per GPU for the MPS pod(s) vs the fleet.

  MPS pod  = 1 GPU running affect+disfluency+emotion (3 procs sharing CUDA MPS)
  Fleet    = the same 3 tasks on 3 dedicated GPUs
VAD is CPU-only and is excluded from the GPU comparison (reported separately).

Done counts are summed per hostname; rates are real deltas / real elapsed, so a
worker that restarts (new UUID timing file) mid-window is still counted, and a
pod that just started (or is glacier-stalled with 0 completions) is surfaced as
stalled rather than silently dropped. Workers idle longer than ``--active-within``
(dead pods that left stale locks/timing files on the shared tree) are ignored.

Example:
    uv run python scripts/mps_vs_fleet_throughput.py \
        --output /efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference \
        --interval 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from audio_classification_playground.acoustic_events.orchestration.heartbeat import (
        load_recent_timings,
        parse_active_locks,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "Could not import orchestration.heartbeat -- run via\n"
        "  uv run python scripts/mps_vs_fleet_throughput.py ...\n"
        f"from the repo root. ({exc})"
    )

GPU_TASKS = ("affect", "disfluency", "emotion")
DEFAULT_OUTPUT = (
    "/efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference"
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _file_task(task_groups: tuple[str, ...], hostname: str) -> str:
    """Task for a (single-task) timing file: its task_group, else inferred from host."""
    for tg in task_groups:
        if tg:
            return tg
    low = hostname.lower()
    for t in (*GPU_TASKS, "vad"):
        if t in low:
            return t
    return "?"


def take_snapshot(output_base: Path) -> dict:
    """One observation: per-file line counts, lock metadata, wall time."""
    timings = load_recent_timings(output_base, tail=1)  # tail=1: done is a full line count
    per_file = {
        wid: {
            "hostname": ti.hostname,
            "task": _file_task(ti.task_groups, ti.hostname),
            "done": ti.done,
            "latest_ts": ti.latest_ts,
        }
        for wid, ti in timings.items()
    }
    lock_meta = {
        host: {
            "count": len(infos),
            "latest": max((li.lock_time for li in infos), default=0.0),
        }
        for host, infos in parse_active_locks(output_base).items()
    }
    return {"ts": time.time(), "per_file": per_file, "lock_meta": lock_meta}


def active_hosts(snap: dict, within: float | None) -> set[str] | None:
    """Hostnames whose newest timing record OR lock is within ``within`` seconds.

    Returns ``None`` (no filtering) when ``within`` is None.
    """
    if within is None:
        return None
    last: dict[str, float] = defaultdict(float)
    for info in snap["per_file"].values():
        if info["latest_ts"]:
            last[info["hostname"]] = max(last[info["hostname"]], info["latest_ts"])
    for host, meta in snap["lock_meta"].items():
        last[host] = max(last[host], meta["latest"])
    cutoff = snap["ts"] - within
    return {h for h, t in last.items() if t >= cutoff}


def compute_deltas(
    s0: dict, s1: dict, mps_match: str, keep: set[str] | None
) -> tuple[list[dict], list[str]]:
    """Per-timing-file completion deltas between two snapshots (active hosts only)."""
    elapsed_h = (s1["ts"] - s0["ts"]) / 3600.0
    needle = mps_match.lower()
    rows: list[dict] = []
    warnings: list[str] = []
    for wid, info in s1["per_file"].items():
        if keep is not None and info["hostname"] not in keep:
            continue
        before = s0["per_file"].get(wid, {}).get("done", 0)
        delta = info["done"] - before
        if delta < 0:
            warnings.append(
                f"{wid}: line count went backwards ({before} -> {info['done']}); clamped to 0"
            )
            delta = 0
        rows.append({
            "worker_id": wid,
            "hostname": info["hostname"],
            "task": info["task"],
            "delta": delta,
            "rate": delta / elapsed_h if elapsed_h > 0 else 0.0,
            "is_mps": needle in info["hostname"].lower(),
        })
    return rows, warnings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(headers: tuple[str, ...], rows: list[tuple]) -> None:
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt(r: tuple) -> str:
        return "  ".join(
            (str(v).ljust(w) if i == 0 else str(v).rjust(w))
            for i, (v, w) in enumerate(zip(r, widths))
        )

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def _short(host: str) -> str:
    """Trim the common ``arno-inference-`` / ``dvir-inference-`` prefix for display."""
    for pre in ("arno-inference-", "dvir-inference-"):
        if host.startswith(pre):
            return host[len(pre):]
    return host


def report(
    rows: list[dict], elapsed_sec: float, lock_counts: dict[str, int], warnings: list[str]
) -> dict:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\nMPS-vs-Fleet throughput   (Δ over {elapsed_sec / 60.0:.1f} min)   {now}")
    print("=" * 78)

    mps_gpu = [r for r in rows if r["is_mps"] and r["task"] in GPU_TASKS]
    fleet_gpu = [r for r in rows if not r["is_mps"] and r["task"] in GPU_TASKS]
    other = [r for r in rows if r["task"] not in GPU_TASKS]  # vad, '?'

    # --- MPS pods (one row per task process, subtotal per pod) ---
    print("\n== MPS-colocated pod(s)  [1 GPU, 3 tasks via CUDA MPS] ==")
    if mps_gpu:
        table: list[tuple] = []
        by_host: dict[str, list[dict]] = defaultdict(list)
        for r in sorted(mps_gpu, key=lambda x: (x["hostname"], x["task"])):
            by_host[r["hostname"]].append(r)
        for host, hrows in by_host.items():
            for r in hrows:
                table.append((_short(host), r["task"], r["delta"], f"{r['rate']:.1f}"))
            table.append((
                f"  └ pod total ({_short(host)})", "",
                sum(r["delta"] for r in hrows), f"{sum(r['rate'] for r in hrows):.1f}",
            ))
        _print_table(("worker", "task", "Δdone", "arc/h"), table)
    else:
        print("  (no MPS pod completions in window)")

    # --- Fleet GPU workers ---
    print("\n== Per-task fleet  [dedicated GPUs] ==")
    if fleet_gpu:
        _print_table(
            ("worker", "task", "Δdone", "arc/h"),
            [(_short(r["hostname"]), r["task"], r["delta"], f"{r['rate']:.1f}")
             for r in sorted(fleet_gpu, key=lambda x: (x["task"], x["hostname"]))],
        )
    else:
        print("  (no fleet GPU completions in window)")

    # --- Per-task comparison ---
    print("\n== Per-task: MPS (shared GPU) vs fleet (dedicated GPU) ==")
    _print_table(
        ("task", "MPS arc/h", "fleet arc/h", "MPS/fleet"),
        [(
            task,
            f"{sum(r['rate'] for r in mps_gpu if r['task'] == task):.1f}",
            f"{sum(r['rate'] for r in fleet_gpu if r['task'] == task):.1f}",
            (f"{sum(r['rate'] for r in mps_gpu if r['task'] == task) / f:.2f}x"
             if (f := sum(r['rate'] for r in fleet_gpu if r['task'] == task)) > 0 else "--"),
        ) for task in GPU_TASKS],
    )

    # --- Per-GPU efficiency (the headline number) ---
    mps_hosts = {r["hostname"] for r in mps_gpu}
    fleet_hosts = {r["hostname"] for r in fleet_gpu}
    mps_total = sum(r["rate"] for r in mps_gpu)
    fleet_total = sum(r["rate"] for r in fleet_gpu)
    mps_per_gpu = mps_total / len(mps_hosts) if mps_hosts else 0.0
    fleet_per_gpu = fleet_total / len(fleet_hosts) if fleet_hosts else 0.0

    print("\n== Aggregate / per-GPU efficiency ==")
    print(f"  MPS:   {mps_total:8.1f} arc/h  over {len(mps_hosts)} GPU(s)  "
          f"= {mps_per_gpu:8.1f} arc/h per GPU")
    print(f"  Fleet: {fleet_total:8.1f} arc/h  over {len(fleet_hosts)} GPU(s)  "
          f"= {fleet_per_gpu:8.1f} arc/h per GPU")
    if mps_per_gpu and fleet_per_gpu:
        eff = mps_per_gpu / fleet_per_gpu
        print(f"  → MPS colocation does {eff:.2f}x the per-GPU throughput of dedicated "
              f"workers ({'MORE' if eff >= 1 else 'LESS'} work per GPU).")

    # --- VAD / other ---
    if other:
        print("\n== Other (CPU/VAD, informational) ==")
        _print_table(
            ("worker", "task", "Δdone", "arc/h"),
            [(_short(r["hostname"]), r["task"], r["delta"], f"{r['rate']:.1f}")
             for r in sorted(other, key=lambda x: x["hostname"])],
        )

    # --- Stalled: active (recent locks) but produced nothing in the window ---
    produced = {r["hostname"] for r in rows if r["delta"] > 0}
    stalled = sorted(h for h in lock_counts if h not in produced)
    if stalled:
        print("\n== Stalled: holds locks, 0 completions in window ==")
        for h in stalled:
            tag = "  [MPS]" if "mps" in h.lower() else ""
            print(f"  {_short(h)}  ({lock_counts[h]} locks){tag}")
        print("  (likely model load on startup, or glacier-cluster starvation — "
              "see _meta/audio_errors)")

    if warnings:
        print("\n== Warnings ==")
        for w in warnings:
            print(f"  ! {w}")

    return {
        "elapsed_sec": elapsed_sec,
        "mps": {"total_arc_h": mps_total, "gpus": len(mps_hosts),
                "per_gpu_arc_h": mps_per_gpu},
        "fleet": {"total_arc_h": fleet_total, "gpus": len(fleet_hosts),
                  "per_gpu_arc_h": fleet_per_gpu},
        "per_worker": rows,
        "stalled": stalled,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Inference output base (contains _meta/).")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="Seconds between samples (default 300).")
    ap.add_argument("--samples", type=int, default=2,
                    help="Number of snapshots; headline uses first vs last (default 2).")
    ap.add_argument("--mps-match", default="mps",
                    help="Substring identifying MPS-colocated hostnames (default 'mps').")
    ap.add_argument("--active-within", type=float, default=1800.0,
                    help="Ignore workers idle longer than this many seconds "
                         "(dead pods with stale locks). Default 1800; 0 = no filter.")
    ap.add_argument("--json", default=None, help="Optional path to dump the result JSON.")
    args = ap.parse_args()

    if args.samples < 2:
        ap.error("--samples must be >= 2")
    output_base = Path(args.output)
    if not (output_base / "_meta" / "timings").is_dir():
        ap.error(f"no _meta/timings/ under {output_base}")
    within = None if args.active_within <= 0 else args.active_within

    snaps: list[dict] = []
    for i in range(args.samples):
        snaps.append(take_snapshot(output_base))
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        n_active = len(active_hosts(snaps[-1], within) or snaps[-1]["lock_meta"])
        print(f"[{stamp}] sample {i + 1}/{args.samples}  "
              f"({len(snaps[-1]['per_file'])} timing files, {n_active} active hosts)",
              flush=True)
        if i < args.samples - 1:
            time.sleep(args.interval)

    keep = active_hosts(snaps[-1], within)

    # Optional stability series across consecutive intervals.
    if args.samples > 2:
        print("\n== Per-interval aggregate (stability check) ==")
        series: list[tuple] = []
        for a, b in zip(snaps, snaps[1:]):
            rs, _ = compute_deltas(a, b, args.mps_match, keep)
            mps = sum(r["rate"] for r in rs if r["is_mps"] and r["task"] in GPU_TASKS)
            flt = sum(r["rate"] for r in rs if not r["is_mps"] and r["task"] in GPU_TASKS)
            series.append((f"{(b['ts'] - a['ts']) / 60.0:.1f} min", f"{mps:.1f}", f"{flt:.1f}"))
        _print_table(("interval", "MPS arc/h", "fleet arc/h"), series)

    rows, warnings = compute_deltas(snaps[0], snaps[-1], args.mps_match, keep)
    lock_counts = {
        h: m["count"] for h, m in snaps[-1]["lock_meta"].items()
        if keep is None or h in keep
    }
    result = report(rows, snaps[-1]["ts"] - snaps[0]["ts"], lock_counts, warnings)

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
