#!/usr/bin/env python3
"""MPS-colocated vs per-task fleet throughput, measured over wall-clock.

The ``orchestration status`` dashboard's ``Pace`` column is a latency-derived
estimate (``3600 / mean(total_sec)`` over a *single* timing file), so it
under-reports a co-located MPS pod ~3x and actively misleads when MPS trades
per-archive latency for aggregate throughput. The trustworthy signal is
completions over real elapsed time.

  MPS pod  = 1 GPU running affect+disfluency+emotion (3 procs sharing CUDA MPS)
  Fleet    = the same 3 tasks on 3 dedicated GPUs
VAD is CPU-only and is excluded from the GPU comparison (reported separately).

Two modes:

* History (default) -- reads the per-completion ``ts`` timestamp already stored
  in every ``_meta/timings/*.jsonl`` record and buckets completions into
  ``--window``-sized bins. True throughput = completions-per-real-time, computed
  from data that already exists, so it works retrospectively (even after the
  queue drains and the counters stop moving) with no waiting. Emits the
  MPS-vs-fleet table over the most recent window plus a throughput-over-time
  curve. Records with no parseable ``ts`` are warned about, never silently
  dropped.
* Live (``--live``) -- samples ``_meta/timings/*.jsonl`` line counts ``--interval``
  apart (exactly the counter behind the dashboard's ``Done`` column) and reports
  real deltas / real elapsed. Use when you want the rate *right now*, byte-for-byte
  parity with the dashboard, or immunity to cross-pod clock skew on tight windows.
  Workers idle longer than ``--active-within`` are ignored; a worker that restarts
  mid-window is still counted; a stalled pod (locks, 0 completions) is surfaced.

Examples:
    # Retrospective, no waiting (default):
    uv run python scripts/mps_vs_fleet_throughput.py --window 5m
    # Live sampling (watch the counters advance):
    uv run python scripts/mps_vs_fleet_throughput.py --live --interval 300 --samples 4
"""
from __future__ import annotations

import argparse
import bisect
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
    # done is a full line count regardless of tail, but task_group/latest_ts come
    # from the parsed tail. A single timing record here is ~2 KB (49 fields) and
    # heartbeat's tail reader only scans the last tail*1024 bytes, so tail=1 (1 KB)
    # truncates the final JSON line -> it fails to parse -> task_group is lost and
    # the worker is misbucketed as '?'. Read enough tail to capture whole records.
    timings = load_recent_timings(output_base, tail=8)
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
# History mode: reconstruct throughput from per-completion ``ts`` timestamps
# ---------------------------------------------------------------------------


def _parse_duration(text: str) -> float:
    """Parse ``'300'`` / ``'90s'`` / ``'5m'`` / ``'1h'`` into seconds."""
    text = str(text).strip().lower()
    mult = {"s": 1.0, "m": 60.0, "h": 3600.0}
    if text and text[-1] in mult:
        return float(text[:-1]) * mult[text[-1]]
    return float(text)


def _parse_ts(ts_str) -> float | None:
    """Completion timestamp ``"%Y-%m-%dT%H:%M:%SZ"`` -> epoch seconds, or None."""
    try:
        return (
            datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except (ValueError, TypeError):
        return None


def collect_history(output_base: Path, mps_match: str) -> tuple[list[dict], int, int]:
    """Read every timing record, grouping completion timestamps per worker file.

    Returns ``(workers, n_records, n_untimestamped)`` where each worker is
    ``{worker_id, hostname, task, is_mps, ts: sorted list[float]}``. ``ts`` is
    the per-completion wall-clock stamp already stored in each record, so rates
    are reconstructable for any past window with no live sampling.
    """
    timings_dir = output_base / "_meta" / "timings"
    needle = mps_match.lower()
    workers: list[dict] = []
    n_records = 0
    n_bad = 0
    for jsonl_path in sorted(timings_dir.iterdir()):
        if not jsonl_path.name.endswith(".jsonl"):
            continue
        worker_id = jsonl_path.stem
        hostname = worker_id.rsplit("_", 1)[0] if "_" in worker_id else worker_id
        ts_list: list[float] = []
        gated_ts: list[float] = []  # subset of ts where the archive was VAD-gated
        task_groups: set[str] = set()
        try:
            with open(jsonl_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n_records += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        n_bad += 1
                        continue
                    if rec.get("task_group"):
                        task_groups.add(str(rec["task_group"]))
                    epoch = _parse_ts(rec.get("ts"))
                    if epoch is None:
                        n_bad += 1
                        continue
                    ts_list.append(epoch)
                    # Gated == ran against a precomputed/cached VAD artifact (vs.
                    # full-timeline). precomputed_vad mirrors timings.derive_vad_mode.
                    if rec.get("precomputed_vad") or rec.get("vad_reused"):
                        gated_ts.append(epoch)
        except OSError:
            continue
        if not ts_list:
            continue
        ts_list.sort()
        gated_ts.sort()
        workers.append({
            "worker_id": worker_id,
            "hostname": hostname,
            "task": _file_task(tuple(sorted(task_groups)), hostname),
            "is_mps": needle in hostname.lower(),
            "ts": ts_list,
            "gated_ts": gated_ts,
        })
    return workers, n_records, n_bad


def _count_in(ts_sorted: list[float], lo: float, hi: float) -> int:
    """Number of timestamps in the half-open window ``[lo, hi)``."""
    return bisect.bisect_left(ts_sorted, hi) - bisect.bisect_left(ts_sorted, lo)


def run_history(
    output_base: Path, window_sec: float, max_bins: int, mps_match: str, json_path
) -> None:
    workers, n_records, n_bad = collect_history(output_base, mps_match)
    if not workers:
        sys.exit("No timestamped completions found in history.")

    t_max = max(w["ts"][-1] for w in workers)
    t_min = min(w["ts"][0] for w in workers)
    window_h = window_sec / 3600.0

    print(f"\nMPS-vs-Fleet throughput  [history]  "
          f"{n_records} records over {(t_max - t_min) / 60.0:.1f} min of data")

    # --- Throughput-over-time curve (most recent max_bins windows) ---
    span_bins = int((t_max - t_min) // window_sec) + 1
    nb = max(1, min(max_bins, span_bins))
    curve: list[tuple] = []  # (window_end_label, mps_rate, fleet_rate)
    for i in range(nb):
        hi_b = t_max - i * window_sec
        lo_b = hi_b - window_sec
        mps = sum(_count_in(w["ts"], lo_b, hi_b) for w in workers
                  if w["is_mps"] and w["task"] in GPU_TASKS) / window_h
        flt = sum(_count_in(w["ts"], lo_b, hi_b) for w in workers
                  if not w["is_mps"] and w["task"] in GPU_TASKS) / window_h
        curve.append((datetime.fromtimestamp(hi_b, timezone.utc).strftime("%H:%M:%S"),
                      mps, flt))
    curve.reverse()  # chronological
    print(f"\n== Throughput over time  [{window_sec / 60.0:.1f} min bins, "
          f"last {nb} of {span_bins}] ==")
    _print_table(("window end (UTC)", "MPS arc/h", "fleet arc/h"),
                 [(e, f"{m:.1f}", f"{f:.1f}") for e, m, f in curve])

    # --- Headline over the most recent full window [t_max - window, t_max] ---
    # Only workers that actually completed something in the window count: stale
    # timing files left by dead pods (a prior run's drained workers) otherwise
    # inflate the per-GPU denominator. This is history's analogue of live mode's
    # --active-within filter.
    lo, hi = t_max - window_sec, t_max
    rows = []
    for w in workers:
        cnt = _count_in(w["ts"], lo, hi)
        if cnt == 0:
            continue
        rows.append({
            "worker_id": w["worker_id"],
            "hostname": w["hostname"],
            "task": w["task"],
            "delta": cnt,
            "rate": cnt / window_h if window_h > 0 else 0.0,
            "is_mps": w["is_mps"],
        })

    warnings: list[str] = []
    if n_bad:
        warnings.append(
            f"{n_bad} of {n_records} records had no parseable ts/JSON; "
            "excluded from history rates"
        )

    start_s = datetime.fromtimestamp(lo, timezone.utc).strftime("%H:%M:%S")
    end_s = datetime.fromtimestamp(hi, timezone.utc).strftime("%H:%M:%S")
    print(f"\n[history] headline window: {start_s} -> {end_s} UTC "
          f"(last {window_sec / 60.0:.1f} min of data; anchored to latest completion)")
    result = report(rows, window_sec, {}, warnings)  # {} locks: no live "stalled" notion

    # --- VAD-gating coverage (are archives actually gated, or full-timeline?) ---
    # Lifetime per GPU worker (not just the headline window): prod running ~100%
    # ungated means the 33-56% gating speedup is unrealized; this surfaces it.
    cov_rows: list[tuple] = []
    coverage: list[dict] = []
    for w in sorted(workers, key=lambda x: (not x["is_mps"], x["task"], x["hostname"])):
        if w["task"] not in GPU_TASKS:
            continue
        total = len(w["ts"])
        if total == 0:
            continue
        gated = len(w["gated_ts"])
        pct = 100.0 * gated / total
        pool = "MPS" if w["is_mps"] else "fleet"
        cov_rows.append((pool, w["task"], _short(w["hostname"]), str(total),
                         str(gated), f"{pct:.0f}%"))
        coverage.append({"pool": pool, "task": w["task"], "hostname": w["hostname"],
                         "n": total, "gated": gated, "gated_pct": pct})
    if cov_rows:
        print("\n== VAD-gating coverage (lifetime, GPU tasks) ==")
        _print_table(("pool", "task", "worker", "n", "gated", "gated%"), cov_rows)
        tot = sum(c["n"] for c in coverage)
        gat = sum(c["gated"] for c in coverage)
        print(f"  overall: {gat}/{tot} = {100.0 * gat / tot:.0f}% gated "
              "(0% => gating speedup unrealized; VAD not leading)")

    result["mode"] = "history"
    result["window_sec"] = window_sec
    result["data_span_sec"] = t_max - t_min
    result["curve"] = [
        {"window_end_utc": e, "mps_arc_h": m, "fleet_arc_h": f} for e, m, f in curve
    ]
    result["gating_coverage"] = coverage

    if json_path:
        Path(json_path).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nWrote {json_path}")


# ---------------------------------------------------------------------------
# Live mode: sample the line-count counters over real elapsed wall-clock
# ---------------------------------------------------------------------------


def run_live(
    output_base: Path, interval: float, samples: int, mps_match: str,
    within: float | None, json_path,
) -> None:
    snaps: list[dict] = []
    for i in range(samples):
        snaps.append(take_snapshot(output_base))
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        n_active = len(active_hosts(snaps[-1], within) or snaps[-1]["lock_meta"])
        print(f"[{stamp}] sample {i + 1}/{samples}  "
              f"({len(snaps[-1]['per_file'])} timing files, {n_active} active hosts)",
              flush=True)
        if i < samples - 1:
            time.sleep(interval)

    keep = active_hosts(snaps[-1], within)

    # Optional stability series across consecutive intervals.
    if samples > 2:
        print("\n== Per-interval aggregate (stability check) ==")
        series: list[tuple] = []
        for a, b in zip(snaps, snaps[1:]):
            rs, _ = compute_deltas(a, b, mps_match, keep)
            mps = sum(r["rate"] for r in rs if r["is_mps"] and r["task"] in GPU_TASKS)
            flt = sum(r["rate"] for r in rs if not r["is_mps"] and r["task"] in GPU_TASKS)
            series.append((f"{(b['ts'] - a['ts']) / 60.0:.1f} min", f"{mps:.1f}", f"{flt:.1f}"))
        _print_table(("interval", "MPS arc/h", "fleet arc/h"), series)

    rows, warnings = compute_deltas(snaps[0], snaps[-1], mps_match, keep)
    lock_counts = {
        h: m["count"] for h, m in snaps[-1]["lock_meta"].items()
        if keep is None or h in keep
    }
    result = report(rows, snaps[-1]["ts"] - snaps[0]["ts"], lock_counts, warnings)
    result["mode"] = "live"

    if json_path:
        Path(json_path).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nWrote {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Inference output base (contains _meta/).")
    ap.add_argument("--live", action="store_true",
                    help="Live sampling (watch counters advance). Default is history mode, "
                         "which reconstructs throughput from stored ts timestamps without waiting.")
    ap.add_argument("--mps-match", default="mps",
                    help="Substring identifying MPS-colocated hostnames (default 'mps').")
    ap.add_argument("--json", default=None, help="Optional path to dump the result JSON.")

    hist = ap.add_argument_group("history mode (default)")
    hist.add_argument("--window", default="5m",
                      help="Analysis/bin window, e.g. 300, 90s, 5m, 1h (default 5m).")
    hist.add_argument("--max-bins", type=int, default=12,
                      help="Max windows in the throughput-over-time curve (default 12).")

    live = ap.add_argument_group("live mode (--live)")
    live.add_argument("--interval", type=float, default=300.0,
                      help="Seconds between samples (default 300).")
    live.add_argument("--samples", type=int, default=2,
                      help="Number of snapshots; headline uses first vs last (default 2).")
    live.add_argument("--active-within", type=float, default=1800.0,
                      help="Ignore workers idle longer than this many seconds "
                           "(dead pods with stale locks). Default 1800; 0 = no filter.")
    args = ap.parse_args()

    output_base = Path(args.output)
    if not (output_base / "_meta" / "timings").is_dir():
        ap.error(f"no _meta/timings/ under {output_base}")

    if args.live:
        if args.samples < 2:
            ap.error("--samples must be >= 2")
        within = None if args.active_within <= 0 else args.active_within
        run_live(output_base, args.interval, args.samples, args.mps_match, within, args.json)
    else:
        window_sec = _parse_duration(args.window)
        if window_sec <= 0:
            ap.error("--window must be > 0")
        run_history(output_base, window_sec, args.max_bins, args.mps_match, args.json)


if __name__ == "__main__":
    main()
