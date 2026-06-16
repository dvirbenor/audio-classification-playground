#!/usr/bin/env python3
"""Post-hoc throughput from STORED completion timestamps — no live sampling, no waiting.

Every inference completion already records a ``ts`` in ``_meta/timings/*.jsonl``. This
reads those, filters by task, and prints real throughput (completions / wall-clock) —
overall, per pod, and as an hourly-ish curve — computed entirely from data on disk. It
returns instantly and works retrospectively (even after the queue drained and the
counters stopped moving). This is the "just tell me the rate" companion to
mps_vs_fleet_throughput.py, which is shaped around the MPS-vs-fleet GPU comparison and
buries CPU/VAD in an "Other" line.

Examples:
    # VAD backfill rate, overall + per pod + 1h curve, from stored data:
    uv run python scripts/throughput_history.py --task vad

    # only the last 2h of activity, 30-min bins:
    uv run python scripts/throughput_history.py --task vad --last 2h --bin 30m

    # everything, regardless of task:
    uv run python scripts/throughput_history.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUTPUT = "/efs/dvir/data/magic-clips-research/acoustic-understanding/models-inference"


def _parse_ts(ts_str) -> float | None:
    try:
        return (datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
                .replace(tzinfo=timezone.utc).timestamp())
    except (ValueError, TypeError):
        return None


def _parse_duration(s: str) -> float:
    """'90s' / '5m' / '2h' / '1d' -> seconds."""
    s = s.strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(s[-1:])
    if mult is None:
        return float(s)  # bare seconds
    return float(s[:-1]) * mult


def collect(output_base: Path, tasks: set[str] | None):
    """Return {hostname: sorted[ts]} and counters, filtered to *tasks* (None = all)."""
    timings_dir = output_base / "_meta" / "timings"
    if not timings_dir.is_dir():
        sys.exit(f"No timings dir at {timings_dir}")
    by_host: dict[str, list[float]] = {}
    by_task: dict[str, list[float]] = {}
    n_records = n_bad = n_filtered = 0
    for p in sorted(timings_dir.iterdir()):
        if not p.name.endswith(".jsonl"):
            continue
        host = p.stem.rsplit("_", 1)[0] if "_" in p.stem else p.stem
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
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
                    tg = str(rec.get("task_group", "") or "?")
                    if tasks is not None and tg not in tasks:
                        n_filtered += 1
                        continue
                    epoch = _parse_ts(rec.get("ts"))
                    if epoch is None:
                        n_bad += 1
                        continue
                    by_host.setdefault(host, []).append(epoch)
                    by_task.setdefault(tg, []).append(epoch)
        except OSError:
            continue
    for ts in by_host.values():
        ts.sort()
    for ts in by_task.values():
        ts.sort()
    return by_host, by_task, n_records, n_bad, n_filtered


def _rate(n: int, span: float) -> float:
    return n / span * 3600.0 if span > 0 else 0.0


def _short(host: str) -> str:
    return host[-28:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Inference output base.")
    ap.add_argument("--task", default=None,
                    help="Comma-separated task_group filter (e.g. vad). Default: all tasks.")
    ap.add_argument("--last", default=None,
                    help="Only completions within this much of the LAST record (e.g. 2h).")
    ap.add_argument("--bin", default="1h", help="Curve bin width (default 1h).")
    args = ap.parse_args()

    tasks = {t.strip() for t in args.task.split(",")} if args.task else None
    by_host, by_task, n_records, n_bad, n_filtered = collect(Path(args.output), tasks)
    if not by_host:
        sys.exit("No timestamped completions found (after filter).")

    all_ts = sorted(t for ts in by_host.values() for t in ts)
    t_max = all_ts[-1]
    t_min = all_ts[0]
    if args.last:
        t_min = max(t_min, t_max - _parse_duration(args.last))
        all_ts = [t for t in all_ts if t >= t_min]
        by_host = {h: [t for t in ts if t >= t_min] for h, ts in by_host.items()}
        by_host = {h: ts for h, ts in by_host.items() if ts}
        by_task = {k: [t for t in ts if t >= t_min] for k, ts in by_task.items()}
        by_task = {k: ts for k, ts in by_task.items() if ts}

    span = t_max - t_min
    bin_w = _parse_duration(args.bin)
    import bisect
    # "Current" = completions in the most recent bin (robust to a history that spans weeks
    # of dead workers, which would dilute a full-span average into a meaningless number).
    recent_lo = t_max - bin_w
    recent_n = len(all_ts) - bisect.bisect_left(all_ts, recent_lo)
    label = ("tasks=" + ",".join(sorted(tasks))) if tasks else "all tasks"
    print(f"\nThroughput from stored timings  [{label}]   (from disk, no sampling)")
    print("=" * 74)
    print(f"  window:   {datetime.fromtimestamp(t_min, timezone.utc):%Y-%m-%d %H:%M} "
          f"-> {datetime.fromtimestamp(t_max, timezone.utc):%H:%M} UTC   "
          f"({span / 3600.0:.2f} h, {len(all_ts):,} completions)")
    print(f"  CURRENT:   {_rate(recent_n, bin_w):,.0f} arc/h   (last {args.bin}, {recent_n:,} completions)")
    print(f"  lifetime:  {_rate(len(all_ts), span):,.0f} arc/h   (whole window — diluted by "
          f"dead workers if it spans days; use --last to scope)")

    print(f"\n  per task ({len(by_task)}):   [CURRENT = last {args.bin}]")
    print(f"    {'task':<14} {'completions':>12} {'CURRENT':>11} {'lifetime':>11}")
    for tg in sorted(by_task, key=lambda k: -len(by_task[k])):
        ts = by_task[tg]
        recent = len(ts) - bisect.bisect_left(ts, t_max - bin_w)
        print(f"    {tg:<14} {len(ts):>12,} {_rate(recent, bin_w):>8,.0f}/h {_rate(len(ts), ts[-1] - ts[0]):>8,.0f}/h")

    print(f"\n  per pod ({len(by_host)}):")
    for host in sorted(by_host, key=lambda h: -len(by_host[h])):
        ts = by_host[host]
        pod_span = ts[-1] - ts[0]
        print(f"    {_short(host):<30} {len(ts):>7,} arc   {_rate(len(ts), pod_span):>7,.0f} arc/h")

    nbins = max(1, int((span + bin_w - 1) // bin_w))
    print(f"\n  curve ({args.bin} bins):")
    for b in range(nbins):
        lo = t_min + b * bin_w
        hi = min(lo + bin_w, t_max + 1)
        cnt = bisect.bisect_left(all_ts, hi) - bisect.bisect_left(all_ts, lo)
        bar = "#" * min(50, int(_rate(cnt, hi - lo) / 50))
        print(f"    {datetime.fromtimestamp(lo, timezone.utc):%m-%d %H:%M}  "
              f"{_rate(cnt, hi - lo):>7,.0f} arc/h  {bar}")

    if n_bad or n_filtered:
        print(f"\n  ({n_records:,} records read; {n_filtered:,} other-task, {n_bad:,} unparseable)")


if __name__ == "__main__":
    main()
