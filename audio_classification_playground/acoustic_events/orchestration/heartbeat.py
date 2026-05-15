"""Fleet heartbeat: per-worker lock/pace dashboard from ``_meta/`` on EFS.

Reads lock files and timing JSONL tails to produce a compact, at-a-glance
view of which pods are alive, how many archives each has completed, and
the current processing pace.  No parquet or manifest required.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .locking import LOCKS_DIR
from .timings import TIMINGS_DIR

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LockInfo:
    session_id: str
    archive_id: str
    lock_time: float
    hostname: str
    pid: int


@dataclass
class WorkerHeartbeat:
    worker_id: str
    hostname: str
    locks: int
    done: int
    last_activity_ts: float | None
    pace_per_hour: float | None


@dataclass
class FleetHeartbeat:
    workers: list[WorkerHeartbeat] = field(default_factory=list)
    fleet_locks: int = 0
    fleet_done: int = 0
    fleet_pace_per_hour: float | None = None


# ---------------------------------------------------------------------------
# Lock parsing
# ---------------------------------------------------------------------------


def parse_active_locks(output_base: Path) -> dict[str, list[LockInfo]]:
    """Read all ``_meta/locks/*.lock`` files and group by hostname.

    Returns ``{hostname: [LockInfo, ...]}``.
    """
    locks_dir = output_base / LOCKS_DIR
    if not locks_dir.is_dir():
        return {}

    by_host: dict[str, list[LockInfo]] = {}
    for lock_file in locks_dir.iterdir():
        if not lock_file.name.endswith(".lock"):
            continue
        stem = lock_file.stem
        if "__" not in stem:
            continue
        session_id, _, archive_id = stem.partition("__")

        hostname = "unknown"
        pid = 0
        lock_time = 0.0
        try:
            text = lock_file.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("worker="):
                    hostname = line[len("worker="):]
                elif line.startswith("pid="):
                    try:
                        pid = int(line[len("pid="):])
                    except ValueError:
                        pass
                elif line.startswith("time="):
                    try:
                        lock_time = float(line[len("time="):])
                    except ValueError:
                        pass
        except OSError:
            continue

        info = LockInfo(
            session_id=session_id,
            archive_id=archive_id,
            lock_time=lock_time,
            hostname=hostname,
            pid=pid,
        )
        by_host.setdefault(hostname, []).append(info)

    return by_host


# ---------------------------------------------------------------------------
# Timing tail-reading
# ---------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    """Count newline characters in a file via raw byte scan."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size == 0:
        return 0
    count = 0
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 16)  # 64 KiB
                if not chunk:
                    break
                count += chunk.count(b"\n")
    except OSError:
        return 0
    return count


def _read_tail_lines(path: Path, n: int) -> list[str]:
    """Read the last *n* non-empty lines of a text file efficiently."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []

    block = min(size, n * 1024)
    try:
        with open(path, "rb") as f:
            f.seek(max(0, size - block))
            raw = f.read()
    except OSError:
        return []

    lines = raw.decode("utf-8", errors="replace").splitlines()
    return [l for l in lines if l.strip()][-n:]


@dataclass
class _WorkerTimingInfo:
    worker_id: str
    hostname: str
    done: int
    recent_records: list[dict]
    latest_ts: float | None


def load_recent_timings(
    output_base: Path,
    tail: int = 20,
) -> dict[str, _WorkerTimingInfo]:
    """Load per-worker timing summaries from ``_meta/timings/*.jsonl``.

    For each file, counts total lines (for ``done``) and parses only the
    last *tail* records for pace calculation.  Returns keyed by
    ``worker_id``.
    """
    timings_dir = output_base / TIMINGS_DIR
    if not timings_dir.is_dir():
        return {}

    result: dict[str, _WorkerTimingInfo] = {}
    for jsonl_path in timings_dir.iterdir():
        if not jsonl_path.name.endswith(".jsonl"):
            continue

        worker_id = jsonl_path.stem
        hostname = worker_id.rsplit("_", 1)[0] if "_" in worker_id else worker_id

        done = _count_lines(jsonl_path)
        tail_lines = _read_tail_lines(jsonl_path, tail)

        records: list[dict] = []
        latest_ts: float | None = None
        for line in tail_lines:
            try:
                rec = json.loads(line)
                records.append(rec)
                ts_str = rec.get("ts")
                if ts_str:
                    try:
                        dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
                            tzinfo=timezone.utc,
                        )
                        epoch = dt.timestamp()
                        if latest_ts is None or epoch > latest_ts:
                            latest_ts = epoch
                    except ValueError:
                        pass
            except (json.JSONDecodeError, TypeError):
                continue

        result[worker_id] = _WorkerTimingInfo(
            worker_id=worker_id,
            hostname=hostname,
            done=done,
            recent_records=records,
            latest_ts=latest_ts,
        )

    return result


# ---------------------------------------------------------------------------
# Fleet heartbeat assembly
# ---------------------------------------------------------------------------


def build_fleet_heartbeat(
    locks: dict[str, list[LockInfo]],
    timings: dict[str, _WorkerTimingInfo],
) -> FleetHeartbeat:
    """Correlate lock ownership and timing data into a fleet snapshot."""
    all_hostnames: set[str] = set(locks.keys())
    for info in timings.values():
        all_hostnames.add(info.hostname)
    all_hostnames.discard("unknown")

    lock_count_by_host: dict[str, int] = {h: len(v) for h, v in locks.items()}
    latest_lock_by_host: dict[str, float] = {}
    for host, infos in locks.items():
        latest_lock_by_host[host] = max(li.lock_time for li in infos)

    timings_by_host: dict[str, list[_WorkerTimingInfo]] = {}
    for info in timings.values():
        timings_by_host.setdefault(info.hostname, []).append(info)

    workers: list[WorkerHeartbeat] = []
    fleet_done = 0
    fleet_locks = 0
    fleet_pace_sum = 0.0
    fleet_pace_count = 0

    for hostname in sorted(all_hostnames):
        host_timings = timings_by_host.get(hostname, [])
        host_locks = lock_count_by_host.get(hostname, 0)
        fleet_locks += host_locks

        if not host_timings:
            last_activity = latest_lock_by_host.get(hostname)
            workers.append(WorkerHeartbeat(
                worker_id=hostname,
                hostname=hostname,
                locks=host_locks,
                done=0,
                last_activity_ts=last_activity,
                pace_per_hour=None,
            ))
            continue

        # A hostname may have multiple timing files when the orchestrator
        # has been restarted (each launch generates a fresh UUID suffix).
        # Collapse them into one row: sum done counts, derive pace from
        # the currently-active instance only.
        total_done = sum(ti.done for ti in host_timings)
        active_ti = max(host_timings, key=lambda t: t.latest_ts or 0.0)

        total_secs = [
            r["total_sec"]
            for r in active_ti.recent_records
            if isinstance(r.get("total_sec"), (int, float))
        ]
        pace: float | None = None
        if total_secs:
            mean_sec = sum(total_secs) / len(total_secs)
            if mean_sec > 0:
                pace = 3600.0 / mean_sec
                fleet_pace_sum += pace
                fleet_pace_count += 1

        all_ts = [ti.latest_ts for ti in host_timings if ti.latest_ts is not None]
        lock_ts = latest_lock_by_host.get(hostname)
        if lock_ts is not None:
            all_ts.append(lock_ts)
        last_activity = max(all_ts) if all_ts else None

        fleet_done += total_done
        workers.append(WorkerHeartbeat(
            worker_id=hostname,
            hostname=hostname,
            locks=host_locks,
            done=total_done,
            last_activity_ts=last_activity,
            pace_per_hour=pace,
        ))

    return FleetHeartbeat(
        workers=workers,
        fleet_locks=fleet_locks,
        fleet_done=fleet_done,
        fleet_pace_per_hour=fleet_pace_sum if fleet_pace_count else None,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _relative_time(epoch: float | None) -> str:
    if epoch is None:
        return "--"
    delta = time.time() - epoch
    if delta < 0:
        return "just now"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{delta / 3600:.1f}h ago"
    return f"{delta / 86400:.1f}d ago"


def format_heartbeat(
    heartbeat: FleetHeartbeat,
    disk_summary: object | None = None,
) -> str:
    """Render a compact fleet dashboard.

    *disk_summary* should be a ``progress.QuickSummary`` instance (or
    ``None`` to omit the footer).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = f"Fleet heartbeat{' ' * 34}{now}"

    col_w = "Worker"
    col_l = "Locks"
    col_d = "Done"
    col_a = "Last activity"
    col_p = "Pace (arc/h)"

    rows: list[tuple[str, str, str, str, str]] = []
    for w in heartbeat.workers:
        pace_str = f"~{w.pace_per_hour:.1f}" if w.pace_per_hour is not None else "--"
        rows.append((
            w.worker_id,
            str(w.locks),
            f"{w.done:,}",
            _relative_time(w.last_activity_ts),
            pace_str,
        ))

    w_widths = [
        max(len(col_w), *(len(r[0]) for r in rows)) if rows else len(col_w),
        max(len(col_l), *(len(r[1]) for r in rows)) if rows else len(col_l),
        max(len(col_d), *(len(r[2]) for r in rows)) if rows else len(col_d),
        max(len(col_a), *(len(r[3]) for r in rows)) if rows else len(col_a),
        max(len(col_p), *(len(r[4]) for r in rows)) if rows else len(col_p),
    ]

    def _fmt_row(vals: tuple[str, str, str, str, str]) -> str:
        return (
            f"{vals[0]:<{w_widths[0]}}  "
            f"{vals[1]:>{w_widths[1]}}  "
            f"{vals[2]:>{w_widths[2]}}  "
            f"{vals[3]:>{w_widths[3]}}  "
            f"{vals[4]:>{w_widths[4]}}"
        )

    lines: list[str] = [title, "=" * len(title), ""]

    if not heartbeat.workers:
        lines.append("No active workers detected.")
    else:
        header = _fmt_row((col_w, col_l, col_d, col_a, col_p))
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)
        for r in rows:
            lines.append(_fmt_row(r))
        lines.append(sep)

        n_workers = len(heartbeat.workers)
        fleet_pace = (
            f"~{heartbeat.fleet_pace_per_hour:.1f}"
            if heartbeat.fleet_pace_per_hour is not None
            else "--"
        )
        fleet_label = f"Fleet ({n_workers} worker{'s' if n_workers != 1 else ''})"
        fleet_row = _fmt_row((
            fleet_label,
            str(heartbeat.fleet_locks),
            f"{heartbeat.fleet_done:,}",
            "",
            fleet_pace,
        ))
        lines.append(fleet_row)

    if disk_summary is not None:
        lines.append("")
        parts = [f"Completed: {disk_summary.complete:,}"]
        parts.append(f"Partial: {disk_summary.partial:,}")
        err_parts: list[str] = []
        if disk_summary.audio_error_records:
            err_parts.append(f"{disk_summary.audio_error_records} audio")
        if disk_summary.inference_error_records:
            err_parts.append(f"{disk_summary.inference_error_records} inference")
        if err_parts:
            parts.append(f"Errors: {', '.join(err_parts)}")
        else:
            parts.append("Errors: 0")
        lines.append("  |  ".join(parts))

    lines.append("")
    return "\n".join(lines)
