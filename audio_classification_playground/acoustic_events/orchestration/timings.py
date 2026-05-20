"""Load, summarise, and format per-archive timing records.

Timing records are written by the worker as append-only JSONL files
under ``_meta/timings/``.  Each worker process writes its own file,
so reading requires globbing all ``*.jsonl`` files in that directory.

This module provides:

* ``load_timing_records`` — read and merge all JSONL files.
* ``summarize_timings`` — compute distributional stats per numeric field.
* ``summarize_timings_by_worker`` — group-by-worker variant.
* ``derive_vad_mode`` — compute ``prefetched``/``cached``/``inline``
  from the stored booleans.
* ``format_timing_summary`` — render a human-readable table.
* ``format_timing_csv`` — render as CSV for notebook/spreadsheet use.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

TIMINGS_DIR = "_meta/timings"

DEFAULT_TIMING_FIELDS: tuple[str, ...] = (
    "audio_duration_sec",
    "prefetch_scheduler_wait_sec",
    "prefetch_get_wait_sec",
    "prefetch_wait_sec",
    "decode_queue_wait_sec",
    "download_decode_sec",
    "vad_queue_wait_sec",
    "vad_precompute_sec",
    "prefetch_submit_to_ready_sec",
    "prefetch_ready_age_sec",
    "vad_sec",
    "affect_sec",
    "disfluency_sec",
    "emotion_sec",
    "inference_sec",
    "total_sec",
)


@dataclass(frozen=True)
class FieldStats:
    """Distributional statistics for a single numeric field."""

    count: int
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_timing_records(output_base: str | Path) -> list[dict]:
    """Read all JSONL timing files under ``_meta/timings/``.

    Corrupt lines are logged and skipped.
    """
    timings_dir = Path(output_base) / TIMINGS_DIR
    if not timings_dir.is_dir():
        return []

    records: list[dict] = []
    for jsonl_path in sorted(timings_dir.glob("*.jsonl")):
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        LOGGER.warning(
                            "Skipping corrupt line %d in %s", line_no, jsonl_path,
                        )
        except OSError:
            LOGGER.warning("Could not read %s", jsonl_path, exc_info=True)
    return records


# ---------------------------------------------------------------------------
# VAD mode derivation
# ---------------------------------------------------------------------------


def derive_vad_mode(record: dict) -> str:
    """Derive the VAD execution mode from stored booleans.

    Returns one of ``"prefetched"``, ``"cached"``, or ``"inline"``.
    """
    if record.get("precomputed_vad", False):
        return "prefetched"
    if record.get("vad_reused", False):
        return "cached"
    return "inline"


# ---------------------------------------------------------------------------
# Percentile helpers (stdlib-only, nearest-rank)
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    idx = max(0, min(int(math.ceil(p / 100.0 * len(sorted_values))) - 1,
                     len(sorted_values) - 1))
    return sorted_values[idx]


def _compute_field_stats(values: list[float]) -> FieldStats:
    if not values:
        return FieldStats(
            count=0, mean=0.0, std=0.0, min=0.0,
            p25=0.0, p50=0.0, p75=0.0, p90=0.0, p95=0.0, p99=0.0, max=0.0,
        )
    n = len(values)
    s = sorted(values)
    mean = sum(s) / n
    if n <= 1:
        std = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in s) / (n - 1)
        std = math.sqrt(variance)
    return FieldStats(
        count=n,
        mean=mean,
        std=std,
        min=s[0],
        p25=_percentile(s, 25),
        p50=_percentile(s, 50),
        p75=_percentile(s, 75),
        p90=_percentile(s, 90),
        p95=_percentile(s, 95),
        p99=_percentile(s, 99),
        max=s[-1],
    )


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def summarize_timings(
    records: list[dict],
    fields: tuple[str, ...] | None = None,
) -> dict[str, FieldStats]:
    """Compute distributional stats for each numeric timing field."""
    fields = fields or DEFAULT_TIMING_FIELDS
    columns: dict[str, list[float]] = {f: [] for f in fields}
    for rec in records:
        for f in fields:
            v = rec.get(f)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                columns[f].append(float(v))
    return {f: _compute_field_stats(columns[f]) for f in fields}


def summarize_timings_by_worker(
    records: list[dict],
    fields: tuple[str, ...] | None = None,
) -> dict[str, dict[str, FieldStats]]:
    """Group records by ``worker_id`` and summarise each group."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        wid = rec.get("worker_id", "unknown")
        groups.setdefault(wid, []).append(rec)
    return {wid: summarize_timings(recs, fields) for wid, recs in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_STAT_COLUMNS = ("count", "mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max")


def format_timing_summary(
    summary: dict[str, FieldStats],
    title: str | None = None,
) -> str:
    """Render a human-readable table of timing statistics."""
    if not summary:
        return "No timing data.\n"

    header_parts = [f"{'field':<22s}"]
    for col in _STAT_COLUMNS:
        header_parts.append(f"{col:>8s}")
    header = " ".join(header_parts)

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    lines.append(header)
    lines.append("-" * len(header))

    for field_name, stats in summary.items():
        parts = [f"{field_name:<22s}"]
        for col in _STAT_COLUMNS:
            val = getattr(stats, col)
            if col == "count":
                parts.append(f"{val:>8d}")
            else:
                parts.append(f"{val:>8.3f}")
        lines.append(" ".join(parts))

    lines.append("")
    return "\n".join(lines)


def format_timing_csv(
    records: list[dict],
    fields: tuple[str, ...] | None = None,
) -> str:
    """Render timing records as CSV with grouping columns."""
    fields = fields or DEFAULT_TIMING_FIELDS
    columns = [
        "worker_id",
        "task_group",
        "session_id",
        "archive_id",
        "ts",
        "vad_mode",
        "s3_key",
        "audio_source_extension",
        "audio_object_size_bytes",
        "audio_storage_class",
        *fields,
    ]
    lines = [",".join(columns)]
    for rec in records:
        row: list[str] = []
        for col in columns:
            if col == "vad_mode":
                row.append(derive_vad_mode(rec))
            else:
                v = rec.get(col, "")
                row.append(str(v))
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"
