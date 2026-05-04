"""Canonical JSON helpers for deterministic review packages."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def jsonable(value: Any) -> Any:
    """Convert dataclasses, numpy values, tuples, and paths to JSON values."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(payload: Mapping) -> bytes:
    """Return compact deterministic JSON bytes for hashing."""
    return json.dumps(
        jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_text(payload: Mapping) -> str:
    """Return readable deterministic JSON for files."""
    return json.dumps(
        jsonable(payload),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
