"""HTTP range-aware audio streaming.

Browsers issue ``Range: bytes=...`` requests when seeking in an ``<audio>``
element. Starlette's ``FileResponse`` ignores them, so seeking on a long
file becomes pathological. This module implements RFC 7233 single-range
responses, which is all browsers need in practice.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import Response, StreamingResponse


_CHUNK = 64 * 1024
_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _iter_file(path: Path, start: int, length: int):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            chunk = f.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def serve_with_range(audio_path: str | Path, request: Request) -> Response:
    """Serve ``audio_path`` honoring an optional ``Range`` request header."""
    path = Path(audio_path)
    if not path.is_file():
        return Response(status_code=404, content=b"audio not found")
    file_size = os.path.getsize(path)
    media_type = _media_type(path)

    range_header = request.headers.get("range") or request.headers.get("Range")
    if not range_header:
        return StreamingResponse(
            _iter_file(path, 0, file_size),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    units, _, rng = range_header.partition("=")
    if units.strip().lower() != "bytes" or not rng:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    start_str, _, end_str = rng.split(",", 1)[0].partition("-")
    try:
        if start_str:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
        else:  # suffix range: bytes=-N
            suffix = int(end_str)
            start = max(0, file_size - suffix)
            end = file_size - 1
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    length = end - start + 1
    return StreamingResponse(
        _iter_file(path, start, length),
        media_type=media_type,
        status_code=206,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )
