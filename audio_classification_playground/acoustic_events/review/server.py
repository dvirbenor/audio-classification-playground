"""FastAPI application backing the package-oriented review UI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..composition.package import (
    clear_package_label,
    load_review_package,
    update_package_label,
)
from .audio_serving import serve_with_range
from .models import VERDICTS, Label
from .waveform import cached_peaks, compute_peaks_window


_STATIC_DIR = Path(__file__).parent / "static"


def make_app(package_path: str | Path) -> FastAPI:
    package_path = Path(package_path).resolve()
    if not package_path.is_dir():
        raise FileNotFoundError(f"Review package directory not found: {package_path}")

    app = FastAPI(title="Acoustic Events Review", version="0.1.0")

    # In-memory state, refreshed on demand from disk so external edits propagate.
    state: dict[str, Any] = {"path": package_path, "package": load_review_package(package_path)}

    def _reload():
        state["package"] = load_review_package(state["path"])
        return state["package"]

    # ---- Static UI -------------------------------------------------------
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR)), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    # ---- Session metadata + events --------------------------------------
    @app.get("/api/session")
    async def get_session() -> JSONResponse:
        pkg = state["package"]
        package = pkg.package
        audio = package["audio"]
        # Strip bulky track arrays — they live behind /api/tracks.
        return JSONResponse({
            "session_id": package["package_id"],
            "schema": package.get("schema", "review_package.v1"),
            "schema_version": None,
            "event_schema": "acoustic_events.v1",
            "package_id": package["package_id"],
            "package_fingerprint": package["package_fingerprint"],
            "recording_id": package["recording_id"],
            "audio_sr": audio["sample_rate"],
            "audio_duration_sec": audio["duration_sec"],
            "session_fingerprint": package["package_id"],
            "producer_runs": package.get("producer_runs", []),
            "tracks_meta": package.get("tracks_meta", {}),
            "vad_intervals": package.get("vad_intervals", []),
            "events": package.get("events", []),
            "labels": pkg.labels,
            "verdicts": list(VERDICTS),
            "created_at": "",
            "last_updated_at": "",
            "notes": "",
        })

    # ---- Signals (full arrays, served once on page load) -----------------
    @app.get("/api/tracks")
    async def get_tracks() -> JSONResponse:
        pkg = state["package"]
        meta = pkg.package.get("tracks_meta", {})
        payload = {}
        for track_id, track_meta in meta.items():
            if track_meta.get("kind") == "marker":
                payload[track_id] = track_meta.get("items", [])
            else:
                data_path = track_meta.get("data_path")
                if not data_path:
                    payload[track_id] = []
                    continue
                npz_path = pkg.path / data_path
                if not npz_path.is_file():
                    raise HTTPException(404, f"tracks data not found at {npz_path}")
                with np.load(npz_path) as arrays:
                    payload[track_id] = (
                        arrays[track_id].astype(np.float32).tolist()
                        if track_id in arrays.files
                        else []
                    )
        return JSONResponse({"tracks": payload, "meta": meta})

    # ---- Audio (range-served) -------------------------------------------
    @app.get("/api/audio")
    async def get_audio(request: Request):
        pkg = state["package"]
        return serve_with_range(pkg.package["audio"]["path"], request)

    # ---- Waveform peaks (cached on disk, or windowed high-res) -----------
    @app.get("/api/waveform")
    async def get_waveform(
        t0: float | None = None,
        t1: float | None = None,
        n_peaks: int | None = None,
    ) -> JSONResponse:
        pkg = state["package"]
        audio_path = pkg.package["audio"]["path"]
        if t0 is not None and t1 is not None:
            capped = min(n_peaks or 2000, 4000)
            peaks = compute_peaks_window(audio_path, t0, t1, n_peaks=capped)
        else:
            cache_path = state["path"] / "waveform.peaks.json"
            peaks = cached_peaks(audio_path, cache_path)
        return JSONResponse(peaks)

    # ---- Labels ---------------------------------------------------------
    @app.post("/api/label/{event_id}")
    async def post_label(event_id: str, body: dict) -> JSONResponse:
        verdict = body.get("verdict", "")
        if verdict and verdict not in VERDICTS:
            raise HTTPException(400, f"verdict must be one of {VERDICTS} or empty")
        label = Label(
            verdict=verdict,
            tags=list(body.get("tags") or []),
            comment=body.get("comment", "") or "",
            labeler=body.get("labeler", "") or "",
        )
        try:
            payload = update_package_label(state["path"], event_id, label.to_dict())
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        _reload()
        return JSONResponse(payload)

    @app.delete("/api/label/{event_id}")
    async def delete_label(event_id: str) -> JSONResponse:
        clear_package_label(state["path"], event_id)
        _reload()
        return JSONResponse({"ok": True})

    # ---- Health ---------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "package": state["package"].package.get("package_id")}

    return app
