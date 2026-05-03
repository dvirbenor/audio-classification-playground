"""FastAPI application backing the review UI.

The app holds one in-memory session at a time (loaded from disk at startup)
and persists every label change immediately. State is intentionally minimal;
all business logic lives in :mod:`storage`, :mod:`inherit`, and :mod:`waveform`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio_serving import serve_with_range
from .models import VERDICTS, Label
from .storage import clear_label, load_session_json, update_label
from .waveform import cached_peaks, compute_peaks_window


_STATIC_DIR = Path(__file__).parent / "static"


def make_app(session_path: str | Path) -> FastAPI:
    session_path = Path(session_path).resolve()
    if not session_path.is_file():
        raise FileNotFoundError(f"Session not found: {session_path}")

    app = FastAPI(title="Affective Events Review", version="0.1.0")

    # In-memory state, refreshed on demand from disk so external edits propagate.
    state: dict[str, Any] = {"path": session_path, "session": load_session_json(session_path)}

    def _reload() -> dict:
        state["session"] = load_session_json(state["path"])
        return state["session"]

    # ---- Static UI -------------------------------------------------------
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR)), name="assets")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    # ---- Session metadata + events --------------------------------------
    @app.get("/api/session")
    async def get_session() -> JSONResponse:
        sess = state["session"]
        # Strip bulky track arrays — they live behind /api/tracks.
        return JSONResponse({
            "session_id": sess["session_id"],
            "schema_version": sess.get("schema_version"),
            "event_schema": sess.get("event_schema", "acoustic_events.v1"),
            "recording_id": sess["recording_id"],
            "audio_sr": sess["audio_sr"],
            "audio_duration_sec": sess["audio_duration_sec"],
            "session_fingerprint": sess.get("session_fingerprint", ""),
            "producer_runs": sess.get("producer_runs", []),
            "tracks_meta": sess.get("tracks_meta", {}),
            "vad_intervals": sess["vad_intervals"],
            "events": sess["events"],
            "labels": sess.get("labels", {}),
            "verdicts": list(VERDICTS),
            "created_at": sess.get("created_at", ""),
            "last_updated_at": sess.get("last_updated_at", ""),
            "notes": sess.get("notes", ""),
        })

    # ---- Signals (full arrays, served once on page load) -----------------
    @app.get("/api/tracks")
    async def get_tracks() -> JSONResponse:
        sess = state["session"]
        npz_path = state["path"].parent / sess["tracks_data_path"]
        if not npz_path.is_file():
            raise HTTPException(404, f"tracks data not found at {npz_path}")
        arrays = np.load(npz_path)
        meta = sess.get("tracks_meta", {})
        payload = {}
        for track_id, track_meta in meta.items():
            if track_meta.get("kind") == "marker":
                payload[track_id] = track_meta.get("items", [])
            elif track_id in arrays.files:
                payload[track_id] = arrays[track_id].astype(np.float32).tolist()
            else:
                payload[track_id] = []
        return JSONResponse({"tracks": payload, "meta": meta})

    # ---- Audio (range-served) -------------------------------------------
    @app.get("/api/audio")
    async def get_audio(request: Request):
        sess = state["session"]
        return serve_with_range(sess["audio_path"], request)

    # ---- Waveform peaks (cached on disk, or windowed high-res) -----------
    @app.get("/api/waveform")
    async def get_waveform(
        t0: float | None = None,
        t1: float | None = None,
        n_peaks: int | None = None,
    ) -> JSONResponse:
        sess = state["session"]
        if t0 is not None and t1 is not None:
            capped = min(n_peaks or 2000, 4000)
            peaks = compute_peaks_window(sess["audio_path"], t0, t1, n_peaks=capped)
        else:
            cache_path = state["path"].with_suffix(".peaks.json")
            peaks = cached_peaks(sess["audio_path"], cache_path)
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
            payload = update_label(state["path"], event_id, label)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        _reload()
        return JSONResponse(payload)

    @app.delete("/api/label/{event_id}")
    async def delete_label(event_id: str) -> JSONResponse:
        clear_label(state["path"], event_id)
        _reload()
        return JSONResponse({"ok": True})

    # ---- Health ---------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "session": state["session"].get("session_id")}

    return app
