"""Review interface for inspecting and labeling acoustic events.

Quick start::

    from audio_classification_playground.acoustic_events import (
        extract_events, tracks_from_signals, Config, Signal, Vad,
    )
    from audio_classification_playground.acoustic_events.review import (
        save_session, launch_review,
    )

    events = extract_events(signals, vad, Config.balanced())
    tracks = tracks_from_signals(signals)
    session_path = save_session(
        events=events,
        tracks=tracks,
        vad=vad,
        config=Config.balanced(),
        audio_path="/path/to/audio.mp3",
        session_dir="./labeling",
    )
    launch_review(session_path)        # blocks; opens server on 127.0.0.1:8765
"""
from __future__ import annotations

from pathlib import Path

from .inherit import inherit_labels
from .models import VERDICTS, Label, LabelingSession
from .server import make_app
from .storage import (
    clear_label,
    config_hash,
    list_sessions,
    load_session,
    load_session_json,
    save_session,
    session_fingerprint,
    update_label,
)


def launch_review(
    session_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    block: bool = True,
) -> "object | None":
    """Start the review server.

    Defaults to blocking (``uvicorn.run``). Set ``block=False`` to spawn the
    server on a daemon thread and return control to the caller — useful from
    a notebook cell, but be aware that the server stops when the parent
    process exits.
    """
    import uvicorn

    from .cli import _print_banner

    session_path = Path(session_path).resolve()
    app = make_app(session_path)
    _print_banner(host, port, session_path)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    if block:
        server.run()
        return None

    import threading

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


__all__ = [
    "VERDICTS",
    "Label",
    "LabelingSession",
    "save_session",
    "load_session",
    "load_session_json",
    "list_sessions",
    "update_label",
    "clear_label",
    "inherit_labels",
    "config_hash",
    "session_fingerprint",
    "make_app",
    "launch_review",
]
