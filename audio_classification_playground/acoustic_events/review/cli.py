"""Entry point: ``python -m audio_classification_playground.acoustic_events.review --package ...``.

The CLI is the canonical way to start the review server. Designed to work
on a remote pod with VS Code port forwarding: bind defaults to 127.0.0.1 on
a fixed port (8765), which VS Code auto-detects.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .server import make_app


def _print_banner(host: str, port: int, package_path: Path) -> None:
    url = f"http://{host}:{port}/"
    print()
    print("=" * 64)
    print("  Acoustic Events Review")
    print("=" * 64)
    print(f"  package : {package_path}")
    print(f"  url     : {url}")
    print()
    print("  In VS Code (remote): the Ports panel will forward this port")
    print("  automatically. Click the 'Open in Browser' icon, or use")
    print("  Cmd/Ctrl+Shift+P → 'Simple Browser: Show' and paste the URL.")
    print("=" * 64)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acoustic-events-review",
        description="Launch the acoustic-events review UI for a review package.",
    )
    parser.add_argument("--package", required=True, type=Path, help="Path to review package directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development).",
    )
    args = parser.parse_args(argv)

    package_path = args.package.resolve()
    if not package_path.is_dir():
        print(f"error: review package directory not found: {package_path}", file=sys.stderr)
        return 2

    import uvicorn

    app = make_app(package_path)
    _print_banner(args.host, args.port, package_path)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
