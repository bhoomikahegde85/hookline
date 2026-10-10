#!/usr/bin/env python3
"""Open the editing assistant over http://127.0.0.1 (use this if the video does not play when you open review.html directly).

    python3 scripts/serve_review.py FC002 --artifacts artifacts/e2e --video 01-Hookline/face/videos/FC002.mp4 --open

Serves only the review page and the one video, on this machine only (127.0.0.1), with seeking support. Press Ctrl+C to stop.
Nothing is uploaded. Stdlib only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.dropoffs import write_dropoffs  # noqa: E402
from hookline.review_page import build_review  # noqa: E402
from hookline.review_server import start_server  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    ap.add_argument("--video", type=Path, required=True, help="the video file to play")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    d = args.artifacts / "features" / args.video_id
    if write_dropoffs(d) is None:
        print(f"no successful TTCC prediction for {args.video_id} in {d}; run scripts/run_e2e.py first", file=sys.stderr)
        return 1
    page = build_review(d, video)
    try:
        server, port = start_server(page, video, args.port)
    except OSError as exc:
        print(f"could not start on port {args.port}: {exc}. Try --port 0 style: pick another port.", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"editing assistant: {url}\n(serving only this page and the video, on this machine; Ctrl+C to stop)")
    if args.open:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", url], check=False)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
