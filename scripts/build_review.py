#!/usr/bin/env python3
"""(Re)build the local review page for a video from its saved artifacts: no Space, no model, no upload.

    python3 scripts/build_review.py FC002 --artifacts artifacts/tribe_test --video 01-Hookline/face/videos/FC002.mp4 --open
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.dropoffs import write_dropoffs  # noqa: E402
from hookline.review_page import build_review  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    ap.add_argument("--video", type=Path, help="the video file the page plays (default: the source path stored with the prediction)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser (macOS: open)")
    args = ap.parse_args()
    d = args.artifacts / "features" / args.video_id
    if write_dropoffs(d) is None:
        print(f"no successful TTCC prediction for {args.video_id} in {d}; run scripts/run_ttcc.py or scripts/run_e2e.py first", file=sys.stderr)
        return 1
    page = build_review(d, args.video)
    print(f"review page: {page.resolve()}")
    if args.open:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(page)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
