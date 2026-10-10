#!/usr/bin/env python3
"""Gemini explains the TTCC drop-off segments of a video and proposes timestamped edits (hypotheses), then the review page is rebuilt.

    .venv-models/bin/python scripts/run_gemini_edit.py FC002 --artifacts artifacts/e2e \
        --video 01-Hookline/face/videos/FC002.mp4 --gemini-allow-remote [--gemini-approve FC002]

THE VIDEO IS UPLOADED TO GOOGLE. Nothing is sent without --gemini-allow-remote; company reels under 01-Hookline/ also need
--gemini-approve <video_id> (only after the organisers allow it). The key is read from GEMINI_API_KEY (environment or the untracked
.env) and is never printed or stored. The uploaded file is deleted from Google after the call. A result with the same video,
settings and evidence is reused (no call, no cost) unless --force.

Needs the TTCC prediction first (scripts/run_e2e.py or scripts/run_ttcc.py) and the models environment (google-genai).
Gemini explains; TTCC predicts. Everything returned is a hypothesis until validated against real audience-retention data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.config import DEFAULT_GEMINI_MODEL, GeminiConfig  # noqa: E402
from hookline.dropoffs import write_dropoffs  # noqa: E402
from hookline.gemini_edit import cached_gemini_edit, extract_gemini_edit, write_gemini_edit  # noqa: E402
from hookline.review_page import build_review  # noqa: E402
from hookline.segment_recommendations import write_segment_recommendations  # noqa: E402
from hookline.tribe import load_secret  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gemini-allow-remote", action="store_true", help="required: the video is uploaded to Google")
    ap.add_argument("--gemini-approve", action="append", default=[], metavar="VIDEO_ID", help="approve uploading this company reel; repeatable")
    ap.add_argument("--gemini-model", default=None, help=f"model id (default: $GEMINI_MODEL, else {DEFAULT_GEMINI_MODEL})")
    ap.add_argument("--gemini-fps", type=float, default=GeminiConfig.fps, help="frames per second Gemini samples (more = more tokens)")
    ap.add_argument("--gemini-resolution", choices=["low", "medium", "high"], default=GeminiConfig.media_resolution)
    ap.add_argument("--max-analysed-s", type=float, default=GeminiConfig.max_analysed_s,
                    help="most seconds of video frames sent to Gemini (cost cap); a longer video is sent as the opening plus the flagged segments")
    ap.add_argument("--context-margin-s", type=float, default=GeminiConfig.context_margin_s, help="seconds of context around each flagged segment")
    ap.add_argument("--hook-s", type=float, default=GeminiConfig.hook_s, help="opening seconds always included")
    ap.add_argument("--force", action="store_true", help="call Gemini again even if an identical result exists")
    args = ap.parse_args()
    d = args.artifacts / "features" / args.video_id
    video = args.video.expanduser().resolve()
    if write_dropoffs(d) is None:
        print(f"no TTCC drop-off analysis for {args.video_id} in {d}; run scripts/run_e2e.py first", file=sys.stderr)
        return 1
    try:
        cfg = GeminiConfig(mode="remote", model=args.gemini_model or load_secret("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                           allow_remote=args.gemini_allow_remote, approved_ids=tuple(args.gemini_approve), fps=args.gemini_fps,
                           media_resolution=args.gemini_resolution, max_analysed_s=args.max_analysed_s,
                           context_margin_s=args.context_margin_s, hook_s=args.hook_s)
    except ValueError as exc:
        ap.error(str(exc))
    art = None if args.force else cached_gemini_edit(d, cfg)
    if art:
        print(f"reusing the earlier Gemini result (same video, settings and evidence): {art['validation']['dropped']} item(s) dropped, no call made")
    else:
        art = extract_gemini_edit(d, video, cfg)
        written = write_gemini_edit(d, art)
        print(f"gemini ({cfg.model}): status={art['status']} sent={art['sent']}" + (f" tokens={art['usage']}" if art.get("usage") else "")
              + (f" in {art['seconds']} s" if art.get("seconds") else ""))
        scope = art.get("analysis_scope")
        if scope:
            print(f"analysed {scope['analysed_s']:g} s of the {scope['predicted_span_s']:g} s predicted span in {len(scope['clips_s'])} clip(s) "
                  f"(~{scope['frames_sampled_approx']} frames at {scope['fps']:g} fps)" + ("" if scope["whole_predicted_span"] else
                  f"; {len(scope['not_analysed'])} segment(s) not analysed"))
        if art.get("reason"):
            print(f"reason: {art['reason']}")
        if not written:
            print("(an earlier successful result was kept)")
    rec_path = write_segment_recommendations(d)
    page = build_review(d, video)
    if art.get("analysis"):
        v = art.get("validation") or {}
        total = sum(len(s["suggestions"]) for s in art["analysis"]["segments"])
        print(f"{len(art['analysis']['segments'])} segment(s) interpreted, {total} suggestion(s) kept, {v.get('dropped', 0)} dropped by validation")
        for issue in (v.get("issues") or [])[:6]:
            print("  validation:", issue)
    print(f"segment recommendations: {rec_path}\nreview page: {page}")
    return 0 if art.get("measured") else 1


if __name__ == "__main__":
    sys.exit(main())
