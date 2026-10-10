#!/usr/bin/env python3
"""The whole pipeline for one video, in one command, ending in a local review page.

    python3 scripts/run_e2e.py /path/to/video.mp4 --allow-remote                    # your own clip
    python3 scripts/run_e2e.py 01-Hookline/face/videos/FC002.mp4 --allow-remote --approve FC002 --open

Steps (each prints its status; a failed step stops the run and says what to fix):
  1. features      OpenCV / FFmpeg / audio features                         (.venv, local)
  2. speech + text Whisper, PaddleOCR, librosa (+ SigLIP with --siglip)      (.venv-models, local, cached models)
  3. retention     TTCC on YOUR Hugging Face Space (needs TTCC_ENDPOINT_URL, HF_TOKEN in .env; uploads the video)
  4. segments      drop-off detection + evidence + rule-based suggestions    (local, from the saved files)
  5. review page   artifacts/<root>/features/<id>/review.html: jump to timestamps, assess each suggestion

The video is uploaded to your Space only with --allow-remote; company reels under 01-Hookline/ also need --approve <video_id>
(only after the organisers allow it). The retention curve is a pretrained model's prediction, not measured retention,
and has not been validated on these videos. Nothing runs a model on this machine except Whisper/OCR/SigLIP in step 2.
Stdlib only: run it with any python3; it calls the project's own environments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV, VENV_MODELS = PROJECT_ROOT / ".venv" / "bin" / "python", PROJECT_ROOT / ".venv-models" / "bin" / "python"


def plan(video: Path, root: Path, *, allow_remote: bool, approve: str | None, siglip: bool, skip_local: bool, open_page: bool,
         gemini: bool = False, gemini_allow_remote: bool = False) -> list[dict]:
    """The ordered steps as commands. A step with ``skip`` is reported and not run."""
    vid = video.stem
    local_common = ["--input", str(video), "--artifacts", str(root)]
    steps = [
        {"name": "features", "cmd": [str(VENV), "scripts/run_pipeline.py", *local_common], "skip": "skipped (--skip-local)" if skip_local else None},
        {"name": "speech + text + audio", "cmd": [str(VENV_MODELS), "scripts/run_speech_ocr.py", *local_common, "--ocr-fps", "1", *(["--siglip"] if siglip else [])],
         "skip": "skipped (--skip-local)" if skip_local else None},
        {"name": "retention (TTCC, remote Space)",
         "cmd": [str(VENV), "scripts/run_ttcc.py", *local_common, "--features", "reuse", "--enable-ttcc", "--ttcc-remote", "--ttcc-allow-remote",
                 *(["--ttcc-approve", approve] if approve else [])],
         "skip": None if allow_remote else "not run: the video is uploaded to your Space only with --allow-remote"},
        *([{"name": "explanation + edit suggestions (Gemini, uploads the video to Google)", "optional": True,
            "cmd": [str(VENV_MODELS), "scripts/run_gemini_edit.py", vid, "--artifacts", str(root), "--video", str(video), "--gemini-allow-remote",
                    *(["--gemini-approve", approve] if approve else [])],
            "skip": None if gemini_allow_remote else "not run: the video is uploaded to Google only with --gemini-allow-remote"}] if gemini else []),
        {"name": "segments + suggestions + review page",
         "cmd": [str(VENV), "scripts/build_review.py", vid, "--artifacts", str(root), "--video", str(video), *(["--open"] if open_page else [])], "skip": None},
    ]
    return steps


def run_e2e(video: Path, root: Path, run=subprocess.run, log=print, **opts) -> int:
    steps = plan(video, root, **opts)
    results = []
    started = time.time()
    for n, step in enumerate(steps, 1):
        label = f"[{n}/{len(steps)}] {step['name']}"
        if step["skip"]:
            log(f"{label}: {step['skip']}")
            results.append((step["name"], "skipped"))
            if n == 3 and not step.get("optional"):  # no retention curve, so the segment and review steps have nothing to work on
                log("stopping: the segment analysis needs the TTCC curve (step 3)")
                break
            continue
        log(f"{label} ...")
        t0 = time.time()
        proc = run(step["cmd"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-4:]
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append((step["name"], status))
        log(f"{label}: {status} in {time.time() - t0:.0f} s")
        for ln in tail:
            log("      " + ln[:240])
        if proc.returncode != 0:
            log(hint(n, proc.returncode, out))
            if step.get("optional"):
                log("      (optional step; continuing without it)")
                continue
            break
    ok = all(s in ("ok", "skipped") or "FAILED" in s and name_is_optional(steps, name) for name, s in results) and len(results) == len(steps)
    page = root / "features" / video.stem / "review.html"
    log(f"\nsummary ({time.time() - started:.0f} s): " + "; ".join(f"{name}: {status}" for name, status in results))
    if ok and page.is_file():
        log(f"review page: {page.resolve()}\nopen it with:  open '{page.resolve()}'")
    return 0 if ok else 1


def name_is_optional(steps: list[dict], name: str) -> bool:
    return any(s["name"] == name and s.get("optional") for s in steps)


def hint(step: int, code: int, out: str) -> str:
    if step == 3:
        if "not reachable" in out or "still downloading" in out or "unavailable" in out:
            return "  -> the Space is not ready or not reachable: check its status and /health, then run again (cached steps are reused)"
        if "not configured" in out:
            return "  -> add TTCC_ENDPOINT_URL and HF_TOKEN to your .env"
        if "--ttcc-approve" in out:
            return "  -> company reel: pass --approve <video_id> (only if the organisers allow uploading it)"
        return "  -> see the lines above; the Space's Container logs hold more detail"
    if step in (1, 2):
        return "  -> a local step failed; check the lines above (is the right environment installed? see the README)"
    return "  -> see the lines above"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "e2e", help="output root (default artifacts/e2e)")
    ap.add_argument("--allow-remote", action="store_true", help="allow uploading the video to your TTCC Space (step 3)")
    ap.add_argument("--approve", metavar="VIDEO_ID", help="approve uploading this company reel (under 01-Hookline/)")
    ap.add_argument("--siglip", action="store_true", help="also run SigLIP in step 2 (model must be cached)")
    ap.add_argument("--skip-local", action="store_true", help="skip steps 1-2 (their artifacts already exist in --artifacts)")
    ap.add_argument("--gemini", action="store_true", help="add the Gemini explanation step (optional; uploads the video to Google)")
    ap.add_argument("--gemini-allow-remote", action="store_true", help="allow uploading the video to Google (needed for --gemini)")
    ap.add_argument("--open", action="store_true", help="open the review page when done")
    args = ap.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    root = args.artifacts if args.artifacts.is_absolute() else PROJECT_ROOT / args.artifacts
    return run_e2e(video, root, allow_remote=args.allow_remote, approve=args.approve, siglip=args.siglip, skip_local=args.skip_local,
                   open_page=args.open, gemini=args.gemini, gemini_allow_remote=args.gemini_allow_remote)


if __name__ == "__main__":
    sys.exit(main())
