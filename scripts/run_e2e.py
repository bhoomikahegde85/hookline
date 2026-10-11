#!/usr/bin/env python3
"""The whole pipeline for one video, in one command, ending in a local review page.

    python3 scripts/run_e2e.py /path/to/video.mp4 --allow-remote                    # your own clip
    python3 scripts/run_e2e.py 01-Hookline/face/videos/FC002.mp4 --allow-remote --approve FC002 --open

Steps (each prints its status; a failed step stops the run and says what to fix):
  1. features      OpenCV / FFmpeg / audio features                         (.venv, local)
  2. speech + text Whisper, PaddleOCR, librosa (+ SigLIP with --siglip)      (.venv-models, local, cached models)
  3. retention     TTCC on YOUR Hugging Face Space (needs TTCC_ENDPOINT_URL, HF_TOKEN in .env; uploads the video)
  4. segments      drop-off detection + evidence + rule-based suggestions    (local, from the saved files)
  5. scores        0-100 hook / breakout positions with the measured accuracy of their models (required step, local)
  6. explain       why the hook is strong or weak; the strongest and weakest moments (required step, local)
  7. review page   artifacts/<root>/features/<id>/review.html: jump to timestamps, assess each suggestion

The video is uploaded to your Space only with --allow-remote; company reels under 01-Hookline/ also need --approve <video_id>
(only after the organisers allow it). The retention curve is a pretrained model's prediction, not measured retention,
and has not been validated on these videos. Nothing runs a model on this machine except Whisper/OCR/SigLIP in step 2.
Stdlib only: run it with any python3; it calls the project's own environments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV, VENV_MODELS = PROJECT_ROOT / ".venv" / "bin" / "python", PROJECT_ROOT / ".venv-models" / "bin" / "python"
VENV_DEMUCS = PROJECT_ROOT / ".venv-demucs" / "bin" / "python"


def plan(video: Path, root: Path, *, allow_remote: bool, approve: str | None, siglip: bool, skip_local: bool, open_page: bool,
         gemini_allow_remote: bool = False, hook_s: int | None = None, full_curve: bool = False) -> list[dict]:
    """The ordered steps as commands. A step with ``skip`` is reported and not run."""
    vid = video.stem
    local_common = ["--input", str(video), "--artifacts", str(root)]
    steps = [
        {"name": "features", "cmd": [str(VENV), "scripts/run_pipeline.py", *local_common], "skip": "skipped (--skip-local)" if skip_local else None},
        {"name": "speech + text + audio", "cmd": [str(VENV_MODELS), "scripts/run_speech_ocr.py", *local_common, "--ocr-fps", "1", *(["--siglip"] if siglip else [])],
         "skip": "skipped (--skip-local)" if skip_local else None},
        {"name": "retention (TTCC, remote Space)",
         "cmd": [str(VENV), "scripts/run_ttcc.py", *local_common, "--features", "reuse", "--enable-ttcc", "--ttcc-remote", "--ttcc-allow-remote",
                 *(["--ttcc-approve", approve] if approve else []), *(["--hook-s", str(hook_s)] if hook_s else []),
                 # whole-video curve: native windows of hook_s seconds every half window, merged (approximate); window 0 is the native hook
                 *(["--long-videos", "windows", "--window-mode", "fixed", "--window-s", str(hook_s), "--stride-s", str(max(1, hook_s // 2)),
                    "--warmup-s", str(hook_s / 4)] if (full_curve and hook_s) else [])],
         "skip": None if allow_remote else "not run: the video is uploaded to your Space only with --allow-remote"},
        # required: Gemini reads the video plus the transcript and measured evidence to write the explanations and edit suggestions
        {"name": "explanation + edit suggestions (AI: video + transcript)",
         "cmd": [str(VENV_MODELS), "scripts/run_gemini_edit.py", vid, "--artifacts", str(root), "--video", str(video), "--gemini-allow-remote",
                 *(["--gemini-approve", approve] if approve else [])],
         "skip": None if gemini_allow_remote else "not run: the video is uploaded to Google only with --gemini-allow-remote"},
        {"name": "music vs voice levels (Demucs, local)",
         "cmd": [str(VENV_DEMUCS), "scripts/run_stems.py", "--input", str(video), "--artifacts", str(root)],
         "skip": None},
        {"name": "popularity model (TRIBE, remote Space; separate from TTCC)",
         "cmd": [str(VENV), "scripts/run_tribe.py", "--input", str(video), "--artifacts", str(root)], "skip": None if allow_remote else "not run: remote processing is not allowed"},
        {"name": "scores: hook, breakout and accuracy (local, from saved files)",
         "cmd": [str(VENV), "scripts/score_video.py", vid, "--artifacts", str(root)], "skip": None},
        {"name": "explain: hook strength and strongest / weakest moments (local)",
         "cmd": [str(VENV), "scripts/explain_video.py", vid, "--artifacts", str(root)], "skip": None},
        {"name": "segments + suggestions + review page",
         "cmd": [str(VENV), "scripts/build_review.py", vid, "--artifacts", str(root), "--video", str(video), *(["--open"] if open_page else [])], "skip": None},
    ]
    return steps


def default_wake(log):
    """Wake the paused or sleeping TTCC / TRIBE Spaces: a request at the start (so waking overlaps the local steps), and a wait before each remote step."""
    def wake(phase: str) -> None:
        cmd = [str(VENV_MODELS), "scripts/wake_spaces.py"] + (["--no-wait"] if phase == "start" else ["--which", phase])
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1000)
        except (subprocess.TimeoutExpired, OSError) as exc:
            log(f"      waking the {phase} service: {type(exc).__name__}")
            return
        for ln in (proc.stdout or "").strip().splitlines()[-3:]:
            log("      " + ln[:200])
    return wake


def run_e2e(video: Path, root: Path, run=subprocess.run, log=print, wake=None, **opts) -> int:
    steps = plan(video, root, **opts)
    if wake:
        wake("start")
    results: dict[int, tuple[str, str]] = {}
    started = time.time()

    def execute(n: int, step: dict) -> bool:
        """Run one step; returns True when the run must stop (a required step failed or could not run)."""
        label = f"[{n}/{len(steps)}] {step['name']}"
        if step["skip"]:
            log(f"{label}: {step['skip']}")
            results[n] = (step["name"], "skipped")
            return step["name"].split(" ")[0] in ("retention", "explanation", "popularity")  # TTCC, Gemini and TRIBE are required for a complete analysis
        log(f"{label} ...")
        if wake and step["name"].split(" ")[0] in ("retention", "popularity"):
            wake("ttcc" if step["name"].startswith("retention") else "tribe")  # wait until that Space answers (it may have been asleep)
        t0 = time.time()
        proc = run(step["cmd"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-4:]
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        if proc.returncode == 0 and step["name"].startswith("retention"):  # run_ttcc exits 0 for 'unavailable' (too short, not approved, Space down): say so
            why = ttcc_not_successful(root / "features" / video.stem / "ttcc_status.json")
            if why:
                status = f"FAILED ({why})"
                out += f"\n{why}"
                proc = subprocess.CompletedProcess(step["cmd"], 1, out, "")
        results[n] = (step["name"], status)
        log(f"{label}: {status} in {time.time() - t0:.0f} s")
        for ln in tail:
            log("      " + ln[:240])
        if proc.returncode != 0:
            log(hint(n, proc.returncode, out))
            return True
        return False

    # Steps run one after another, in plan order (each later step reads the files the earlier ones wrote).
    for n, step in enumerate(steps, 1):
        if execute(n, step):
            log("stopping: a required step failed, so the later steps have nothing to work on")
            break
    results = sorted(results.items())
    results = [r for _, r in results]
    ok = all(s in ("ok", "skipped") for _, s in results) and len(results) == len(steps)
    page = root / "features" / video.stem / "review.html"
    log(f"\nsummary ({time.time() - started:.0f} s): " + "; ".join(f"{name}: {status}" for name, status in results))
    if ok and page.is_file():
        log(f"review page: {page.resolve()}\nplay it with the video (a file:// page cannot load local video):\n  python3 scripts/serve_review.py {video.stem} --artifacts {root} --video {video} --open")
    return 0 if ok else 1


def ttcc_not_successful(status_path: Path) -> str | None:
    """Why the TTCC step produced no prediction, from its status file; None when it succeeded."""
    try:
        st = json.loads(status_path.read_text())
    except (OSError, ValueError):
        return "TTCC wrote no status for this video"
    if st.get("status") == "success":
        return None
    return f"TTCC {st.get('status')}: {str(st.get('reason') or 'no reason recorded')[:220]}"


def hint(step: int, code: int, out: str) -> str:
    if step == 3:
        if "not reachable" in out or "still downloading" in out or "unavailable" in out:
            return "  -> the Space is not ready or not reachable: check its status and /health, then run again (cached steps are reused)"
        if "not configured" in out:
            return "  -> add TTCC_ENDPOINT_URL and HF_TOKEN to your .env"
        if "--ttcc-approve" in out:
            return "  -> company reel: pass --approve <video_id> (only if the organisers allow uploading it)"
        return "  -> see the lines above; the Space's Container logs hold more detail"
    if step == 4:
        if "GEMINI_API_KEY" in out or "API key" in out or "API_KEY_INVALID" in out or "PERMISSION_DENIED" in out:
            return "  -> check GEMINI_API_KEY in your .env (use the new key, not the revoked one)"
        return "  -> the AI step is required: see the lines above (network, quota or an unreadable video); cached steps are reused on a rerun"
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
    ap.add_argument("--hook-s", type=int, default=None, help="seconds of the opening TTCC analyses (default 30; 10 fits a 24 GB GPU and matches the reference run)")
    ap.add_argument("--full-curve", action="store_true", help="predict the whole video in overlapping hook-s windows (needs --hook-s); the first window is the native hook")
    ap.add_argument("--gemini-allow-remote", action="store_true", help="allow uploading the video to Google (required: Gemini writes the explanations from the video and transcript)")
    ap.add_argument("--open", action="store_true", help="open the review page when done")
    args = ap.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    root = args.artifacts if args.artifacts.is_absolute() else PROJECT_ROOT / args.artifacts
    return run_e2e(video, root, wake=default_wake(print) if args.allow_remote else None, allow_remote=args.allow_remote, approve=args.approve, siglip=args.siglip, skip_local=args.skip_local,
                   open_page=args.open, gemini_allow_remote=args.gemini_allow_remote, hook_s=args.hook_s, full_curve=args.full_curve)


if __name__ == "__main__":
    sys.exit(main())
