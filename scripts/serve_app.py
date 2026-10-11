#!/usr/bin/env python3
"""Hookline on your own video: upload page, live progress, then the analysis page. Runs on this machine only (http://127.0.0.1).

    python3 scripts/serve_app.py --open                 # then choose a video in the browser
    python3 scripts/serve_app.py --hook-s 10 --port 8770

Per upload the page asks what may be sent where: your TTCC Space and Google Gemini (both required: retention prediction, and the explanations from the video and transcript).
Behind the page is the same pipeline as scripts/run_e2e.py. Needs TTCC_ENDPOINT_URL and HF_TOKEN (and GEMINI_API_KEY for Gemini) in the
untracked .env, and the TTCC Space running. Everything shown is a model prediction, not audience data. Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.app_server import App, Job, start_server  # noqa: E402


def load_e2e():
    spec = importlib.util.spec_from_file_location("run_e2e", PROJECT_ROOT / "scripts" / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_runner(root: Path, hook_s: int | None, e2e=None):
    e2e = e2e or load_e2e()

    def runner(job: Job) -> int:
        return e2e.run_e2e(job.video, root, run=subprocess.run, log=job.add, wake=e2e.default_wake(job.add) if hasattr(e2e, "default_wake") else None, allow_remote=True, approve=None, siglip=True, skip_local=False,
                           open_page=False, gemini_allow_remote=True, hook_s=hook_s, full_curve=True)
    return runner


def make_whatif_runner(root: Path, hook_s: int | None, run=subprocess.run):
    """'What if I make these edits?': cut the trims with ffmpeg, predict the edited video with TTCC on your Space (same windows as the original), compare."""
    from hookline.whatif import apply_trims, compare_curves, normalise_trims

    hs = hook_s or 10

    def runner(job_id: str, trims: list[dict]) -> dict:
        feats = root / "features" / job_id
        orig = json.loads((feats / "ttcc_retention.json").read_text())
        video = next(p for p in (root / "uploads").glob(f"{job_id}.*") if p.suffix.lower() in (".mp4", ".mov", ".m4v", ".webm", ".mkv"))
        ranges, warnings = normalise_trims(trims, float(orig["duration_s"]))
        work = root / "whatif" / job_id
        work.mkdir(parents=True, exist_ok=True)
        edited = apply_trims(video, ranges, work / f"{job_id}_edit.mp4")
        cmd = [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "scripts/run_ttcc.py", "--input", str(edited), "--artifacts", str(work), "--features", "skip",
               "--enable-ttcc", "--ttcc-remote", "--ttcc-allow-remote", "--hook-s", str(hs), "--long-videos", "windows", "--window-mode", "fixed",
               "--window-s", str(hs), "--stride-s", str(max(1, hs // 2)), "--warmup-s", str(hs / 4)]
        proc = run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        art_path = work / "features" / f"{job_id}_edit" / "ttcc_retention.json"
        st = {}
        try:
            st = json.loads((work / "features" / f"{job_id}_edit" / "ttcc_status.json").read_text())
        except (OSError, ValueError):
            pass
        if proc.returncode != 0 or not art_path.is_file() or st.get("status", "success") != "success":
            tail = ([f"TTCC {st.get('status')}: {st.get('reason')}"] if st.get("status", "success") != "success" else []) + ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-3:]
            raise RuntimeError("TTCC could not predict the edited video: " + " | ".join(tail)[:300])
        edit = json.loads(art_path.read_text())
        res = compare_curves(orig["retention"], edit["retention"], ranges, original_is_stitched=bool(orig.get("stitched")), edited_is_stitched=bool(edit.get("stitched")))
        res |= {"warnings": warnings, "original_retention": orig["retention"], "edited_retention": edit["retention"], "video_id": job_id}
        (work / "whatif.json").write_text(json.dumps(res, indent=2))
        return res
    return runner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "app", help="where uploads and analyses are stored")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--hook-s", type=int, default=10, help="seconds of the opening TTCC analyses (10 matches the reference run and fits a 24 GB GPU)")
    ap.add_argument("--max-upload-mb", type=int, default=300)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args(argv)
    args.artifacts = args.artifacts if args.artifacts.is_absolute() else (Path.cwd() / args.artifacts).resolve()
    app = App(args.artifacts, make_runner(args.artifacts, args.hook_s), args.max_upload_mb, args.hook_s, make_whatif_runner(args.artifacts, args.hook_s))
    try:
        server, port = start_server(app, args.port)
    except OSError as exc:
        print(f"could not start on port {args.port}: {exc}. Pick another with --port.", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"Hookline: {url}\n(this machine only; Ctrl+C to stop)")
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
