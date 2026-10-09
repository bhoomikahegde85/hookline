#!/usr/bin/env python3
"""Timestamped speech (Whisper), on-screen text (PaddleOCR), librosa audio features and, if
enabled, a Gemini interpretation on top of the cached features; then the merged evidence
timeline and evidence-linked edit recommendations.

    python scripts/run_speech_ocr.py --download-models            # first run: fetch models
    python scripts/run_speech_ocr.py --input path/to/dir_or_file  # repeatable; default data/raw + data/sample
    python scripts/run_speech_ocr.py --skip-ocr --asr-model base
    python scripts/run_speech_ocr.py --skip-speech --ocr-fps 1 --ocr-max-seconds 10
    python scripts/run_speech_ocr.py --input 01-Hookline --artifacts artifacts/company --limit 5
    python scripts/run_speech_ocr.py --input data/sample/syn_av_testpattern_8s.mp4 --gemini offline
    python scripts/run_speech_ocr.py --input my_clip.mp4 --gemini remote --gemini-allow-remote   # uploads my_clip.mp4
    python scripts/run_speech_ocr.py --input data/sample/syn_av_testpattern_8s.mp4 --siglip --download-models  # first SigLIP run (~1.5 GB)

Gemini is off by default. "offline" never uses the network. "remote" uploads the video to the
Gemini API and needs --gemini-allow-remote plus GEMINI_API_KEY (environment or untracked .env);
company videos under 01-Hookline/ additionally need --gemini-approve <video_id>.

Needs requirements-models.txt, installed in a separate venv (.venv-models; see README).
Run it as .venv-models/bin/python scripts/run_speech_ocr.py. Models are never downloaded unless
--download-models is given. Preprocessing/features are reused from the cache
(computed first if missing): pass the same --window/--proxy-fps as run_pipeline.py.
Writes artifacts/features/<video_id>/{speech.json, ocr.json, librosa.json, gemini.json (if enabled),
segments_merged.csv, words.csv, text_observations.csv, opening_multimodal.json, recommendations.json}
and run-level segments_merged_all.csv,
first5s_multimodal.csv, speech_ocr_run_summary.json.
Exit code: 0 if no video failed, 1 if any failed, 2 if FFmpeg is missing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hookline.config import GeminiConfig, LibrosaConfig, OcrConfig, PipelineConfig, SiglipConfig, SpeechConfig  # noqa: E402
from hookline.context import LazyBackend  # noqa: E402
from hookline.gemini import default_model  # noqa: E402
from hookline.media import find_tool  # noqa: E402
from hookline.multimodal import Backends, run_multimodal  # noqa: E402
from hookline.ocr import PaddleOcrReader  # noqa: E402
from hookline.pipeline import default_inputs  # noqa: E402
from hookline.siglip import HfSiglipEmbedder  # noqa: E402
from hookline.speech import FasterWhisperTranscriber  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", type=Path, help="video file or directory (repeatable)")
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts", help="output root (default: artifacts/)")
    ap.add_argument("--limit", type=int, help="process at most N videos")
    ap.add_argument("--force", action="store_true", help="redo speech/OCR even if cached (features stay cached)")
    ap.add_argument("--download-models", action="store_true", help="allow downloading Whisper/PaddleOCR models")
    ap.add_argument("--skip-speech", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--skip-librosa", action="store_true", help="skip librosa audio features")
    g = ap.add_argument_group("features (must match run_pipeline.py to reuse its cache)")
    g.add_argument("--window", type=float)
    g.add_argument("--proxy-fps", type=int)
    s = ap.add_argument_group("speech")
    s.add_argument("--asr-model", default=SpeechConfig.model, help="Whisper size: tiny, base, small, medium, large-v3 (default small)")
    s.add_argument("--asr-device", default=SpeechConfig.device, help="cpu, cuda or auto")
    s.add_argument("--asr-compute-type", default=SpeechConfig.compute_type, help="int8 (CPU) or float16 (CUDA)")
    s.add_argument("--language", help="force a language code (default: detect per video)")
    s.add_argument("--beam-size", type=int, default=SpeechConfig.beam_size)
    s.add_argument("--no-vad", action="store_true", help="disable the Silero VAD pre-filter")
    o = ap.add_argument_group("on-screen text")
    o.add_argument("--ocr-fps", type=float, default=OcrConfig.sample_fps, help="samples per second (default 2)")
    o.add_argument("--ocr-max-seconds", type=float, help="OCR only the first N seconds")
    o.add_argument("--ocr-short-side", type=int, default=OcrConfig.frame_short_side)
    o.add_argument("--ocr-lang", help="PaddleOCR language (e.g. hi); picks models unless --ocr-det-model/--ocr-rec-model given")
    o.add_argument("--ocr-det-model", help=f"detection model name (default {OcrConfig.det_model})")
    o.add_argument("--ocr-rec-model", help=f"recognition model name (default {OcrConfig.rec_model})")
    o.add_argument("--ocr-device", default=OcrConfig.device, help="cpu or gpu:0")
    o.add_argument("--ocr-min-score", type=float, default=OcrConfig.min_score)
    v = ap.add_argument_group("SigLIP 2 visual-semantic signals (off by default)")
    v.add_argument("--siglip", action="store_true", help="embed sampled proxy frames with google/siglip2-base-patch16-224")
    v.add_argument("--siglip-fps", type=float, default=SiglipConfig.sample_fps, help="proxy frames per second to embed")
    v.add_argument("--siglip-device", choices=["auto", "mps", "cpu"], default=SiglipConfig.device)
    v.add_argument("--siglip-batch-size", type=int, default=SiglipConfig.batch_size)
    m = ap.add_argument_group("Gemini interpretation (off by default)")
    m.add_argument("--gemini", choices=["off", "offline", "remote"], default="off",
                   help="off; offline = no network (validates --gemini-offline-response or an empty skeleton); remote = upload")
    m.add_argument("--gemini-model", default=None, help="model id (default: $GEMINI_MODEL or %s)" % GeminiConfig.model)
    m.add_argument("--gemini-allow-remote", action="store_true", help="required to upload any video in remote mode")
    m.add_argument("--gemini-approve", action="append", default=[], metavar="VIDEO_ID",
                   help="allow uploading this company video (under 01-Hookline/); repeatable")
    m.add_argument("--gemini-timeout", type=float, default=GeminiConfig.timeout_s, help="seconds per API request")
    m.add_argument("--gemini-max-retries", type=int, default=GeminiConfig.max_retries)
    m.add_argument("--gemini-fps", type=float, default=GeminiConfig.fps, help="frames per second Gemini samples")
    m.add_argument("--gemini-media-resolution", choices=["low", "medium", "high"], default=GeminiConfig.media_resolution)
    m.add_argument("--gemini-offline-response", type=Path, help="offline mode: JSON response to validate instead of calling the API")
    args = ap.parse_args(argv)

    if args.skip_speech and args.skip_ocr and args.skip_librosa and args.gemini == "off" and not args.siglip:
        ap.error("nothing to do: every stage skipped")
    if not (find_tool("ffmpeg") and find_tool("ffprobe")):
        print("ffmpeg and ffprobe are required (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg)", file=sys.stderr)
        return 2

    cfg = PipelineConfig()
    if args.window:
        cfg = replace(cfg, window_seconds=args.window)
    if args.proxy_fps:
        cfg = replace(cfg, proxy_fps=args.proxy_fps)
    speech_cfg = SpeechConfig(
        model=args.asr_model, device=args.asr_device, compute_type=args.asr_compute_type,
        language=args.language, beam_size=args.beam_size, vad_filter=not args.no_vad,
    )
    det, rec = args.ocr_det_model, args.ocr_rec_model
    if not args.ocr_lang and det is None and rec is None:
        det, rec = OcrConfig.det_model, OcrConfig.rec_model
    try:
        ocr_cfg = OcrConfig(
            sample_fps=args.ocr_fps, max_seconds=args.ocr_max_seconds, frame_short_side=args.ocr_short_side,
            det_model=det, rec_model=rec, lang=args.ocr_lang, device=args.ocr_device, min_score=args.ocr_min_score,
        )
    except ValueError as exc:
        ap.error(str(exc))

    try:
        gemini_cfg = GeminiConfig(
            mode=args.gemini, model=args.gemini_model or default_model(), allow_remote=args.gemini_allow_remote,
            approved_ids=tuple(args.gemini_approve), timeout_s=args.gemini_timeout, max_retries=args.gemini_max_retries,
            fps=args.gemini_fps, media_resolution=args.gemini_media_resolution,
            offline_response=str(args.gemini_offline_response) if args.gemini_offline_response else None,
        )
    except ValueError as exc:
        ap.error(str(exc))
    try:
        siglip_cfg = SiglipConfig(sample_fps=args.siglip_fps, device=args.siglip_device, batch_size=args.siglip_batch_size)
    except ValueError as exc:
        ap.error(str(exc))
    if args.gemini == "remote" and not args.gemini_allow_remote:
        print("gemini: remote mode without --gemini-allow-remote: nothing will be uploaded", file=sys.stderr)

    backends = Backends(
        speech=None if args.skip_speech else LazyBackend(lambda: FasterWhisperTranscriber(speech_cfg, args.download_models)),
        ocr=None if args.skip_ocr else LazyBackend(lambda: PaddleOcrReader(ocr_cfg, args.download_models)),
        siglip=LazyBackend(lambda: HfSiglipEmbedder(siglip_cfg, args.download_models)) if args.siglip else None,
    )
    missing = [str(p) for p in args.input or [] if not p.expanduser().exists()]
    if missing:
        print(f"input not found: {', '.join(missing)} (paths are relative to {Path.cwd()})", file=sys.stderr)
        return 2
    inputs = args.input or default_inputs()
    print("inputs: " + ", ".join(str(p) for p in inputs))
    summary = run_multimodal(
        inputs, args.artifacts, cfg, speech_cfg, ocr_cfg, backends,
        run_speech=not args.skip_speech, run_ocr=not args.skip_ocr, force=args.force, limit=args.limit,
        librosa_cfg=LibrosaConfig(), gemini_cfg=gemini_cfg, run_librosa=not args.skip_librosa,
        siglip_cfg=siglip_cfg, run_siglip=args.siglip,
    )
    c = summary["counts"]
    print(f"\n{c['videos']} videos: {c['ok']} ok, {c['partial']} partial, {c['failed']} failed")
    for name, backend in (("speech", backends.speech), ("ocr", backends.ocr), ("siglip", backends.siglip)):
        if backend is not None and backend.error:
            print(f"{name} model unavailable: {backend.error}")
    if args.gemini != "off":
        tokens = summary.get("gemini_tokens")
        print(f"gemini ({args.gemini}): {summary['gemini_api_calls']} API calls"
              + (f", {tokens} tokens reported by the API" if tokens else ", token usage unavailable")
              + "; cost not computed (the API returns token counts, not prices)")
    print(f"outputs: {Path(summary['artifacts_root']) / 'features'}")
    return 1 if c["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
