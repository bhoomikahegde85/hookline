"""On-screen text: PaddleOCR on frames sampled from the original video.

Sampling follows the proxy's convention: FFmpeg's ``fps`` filter resamples the
original (including variable-frame-rate sources) to a constant ``sample_fps``
grid on the project timeline, and each frame time is read back with ffprobe.
Sample ``t`` stands for ``[t, t + 1/sample_fps)``. Frames are larger than the
analysis proxy (``frame_short_side``) because small captions need resolution.

Repeated text is not stored as new events: detections are linked across
samples into **events** (same text within ``track_text_similarity``; boxes
must overlap by ``track_iou`` unless the text was seen in the previous sample,
so moving text stays one event; up to ``track_max_gap_samples`` missed samples
are bridged). Detections without a letter or digit are dropped.
Per-frame records keep the timestamp, box and event id. Text that disappears
and comes back later is a new event with ``repeat_of`` pointing to the earlier one.

Per-frame status distinguishes an empty detection (``available`` with no
detections) from an OCR call that raised (``failed``).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import OCR_VERSION, SCHEMA_VERSION, OcrConfig, stable_hash
from .context import BackendUnavailable, LazyBackend, MediaContext, clean, covered_length, peak_rss_mb, segment_locator
from .media import find_tool, probe_video
from .preprocess import _atomic_ffmpeg, _run, proxy_dimensions, proxy_frame_times
from .status import ABSENT, AVAILABLE, FAILED, MEASURED_STATUSES, UNCERTAIN, expects_data, window_status
from .visual import FrameReader

TEXT_COLUMNS = [
    "text_status", "text_evidence", "text_samples", "text_present", "text_coverage", "text_event_count", "text_new_count",
    "text_area_ratio", "on_screen_text", "text_event_ids",
]
OBSERVATION_COLUMNS = [
    "video_id", "t_s", "segment_index", "sample_status", "event_id", "text", "score", "x0", "y0", "x1", "y1", "error",
]
TIMELINE_NOTE = "seconds from start of original playback; sample t stands for [t, t + interval_s)"


class TextReader(Protocol):
    def info(self) -> dict: ...

    def read(self, image_bgr: np.ndarray) -> list[dict]:
        """Return [{text, score, box: [x0, y0, x1, y1], poly: [[x, y], ...] | None}] in image pixels."""
        ...


def paddlex_model_dir() -> Path:
    return Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")) / "official_models"


def _poly_to_box(poly) -> list[float] | None:
    if poly is None or len(poly) == 0:
        return None
    a = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def parse_paddle_result(result: Any) -> list[dict]:
    """Normalise PaddleOCR output: 3.x ``predict`` (dict-like results with rec_texts/rec_scores/
    rec_polys/rec_boxes) or 2.x ``ocr`` (per image: [[poly, (text, score)], ...] or None)."""
    dets: list[dict] = []
    for item in list(result) if result is not None else []:
        if item is None:
            continue
        if hasattr(item, "keys") and "rec_texts" in item:
            texts, scores = item["rec_texts"], item["rec_scores"]
            polys, boxes = item.get("rec_polys"), item.get("rec_boxes")
            for i, text in enumerate(texts):
                poly = np.asarray(polys[i], dtype=float).tolist() if polys is not None and len(polys) > i else None
                box = (np.asarray(boxes[i], dtype=float).tolist() if boxes is not None and len(boxes) > i else _poly_to_box(poly))
                dets.append({"text": str(text), "score": float(scores[i]), "box": box, "poly": poly})
        elif isinstance(item, (list, tuple)):
            for line in item:
                if not line or len(line) < 2:
                    continue
                poly, (text, score) = line[0], line[1]
                poly = np.asarray(poly, dtype=float).tolist()
                dets.append({"text": str(text), "score": float(score), "box": _poly_to_box(poly), "poly": poly})
    return dets


class PaddleOcrReader:
    backend = "paddleocr"

    def __init__(self, cfg: OcrConfig, allow_download: bool = False):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            import paddleocr
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise BackendUnavailable("paddleocr/paddlepaddle are not installed (use .venv-models, see requirements-models.txt)") from exc
        self.cfg = cfg
        names = (cfg.det_model, cfg.rec_model)
        if cfg.det_model is None:
            try:  # PaddleOCR's own lang/version -> model-name mapping
                names = PaddleOCR._get_ocr_model_names(None, cfg.lang, cfg.ocr_version)
            except Exception:
                names = (None, None)
        if not allow_download:
            missing = [n or "<unresolved>" for n in names if not n or not (paddlex_model_dir() / n).exists()]
            if missing:
                raise BackendUnavailable(
                    f"OCR models not in {paddlex_model_dir()}: {', '.join(missing)}; re-run with --download-models to fetch them"
                )
        kwargs: dict[str, Any] = dict(
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device=cfg.device,
        )
        if cfg.det_model is not None:
            kwargs.update(text_detection_model_name=cfg.det_model, text_recognition_model_name=cfg.rec_model)
        else:
            kwargs.update(lang=cfg.lang, ocr_version=cfg.ocr_version)
        try:
            self.ocr = PaddleOCR(**kwargs)
        except Exception as exc:
            raise BackendUnavailable(f"could not initialise PaddleOCR: {type(exc).__name__}: {exc}") from exc
        try:
            import paddle

            paddle_version = paddle.__version__
        except Exception:
            paddle_version = None
        self._info = {
            "backend": self.backend, "backend_version": getattr(paddleocr, "__version__", None),
            "paddle_version": paddle_version, "det_model": names[0], "rec_model": names[1],
            "lang": cfg.lang, "ocr_version": cfg.ocr_version, "device": cfg.device,
        }

    def info(self) -> dict:
        return dict(self._info)

    def read(self, image_bgr: np.ndarray) -> list[dict]:
        if hasattr(self.ocr, "predict"):
            return parse_paddle_result(self.ocr.predict(image_bgr))
        return parse_paddle_result(self.ocr.ocr(image_bgr, cls=False))


def ocr_key(ctx: MediaContext, cfg: OcrConfig) -> str:
    return stable_hash("ocr", ctx.source_sha256 or ctx.source_path, cfg.cache_dict())


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def sample_frames(ctx: MediaContext, cfg: OcrConfig, width: int | None, height: int | None, work_dir: Path) -> tuple[Path, list[float], int, int]:
    """Encode constant-rate OCR frames from the original into ``work_dir`` and read back their times.

    ``round=up`` makes the frame at tick ``t`` the one on screen at playback time ``t``; the
    default rounding emits a frame from up to half an interval later under the earlier label.
    Frames are encoded losslessly (with libx264) so OCR input does not depend on the sampling rate.
    """
    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required")
    dims = proxy_dimensions(width, height, cfg.frame_short_side)
    scale = f"scale={dims[0]}:{dims[1]}:flags=area" if dims else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    code, out, _ = _run([ffmpeg, "-hide_banner", "-encoders"], timeout=20)
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0"] if code == 0 and " libx264 " in out else ["-c:v", "mpeg4", "-q:v", "1"]
    dest = work_dir / "ocr_frames.mp4"
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y", "-i", ctx.source_path, "-map", "0:v:0", "-an", "-sn", "-dn",
           "-vf", f"fps={cfg.sample_fps}:round=up,{scale},setsar=1", *enc, "-pix_fmt", "yuv420p"]
    if cfg.max_seconds:
        cmd += ["-t", str(cfg.max_seconds)]
    ok, err = _atomic_ffmpeg(cmd, dest)
    if not ok:
        raise RuntimeError(f"OCR frame sampling failed: {err[-300:]}")
    times = proxy_frame_times(ffprobe, dest)
    probe = probe_video(dest, ffprobe=ffprobe)
    if not times or not probe.width or not probe.height:
        raise RuntimeError("OCR frame sampling produced no decodable frames")
    return dest, times, probe.width, probe.height


def _base(ctx: MediaContext, cfg: OcrConfig, key: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ocr_version": OCR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": ctx.video_id,
        "source_path": ctx.source_path,
        "source_sha256": ctx.source_sha256,
        "cache_key": key,
        "status": FAILED,
        "measured": False,
        "text_detected": None,
        "reason": None,
        "model": None,
        "config": cfg.to_dict(),
        "sampling": None,
        "timing": None,
        "warnings": [],
        "errors": [],
        "events": [],
        "frames": [],
    }


def build_events(frames: list[dict], cfg: OcrConfig, interval: float, range_end: float) -> list[dict]:
    """Link per-frame detections into events; sets ``event_id`` on each detection."""
    events: list[dict] = []
    for fi, frame in enumerate(frames):
        if frame["status"] != AVAILABLE:
            continue
        taken: set[int] = set()
        for det in sorted(frame["detections"], key=lambda d: -d["score"]):
            norm = normalise_text(det["text"])
            best, best_score = None, 0.0
            for ev in events:
                if ev["event_id"] in taken or ev["_last_index"] < fi - 1 - cfg.track_max_gap_samples:
                    continue
                sim = SequenceMatcher(None, ev["norm_text"], norm).ratio()
                iou = box_iou(ev["_last_box"], det["box_norm"])
                # Text seen in the previous sample may have moved (camera/scroll); bridging a gap needs overlap.
                placed = iou >= cfg.track_iou or ev["_last_index"] == fi - 1
                if sim >= cfg.track_text_similarity and placed and sim + iou > best_score:
                    best, best_score = ev, sim + iou
            if best is None:
                earlier = [e for e in events if e["_last_index"] < fi
                           and SequenceMatcher(None, e["norm_text"], norm).ratio() >= cfg.track_text_similarity]
                best = {
                    "event_id": len(events), "norm_text": norm, "first_seen_s": frame["t_s"],
                    "repeat_of": earlier[0]["event_id"] if earlier else None,
                    "_texts": Counter(), "_scores": [], "_boxes": [], "_last_index": fi,
                }
                events.append(best)
            best["_texts"][det["text"]] += 1
            best["_scores"].append(det["score"])
            best["_boxes"].append(det["box_norm"])
            best["_last_index"], best["_last_box"], best["last_seen_s"] = fi, det["box_norm"], frame["t_s"]
            det["event_id"] = best["event_id"]
            taken.add(best["event_id"])
    for ev in events:
        ev["text"] = ev["_texts"].most_common(1)[0][0]
        ev["frames"] = len(ev["_scores"])
        ev["score_mean"] = round(float(np.mean(ev["_scores"])), 4)
        ev["box_norm"] = [round(float(v), 4) for v in np.median(np.array(ev["_boxes"]), axis=0)]
        ev["visible_until_s"] = round(min(ev["last_seen_s"] + interval, range_end), 3)
        for k in ("_texts", "_scores", "_boxes", "_last_index", "_last_box"):
            ev.pop(k, None)
    by_id = {ev["event_id"]: ev for ev in events}
    for frame in frames:
        for det in frame["detections"]:
            if det["text"] == by_id[det["event_id"]]["text"]:
                det.pop("text")  # stored once on the event; per-frame text only when it differs
    order = ["event_id", "text", "norm_text", "first_seen_s", "last_seen_s", "visible_until_s", "frames", "score_mean", "box_norm", "repeat_of"]
    return [{k: ev[k] for k in order} for ev in events]


def extract_ocr(ctx: MediaContext, cfg: OcrConfig, backend: LazyBackend | None, work_root: Path) -> dict:
    """Sample frames, OCR them one at a time and build deduplicated text events."""
    art = _base(ctx, cfg, ocr_key(ctx, cfg))
    if ctx.visual_stream != AVAILABLE:
        art["status"] = ctx.visual_stream
        art["reason"] = {
            ABSENT: "no video stream",
            UNCERTAIN: "video presence unknown (inspection failed)",
            FAILED: "video stream could not be decoded; on-screen text not measured",
        }.get(ctx.visual_stream, f"video stream status {ctx.visual_stream}")
        return art
    if not Path(ctx.source_path).is_file():
        art["reason"] = "source video not found"
        return art
    if backend is None:
        art["reason"] = "no OCR backend configured"
        return art
    try:
        reader = backend.get()
    except BackendUnavailable as exc:
        art["reason"] = f"model unavailable: {exc}"
        return art
    art["model"] = reader.info()

    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"ocr_{ctx.video_id}_", dir=work_root))
    try:
        try:
            path, times, fw, fh = sample_frames(ctx, cfg, ctx.width, ctx.height, work_dir)
        except Exception as exc:
            art["reason"] = str(exc)
            art["errors"].append(art["reason"])
            return art

        interval = 1.0 / cfg.sample_fps
        bounds = [x for x in (ctx.duration, cfg.max_seconds, ctx.video_span[1] if ctx.video_span else None) if x]
        limit = min(bounds) if bounds else None
        frames: list[dict] = []
        started = time.perf_counter()
        frame_reader = FrameReader(path, fw, fh)
        for t, img in zip(times, frame_reader):
            if limit is not None and t >= limit - 1e-6:
                break
            frame: dict[str, Any] = {"t_s": round(float(t), 3), "status": AVAILABLE, "detections": []}
            try:
                for d in reader.read(img):
                    text = (d.get("text") or "").strip()
                    if not any(ch.isalnum() for ch in text):  # e.g. "-", "***": edges and textures, not text
                        continue
                    if d.get("score") is None or d["score"] < cfg.min_score or not d.get("box"):
                        continue
                    x0, y0, x1, y1 = d["box"]
                    frame["detections"].append({
                        "text": text,
                        "score": round(float(d["score"]), 4),
                        "box_norm": [round(max(0.0, min(1.0, v)), 4) for v in (x0 / fw, y0 / fh, x1 / fw, y1 / fh)],
                        "poly_px": [[round(float(x), 1), round(float(y), 1)] for x, y in d["poly"]] if d.get("poly") else None,
                    })
            except Exception as exc:
                frame.update(status=FAILED, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            frames.append(frame)
        elapsed = time.perf_counter() - started
        art["errors"].extend(frame_reader.errors)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    range_end = round(limit if limit is not None else (frames[-1]["t_s"] + interval if frames else 0.0), 3)
    ok = [f for f in frames if f["status"] == AVAILABLE]
    art["sampling"] = {
        "convention": TIMELINE_NOTE,
        "fps": cfg.sample_fps,
        "interval_s": round(interval, 6),
        "frame_size": [fw, fh],
        "range_s": [0.0, range_end],
        "frames": len(frames),
        "failed_frames": len(frames) - len(ok),
        "first_frame_s": frames[0]["t_s"] if frames else None,
        "max_deviation_from_grid_s": round(max(abs(f["t_s"] - (frames[0]["t_s"] + i * interval)) for i, f in enumerate(frames)), 4) if frames else None,
        "source_fps": ctx.source_fps,
        "source_variable_frame_rate": ctx.source_vfr,
    }
    art["timing"] = {"frames": len(frames), "processing_seconds": round(elapsed, 3),
                     "seconds_per_frame": round(elapsed / len(frames), 4) if frames else None,
                     "process_peak_rss_mb": peak_rss_mb()}
    if not ok:
        art["reason"] = "OCR failed on every sampled frame" if frames else "no frames sampled"
        art["frames"] = frames
        return art
    art["events"] = build_events(frames, cfg, interval, range_end)
    art["frames"] = frames
    art.update(status=AVAILABLE, measured=True, text_detected=bool(art["events"]))
    if len(ok) < len(frames):
        art["warnings"].append(f"OCR failed on {len(frames) - len(ok)} of {len(frames)} sampled frames")
    return art


def _area(frame: dict) -> float:
    return sum((d["box_norm"][2] - d["box_norm"][0]) * (d["box_norm"][3] - d["box_norm"][1]) for d in frame["detections"])


def text_window(art: dict | None, seg: dict, video_span: list[float] | None, min_coverage: float) -> tuple[dict, list[str]]:
    """On-screen-text columns for one existing segment.

    ``available`` only when a successful OCR sample lies inside the segment (values come
    from those samples, ``text_evidence="observed"``). A segment between samples is
    ``uncertain``: its values are inferred from the preceding sample, which stands for
    ``[t, t + interval)`` (``text_evidence="inferred"``).
    """
    out: dict[str, Any] = {k: None for k in TEXT_COLUMNS}
    if art is None:
        out["text_status"] = UNCERTAIN
        return out, ["text=not extracted (stage skipped)"]
    if not art.get("measured"):
        out["text_status"] = art["status"]
        return out, []
    if seg.get("visual_status") == ABSENT:
        out["text_status"] = ABSENT
        return out, []
    start, end = seg["start_s"], seg["end_s"]
    samp = art["sampling"]
    interval, range_end = samp["interval_s"], samp["range_s"][1]
    if start >= range_end - 1e-6:
        out["text_status"] = UNCERTAIN
        return out, ["text=not sampled (beyond OCR range)"]
    inside = [f for f in art["frames"] if start <= f["t_s"] < end]
    observed = [f for f in inside if f["status"] == AVAILABLE]
    held = [f for f in art["frames"] if f["status"] == AVAILABLE and f["t_s"] < end and min(f["t_s"] + interval, range_end) > start]
    spans = [(f["t_s"], min(f["t_s"] + interval, range_end)) for f in held]
    coverage = covered_length(spans, start, end) / (end - start) if end > start else 0.0
    out["text_coverage"] = round(coverage, 4)
    notes: list[str] = []
    if observed:
        status = AVAILABLE
    elif inside:
        status = FAILED
    else:
        span = [video_span[0], min(video_span[1], range_end)] if video_span else [0.0, range_end]
        status = window_status(AVAILABLE, coverage, expects_data(start, end, span, min_coverage), min_coverage)
        if status in MEASURED_STATUSES:
            status = UNCERTAIN
            notes.append("text=inferred between samples (no OCR sample in segment)")
    out["text_status"] = status
    if status not in MEASURED_STATUSES:
        return out, notes
    by_id = {e["event_id"]: e for e in art["events"]}
    if observed:
        ids = {d["event_id"] for f in observed for d in f["detections"]}
        visible = sorted((by_id[i] for i in ids), key=lambda e: e["first_seen_s"])
        frames = observed
    else:
        visible = sorted((e for e in art["events"] if e["first_seen_s"] < end and e["visible_until_s"] > start),
                         key=lambda e: e["first_seen_s"])
        frames = held
    areas = [_area(f) for f in frames]
    out.update(
        text_evidence="observed" if observed else "inferred",
        text_samples=len(observed),
        text_present=bool(visible),
        text_event_count=len(visible),
        text_new_count=sum(1 for e in visible if start <= e["first_seen_s"] < end and e["repeat_of"] is None),
        text_area_ratio=round(min(1.0, max(areas)), 4) if areas else None,
        on_screen_text=" | ".join(e["text"] for e in visible) or None,
        text_event_ids=[e["event_id"] for e in visible] or None,
    )
    return out, notes


def observation_rows(art: dict | None, segments: list[dict]) -> list[dict]:
    """One row per OCR detection (or per sample without detections), on the original timeline,
    with the segment containing the sample time. Failed samples are kept as ``failed`` rows."""
    if not art or not art.get("frames"):
        return []
    locate = segment_locator(segments)
    text_of = {e["event_id"]: e["text"] for e in art.get("events") or []}
    rows = []
    for f in art["frames"]:
        base = {"video_id": art["video_id"], "t_s": f["t_s"], "segment_index": locate(f["t_s"]),
                "sample_status": f["status"], "error": f.get("error")}
        if not f["detections"]:
            rows.append(base)
        for d in f["detections"]:
            x0, y0, x1, y1 = d["box_norm"]
            rows.append(base | {"event_id": d.get("event_id"), "text": d.get("text", text_of.get(d.get("event_id"))),
                                "score": d["score"], "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return rows


def opening_text(art: dict | None, end: float, max_texts: int = 10) -> dict:
    """On-screen text in [0, end). "No text" is only claimed if every sample in the opening
    succeeded and sampling reached the end of the opening (no text *observed* at that rate)."""
    summary: dict[str, Any] = {"status": UNCERTAIN if art is None else art["status"], "text_detected": None,
                               "samples": None, "sample_interval_s": None, "first_text_s": None, "events": None,
                               "new_events": None, "single_sample_events": None, "screen_coverage": None, "texts": None}
    if art is None or not art.get("measured") or end <= 0:
        return summary
    visible = sorted((e for e in art["events"] if e["first_seen_s"] < end), key=lambda e: e["first_seen_s"])
    opening = [f for f in art["frames"] if f["t_s"] < end]
    fully_measured = (bool(opening) and all(f["status"] == AVAILABLE for f in opening)
                      and art["sampling"]["range_s"][1] >= end - 1e-6)
    summary.update(samples=len(opening), sample_interval_s=art["sampling"]["interval_s"])
    # Most persistent text first: single-sample events are often scene text or misreads.
    samples = Counter(d["event_id"] for f in art["frames"] if f["t_s"] < end for d in f["detections"] if "event_id" in d)
    ranked = sorted(visible, key=lambda e: (-samples[e["event_id"]], e["first_seen_s"]))
    texts: list[str] = []
    for e in ranked:
        if e["text"] not in texts:
            texts.append(e["text"])
    summary.update(
        text_detected=True if visible else (False if fully_measured else None),
        first_text_s=visible[0]["first_seen_s"] if visible else None,
        events=len(visible),
        new_events=sum(1 for e in visible if e["repeat_of"] is None),
        single_sample_events=sum(1 for e in visible if e["frames"] == 1),
        screen_coverage=clean(covered_length([(e["first_seen_s"], e["visible_until_s"]) for e in visible], 0.0, end) / end),
        texts=texts[:max_texts] or None,
    )
    return summary
