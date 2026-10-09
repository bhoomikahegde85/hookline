"""Speech + on-screen-text stage on top of the cached visual/audio features.

The existing segments (0.5 s grid split at scene cuts) stay the shared timeline:
speech words and OCR samples are mapped onto them, never the other way round.

Per video, in ``<artifacts>/features/<video_id>/``:
- ``speech.json``, ``ocr.json``: structured artifacts, cached by their own keys
- ``segments_merged.csv``: one row per existing segment with every ``segments.csv``
  column (visual, audio, scene) plus speech and on-screen-text columns
- ``words.csv``: one row per word, exact times + the segment containing its midpoint
- ``text_observations.csv``: one row per OCR sample/detection, sample time, box, segment
- ``opening_multimodal.json``: first-5-s summary combining visual, audio, speech and text
- ``librosa.json``: librosa audio features per segment, low-energy intervals, global tempo
- ``gemini.json``: Gemini interpretation (only when enabled; off by default)
- ``recommendations.json``: rule-based and Gemini edit suggestions, each citing evidence ids
- ``siglip.json`` + ``siglip_embeddings.npy``: SigLIP 2 frame embeddings and description scores (opt-in)
Run level, in ``<artifacts>/features/``: ``segments_merged_all.csv``,
``first5s_multimodal.csv``, ``speech_ocr_run_summary.json``.

Nothing here scores attention or predicts engagement.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import GeminiConfig, LibrosaConfig, OcrConfig, PipelineConfig, SiglipConfig, SpeechConfig
from .context import LazyBackend, MediaContext
from .features import OPENING_NOTE, SEGMENT_COLUMNS, _csv_value
from .gemini import GEMINI_COLUMNS, GeminiTransport, extract_gemini, gemini_key, gemini_window
from .librosa_audio import LIBROSA_COLUMNS, extract_librosa, librosa_key, librosa_window, opening_librosa
from .ocr import OBSERVATION_COLUMNS, TEXT_COLUMNS, extract_ocr, observation_rows, ocr_key, opening_text, text_window
from .paths import artifacts_dir
from .pipeline import VideoInput, _write_csv, default_inputs, discover, process_video
from .recommendations import build_recommendations, evidence_index, invalid_evidence, recommendation_ids
from .siglip import SIGLIP_COLUMNS, extract_siglip, frame_selection, read_frame_times, siglip_key, siglip_window, write_siglip
from .segments import overlap
from .speech import SPEECH_COLUMNS, WORD_COLUMNS, extract_speech, opening_speech, speech_key, speech_window, word_rows
from .status import ABSENT, ALL_STATUSES, AVAILABLE, FAILED, UNCERTAIN

# Links into words.csv (word_index) and text_observations.csv / ocr.json (event_id).
WORD_LINK_COLUMNS = ["speech_word_index_first", "speech_word_index_last"]
MERGED_COLUMNS = ([c for c in SEGMENT_COLUMNS if c != "notes"] + SPEECH_COLUMNS + WORD_LINK_COLUMNS + TEXT_COLUMNS
                  + LIBROSA_COLUMNS + SIGLIP_COLUMNS + GEMINI_COLUMNS + ["recommendation_ids", "notes"])
LEGACY_OUTPUTS = ("segments_speech_text.csv", "segments_speech_text_all.csv")
OPENING_MM_COLUMNS = [
    "video_id", "stage_status", "features_status", "duration_s", "analysed_seconds", "shorter_than_opening_window",
    "visual_status_coverage", "scene_cuts", "first_scene_cut_s",
    "audio_status_coverage", "first_non_silent_s", "near_silence_ratio",
    "speech_status", "speech_status_coverage", "speech_language", "speech_detected", "first_word_s",
    "word_count", "words_per_s", "speech_coverage", "transcript",
    "text_status", "text_status_coverage", "text_samples", "text_detected", "first_text_s", "text_events",
    "new_text_events", "single_sample_text_events", "text_screen_coverage", "texts",
]


@dataclass
class Backends:
    speech: LazyBackend | None = None
    ocr: LazyBackend | None = None
    gemini: Callable[[], GeminiTransport] | None = None  # None: offline fixture or the real SDK, per GeminiConfig.mode
    siglip: LazyBackend | None = None


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _cached(path: Path, key: str, force: bool) -> dict | None:
    """Reuse an artifact with the same key; failures are always retried."""
    if force:
        return None
    art = _load_json(path)
    if art and art.get("cache_key") == key and art.get("status") != FAILED:
        return art
    return None


def merged_rows(ctx: MediaContext, speech: dict | None, ocr: dict | None, min_coverage: float,
                words: list[dict] | None = None, librosa: dict | None = None, gemini: dict | None = None,
                siglip: dict | None = None) -> list[dict]:
    """Existing segment rows (visual/audio untouched) extended with speech, text, librosa and Gemini
    columns. ``words`` are ``word_rows(speech, ctx.segments)``; computed here if not given.
    ``recommendation_ids`` is filled in by ``process_video_multimodal``."""
    rows = []
    last = len(ctx.segments) - 1
    by_segment: dict[int, list[int]] = {}
    for w in word_rows(speech, ctx.segments) if words is None else words:
        by_segment.setdefault(w["segment_index"], []).append(w["word_index"])
    for i, seg in enumerate(ctx.segments):
        s, s_notes = speech_window(speech, seg, last=i == last)
        t, t_notes = text_window(ocr, seg, ctx.video_span, min_coverage)
        lr, lr_notes = librosa_window(librosa, seg)
        sg, sg_notes = siglip_window(siglip, seg)
        g, g_notes = gemini_window(gemini, seg)
        idx = by_segment.get(seg["segment_index"]) if s["speech_status"] in (AVAILABLE, UNCERTAIN) else None
        link = {"speech_word_index_first": min(idx) if idx else None, "speech_word_index_last": max(idx) if idx else None}
        row = {k: seg.get(k) for k in MERGED_COLUMNS if k != "notes"} | s | link | t | lr | sg | g
        row["notes"] = "; ".join([n for n in [seg.get("notes")] if n] + s_notes + t_notes + lr_notes + sg_notes + g_notes)
        rows.append(row)
    return rows


def _status_coverage(rows: list[dict], column: str, end: float) -> dict[str, float] | None:
    if end <= 0 or not rows:
        return None
    seconds = {s: 0.0 for s in ALL_STATUSES}
    for r in rows:
        seconds[r[column]] += overlap(r["start_s"], r["end_s"], 0.0, end)
    return {s: round(sec / end, 4) for s, sec in seconds.items() if sec > 0}


def opening_multimodal(ctx: MediaContext, speech: dict | None, ocr: dict | None, rows: list[dict], opening_seconds: float,
                       librosa: dict | None = None, gemini: dict | None = None) -> dict:
    duration = ctx.duration or 0.0
    end = min(opening_seconds, duration)
    in_open = [r for r in rows if overlap(r["start_s"], r["end_s"], 0.0, end) > 0]
    o = ctx.opening or {}
    v, a, cov = o.get("visual") or {}, o.get("audio") or {}, o.get("coverage") or {}
    return {
        "video_id": ctx.video_id,
        "window_s": [0.0, round(end, 6)],
        "analysed_seconds": round(end, 6),
        "shorter_than_opening_window": duration < opening_seconds,
        "visual": {
            "stream_status": ctx.visual_stream, "status_coverage": cov.get("visual"),
            "scene_cuts": v.get("scene_cuts"), "first_scene_cut_s": v.get("first_scene_cut_s"),
            "brightness_mean": v.get("brightness_mean"), "motion_mean": v.get("motion_mean"),
        },
        "audio": {
            "stream_status": ctx.audio_stream, "status_coverage": cov.get("audio"),
            "first_non_silent_s": a.get("first_non_silent_s"), "near_silence_ratio": a.get("near_silence_ratio"),
            "rms_dbfs": a.get("rms_dbfs"), "onset_count": a.get("onset_count"),
        },
        "speech": opening_speech(speech, in_open, end) | {"status_coverage": _status_coverage(in_open, "speech_status", end)},
        "text": opening_text(ocr, end) | {"status_coverage": _status_coverage(in_open, "text_status", end)},
        "librosa": opening_librosa(librosa, end),
        "gemini": {"status": None if gemini is None else gemini["status"],
                   "opening_hook": ((gemini or {}).get("analysis") or {}).get("opening_hook"),
                   "kind": "model interpretation, not measurement"},
        "note": OPENING_NOTE,
    }


def stage_status(artifacts: list[dict], rows: list[dict]) -> str:
    """ok: every requested modality measured or absent, no failed windows;
    failed: every requested modality failed or uncertain; partial otherwise."""
    statuses = [a["status"] for a in artifacts]
    window_failed = any(r.get(c) == FAILED for r in rows
                        for c in ("speech_status", "text_status", "librosa_status", "siglip_status"))
    if statuses and all(s in (AVAILABLE, ABSENT) for s in statuses) and not window_failed:
        return "ok"
    if any(a.get("measured") or a["status"] == ABSENT for a in artifacts):
        return "partial"
    return "failed"


def _write_unless_downgrade(path: Path, art: dict) -> bool:
    """Write an artifact, except a failure over an existing usable one (e.g. a rate-limited
    re-run must not destroy an earlier good Gemini result). Returns True if written."""
    if art["status"] == FAILED:
        old = _load_json(path)
        if old and old.get("measured"):
            return False
    _write_json(path, art)
    return True


def process_video_multimodal(
    item: VideoInput,
    root: str | Path | None = None,
    cfg: PipelineConfig | None = None,
    speech_cfg: SpeechConfig | None = None,
    ocr_cfg: OcrConfig | None = None,
    backends: Backends | None = None,
    run_speech: bool = True,
    run_ocr: bool = True,
    force: bool = False,
    librosa_cfg: LibrosaConfig | None = None,
    gemini_cfg: GeminiConfig | None = None,
    run_librosa: bool = True,
    siglip_cfg: SiglipConfig | None = None,
    run_siglip: bool = False,
) -> dict:
    """Speech, OCR, librosa and (if enabled) SigLIP and Gemini for one video, then the merged
    timeline and recommendations. Reuses cached preprocessing/features (computes them if missing)."""
    cfg, speech_cfg, ocr_cfg = cfg or PipelineConfig(), speech_cfg or SpeechConfig(), ocr_cfg or OcrConfig()
    librosa_cfg, gemini_cfg = librosa_cfg or LibrosaConfig(), gemini_cfg or GeminiConfig()
    siglip_cfg = siglip_cfg or SiglipConfig()
    backends = backends or Backends()
    root = Path(root) if root else artifacts_dir()
    report = process_video(item, cfg, root)
    features_cached = (report.pop("_cache", None) or {}).get("features")
    ctx = MediaContext.from_report(report)
    out_dir = root / "features" / item.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, Any] = {"features": features_cached, "speech": None, "ocr": None, "librosa": None, "gemini": None}
    speech = ocr = librosa = gemini = siglip = None
    if run_speech:
        path = out_dir / "speech.json"
        speech = _cached(path, speech_key(ctx, speech_cfg), force)
        cache["speech"] = speech is not None
        if speech is None:
            speech = extract_speech(ctx, speech_cfg, backends.speech)
            _write_json(path, speech)
    if run_ocr:
        path = out_dir / "ocr.json"
        ocr = _cached(path, ocr_key(ctx, ocr_cfg), force)
        cache["ocr"] = ocr is not None
        if ocr is None:
            ocr = extract_ocr(ctx, ocr_cfg, backends.ocr, root / "cache" / "_ocr_tmp")
            _write_json(path, ocr)
    if run_librosa:
        path = out_dir / "librosa.json"
        librosa = _cached(path, librosa_key(ctx, librosa_cfg), force)
        cache["librosa"] = librosa is not None
        if librosa is None:
            librosa = extract_librosa(ctx, librosa_cfg)
            _write_json(path, librosa)
    if run_siglip:
        frames_csv = out_dir / "frames.csv"
        frame_times = read_frame_times(frames_csv) if frames_csv.is_file() else []
        selected = [frame_times[i] for i in frame_selection(frame_times, ctx.segments, ctx.proxy_fps or cfg.proxy_fps, siglip_cfg)]
        path = out_dir / "siglip.json"
        siglip = _cached(path, siglip_key(ctx, siglip_cfg, selected), force)
        cache["siglip"] = siglip is not None
        if siglip is None:
            siglip, emb = extract_siglip(ctx, siglip_cfg, backends.siglip, frame_times)
            cache["siglip_written"] = write_siglip(out_dir, siglip, emb, _write_json)
    if gemini_cfg.mode != "off":
        path = out_dir / "gemini.json"
        gemini = _cached(path, gemini_key(ctx, gemini_cfg), force)
        cache["gemini"] = gemini is not None
        if gemini is None:
            gemini = extract_gemini(ctx, gemini_cfg, backends.gemini)
            cache["gemini_written"] = _write_unless_downgrade(path, gemini)

    words = word_rows(speech, ctx.segments)
    rows = merged_rows(ctx, speech, ocr, cfg.min_coverage, words, librosa, gemini, siglip)
    recs = build_recommendations(rows, words, ocr, librosa, gemini, ctx.duration)
    problems = invalid_evidence(recs, evidence_index(rows, words, ocr, librosa, gemini))
    for row in rows:
        row["recommendation_ids"] = recommendation_ids(recs, row)
    status = stage_status([a for a in (speech, ocr, librosa, siglip, gemini) if a is not None], rows)
    opening = opening_multimodal(ctx, speech, ocr, rows, cfg.opening_seconds, librosa, gemini)
    opening.update(stage_status=status, features_status=ctx.processing_status, duration_s=ctx.duration)
    _write_csv(out_dir / "segments_merged.csv", MERGED_COLUMNS, rows)
    _write_csv(out_dir / "words.csv", WORD_COLUMNS, words)
    _write_csv(out_dir / "text_observations.csv", OBSERVATION_COLUMNS, observation_rows(ocr, ctx.segments))
    _write_json(out_dir / "opening_multimodal.json", opening)
    _write_json(out_dir / "recommendations.json", {
        "video_id": item.video_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Edit suggestions grounded in the listed evidence. Not a retention, engagement or virality prediction.",
        "inputs": {name: None if a is None else a["status"] for name, a in
                   (("speech", speech), ("ocr", ocr), ("librosa", librosa), ("siglip", siglip), ("gemini", gemini))},
        "evidence_problems": problems,
        "recommendations": recs,
    })
    (out_dir / LEGACY_OUTPUTS[0]).unlink(missing_ok=True)
    return {
        "video_id": item.video_id,
        "source_path": str(item.path),
        "status": status,
        "features_status": ctx.processing_status,
        "speech": speech,
        "ocr": ocr,
        "librosa": librosa,
        "siglip": siglip,
        "gemini": gemini,
        "recommendations": recs,
        "rows": rows,
        "opening": opening,
        "cache": cache,
    }


def _fmt_cov(cov: dict | None) -> str | None:
    return "|".join(f"{k}:{v}" for k, v in sorted(cov.items())) if cov else None


def opening_row(o: dict) -> dict:
    v, a, s, t = o["visual"], o["audio"], o["speech"], o["text"]
    return {
        "video_id": o["video_id"], "stage_status": o.get("stage_status"), "features_status": o.get("features_status"),
        "duration_s": o.get("duration_s"), "analysed_seconds": o["analysed_seconds"],
        "shorter_than_opening_window": o["shorter_than_opening_window"],
        "visual_status_coverage": _fmt_cov(v["status_coverage"]), "scene_cuts": v["scene_cuts"],
        "first_scene_cut_s": v["first_scene_cut_s"],
        "audio_status_coverage": _fmt_cov(a["status_coverage"]), "first_non_silent_s": a["first_non_silent_s"],
        "near_silence_ratio": a["near_silence_ratio"],
        "speech_status": s["status"], "speech_status_coverage": _fmt_cov(s["status_coverage"]),
        "speech_language": s["language"], "speech_detected": s["speech_detected"], "first_word_s": s["first_word_s"],
        "word_count": s["word_count"], "words_per_s": s["words_per_s"], "speech_coverage": s["speech_coverage"],
        "transcript": s["transcript"],
        "text_status": t["status"], "text_status_coverage": _fmt_cov(t["status_coverage"]),
        "text_samples": t["samples"], "text_detected": t["text_detected"], "first_text_s": t["first_text_s"], "text_events": t["events"],
        "new_text_events": t["new_events"], "single_sample_text_events": t["single_sample_events"],
        "text_screen_coverage": t["screen_coverage"], "texts": t["texts"],
    }


def _record(res: dict) -> dict:
    s, o = res.get("speech") or {}, res.get("ocr") or {}
    return {
        "video_id": res["video_id"],
        "source_path": res["source_path"],
        "status": res["status"],
        "features_status": res.get("features_status"),
        "speech_status": s.get("status"),
        "speech_detected": s.get("speech_detected"),
        "speech_language": s.get("language"),
        "speech_reason": s.get("reason"),
        "speech_timing": s.get("timing"),
        "text_status": o.get("status"),
        "text_detected": o.get("text_detected"),
        "text_events": len(o.get("events") or []) if o else None,
        "text_reason": o.get("reason"),
        "ocr_timing": o.get("timing"),
        "librosa_status": (res.get("librosa") or {}).get("status"),
        "librosa_reason": (res.get("librosa") or {}).get("reason"),
        "tempo_bpm": ((res.get("librosa") or {}).get("global") or {}).get("tempo_bpm"),
        "siglip_status": (res.get("siglip") or {}).get("status"),
        "siglip_reason": (res.get("siglip") or {}).get("reason"),
        "siglip_timing": (res.get("siglip") or {}).get("timing"),
        "gemini_status": (res.get("gemini") or {}).get("status"),
        "gemini_reason": (res.get("gemini") or {}).get("reason"),
        "gemini_source": (res.get("gemini") or {}).get("source"),
        "gemini_api_calls": (res.get("gemini") or {}).get("api_calls"),
        "gemini_usage": (res.get("gemini") or {}).get("usage"),
        "recommendations": len(res.get("recommendations") or []) if res.get("rows") else None,
        "cache": res.get("cache"),
        "error": res.get("error"),
    }


def _log_line(i: int, n: int, rec: dict) -> str:
    def part(name: str, status: str | None, detail: str) -> str:
        return f"{name}={status}" + (f"({detail})" if detail and status else "")

    s_detail = {True: "speech", False: "no speech", None: ""}[rec["speech_detected"]]
    if rec["speech_language"] and rec["speech_detected"]:
        s_detail += f", {rec['speech_language']}"
    t_detail = f"{rec['text_events']} text events" if rec["text_detected"] is not None else ""
    cache = rec.get("cache") or {}
    cached = ",".join(k for k in ("speech", "ocr", "librosa", "siglip", "gemini") if cache.get(k))
    reason = rec.get("error") or next((rec[f"{m}_reason"] for m, st in (("speech", "speech_status"), ("text", "text_status"),
                                                                        ("librosa", "librosa_status"), ("siglip", "siglip_status"),
                                                                        ("gemini", "gemini_status"))
                                       if rec.get(st) == FAILED and rec.get(f"{m}_reason")), None)
    extra = ""
    if rec.get("librosa_status"):
        extra += ", " + part("librosa", rec["librosa_status"], f"{rec['tempo_bpm']:g} bpm" if rec.get("tempo_bpm") else "")
    if rec.get("siglip_status"):
        t = rec.get("siglip_timing") or {}
        extra += ", " + part("siglip", rec["siglip_status"], f"{t['frames']} frames on {t['device']}" if t.get("frames") else "")
    if rec.get("gemini_status"):
        extra += ", " + part("gemini", rec["gemini_status"], f"{rec['gemini_source']}, {rec['gemini_api_calls']} API calls"
                             if rec.get("gemini_source") else "")
    if rec.get("recommendations") is not None:
        extra += f", {rec['recommendations']} recommendations"
    return (f"[{i}/{n}] {rec['video_id']}: {rec['status']} ({part('speech', rec['speech_status'], s_detail)}, "
            f"{part('text', rec['text_status'], t_detail)}{extra}{', cached ' + cached if cached else ''})"
            + (f" reason: {reason[:160]}" if reason and rec["status"] != "ok" else ""))


def run_multimodal(
    inputs: Iterable[str | Path] | None = None,
    root: str | Path | None = None,
    cfg: PipelineConfig | None = None,
    speech_cfg: SpeechConfig | None = None,
    ocr_cfg: OcrConfig | None = None,
    backends: Backends | None = None,
    run_speech: bool = True,
    run_ocr: bool = True,
    force: bool = False,
    limit: int | None = None,
    log=print,
    librosa_cfg: LibrosaConfig | None = None,
    gemini_cfg: GeminiConfig | None = None,
    run_librosa: bool = True,
    siglip_cfg: SiglipConfig | None = None,
    run_siglip: bool = False,
) -> dict:
    """Batch version of ``process_video_multimodal``; one video in memory at a time."""
    cfg, speech_cfg, ocr_cfg = cfg or PipelineConfig(), speech_cfg or SpeechConfig(), ocr_cfg or OcrConfig()
    librosa_cfg, gemini_cfg = librosa_cfg or LibrosaConfig(), gemini_cfg or GeminiConfig()
    siglip_cfg = siglip_cfg or SiglipConfig()
    root = Path(root) if root else artifacts_dir()
    inputs = [Path(p) for p in (inputs or default_inputs())]
    videos = discover(inputs)[: limit or None]
    features_root = root / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    seg_path, open_path = features_root / "segments_merged_all.csv", features_root / "first5s_multimodal.csv"
    seg_tmp, open_tmp = seg_path.with_suffix(".csv.tmp"), open_path.with_suffix(".csv.tmp")

    records = []
    with seg_tmp.open("w", newline="") as fs, open_tmp.open("w", newline="") as fo:
        seg_w = csv.DictWriter(fs, fieldnames=MERGED_COLUMNS, extrasaction="ignore")
        open_w = csv.DictWriter(fo, fieldnames=OPENING_MM_COLUMNS, extrasaction="ignore")
        seg_w.writeheader()
        open_w.writeheader()
        for i, item in enumerate(videos, 1):
            try:
                res = process_video_multimodal(item, root, cfg, speech_cfg, ocr_cfg, backends, run_speech, run_ocr, force,
                                               librosa_cfg, gemini_cfg, run_librosa, siglip_cfg, run_siglip)
            except Exception as exc:  # isolate per-video failures
                res = {"video_id": item.video_id, "source_path": str(item.path), "status": "failed",
                       "error": f"pipeline error: {type(exc).__name__}: {exc}", "rows": [], "opening": None}
            for row in res["rows"]:
                seg_w.writerow({k: _csv_value(row.get(k)) for k in MERGED_COLUMNS})
            if res["opening"]:
                open_w.writerow({k: _csv_value(v) for k, v in opening_row(res["opening"]).items()})
            rec = _record(res)
            records.append(rec)
            log(_log_line(i, len(videos), rec))
    seg_tmp.replace(seg_path)
    open_tmp.replace(open_path)
    (features_root / LEGACY_OUTPUTS[1]).unlink(missing_ok=True)

    counts = {s: sum(1 for r in records if r["status"] == s) for s in ("ok", "partial", "failed")}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [str(p) for p in inputs],
        "artifacts_root": str(root),
        "stages": {"speech": run_speech, "ocr": run_ocr, "librosa": run_librosa, "siglip": run_siglip, "gemini": gemini_cfg.mode},
        "config": {"pipeline": cfg.to_dict(), "speech": speech_cfg.to_dict(), "ocr": ocr_cfg.to_dict(),
                   "librosa": librosa_cfg.to_dict(), "siglip": siglip_cfg.to_dict(),
                   "gemini": {k: v for k, v in gemini_cfg.to_dict().items() if k != "protected_roots"}},
        "counts": {"videos": len(records), **counts},
        "gemini_api_calls": sum(r.get("gemini_api_calls") or 0 for r in records),
        "gemini_tokens": sum(((r.get("gemini_usage") or {}).get("total_token_count") or 0) for r in records) or None,
        "videos": records,
    }
    _write_json(features_root / "speech_ocr_run_summary.json", summary)
    return summary
