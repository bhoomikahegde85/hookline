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
- ``recommendations.json``: rule-based edit observations, each citing evidence ids
- ``siglip.json`` + ``siglip_embeddings.npy``: SigLIP 2 frame embeddings and description scores (opt-in)
- ``tribe.json`` + ``tribe_preds.npy``: TRIBE v2 predicted brain responses from a remote endpoint (opt-in;
  kept out of the segment timeline, scores and status: a separate model output, not a feature)
Run level, in ``<artifacts>/features/``: ``segments_merged_all.csv``,
``first5s_multimodal.csv``, ``speech_ocr_run_summary.json``.

Nothing here scores attention or predicts engagement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import LibrosaConfig, OcrConfig, PipelineConfig, SiglipConfig, SpeechConfig, TribeConfig
from .context import LazyBackend, MediaContext
from .features import OPENING_NOTE, SEGMENT_COLUMNS
from .librosa_audio import LIBROSA_COLUMNS, extract_librosa, librosa_key, librosa_window, opening_librosa
from .ocr import OBSERVATION_COLUMNS, TEXT_COLUMNS, extract_ocr, observation_rows, ocr_key, opening_text, text_window
from .paths import artifacts_dir
from .pipeline import (
    VideoInput,
    _load_report,
    _write_csv,
    concat_video_csvs,
    default_inputs,
    discover,
    load_features,
    merge_run_summary,
    video_dirs,
)
from .recommendations import build_recommendations, evidence_index, invalid_evidence, recommendation_ids
from .scoring import SCORE_COLUMNS, score_video
from .siglip import SIGLIP_COLUMNS, extract_siglip, frame_selection, read_frame_times, siglip_key, siglip_window, write_siglip
from .segments import overlap
from .speech import SPEECH_COLUMNS, WORD_COLUMNS, extract_speech, opening_speech, speech_key, speech_window, word_rows
from .status import ABSENT, ALL_STATUSES, AVAILABLE, FAILED, UNCERTAIN
from .tribe import cached_tribe, extract_tribe, tribe_key, write_tribe

# Links into words.csv (word_index) and text_observations.csv / ocr.json (event_id).
WORD_LINK_COLUMNS = ["speech_word_index_first", "speech_word_index_last"]
MERGED_COLUMNS = ([c for c in SEGMENT_COLUMNS if c != "notes"] + SPEECH_COLUMNS + WORD_LINK_COLUMNS + TEXT_COLUMNS
                  + LIBROSA_COLUMNS + SIGLIP_COLUMNS + ["recommendation_ids", "notes"])
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
    siglip: LazyBackend | None = None
    tribe: Callable[[str, str], Any] | None = None  # (url, token) -> transport; None: HTTPS to the endpoint


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
                words: list[dict] | None = None, librosa: dict | None = None,
                siglip: dict | None = None) -> list[dict]:
    """Existing segment rows (visual/audio untouched) extended with speech, text, librosa and SigLIP
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
        idx = by_segment.get(seg["segment_index"]) if s["speech_status"] in (AVAILABLE, UNCERTAIN) else None
        link = {"speech_word_index_first": min(idx) if idx else None, "speech_word_index_last": max(idx) if idx else None}
        row = {k: seg.get(k) for k in MERGED_COLUMNS if k != "notes"} | s | link | t | lr | sg
        row["notes"] = "; ".join([n for n in [seg.get("notes")] if n] + s_notes + t_notes + lr_notes + sg_notes)
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
                       librosa: dict | None = None) -> dict:
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
    run_librosa: bool = True,
    siglip_cfg: SiglipConfig | None = None,
    run_siglip: bool = False,
    tribe_cfg: TribeConfig | None = None,
) -> dict:
    """Speech, OCR, librosa and (if enabled) SigLIP and TRIBE v2 for one video, then the merged
    timeline and recommendations. Reuses cached preprocessing/features (computes them if missing)."""
    cfg, speech_cfg, ocr_cfg = cfg or PipelineConfig(), speech_cfg or SpeechConfig(), ocr_cfg or OcrConfig()
    librosa_cfg = librosa_cfg or LibrosaConfig()
    siglip_cfg, tribe_cfg = siglip_cfg or SiglipConfig(), tribe_cfg or TribeConfig()
    backends = backends or Backends()
    root = Path(root) if root else artifacts_dir()
    report = load_features(item, cfg, root)  # read-only: Task 2 runs in the feature environment
    features_cached = (report.pop("_cache", None) or {}).get("features")
    ctx = MediaContext.from_report(report)
    out_dir = root / "features" / item.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, Any] = {"features": features_cached, "speech": None, "ocr": None, "librosa": None}
    speech = ocr = librosa = siglip = None
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
    words = word_rows(speech, ctx.segments)
    tribe = None
    if tribe_cfg.enabled:  # remote model output kept apart from the timeline; a failure changes nothing else
        tribe = None if force else cached_tribe(out_dir, tribe_key(ctx, tribe_cfg))
        cache["tribe"] = tribe is not None
        if tribe is None:
            tribe, preds = extract_tribe(ctx, tribe_cfg, backends.tribe)
            cache["tribe_written"] = write_tribe(out_dir, tribe, preds, _write_json)
    rows = merged_rows(ctx, speech, ocr, cfg.min_coverage, words, librosa, siglip)
    recs = build_recommendations(rows, words, ocr, librosa, ctx.duration)
    problems = invalid_evidence(recs, evidence_index(rows, words, ocr, librosa))
    for row in rows:
        row["recommendation_ids"] = recommendation_ids(recs, row)
    status = stage_status([a for a in (speech, ocr, librosa, siglip) if a is not None], rows)
    opening = opening_multimodal(ctx, speech, ocr, rows, cfg.opening_seconds, librosa)
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
                   (("speech", speech), ("ocr", ocr), ("librosa", librosa), ("siglip", siglip))},
        "evidence_problems": problems,
        "recommendations": recs,
    })
    scores = score_video(item.video_id, rows, recs)
    _write_json(out_dir / "scores.json", scores)
    _write_csv(out_dir / "segment_scores.csv", SCORE_COLUMNS, scores["segments"])
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
        "tribe": tribe,
        "recommendations": recs,
        "scores": scores,
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
        "speech_result": s.get("speech_result"),
        "speech_flags": s.get("flags") or None,
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
        "tribe_status": (res.get("tribe") or {}).get("status"),
        "tribe_reason": (res.get("tribe") or {}).get("reason"),
        "tribe_sent": (res.get("tribe") or {}).get("sent"),
        "tribe_cached": (res.get("cache") or {}).get("tribe"),
        "score_basis": (res.get("scores") or {}).get("basis"),
        "highest_attention_risk": ((res.get("scores") or {}).get("highest_attention_risk") or [])[:3],
        "recommendations": len(res.get("recommendations") or []) if res.get("rows") else None,
        "cache": res.get("cache"),
        "error": res.get("error"),
    }


def _log_line(i: int, n: int, rec: dict) -> str:
    def part(name: str, status: str | None, detail: str) -> str:
        return f"{name}={status}" + (f"({detail})" if detail and status else "")

    s_detail = {True: "speech", False: "no speech", None: ""}[rec["speech_detected"]]
    if rec.get("speech_result") == "uncertain_speech":
        s_detail = "uncertain speech: every segment flagged"
    if rec["speech_language"] and rec["speech_detected"]:
        s_detail += f", {rec['speech_language']}"
    t_detail = f"{rec['text_events']} text events" if rec["text_detected"] is not None else ""
    cache = rec.get("cache") or {}
    cached = ",".join(k for k in ("speech", "ocr", "librosa", "siglip", "tribe") if cache.get(k))
    reason = rec.get("error") or next((rec[f"{m}_reason"] for m, st in (("speech", "speech_status"), ("text", "text_status"),
                                                                        ("librosa", "librosa_status"), ("siglip", "siglip_status"))
                                       if rec.get(st) == FAILED and rec.get(f"{m}_reason")), None)
    extra = ""
    if rec.get("librosa_status"):
        extra += ", " + part("librosa", rec["librosa_status"], f"{rec['tempo_bpm']:g} bpm" if rec.get("tempo_bpm") else "")
    if rec.get("siglip_status"):
        t = rec.get("siglip_timing") or {}
        extra += ", " + part("siglip", rec["siglip_status"], f"{t['frames']} frames on {t['device']}" if t.get("frames") else "")
    if rec.get("tribe_status"):
        t_detail = "cached" if rec.get("tribe_cached") else (rec.get("tribe_reason") or "")[:80]
        extra += ", " + part("tribe", rec["tribe_status"], t_detail)
    if rec.get("recommendations") is not None:
        extra += f", {rec['recommendations']} recommendations"
    return (f"[{i}/{n}] {rec['video_id']}: {rec['status']} ({part('speech', rec['speech_status'], s_detail)}, "
            f"{part('text', rec['text_status'], t_detail)}{extra}{', cached ' + cached if cached else ''})"
            + (f" reason: {reason[:160]}" if reason and rec["status"] != "ok" else ""))


def write_multimodal_aggregates(features_root: Path) -> list[str]:
    """``segments_merged_all.csv`` and ``first5s_multimodal.csv`` from every video in the root
    (sorted by id). Videos whose per-video files use an older schema are left out and returned."""
    stale = concat_video_csvs(features_root, "segments_merged.csv", MERGED_COLUMNS, features_root / "segments_merged_all.csv")
    stale += [v for v in concat_video_csvs(features_root, "segment_scores.csv", SCORE_COLUMNS,
                                           features_root / "segment_scores_all.csv") if v not in stale]
    rows = []
    for d in video_dirs(features_root, "opening_multimodal.json"):
        o = _load_report(d / "opening_multimodal.json")
        try:
            rows.append(opening_row(o))
        except (KeyError, TypeError):
            if d.name not in stale:
                stale.append(d.name)
    _write_csv(features_root / "first5s_multimodal.csv", OPENING_MM_COLUMNS, rows)
    return sorted(stale)


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
    run_librosa: bool = True,
    siglip_cfg: SiglipConfig | None = None,
    run_siglip: bool = False,
    tribe_cfg: TribeConfig | None = None,
) -> dict:
    """Batch version of ``process_video_multimodal``; one video in memory at a time."""
    cfg, speech_cfg, ocr_cfg = cfg or PipelineConfig(), speech_cfg or SpeechConfig(), ocr_cfg or OcrConfig()
    librosa_cfg = librosa_cfg or LibrosaConfig()
    siglip_cfg, tribe_cfg = siglip_cfg or SiglipConfig(), tribe_cfg or TribeConfig()
    root = Path(root) if root else artifacts_dir()
    inputs = [Path(p) for p in (inputs or default_inputs())]
    videos = discover(inputs)[: limit or None]
    features_root = root / "features"
    features_root.mkdir(parents=True, exist_ok=True)

    records = []
    for i, item in enumerate(videos, 1):
        try:
            res = process_video_multimodal(item, root, cfg, speech_cfg, ocr_cfg, backends, run_speech, run_ocr, force,
                                           librosa_cfg, run_librosa, siglip_cfg, run_siglip, tribe_cfg)
        except Exception as exc:  # isolate per-video failures
            res = {"video_id": item.video_id, "source_path": str(item.path), "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}", "rows": [], "opening": None}
        rec = _record(res)
        records.append(rec)
        log(_log_line(i, len(videos), rec))
        del res  # one video in memory at a time

    stale = write_multimodal_aggregates(features_root)
    if stale:
        log(f"run-level files exclude {len(stale)} videos with older per-video outputs (re-run them): {', '.join(stale[:10])}")
    (features_root / LEGACY_OUTPUTS[1]).unlink(missing_ok=True)
    backfill = {}
    for d in video_dirs(features_root, "opening_multimodal.json"):
        o = _load_report(d / "opening_multimodal.json") or {}
        backfill[d.name] = {"video_id": d.name, "status": o.get("stage_status"), "features_status": o.get("features_status")}
    summary = merge_run_summary(
        features_root / "speech_ocr_run_summary.json", records, "status", backfill=backfill,
        last_run={
            "inputs": [str(p) for p in inputs],
            "stages": {"speech": run_speech, "ocr": run_ocr, "librosa": run_librosa, "siglip": run_siglip,
                       "tribe": tribe_cfg.enabled},
            "config": {"pipeline": cfg.to_dict(), "speech": speech_cfg.to_dict(), "ocr": ocr_cfg.to_dict(),
                       "librosa": librosa_cfg.to_dict(), "siglip": siglip_cfg.to_dict(),
                       "tribe": {k: v for k, v in tribe_cfg.to_dict().items() if k != "protected_roots"}},
            "tribe_requests_sent": sum(1 for r in records if r.get("tribe_sent") and not r.get("tribe_cached")),
        },
        extra={"excluded_stale_schema": stale},
    )
    return summary
