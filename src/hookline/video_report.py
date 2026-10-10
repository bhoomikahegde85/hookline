"""Readable per-video report (``<artifacts>/features/<video_id>/video_report.md``).

Built only from artifacts already on disk; it computes nothing. Four kinds of output stay in
separate sections and are never combined into one number:

1. measured multimodal features (Task 2 visual/audio, Whisper, PaddleOCR, librosa, SigLIP)
2. heuristic ``attention_risk`` / ``moment_quality`` scores (hand-set weights, not predictions)
3. TTCC-predicted retention (a pretrained model's prediction, not measured retention)
4. rule-based edit observations, each citing measured evidence

A missing artifact is reported as "not run" or "not available", never as a negative result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .dropoffs import DROPOFFS_FILE
from .ttcc import PLOT_FILE, RAW_FILE, RETENTION_FILE, STATUS_FILE, SUCCESS

REPORT_FILE = "video_report.md"


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _f(v, fmt: str = "{:.2f}", none: str = "n/a") -> str:
    if v is None:
        return none
    return fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _s(v) -> str:
    return "n/a" if v is None else f"{v:.2f} s"


def _measured(d: Path) -> list[str]:
    rep, opening = _load(d / "report.json"), _load(d / "opening_multimodal.json")
    out = ["## 1. Measured multimodal features", ""]
    if not rep:
        return out + ["Not available: no Task 2 features for this video (run `.venv/bin/python scripts/run_pipeline.py`).", ""]
    mods = rep.get("modalities") or {}
    o = rep.get("opening") or {}
    out += [f"- Feature extraction: **{rep.get('processing_status')}**; visual stream "
            f"{(mods.get('visual') or {}).get('stream_status')}, audio stream {(mods.get('audio') or {}).get('stream_status')}",
            f"- Scene cuts: {len(rep.get('scene_cut_times') or [])} in total; first 5 s: "
            f"{_f((o.get('visual') or {}).get('scene_cuts'), '{}')} (first at {_s((o.get('visual') or {}).get('first_scene_cut_s'))})",
            f"- First non-silent audio: {_s((o.get('audio') or {}).get('first_non_silent_s'))}"]
    if opening:
        s, t = opening.get("speech") or {}, opening.get("text") or {}
        out.append(f"- Speech (Whisper): {s.get('status') or 'not run'}; first word at {_s(s.get('first_word_s'))}, "
                   f"{_f(s.get('word_count'), '{}')} words in the first 5 s")
        out.append(f"- On-screen text (PaddleOCR): {t.get('status') or 'not run'}; first text at {_s(t.get('first_text_s'))}, "
                   f"{_f(t.get('events'), '{}')} text events in the first 5 s")
    else:
        out.append("- Speech, on-screen text: not run (`scripts/run_speech_ocr.py`)")
    for name, label in (("librosa", "librosa audio features"), ("siglip", "SigLIP 2 visual signals")):
        art = _load(d / f"{name}.json")
        out.append(f"- {label}: {art.get('status') if art else 'not run'}")
    for w in (rep.get("warnings") or [])[:5]:
        out.append(f"- Warning: {w}")
    return out + [""]


def _heuristics(d: Path) -> list[str]:
    scores = _load(d / "scores.json")
    out = ["## 2. Heuristic attention and moment-quality scores", ""]
    if not scores:
        return out + ["Not computed for this video (`scripts/run_speech_ocr.py` writes `scores.json`).", ""]
    out += [f"Basis: `{scores.get('basis')}`. {scores.get('note')}", ""]
    for key, title in (("highest_attention_risk", "attention_risk"), ("highest_moment_quality", "moment_quality")):
        spans = scores.get(key) or []
        out.append(f"Highest `{title}` spans:" if spans else f"No segment has `{title}` evidence.")
        out += [f"- {_f(sp['start_s'])}-{_f(sp['end_s'])} s: {sp.get(title)} ({sp.get('reason')})" for sp in spans[:5]]
        out.append("")
    return out


def _ttcc(d: Path) -> list[str]:
    st, art = _load(d / STATUS_FILE), _load(d / RETENTION_FILE)
    out = ["## 3. TTCC-predicted retention", ""]
    if st:
        out.append(f"Latest attempt ({st.get('attempted_at')}): **{st.get('status')}**"
                   + (f" ({st.get('stage')}): {st.get('reason')}" if st.get("reason") else "")
                   + (" (cached result reused)" if st.get("cached") else ""))
    else:
        out.append("TTCC has not been run for this video (it is off by default; see docs/ttcc.md).")
    if not art or art.get("status") != SUCCESS:
        return out + ["", "No TTCC prediction is available, so no retention curve is shown.", ""]
    current = bool(st and st.get("prediction_is_current"))
    m, s = art.get("model") or {}, art.get("summary") or {}
    out += ["",
            f"Prediction from {art.get('generated_at')}"
            + ("" if current else " (**not confirmed current**: the latest attempt did not reproduce or revalidate it)") + ".",
            *([f"- **Partial prediction:** the video is {_f(art.get('duration_s'), '{:.1f}')} s; the model saw only its "
               f"first {art.get('horizon_s')} s, and {_f(((art.get('coverage') or {}).get('unpredicted_s') or [None])[0], '{:.0f}')}-"
               f"{_f(((art.get('coverage') or {}).get('unpredicted_s') or [None, None])[1], '{:.1f}')} s has no prediction"]
              if art.get("partial_coverage") else []),
            *[f"- Differs from the published GPU evaluation: {d}" for d in
              ((art.get("inference") or {}).get("deviations_from_upstream_eval") or []) if not d.startswith("video is ")],
            f"- Model: `{m.get('checkpoint_repo')}` @ `{(m.get('checkpoint_revision') or '')[:12]}`, {m.get('head_type')} head",
            f"- Horizon: t = 0..{art.get('horizon_s')} s at 1 s resolution (video {_f(art.get('duration_s'))} s); "
            f"{(art.get('convention') or {}).get('t0')}",
            f"- Predicted R({art.get('horizon_s')}) = {_f(s.get('retention_at_horizon'), '{:.3f}')}; steepest one-second drop "
            f"{_f((s.get('steepest_one_second_drop') or {}).get('drop'), '{:.3f}')} between "
            f"{(s.get('steepest_one_second_drop') or {}).get('from_s')} and {(s.get('steepest_one_second_drop') or {}).get('to_s')} s",
            f"- Plot: [{PLOT_FILE}]({PLOT_FILE}); raw output: [{RAW_FILE}]({RAW_FILE}); full record: [{RETENTION_FILE}]({RETENTION_FILE})",
            "", "| t (s) | " + " | ".join(str(t) for t in art["timestamps_s"]) + " |",
            "|---|" + "---|" * len(art["timestamps_s"]),
            "| R(t) | " + " | ".join(f"{r:.3f}" for r in art["retention"]) + " |",
            "", f"{art.get('note')}", ""]
    return out + _dropoffs(d)


def _obs(o: dict) -> str:
    parts = []
    w, t, c, q = o.get("words"), o.get("on_screen_text"), o.get("scene_cuts"), o.get("quiet_audio")
    parts.append("words: " + ("not run" if w == "not run" else f"{w['count']}" + (f" ({w['text']})" if w["count"] else "")))
    parts.append("text: " + ("not run" if t == "not run" else ", ".join(x["text"] for x in t) or "none"))
    parts.append("cuts: " + ("not run" if c == "not run" else str(c["count"])))
    parts.append("quiet audio: " + ("not run" if q == "not run" else str(len(q)) + " interval(s)"))
    if o.get("motion_mean") is not None:
        parts.append(f"motion {o['motion_mean']}")
    return "; ".join(parts)


def _dropoffs(d: Path) -> list[str]:
    dr = _load(d / DROPOFFS_FILE)
    if not dr:
        return []
    out = ["### Predicted drop-offs (derived from the TTCC curve above)", "",
           f"Total predicted fall {dr['curve']['total_predicted_loss']:.3f}; 50 % of it by {dr['curve']['50pct_of_predicted_loss_by_s']} s, "
           f"90 % by {dr['curve']['90pct_of_predicted_loss_by_s']} s. Evidence sources: "
           + ", ".join(f"{k} {v}" for k, v in dr["evidence_availability"].items()) + ".", ""]
    for iv in dr["intervals"]:
        share = iv["share_of_total_predicted_loss"]
        out.append(f"- **{iv['kind'].replace('_', ' ')}, {iv['start_s']}-{iv['end_s']} s**: R {iv['retention_start']:.3f} to "
                   f"{iv['retention_end']:.3f}" + (f" ({100 * share:.0f} % of the predicted fall)" if share is not None else ""))
        out.append(f"  - observed: {_obs(iv['observed'])}")
        for c in iv["candidate_causes"][:3]:
            out.append(f"  - candidate cause (inferred, unvalidated): [{c['rule']}] {c['suggestion']}")
    return out + ["", f"{dr['note']} Confidence: {dr['confidence_note']}.", ""]


def _recommendations(d: Path) -> list[str]:
    recs = _load(d / "recommendations.json")
    out = ["## 4. Evidence-backed editing recommendations", ""]
    if not recs:
        return out + ["Not computed for this video (`scripts/run_speech_ocr.py` writes `recommendations.json`).", ""]
    items = recs.get("recommendations") or []
    if not items:
        out.append("No rule fired on the measured evidence.")
    for r in items:
        out.append(f"- [{r.get('source')}] {_f(r.get('start_s'))}-{_f(r.get('end_s'))} s: {r.get('suggestion')} "
                   f"Why: {r.get('why')} ({len(r.get('evidence') or [])} evidence items)")
    return out + ["", f"{recs.get('note')}", ""]


def write_video_report(out_dir: Path, video_id: str) -> Path:
    rep = _load(out_dir / "report.json") or {}
    lines = [f"# {video_id}", "",
             f"Source: `{rep.get('source_path') or 'n/a'}` (sha256 `{(rep.get('source_sha256') or 'n/a')[:16]}`), "
             f"duration {_f(rep.get('duration_s'))} s. Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
             "from the artifacts in this folder.", "",
             "Sections 2-4 are separate kinds of output: heuristics are hand-weighted evidence counts, TTCC is a "
             "pretrained model's retention prediction, and recommendations cite measured evidence. None of them "
             "guarantees views or virality.", ""]
    lines += _measured(out_dir) + _heuristics(out_dir) + _ttcc(out_dir) + _recommendations(out_dir)
    path = out_dir / REPORT_FILE
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n")
    tmp.replace(path)
    return path
