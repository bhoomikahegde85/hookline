"""Gemini explains the predicted drop-off segments and proposes timestamped edits (``gemini_edit.json``).

Division of labour: TTCC *predicts* a retention curve; Gemini *interprets the video content* around the segments TTCC flags and
proposes edits, citing measured evidence. Gemini sees the video (uploaded to Google) plus an evidence pack with citable ids: the
TTCC curve and segments, Whisper words, PaddleOCR text, scene cuts, quiet audio and the rule-based observations. Everything it
returns is a HYPOTHESIS: no claim about why real viewers left, and nothing here has been validated against audience data.

Safeguards:
- cost: at most ``max_analysed_s`` seconds of frames are analysed. A video whose predicted span fits is sent whole; a longer one is sent as
  clips (the opening plus the flagged segments with context), so a long video costs about the same as a 2-minute one;
- off by default; the video is uploaded only with ``allow_remote``, and company reels need their id in ``approved_ids``;
- the API key comes from GEMINI_API_KEY (environment or untracked ``.env``) and never appears in an artifact or message;
- the uploaded file is deleted from Google after the call;
- the answer is validated strictly: unknown segments, invented evidence ids, timestamps outside the predicted span and
  suggestions without cited evidence are dropped and counted, never repaired silently;
- failures never raise and never replace an earlier usable result.
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import GEMINI_EDIT_VERSION, GeminiConfig, file_sha256, stable_hash
from .paths import PROJECT_ROOT
from .tribe import load_secret

ARTIFACT = "gemini_edit.json"
CATEGORIES = ("tighten_hook", "remove_dead_space", "adjust_pacing", "change_cuts", "add_caption", "fix_audio", "other")
EDIT_TYPES = ("trim", "add", "review")
NOTE = ("Model interpretation (Gemini) of a video segment and unvalidated edit hypotheses. TTCC's retention curve is a prediction, not audience "
        "data; neither model has been validated on these videos.")
MAX_WORDS, MAX_CUTS, MAX_TEXT, MAX_RULES = 300, 80, 60, 40
WORDS_PER_ANALYSED_S = 4  # transcript budget scales with the seconds sent, so a longer clip set keeps its late words too


class GeminiUnavailable(RuntimeError):
    """Credentials, SDK or remote permission missing: nothing was sent."""


class Transport(Protocol):
    def generate(self, video_path: Path, system: str, prompt: str, schema: dict, ranges: list[tuple[float, float]]) -> tuple[str, dict]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------- evidence pack

def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


FLAGGED_KINDS = ("steep_drop", "sustained_decline")


def _union(ranges: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1.0:  # touching or one second apart: one clip
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def select_ranges(segments: list[dict], span: float, cfg: GeminiConfig | None) -> dict:
    """The seconds of video Gemini is given. Whole predicted span when it fits ``cfg.max_analysed_s``; otherwise the opening plus every
    flagged segment with margin, largest predicted fall first, each added only if the total stays within the budget."""
    ids = [s["id"] for s in segments]
    if cfg is None or span <= cfg.max_analysed_s:
        return {"ranges": [[0.0, float(span)]], "full": True, "analysed": ids, "not_analysed": []}
    chosen = _union([[0.0, min(cfg.hook_s, float(span))]])
    for s in sorted((s for s in segments if s["kind"] in FLAGGED_KINDS), key=lambda s: s.get("rank_by_predicted_loss") or 10**9):
        trial = _union(chosen + [[max(0.0, s["start_s"] - cfg.context_margin_s), min(float(span), s["end_s"] + cfg.context_margin_s)]])
        if sum(b - a for a, b in trial) <= cfg.max_analysed_s:
            chosen = trial
    covered = lambda s: any(a - 1e-6 <= s["start_s"] and s["end_s"] <= b + 1e-6 for a, b in chosen)  # noqa: E731
    analysed = [s["id"] for s in segments if covered(s)]
    why = lambda s: (f"over the analysis budget (max_analysed_s={cfg.max_analysed_s:g} s)" if s["kind"] in FLAGGED_KINDS  # noqa: E731
                     else "no predicted drop here and outside the analysed clips")
    return {"ranges": chosen, "full": False, "analysed": analysed,
            "not_analysed": [{"id": s["id"], "reason": why(s)} for s in segments if s["id"] not in analysed]}


def build_evidence_pack(d: Path, cfg: GeminiConfig | None = None) -> dict | None:
    """Everything Gemini may cite, from the saved artifacts, restricted to the analysed clips (the whole predicted span for a video that
    fits ``cfg.max_analysed_s``). None without a drop-off analysis."""
    dr = _load(d / "ttcc_dropoffs.json")
    art = _load(d / "ttcc_retention.json")
    if not dr or not art or art.get("status") != "success":
        return None
    span = float(art["horizon_s"])
    sel = select_ranges(dr["segments"], span, cfg)
    ranges = sel["ranges"]
    inside = lambda t0, t1: any(t0 < b and t1 >= a for a, b in ranges)  # noqa: E731  (point times are inside when a <= t < b)
    analysed = set(sel["analysed"])
    pack: dict[str, Any] = {
        "video_id": art["video_id"], "video_duration_s": art.get("duration_s"), "predicted_span_s": [0.0, span],
        "analysis_scope": {"clips_s": [[round(a, 2), round(b, 2)] for a, b in ranges], "whole_predicted_span": sel["full"],
                           "analysed_segments": sel["analysed"], "not_analysed": sel["not_analysed"]},
        "ttcc": {"note": "TTCC is a pretrained model's PREDICTION of the share of viewers still watching at each second; it is not audience data "
                         "and is unvalidated on this video.",
                 "retention_by_second": [round(r, 3) for r in art["retention"]],
                 "segments": [{"id": s["id"], "kind": s["kind"], "start_s": s["start_s"], "end_s": s["end_s"],
                               "retention_start": round(s["retention_start"], 3), "retention_end": round(s["retention_end"], 3),
                               "predicted_fall": round(s["predicted_loss"], 3),
                               "share_of_total_predicted_fall": None if s["share_of_total_predicted_loss"] is None
                               else round(s["share_of_total_predicted_loss"], 3),
                               "rank_by_predicted_fall": s["rank_by_predicted_loss"],
                               "heuristic_attention_risk": s["heuristic"]["attention_risk"], "heuristic_moment_quality": s["heuristic"]["moment_quality"],
                               "analysed": s["id"] in analysed}
                              for s in dr["segments"]]},
        "evidence_availability": dr["evidence_availability"], "speech_language": dr.get("speech_language"),
    }
    ids: set[str] = {s["id"] for s in dr["segments"]} | {f"ttcc:t={t}" for t in range(len(art["retention"]))}
    truncated: dict[str, dict] = {}
    max_words = max(MAX_WORDS, math.ceil(WORDS_PER_ANALYSED_S * sum(b - a for a, b in ranges)))

    def capped(name: str, items: list[dict], cap: int, t_key: str = "t0") -> list[dict]:
        if len(items) > cap:  # never silent: the pack says what was left out, and the prompt tells the model absence is not evidence
            truncated[name] = {"kept": cap, "total": len(items), "last_kept_s": items[cap - 1][t_key]}
        return items[:cap]

    words = []
    wp = d / "words.csv"
    if wp.is_file() and dr["evidence_availability"].get("speech") == "available":
        with wp.open(newline="") as f:
            for r in csv.DictReader(f):
                if float(r["start_s"]) < span and inside(float(r["start_s"]), float(r["end_s"])):
                    words.append({"id": f"w{r['word_index']}", "t0": round(float(r["start_s"]), 2), "t1": round(float(r["end_s"]), 2), "text": r["word"].strip()})
        words = capped("words", words, max_words)
    pack["words"] = words if dr["evidence_availability"].get("speech") == "available" else "not run"
    ocr = _load(d / "ocr.json")
    texts = [{"id": f"t{e['event_id']}", "t0": e["first_seen_s"], "t1": e["visible_until_s"], "text": e["text"][:MAX_TEXT]}
             for e in (ocr or {}).get("events") or [] if e["first_seen_s"] < span and inside(e["first_seen_s"], e["visible_until_s"])] if ocr and ocr.get("measured") else None
    pack["on_screen_text"] = texts if texts is not None else "not run"
    rep = _load(d / "report.json")
    cuts = [{"id": f"c{i}", "t": round(t, 2)} for i, t in enumerate((rep or {}).get("scene_cut_times") or []) if t < span and inside(t, t)] if rep else None
    cuts = capped("scene_cuts", cuts, MAX_CUTS, "t") if cuts is not None else None
    pack["scene_cuts"] = cuts if cuts is not None else "not run"
    lib = _load(d / "librosa.json")
    quiet = [{"id": f"q{q['interval_id']}", "t0": q["start_s"], "t1": q["end_s"]} for q in (lib or {}).get("low_energy_intervals") or []
             if q["start_s"] < span and inside(q["start_s"], q["end_s"])] if lib and lib.get("measured") else None
    pack["quiet_audio"] = quiet if quiet is not None else "not run"
    recs = (_load(d / "recommendations.json") or {}).get("recommendations") or []
    pack["rule_observations"] = [{"id": f"r{i}", "rule": r["rule"], "t0": r["start_s"], "t1": r["end_s"], "text": r["suggestion"]}
                                 for i, r in enumerate(recs) if r["start_s"] < span and inside(r["start_s"], r["end_s"])]
    pack["rule_observations"] = capped("rule_observations", pack["rule_observations"], MAX_RULES)
    if truncated:
        pack["evidence_truncated"] = truncated
    for key in ("words", "on_screen_text", "scene_cuts", "quiet_audio", "rule_observations"):
        for item in pack[key] if isinstance(pack[key], list) else []:
            ids.add(item["id"])
    pack["_ids"] = sorted(ids)  # kept for validation; removed before the pack is sent
    pack["_ranges"], pack["_analysed"] = [list(r) for r in ranges], sorted(analysed)
    return pack


def pack_for_prompt(pack: dict) -> dict:
    return {k: v for k, v in pack.items() if not k.startswith("_")}


def evidence_digest(pack: dict | None) -> str | None:
    return stable_hash(pack_for_prompt(pack)) if pack else None


# ---------------------------------------------------------------- prompt and schema

SYSTEM = (
    "You are a short-form video editing analyst. You explain what is observable in a video segment and propose specific, timestamped edits. "
    "TTCC, a separate pretrained model, PREDICTS a retention curve; you do not predict retention. You do not know why real viewers leave, and "
    "nothing here has been validated against audience data, so every explanation must be phrased as a hypothesis (may, could, possibly). "
    "Ground every observation and suggestion in the evidence pack: cite its ids. Never invent timestamps outside the analysed clips. "
    "Report every timestamp as ORIGINAL video seconds, never as time inside a clip. "
    "If the evidence is insufficient for a segment, say so and return no suggestions for it. Quote transcript text as given (it may not be English)."
)

PROMPT = """Analyse the attached video ONLY within seconds {ranges_text}. TTCC's prediction and the measured evidence are below as JSON.
{clips_note}
TASK, for every TTCC segment with "analysed": true (including segments with no predicted drop, where only brief notes are needed). Use the predicted
retention trend (retention_by_second, the whole video) to decide what matters most: a segment with a larger predicted fall deserves more attention:
1. observations: what is visible, audible or on screen in that segment. Each has "kind": "observation" (directly seen/heard/measured; must cite at least one
   evidence id) or "interpretation" (your reading). Cite ids from the pack in "evidence_refs".
2. hypothesis: one hedged sentence on why attention might drop in this segment, or "no hypothesis: <reason>". Give a confidence 0 to 1 that reflects how well the
   cited evidence supports it (not a retention probability).
3. suggestions (at most {max_suggestions} per segment): specific edits a creator can apply, each with a category from {categories}, an "action" sentence naming
   what to change and where, an "edit" {{"type": "trim"|"add"|"review", "start_s", "end_s"}} (trim = remove that range; add = insert something at start_s, set
   end_s = start_s; review = look at that range), a "rationale" citing evidence ids, an "expected_effect" hedged as a hypothesis, and a "confidence" 0 to 1.
   Timestamps are seconds from the start of the ORIGINAL video and must lie inside the analysed clips ({ranges_text}). Prefer concrete edits (remove dead space between X s and Y s, bring the
   first spoken line to 0.5 s, cut the repeated text at Z s) over general advice.
Return JSON only, matching the schema. Include "overall": a hook assessment (is the first 3 seconds' content observable and what does it contain) and the
segment ids you consider the top priorities, in order.

If the pack has "evidence_truncated", the listed sources were cut after "last_kept_s": missing words, cuts or notes after that second mean "not included", not silence or no cut.

EVIDENCE PACK:
{pack}
"""


MULTI_CLIP_NOTE = (
    "\nThe video is given as {n} separate clips, each preceded by a text label with the ORIGINAL second where that clip starts. Time inside a clip is "
    "(original second) minus (the clip's start): convert back, and report ORIGINAL video seconds only. Segments with \"analysed\": false lie outside the "
    "clips: their retention values are context only, you cannot see them, so return NO entry for them and do not describe their content.\n")


def ranges_text(ranges: list) -> str:
    parts = [f"{a:g} to {b:g}" for a, b in ranges]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def response_schema() -> dict:
    ref = {"type": "array", "items": {"type": "string"}}
    num = {"type": "number"}
    edit = {"type": "object", "properties": {"type": {"type": "string", "enum": list(EDIT_TYPES)}, "start_s": num, "end_s": num},
            "required": ["type", "start_s", "end_s"]}
    suggestion = {"type": "object", "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)}, "action": {"type": "string"}, "edit": edit, "rationale": {"type": "string"},
        "expected_effect": {"type": "string"}, "confidence": num, "evidence_refs": ref},
        "required": ["category", "action", "edit", "rationale", "expected_effect", "confidence", "evidence_refs"]}
    observation = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["observation", "interpretation"]}, "text": {"type": "string"},
                                                     "evidence_refs": ref}, "required": ["kind", "text", "evidence_refs"]}
    segment = {"type": "object", "properties": {
        "segment_id": {"type": "string"}, "summary": {"type": "string"}, "observations": {"type": "array", "items": observation},
        "hypothesis": {"type": "string"}, "hypothesis_confidence": num, "suggestions": {"type": "array", "items": suggestion}},
        "required": ["segment_id", "summary", "observations", "hypothesis", "hypothesis_confidence", "suggestions"]}
    return {"type": "object", "properties": {
        "segments": {"type": "array", "items": segment},
        "overall": {"type": "object", "properties": {"hook_assessment": {"type": "string"}, "top_priorities": {"type": "array", "items": {"type": "string"}}},
                    "required": ["hook_assessment", "top_priorities"]}}, "required": ["segments", "overall"]}


# ---------------------------------------------------------------- validation

def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def _clip_text(s: Any, n: int) -> str:
    return " ".join(str(s or "").split())[:n]


def validate(raw: Any, pack: dict, cfg: GeminiConfig) -> tuple[dict, dict]:
    """Strict validation. Returns (clean analysis, {"issues": [...], "dropped": n}); never repairs, only drops."""
    issues: list[str] = []
    ids = set(pack["_ids"])
    segs = {s["id"]: s for s in pack["ttcc"]["segments"]}
    span = pack["predicted_span_s"][1]
    analysed, ranges = set(pack.get("_analysed") or segs), pack.get("_ranges") or [[0.0, span]]
    clean_segments: dict[str, dict] = {}
    dropped = 0
    if not isinstance(raw, dict) or not isinstance(raw.get("segments"), list):
        return {"segments": [], "overall": None}, {"issues": ["response has no 'segments' list"], "dropped": 1}
    for entry in raw["segments"]:
        sid = entry.get("segment_id") if isinstance(entry, dict) else None
        if sid not in segs:
            issues.append(f"unknown segment_id {sid!r}")
            dropped += 1
            continue
        if sid not in analysed:
            issues.append(f"{sid}: not part of the analysed clips; entry ignored")
            dropped += 1
            continue
        if sid in clean_segments:
            issues.append(f"{sid}: duplicate entry ignored")
            dropped += 1
            continue
        seg = segs[sid]
        obs = []
        for o in entry.get("observations") or []:
            refs = [r for r in (o.get("evidence_refs") or []) if r in ids] if isinstance(o, dict) else []
            bad = [r for r in (o.get("evidence_refs") or []) if r not in ids] if isinstance(o, dict) else []
            if bad:
                issues.append(f"{sid}: dropped unknown evidence ids {bad[:3]}")
            text = _clip_text(o.get("text") if isinstance(o, dict) else "", 400)
            kind = o.get("kind") if isinstance(o, dict) else None
            if not text or kind not in ("observation", "interpretation"):
                dropped += 1
                continue
            if kind == "observation" and not refs:
                issues.append(f"{sid}: observation without valid evidence dropped")
                dropped += 1
                continue
            obs.append({"kind": kind, "text": text, "evidence_refs": refs})
        sugs = []
        for s in entry.get("suggestions") or []:
            edit = s.get("edit") if isinstance(s, dict) else None
            a, b = (_num(edit.get("start_s")), _num(edit.get("end_s"))) if isinstance(edit, dict) else (None, None)
            refs = [r for r in (s.get("evidence_refs") or []) if r in ids] if isinstance(s, dict) else []
            why = None
            if not isinstance(s, dict) or s.get("category") not in CATEGORIES or not isinstance(edit, dict) or edit.get("type") not in EDIT_TYPES:
                why = "bad category or edit type"
            elif a is None or b is None or a < 0 or b > span + 1e-6 or b < a:
                why = f"timestamps outside the predicted span 0-{span:g} s"
            elif not any(a >= r0 - 1.0 and b <= r1 + 1.0 for r0, r1 in ranges):
                why = "edit lies outside the analysed clips"
            elif edit["type"] == "trim" and not 0.2 <= b - a <= 30:
                why = "trim must be 0.2 to 30 s long"
            elif edit["type"] != "add" and not (a < seg["end_s"] + 1.0 and b > seg["start_s"] - 1.0):
                why = "edit does not overlap the segment"
            elif edit["type"] == "add" and not (seg["start_s"] - 1.0 <= a <= seg["end_s"] + 1.0):
                why = "insert point is outside the segment"
            elif not refs:
                why = "no valid evidence cited"
            elif not _clip_text(s.get("action"), 400):
                why = "empty action"
            if why:
                issues.append(f"{sid}: suggestion dropped ({why})")
                dropped += 1
                continue
            conf = _num(s.get("confidence"))
            sugs.append({"category": s["category"], "action": _clip_text(s["action"], 400), "rationale": _clip_text(s.get("rationale"), 500),
                         "expected_effect": _clip_text(s.get("expected_effect"), 300), "confidence": None if conf is None else min(1.0, max(0.0, conf)),
                         "evidence_refs": refs, "edit": {"type": edit["type"], "start_s": a, "end_s": a if edit["type"] == "add" else b},
                         "status": "hypothesis (Gemini), unvalidated"})
        if len(sugs) > cfg.max_suggestions_per_segment:
            issues.append(f"{sid}: {len(sugs) - cfg.max_suggestions_per_segment} suggestion(s) beyond the cap of {cfg.max_suggestions_per_segment} dropped")
            dropped += len(sugs) - cfg.max_suggestions_per_segment
            sugs = sugs[:cfg.max_suggestions_per_segment]
        hc = _num(entry.get("hypothesis_confidence"))
        clean_segments[sid] = {"segment_id": sid, "summary": _clip_text(entry.get("summary"), 500), "observations": obs,
                               "hypothesis": _clip_text(entry.get("hypothesis"), 500),
                               "hypothesis_confidence": None if hc is None else min(1.0, max(0.0, hc)), "suggestions": sugs}
    ov = raw.get("overall") if isinstance(raw.get("overall"), dict) else None
    overall = {"hook_assessment": _clip_text(ov.get("hook_assessment"), 600),
               "top_priorities": [p for p in (ov.get("top_priorities") or []) if p in analysed]} if ov else None
    missing = [sid for sid in segs if sid in analysed and sid not in clean_segments]
    if missing:
        issues.append(f"no usable entry for segment(s) {missing}")
    return {"segments": list(clean_segments.values()), "overall": overall}, {"issues": issues[:60], "dropped": dropped}


# ---------------------------------------------------------------- permission and transport

def remote_block_reason(video_id: str, video_path: Path, cfg: GeminiConfig, root: Path | None = None) -> str | None:
    if not cfg.enabled:
        return f"gemini mode is {cfg.mode!r}; nothing is uploaded"
    if not cfg.allow_remote:
        return "remote processing not enabled (--gemini-allow-remote)"
    root = root or PROJECT_ROOT
    src = Path(video_path).resolve()
    for protected in cfg.protected_roots:
        base = (root / protected).resolve()
        if (src == base or base in src.parents) and video_id not in cfg.approved_ids:
            return f"{video_id} is under protected {protected}/; approve it explicitly (--gemini-approve {video_id})"
    return None


def _sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiUnavailable("google-genai is not installed in this environment (use .venv-models: pip install -r requirements-models.txt)") from exc
    return genai, types


def _retryable(exc: BaseException) -> bool:
    code = getattr(exc, "code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return code in (408, 429, 500, 502, 503, 504) or isinstance(exc, (TimeoutError, ConnectionError))


class GenaiTransport:
    """google-genai: one Files API upload per instance (reused across retries), wait until ACTIVE, ``generate_content`` with a JSON
    schema, and ``close()`` deletes the upload. Created only after the permission and credential checks."""

    def __init__(self, api_key: str, cfg: GeminiConfig, sleep: Callable[[float], None] = time.sleep, client: Any = None):
        self.cfg, self.sleep = cfg, sleep
        self.genai, self.types = _sdk()
        self.client = client or self.genai.Client(api_key=api_key, http_options=self.types.HttpOptions(timeout=int(cfg.timeout_s * 1000)))
        self.uploaded: Any = None
        self.requests: list[dict] = []

    def _upload(self, path: Path) -> Any:
        if self.uploaded is None:
            self.uploaded = self.client.files.upload(file=str(path), config=self.types.UploadFileConfig(mime_type="video/mp4"))
        end = time.monotonic() + self.cfg.upload_wait_s
        while True:
            f = self.client.files.get(name=self.uploaded.name)
            state = str(getattr(f, "state", "")).upper()
            if state.endswith("ACTIVE"):
                self.uploaded = f
                return f
            if state.endswith("FAILED"):
                raise GeminiUnavailable("Google could not process the uploaded video (file state FAILED)")
            if time.monotonic() >= end:
                raise TimeoutError(f"uploaded file still not ACTIVE after {self.cfg.upload_wait_s:.0f} s")
            self.sleep(2.0)

    def generate(self, video_path: Path, system: str, prompt: str, schema: dict, ranges: Any) -> tuple[str, dict]:
        """``ranges``: [(start_s, end_s), ...] on the original timeline (a bare number n means 0 to n). One Part per range, each preceded by a
        label that tells the model which original second the clip starts at."""
        t = self.types
        ranges = [(0.0, float(ranges))] if isinstance(ranges, (int, float)) else [(float(a), float(b)) for a, b in ranges]
        f = self._upload(video_path)
        mime = getattr(f, "mime_type", None) or "video/mp4"
        parts = []
        for i, (a, b) in enumerate(ranges, 1):
            if len(ranges) > 1:
                parts.append(t.Part(text=f"CLIP {i} of {len(ranges)}: original video seconds {a:g} to {b:g} (clip time 0 = original second {a:g})."))
            parts.append(t.Part(file_data=t.FileData(file_uri=f.uri, mime_type=mime),
                                video_metadata=t.VideoMetadata(start_offset=f"{a:g}s", end_offset=f"{b:g}s", fps=self.cfg.fps)))
        contents = [t.Content(role="user", parts=[*parts, t.Part(text=prompt)])]
        config = t.GenerateContentConfig(system_instruction=system, temperature=self.cfg.temperature, response_mime_type="application/json",
                                         response_json_schema=schema, media_resolution=f"MEDIA_RESOLUTION_{self.cfg.media_resolution.upper()}")
        wait = 2.0
        for attempt in range(1, self.cfg.max_retries + 2):
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(model=self.cfg.model, contents=contents, config=config)
                self.requests.append({"attempt": attempt, "ok": True, "seconds": round(time.perf_counter() - t0, 2)})
                um = getattr(resp, "usage_metadata", None)
                usage = {k: getattr(um, k, None) for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")} if um else {}
                return resp.text or "", usage
            except Exception as exc:
                self.requests.append({"attempt": attempt, "ok": False, "error": type(exc).__name__, "seconds": round(time.perf_counter() - t0, 2)})
                if not _retryable(exc) or attempt > self.cfg.max_retries:
                    raise
                self.sleep(wait)
                wait *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    def close(self) -> None:
        if self.uploaded is not None:
            try:
                self.client.files.delete(name=self.uploaded.name)
            except Exception:  # best effort; the file expires on Google's side anyway
                pass
            self.uploaded = None


# ---------------------------------------------------------------- stage

def gemini_key(d: Path, cfg: GeminiConfig, pack: dict | None) -> str:
    art = _load(d / "ttcc_retention.json") or {}
    return stable_hash("gemini_edit", art.get("source_sha256"), cfg.cache_dict(), evidence_digest(pack))


def _scrub(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def extract_gemini_edit(d: Path, video_path: Path, cfg: GeminiConfig, transport_factory: Callable[[str], Transport] | None = None,
                        env: dict | None = None, root: Path | None = None, dotenv: Path | None = None) -> dict | None:
    """One Gemini interpretation of the video's drop-off segments. None when disabled. Never raises: every problem is the artifact's ``reason``."""
    if not cfg.enabled:
        return None
    pack = build_evidence_pack(d, cfg)
    art = _load(d / "ttcc_retention.json") or {}
    base = {"schema": "hookline.gemini_edit", "version": GEMINI_EDIT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "video_id": art.get("video_id") or d.name, "source_sha256": art.get("source_sha256"), "cache_key": gemini_key(d, cfg, pack),
            "ttcc_cache_key": art.get("cache_key"),
            "status": "failed", "measured": False, "sent": False, "reason": None, "model": cfg.model,
            "config": {k: v for k, v in cfg.to_dict().items() if k not in ("approved_ids", "protected_roots")},
            "evidence_digest": evidence_digest(pack), "note": NOTE, "usage": None, "requests": [], "upload_mb": None}
    if pack is None:
        base["reason"] = "no TTCC drop-off analysis to interpret (run scripts/run_ttcc.py or scripts/run_e2e.py first)"
        return base
    blocked = remote_block_reason(base["video_id"], video_path, cfg, root)
    if blocked:
        base["reason"] = f"not sent: {blocked}"
        return base
    key = load_secret("GEMINI_API_KEY", env, dotenv)
    if not key:
        base["reason"] = "not sent: GEMINI_API_KEY is not set (environment or untracked .env)"
        return base
    video_path = Path(video_path)
    if not video_path.is_file():
        base["reason"] = f"not sent: video not found at {video_path}"
        return base
    if art.get("source_sha256") and file_sha256(video_path) != art["source_sha256"]:
        base["reason"] = "not sent: the video file changed since its TTCC prediction was made"
        return base
    size = video_path.stat().st_size
    base["upload_mb"] = round(size / 2**20, 2)
    if size > cfg.max_upload_mb * 2**20:
        base["reason"] = f"not sent: {base['upload_mb']} MB exceeds max_upload_mb={cfg.max_upload_mb}"
        return base
    ranges = [tuple(r) for r in pack["_ranges"]]
    analysed_s = sum(b - a for a, b in ranges)
    base["analysis_scope"] = {
        "clips_s": [list(r) for r in ranges], "analysed_s": round(analysed_s, 2), "predicted_span_s": pack["predicted_span_s"][1],
        "whole_predicted_span": pack["analysis_scope"]["whole_predicted_span"], "frames_sampled_approx": round(analysed_s * cfg.fps),
        "fps": cfg.fps, "media_resolution": cfg.media_resolution, "max_analysed_s": cfg.max_analysed_s,
        "analysed_segment_ids": pack["analysis_scope"]["analysed_segments"], "not_analysed": pack["analysis_scope"]["not_analysed"],
        "upload": "the whole video file is uploaded to Google's Files API for the call and deleted afterwards; only these clips are given to the model as frames"}
    prompt = PROMPT.format(ranges_text=ranges_text(ranges), clips_note=MULTI_CLIP_NOTE.format(n=len(ranges)) if len(ranges) > 1 else "",
                           max_suggestions=cfg.max_suggestions_per_segment, categories=", ".join(CATEGORIES),
                           pack=json.dumps(pack_for_prompt(pack), ensure_ascii=False))
    try:
        transport = (transport_factory or (lambda k: GenaiTransport(k, cfg)))(key)
    except GeminiUnavailable as exc:
        base["reason"] = f"not sent: {_scrub(str(exc), key)}"
        return base
    t0 = time.perf_counter()
    try:
        base["sent"] = True
        text, usage = transport.generate(video_path, SYSTEM, prompt, response_schema(), ranges)
        base["usage"], base["requests"] = usage, getattr(transport, "requests", [])
        try:
            raw = json.loads(text)
        except ValueError as exc:
            base["reason"] = f"Gemini returned text that is not valid JSON ({exc})"
            return base
        analysis, validation = validate(raw, pack, cfg)
    except Exception as exc:
        base["requests"] = getattr(transport, "requests", [])
        base["reason"] = f"{type(exc).__name__}: {_scrub(str(exc), key)[:300]}"  # scrub the whole message first: a cut could split the key
        return base
    finally:
        try:
            transport.close()
        except Exception:
            pass
    usable = sum(len(s["suggestions"]) + len(s["observations"]) for s in analysis["segments"])
    base.update(analysis=analysis, validation=validation, seconds=round(time.perf_counter() - t0, 2), measured=usable > 0,
                status=("uncertain" if validation["dropped"] else "available") if usable else "failed",
                reason=None if usable else "Gemini's answer contained nothing that passed validation")
    return base


def write_gemini_edit(d: Path, art: dict) -> bool:
    """Write ``gemini_edit.json``; a failure never replaces an earlier usable result. Returns True if written."""
    path = d / ARTIFACT
    if not art.get("measured"):
        old = _load(path)
        if old and old.get("measured"):
            return False
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(art, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return True


def cached_gemini_edit(d: Path, cfg: GeminiConfig) -> dict | None:
    """The stored successful result if its cache key still matches (same video, config and evidence)."""
    art = _load(d / ARTIFACT)
    if art and art.get("measured") and art.get("cache_key") == gemini_key(d, cfg, build_evidence_pack(d, cfg)):
        return art
    return None
