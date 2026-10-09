"""Gemini video interpretation: a structured, timestamped reading of one video.

This is an interpretation layer, kept separate from measured evidence (Task 2 features,
librosa, Whisper words, PaddleOCR observations). It never replaces those, and it is not a
retention, engagement or virality measurement.

Safety and cost controls:
- ``GeminiConfig.mode`` is ``off`` by default. ``offline`` never touches the network (it
  validates a local JSON file instead of an API response). ``remote`` uploads the video.
- Uploads also need ``allow_remote=True``; videos under a protected root (the company
  dataset, ``01-Hookline/``) additionally need their ``video_id`` in ``approved_ids``.
- The API key comes from ``GEMINI_API_KEY`` (environment, else an untracked ``.env``). It is
  never stored, logged or written into artifacts; error messages are scrubbed of it.
- Results are cached by source SHA-256 + model + prompt/schema version + config
  (see ``gemini_key``); unchanged inputs never trigger a second API call.

The SDK call (``google-genai``, ``client.models.generate_content`` with a Files API upload
and a JSON schema) lives only in ``GenaiTransport``; tests replace it with fakes.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import GEMINI_PROMPT_VERSION, GEMINI_SCHEMA_VERSION, SCHEMA_VERSION, GeminiConfig, stable_hash
from .context import MediaContext
from .paths import PROJECT_ROOT
from .status import ABSENT, AVAILABLE, FAILED, UNCERTAIN

GEMINI_COLUMNS = ["gemini_status", "gemini_moment_ids", "gemini_suggestion_ids"]
KINDS = ["observation", "interpretation"]
MODALITIES = ["visual", "audio", "speech", "music", "text", "overall"]
CATEGORIES = ["hook", "visual_change", "scene_clarity", "pacing", "narrative", "audio", "music", "speech",
              "on_screen_text", "other"]
RETRY_CODES = {408, 429, 500, 502, 503, 504}
TIME_TOLERANCE_S = 0.5  # end times up to this far past the duration are clamped, not rejected
POINT_WIDTH_S = 0.5  # a point timestamp (start == end) is widened to this length

_EVIDENCE = {
    "type": "object",
    "properties": {
        "start_s": {"type": "number", "description": "seconds from the start of the video"},
        "end_s": {"type": "number", "description": "seconds from the start of the video; > start_s"},
        "kind": {"type": "string", "enum": KINDS,
                 "description": "observation = directly visible/audible; interpretation = your judgement"},
        "modality": {"type": "string", "enum": MODALITIES},
        "description": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["start_s", "end_s", "kind", "modality", "description", "confidence"],
}
_MOMENT = {
    "type": "object",
    "properties": _EVIDENCE["properties"] | {"category": {"type": "string", "enum": CATEGORIES}},
    "required": _EVIDENCE["required"] + ["category"],
}
_SUGGESTION = {
    "type": "object",
    "properties": {
        "start_s": {"type": "number"},
        "end_s": {"type": "number"},
        "action": {"type": "string", "description": "a concrete edit to this interval"},
        "rationale": {"type": "string", "description": "what in the video motivates the edit"},
        "expected_effect": {"type": "string", "description": "why the edit might help; no retention numbers"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_moment_indices": {"type": "array", "items": {"type": "integer"},
                                    "description": "0-based indices into moments that support this suggestion"},
    },
    "required": ["start_s", "end_s", "action", "rationale", "expected_effect", "confidence", "evidence_moment_indices"],
}
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "video_summary": {"type": "string"},
        "opening_hook": {
            "type": "object",
            "properties": {"assessment": {"type": "string"}, "evidence": {"type": "array", "items": _EVIDENCE}},
            "required": ["assessment", "evidence"],
        },
        "narrative_progression": {"type": "string"},
        "pacing": {"type": "string"},
        "moments": {"type": "array", "items": _MOMENT},
        "strengths": {"type": "array", "items": _EVIDENCE},
        "weaknesses": {"type": "array", "items": _EVIDENCE},
        "edit_suggestions": {"type": "array", "items": _SUGGESTION},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["video_summary", "opening_hook", "narrative_progression", "pacing", "moments", "strengths",
                 "weaknesses", "edit_suggestions", "uncertainties"],
}

PROMPT = """You are reviewing a short-form video before it is published, to help its creator edit it.
The video is {duration:.2f} seconds long. Give every time as seconds from the start of the video
(a number such as 3.5, between 0 and {duration:.2f}); intervals need end_s > start_s.
Detected scene cuts (seconds): {cuts}.

Return JSON matching the schema:
- video_summary: what the video shows and how it is built, in 2-3 sentences.
- opening_hook: an assessment of roughly the first 3-5 seconds, with timestamped evidence.
- narrative_progression and pacing: short assessments.
- moments: notable moments in time order (visual changes, scene clarity, pacing, narrative steps,
  audio/music/speech events), each with start_s/end_s.
- strengths and weaknesses, each tied to an interval.
- edit_suggestions: concrete edits tied to intervals, each citing supporting moments by index.
- uncertainties: anything you could not judge.

Rules:
- Mark each item kind="observation" only if it is directly visible or audible; otherwise "interpretation".
- Do not transcribe or quote dialogue or on-screen text; describe it instead (e.g. "a caption appears").
  Separate tools record exact speech and text.
- Do not claim or estimate viewer retention, watch time, virality or audience reactions.
  "Might help" is fine; numbers about viewers are not.
- If audio is not available to you or unclear, say so in uncertainties instead of guessing.
- Use confidence (0-1) honestly; prefer fewer, well-supported items.
"""


class GeminiTransport(Protocol):
    def analyse(self, video_path: str, prompt: str, schema: dict, cfg: GeminiConfig) -> tuple[str, dict | None]:
        """Return (response text, usage metadata or None)."""


class GeminiUnavailable(RuntimeError):
    """Credentials, SDK or remote permission missing: nothing was sent."""


def load_api_key(env: dict | None = None, dotenv: Path | None = None) -> str | None:
    """``GEMINI_API_KEY`` from the environment, else from an untracked ``.env`` (KEY=VALUE lines)."""
    env = os.environ if env is None else env
    if env.get("GEMINI_API_KEY"):
        return env["GEMINI_API_KEY"].strip()
    dotenv = PROJECT_ROOT / ".env" if dotenv is None else dotenv
    try:
        for line in dotenv.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip().removeprefix("export ").strip() == "GEMINI_API_KEY":
                return value.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def default_model(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    return env.get("GEMINI_MODEL") or GeminiConfig.model


def _scrub(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


class GenaiTransport:
    """``google-genai``: upload via the Files API, wait until ACTIVE, ``generate_content`` with a
    JSON schema, delete the upload. Created only in ``remote`` mode, after the permission checks."""

    def __init__(self, api_key: str, cfg: GeminiConfig, sleep: Callable[[float], None] = time.sleep):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiUnavailable("google-genai is not installed (pip install -r requirements-models.txt)") from exc
        self._types = types
        self._sleep = sleep
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(cfg.timeout_s * 1000)))

    def analyse(self, video_path: str, prompt: str, schema: dict, cfg: GeminiConfig) -> tuple[str, dict | None]:
        t = self._types
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        uploaded = self._client.files.upload(file=video_path, config={"mime_type": mime})
        try:
            waited = 0.0
            while getattr(uploaded.state, "name", str(uploaded.state)) != "ACTIVE":
                state = getattr(uploaded.state, "name", str(uploaded.state))
                if state == "FAILED":
                    raise RuntimeError(f"uploaded file processing failed: {uploaded.error}")
                if waited >= cfg.upload_wait_s:
                    raise TimeoutError(f"uploaded file not ACTIVE after {cfg.upload_wait_s:g} s")
                self._sleep(2.0)
                waited += 2.0
                uploaded = self._client.files.get(name=uploaded.name)
            video = t.Part(file_data=t.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type or mime),
                           video_metadata=t.VideoMetadata(fps=cfg.fps))
            resolution = {"low": t.MediaResolution.MEDIA_RESOLUTION_LOW, "medium": t.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                          "high": t.MediaResolution.MEDIA_RESOLUTION_HIGH}[cfg.media_resolution]
            response = self._client.models.generate_content(
                model=cfg.model,
                contents=[video, prompt],
                config=t.GenerateContentConfig(response_mime_type="application/json", response_json_schema=schema,
                                               temperature=cfg.temperature, media_resolution=resolution),
            )
        finally:
            try:
                self._client.files.delete(name=uploaded.name)
            except Exception:  # best effort; uploads expire after 48 h anyway
                pass
        usage = getattr(response, "usage_metadata", None)
        usage_d = None
        if usage is not None:
            usage_d = {k: getattr(usage, k, None) for k in ("prompt_token_count", "candidates_token_count",
                                                            "thoughts_token_count", "total_token_count")}
        return response.text or "", usage_d


class OfflineTransport:
    """No network: returns the JSON in ``cfg.offline_response`` (or an empty skeleton)."""

    def analyse(self, video_path: str, prompt: str, schema: dict, cfg: GeminiConfig) -> tuple[str, dict | None]:
        if cfg.offline_response:
            return Path(cfg.offline_response).read_text(), None
        return json.dumps(empty_analysis("offline mode: no model output")), None


def empty_analysis(note: str) -> dict:
    return {"video_summary": "", "opening_hook": {"assessment": "", "evidence": []}, "narrative_progression": "",
            "pacing": "", "moments": [], "strengths": [], "weaknesses": [], "edit_suggestions": [], "uncertainties": [note]}


def _file_digest(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def gemini_key(ctx: MediaContext, cfg: GeminiConfig) -> str:
    return stable_hash("gemini", ctx.source_sha256 or ctx.source_path, cfg.cache_dict(), _file_digest(cfg.offline_response))


def remote_block_reason(ctx: MediaContext, cfg: GeminiConfig, root: Path | None = None) -> str | None:
    """Why this video may not be uploaded, or None if it may."""
    if cfg.mode != "remote":
        return f"gemini mode is {cfg.mode!r}; nothing is uploaded"
    if not cfg.allow_remote:
        return "remote processing not enabled (--gemini-allow-remote)"
    root = root or PROJECT_ROOT
    src = Path(ctx.source_path).resolve()
    for protected in cfg.protected_roots:
        base = (root / protected).resolve()
        if src == base or base in src.parents:
            if ctx.video_id not in cfg.approved_ids:
                return f"{ctx.video_id} is under protected {protected}/; approve it explicitly (--gemini-approve {ctx.video_id})"
    return None


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return float(x)


def _interval(item: dict, duration: float | None, issues: list[str], where: str) -> tuple[float, float] | None:
    s, e = _num(item.get("start_s")), _num(item.get("end_s"))
    if s is None or e is None:
        issues.append(f"{where}: dropped, start_s/end_s not numbers")
        return None
    limit = duration if duration else math.inf
    if s < -1e-6 or s >= limit:
        issues.append(f"{where}: dropped, start_s {s:g} outside [0, {limit:g})")
        return None
    if e > limit + TIME_TOLERANCE_S:
        issues.append(f"{where}: dropped, end_s {e:g} beyond duration {limit:g}")
        return None
    if e < s:
        issues.append(f"{where}: dropped, end_s {e:g} < start_s {s:g}")
        return None
    if e == s:
        e = s + POINT_WIDTH_S
        issues.append(f"{where}: point timestamp widened to {POINT_WIDTH_S:g} s")
    return round(max(0.0, s), 3), round(min(e, limit), 3)


_QUOTE = re.compile(r"[\"“”']([^\"“”']{12,})[\"“”']")


def _item(raw: Any, fields: dict, duration: float | None, issues: list[str], where: str) -> dict | None:
    if not isinstance(raw, dict):
        issues.append(f"{where}: dropped, not an object")
        return None
    iv = _interval(raw, duration, issues, where)
    if iv is None:
        return None
    out: dict[str, Any] = {"start_s": iv[0], "end_s": iv[1]}
    for name, allowed in fields.items():
        v = raw.get(name)
        if allowed is str:
            if not isinstance(v, str) or not v.strip():
                issues.append(f"{where}: dropped, {name} missing")
                return None
            out[name] = v.strip()
        elif allowed is float:
            n = _num(v)
            if n is None or not 0 <= n <= 1:
                issues.append(f"{where}: dropped, {name} not in [0, 1]")
                return None
            out[name] = round(n, 3)
        elif isinstance(allowed, list):
            if v not in allowed:
                issues.append(f"{where}: dropped, {name}={v!r} not one of {allowed}")
                return None
            out[name] = v
    text = " ".join(str(out.get(k, "")) for k in ("description", "action", "rationale"))
    if _QUOTE.search(text):
        out["possible_quote"] = True
        issues.append(f"{where}: contains a quoted passage; treat as unverified, use Whisper/OCR for exact words")
    return out


def validate_analysis(data: Any, duration: float | None) -> tuple[dict | None, list[str], str | None]:
    """Check a model response against the local schema. Returns (analysis, issues, fatal_reason).

    Malformed top-level structure is fatal. Individual items with invalid timestamps or fields
    are dropped and listed in ``issues``. Ids: moments m*, hook evidence h*, strengths s*,
    weaknesses w*, suggestions e*; suggestions link to moment ids."""
    issues: list[str] = []
    if not isinstance(data, dict):
        return None, issues, "response is not a JSON object"
    missing = [k for k in RESPONSE_SCHEMA["required"] if k not in data]
    if missing:
        return None, issues, f"response is missing required fields: {missing}"
    extra = sorted(set(data) - set(RESPONSE_SCHEMA["properties"]))
    if extra:
        issues.append(f"ignored unexpected fields: {extra}")
    for k in ("video_summary", "narrative_progression", "pacing"):
        if not isinstance(data[k], str):
            return None, issues, f"{k} is not a string"
    hook = data["opening_hook"]
    if not isinstance(hook, dict) or not isinstance(hook.get("assessment"), str) or not isinstance(hook.get("evidence"), list):
        return None, issues, "opening_hook is malformed"
    for k in ("moments", "strengths", "weaknesses", "edit_suggestions", "uncertainties"):
        if not isinstance(data[k], list):
            return None, issues, f"{k} is not a list"

    ev_fields = {"kind": KINDS, "modality": MODALITIES, "description": str, "confidence": float}

    def items(raw: list, prefix: str, fields: dict, label: str) -> tuple[list[dict], dict[int, str]]:
        out, index_map = [], {}
        for i, r in enumerate(raw):
            it = _item(r, fields, duration, issues, f"{label}[{i}]")
            if it is not None:
                it = {"id": f"{prefix}{len(out)}", **it}
                index_map[i] = it["id"]
                out.append(it)
        return out, index_map

    moments, moment_ids = items(data["moments"], "m", ev_fields | {"category": CATEGORIES}, "moments")
    hook_ev, _ = items(hook["evidence"], "h", ev_fields, "opening_hook.evidence")
    strengths, _ = items(data["strengths"], "s", ev_fields, "strengths")
    weaknesses, _ = items(data["weaknesses"], "w", ev_fields, "weaknesses")
    suggestions, sugg_ids = items(data["edit_suggestions"], "e",
                                  {"action": str, "rationale": str, "expected_effect": str, "confidence": float},
                                  "edit_suggestions")
    by_id = {sg["id"]: sg for sg in suggestions}
    for raw_index, sid in sugg_ids.items():
        refs = data["edit_suggestions"][raw_index].get("evidence_moment_indices")
        links = []
        for ref in refs if isinstance(refs, list) else []:
            if isinstance(ref, int) and not isinstance(ref, bool) and ref in moment_ids:
                links.append(moment_ids[ref])
            else:
                issues.append(f"{sid}: evidence moment index {ref!r} does not refer to a kept moment")
        by_id[sid]["evidence_moment_ids"] = links
    uncertainties = [u for u in data["uncertainties"] if isinstance(u, str)]
    analysis = {
        "video_summary": data["video_summary"].strip(),
        "opening_hook": {"assessment": hook["assessment"].strip(), "evidence": hook_ev},
        "narrative_progression": data["narrative_progression"].strip(),
        "pacing": data["pacing"].strip(),
        "moments": sorted(moments, key=lambda m: (m["start_s"], m["id"])),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "edit_suggestions": suggestions,
        "uncertainties": uncertainties,
    }
    return analysis, issues, None


def _retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in RETRY_CODES:
        return True
    return isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__


def _base(ctx: MediaContext, cfg: GeminiConfig, key: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gemini_prompt_version": GEMINI_PROMPT_VERSION,
        "gemini_schema_version": GEMINI_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": ctx.video_id,
        "source_sha256": ctx.source_sha256,
        "cache_key": key,
        "status": FAILED,
        "measured": False,
        "reason": None,
        "source": None,  # "gemini_api" or "offline_fixture"
        "model": cfg.model if cfg.mode == "remote" else None,
        "config": {k: v for k, v in cfg.to_dict().items() if k not in ("approved_ids", "protected_roots")},
        "timeline": {"convention": "seconds from start of original playback, as returned by the model and validated",
                     "duration_s": ctx.duration},
        "note": "Model interpretation, not measurement. Not a retention, engagement or virality prediction.",
        "api_calls": 0,
        "usage": None,
        "cost": {"usd": None, "note": "not computed: the API returns token counts, not prices"},
        "validation": {"issues": [], "dropped": 0},
        "analysis": None,
        "timing": None,
    }


def extract_gemini(
    ctx: MediaContext,
    cfg: GeminiConfig,
    transport_factory: Callable[[], GeminiTransport] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    env: dict | None = None,
) -> dict | None:
    """One Gemini interpretation. Returns None when mode is off. Never raises."""
    if cfg.mode == "off":
        return None
    art = _base(ctx, cfg, gemini_key(ctx, cfg))
    if ctx.visual_stream == UNCERTAIN and ctx.audio_stream == UNCERTAIN:
        art.update(status=UNCERTAIN, reason="video could not be inspected; not sent")
        return art
    if ctx.visual_stream == ABSENT and ctx.audio_stream == ABSENT:
        art.update(status=ABSENT, reason="no video or audio stream")
        return art
    secret = None
    if cfg.mode == "offline":
        art["source"] = "offline_fixture"
        factory = transport_factory or OfflineTransport
    else:
        blocked = remote_block_reason(ctx, cfg)
        if blocked:
            art["reason"] = f"not sent: {blocked}"
            return art
        secret = load_api_key(env)
        if not secret:
            art["reason"] = "not sent: GEMINI_API_KEY is not set (environment or untracked .env)"
            return art
        art["source"] = "gemini_api"
        factory = transport_factory or (lambda: GenaiTransport(secret, cfg, sleep))
    t0 = time.perf_counter()
    prompt = PROMPT.format(duration=ctx.duration or 0.0, cuts=", ".join(
        f"{s['start_s']:g}" for s in ctx.segments if s.get("scene_cut_at_start") is True) or "none detected")
    try:
        transport = factory()
    except Exception as exc:
        art["reason"] = _scrub(f"not sent: {exc}", secret)
        return art
    text, usage, last_exc = None, None, None
    for attempt in range(cfg.max_retries + 1):
        try:
            if cfg.mode == "remote":
                art["api_calls"] += 1
            text, usage = transport.analyse(ctx.source_path, prompt, RESPONSE_SCHEMA, cfg)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < cfg.max_retries and _retryable(exc):
                sleep(min(60.0, 2.0 * 2**attempt))
                continue
            break
    art["timing"] = {"processing_seconds": round(time.perf_counter() - t0, 3)}
    if text is None:
        code = getattr(last_exc, "code", None)
        art["reason"] = _scrub(f"API call failed{f' ({code})' if code else ''}: {type(last_exc).__name__}: {last_exc}", secret)[:500]
        return art
    art["usage"] = usage
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        art["reason"] = f"model output is not valid JSON: {exc}"
        return art
    analysis, issues, fatal = validate_analysis(data, ctx.duration)
    art["validation"]["issues"] = issues
    art["validation"]["dropped"] = sum(1 for i in issues if ": dropped" in i)
    if fatal:
        art["reason"] = f"model output failed validation: {fatal}"
        return art
    art.update(analysis=analysis, measured=True,
               status=UNCERTAIN if art["validation"]["dropped"] else AVAILABLE,
               reason="some items dropped by validation" if art["validation"]["dropped"] else None)
    return art


def gemini_window(art: dict | None, seg: dict) -> tuple[dict, list[str]]:
    """Ids of Gemini moments and suggestions overlapping one segment (details stay in gemini.json)."""
    out: dict[str, Any] = {k: None for k in GEMINI_COLUMNS}
    if art is None:
        out["gemini_status"] = UNCERTAIN
        return out, ["gemini=not run (disabled)"]
    out["gemini_status"] = art["status"]
    if not art.get("measured"):
        return out, []
    a = art["analysis"]
    s0, s1 = seg["start_s"], seg["end_s"]

    def ids(items: list[dict]) -> list[str] | None:
        return [i["id"] for i in items if i["start_s"] < s1 and i["end_s"] > s0] or None

    out.update(gemini_moment_ids=ids(a["moments"]), gemini_suggestion_ids=ids(a["edit_suggestions"]))
    return out, []
