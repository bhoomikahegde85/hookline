"""TRIBE v2 brain-response predictions from a remote Hugging Face Inference Endpoint (opt-in).

TRIBE v2 (``facebook/tribev2``, CC BY-NC 4.0) predicts fMRI responses of an "average subject" to a
video, on the fsaverage5 cortical mesh: one row per 1-TR segment, one column per cortical vertex.
It is a brain-encoding model. Its outputs are not retention, engagement, attention or virality
predictions, and Hookline does not turn them into any score.

The model never runs here. ``HttpTransport`` posts the video (base64 JSON) to the endpoint that
runs ``endpoint/tribev2/handler.py`` and validates what comes back. Nothing is sent unless TRIBE is
enabled, ``allow_remote`` is set, the endpoint URL and token are configured, and (for company reels
under ``01-Hookline/``) the video id is approved.

Per video, in ``<artifacts>/features/<video_id>/``:
- ``tribe.json``: status, reason, model revision, endpoint/handler versions, timestamps, output
  meaning and provenance. Never contains the endpoint URL or token.
- ``tribe_preds.npy``: float32 (n_segments, n_vertices) predictions exactly as returned.
A failed or blocked attempt never replaces an earlier successful result, and never touches any
other artifact.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .config import TRIBE_VERSION, TribeConfig, stable_hash
from .context import MediaContext
from .paths import PROJECT_ROOT
from .status import ABSENT, AVAILABLE, FAILED

ARTIFACT, PREDS_FILE = "tribe.json", "tribe_preds.npy"
FSAVERAGE5_VERTICES = 20484  # 10242 per hemisphere
RETRY_STATUSES = {429, 500, 502, 503, 504}
OUTPUT_MEANING = {
    "quantity": "TRIBE v2 predicted fMRI (BOLD) response of an average subject to the video",
    "space": "fsaverage5 cortical surface mesh, 20484 vertices (both hemispheres)",
    "units": "arbitrary; training targets were z-scored per sample (config.yaml: data.neuro.cleaning.standardize = zscore_sample)",
    "rows": "one per kept segment of 1 TR; segments with no stimulus event are dropped upstream (TribeModel.predict)",
    "time": "timestamps_s are segment starts in seconds as returned by TribeModel.predict; Hookline does not shift them "
            "(config.yaml sets data.neuro.offset = 5.0 s for training alignment)",
    "not": "not a retention, engagement, attention or virality prediction; not measured brain activity",
}
NOTE = "Predicted brain response from a pretrained encoding model (TRIBE v2); not a measurement and not an engagement score."


class TribeRequestError(RuntimeError):
    """The endpoint could not be reached or answered with an HTTP error."""


class TribeOutputError(ValueError):
    """The endpoint's response is malformed or not what the pinned TRIBE v2 handler returns."""


def load_secret(name: str, env: dict | None = None, dotenv: Path | None = None) -> str | None:
    """``name`` from the environment, else from the untracked ``.env`` (KEY=VALUE lines)."""
    env = os.environ if env is None else env
    if env.get(name):
        return env[name].strip()
    dotenv = PROJECT_ROOT / ".env" if dotenv is None else dotenv
    try:
        for line in dotenv.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip().removeprefix("export ").strip() == name:
                return value.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def remote_block_reason(ctx: MediaContext, cfg: TribeConfig, root: Path | None = None) -> str | None:
    """Why this video may not be uploaded to the endpoint, or None if it may."""
    if not cfg.allow_remote:
        return "remote processing not enabled (--tribe-allow-remote)"
    root = root or PROJECT_ROOT
    src = Path(ctx.source_path).resolve()
    for protected in cfg.protected_roots:
        base = (root / protected).resolve()
        if (src == base or base in src.parents) and ctx.video_id not in cfg.approved_ids:
            return f"{ctx.video_id} is under protected {protected}/; approve it explicitly (--tribe-approve {ctx.video_id})"
    return None


def tribe_key(ctx: MediaContext, cfg: TribeConfig) -> str:
    return stable_hash("tribe", ctx.source_sha256 or ctx.source_path, cfg.cache_dict())


# ---------------------------------------------------------------- transport

class Transport(Protocol):
    def post(self, payload: dict) -> dict: ...


class HttpTransport:
    """POST JSON to the endpoint with a bearer token; retries 429/5xx (503 while a scaled-to-zero
    endpoint starts) and network errors with exponential backoff. ``requests`` records each attempt."""

    def __init__(self, url: str, token: str, cfg: TribeConfig, sleep: Callable[[float], None] = time.sleep,
                 opener: Callable[..., Any] = urllib.request.urlopen):
        self.url, self.token, self.cfg, self.sleep, self.opener = url, token, cfg, sleep, opener
        self.requests: list[dict] = []

    def post(self, payload: dict) -> dict:
        return self.request("POST", "", payload)

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """``method`` ``url + path`` with an optional JSON body; returns the parsed JSON response."""
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        wait = self.cfg.retry_wait_s
        for attempt in range(1, self.cfg.max_retries + 2):
            req = urllib.request.Request(self.url.rstrip("/") + path if path else self.url, data=body, method=method,
                                         headers=headers)
            t0 = time.perf_counter()
            status, error = None, None
            try:
                with self.opener(req, timeout=self.cfg.timeout_s) as resp:
                    status = getattr(resp, "status", 200)
                    text = resp.read().decode()
                self.requests.append({"attempt": attempt, "http_status": status, "seconds": round(time.perf_counter() - t0, 2)})
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise TribeOutputError(f"endpoint returned non-JSON ({exc}): {text[:200]!r}") from exc
            except urllib.error.HTTPError as exc:
                status = exc.code
                detail = exc.read().decode(errors="replace")[:300] if exc.fp else ""
                error = f"HTTP {exc.code} {exc.reason}: {detail}".strip()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
            self.requests.append({"attempt": attempt, "http_status": status, "error": error,
                                  "seconds": round(time.perf_counter() - t0, 2)})
            if status is not None and status not in RETRY_STATUSES:
                raise TribeRequestError(_explain(status, error))
            if attempt > self.cfg.max_retries:
                raise TribeRequestError(f"gave up after {attempt} attempts; last error: {error}")
            self.sleep(wait)
            wait = min(wait * 2, 120.0)
        raise TribeRequestError("unreachable")  # pragma: no cover


class JobTransport(HttpTransport):
    """Job API (a Hugging Face Space running ``endpoint/tribev2_space/app.py``): ``POST {url}/analyze``
    returns a job id, then ``GET {url}/report/{id}`` is polled until the job is ``done`` or ``failed``.
    Returns the same response object as the synchronous handler, so validation is shared."""

    def __init__(self, *args: Any, clock: Callable[[], float] = time.monotonic, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.clock = clock

    def post(self, payload: dict) -> dict:
        submitted = self.request("POST", "/analyze", payload)
        job_id = submitted.get("job_id") if isinstance(submitted, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise TribeOutputError(f"/analyze returned no job_id: {str(submitted)[:200]!r}")
        deadline = self.clock() + self.cfg.job_wait_s
        while True:
            try:
                report = self.request("GET", f"/report/{job_id}")
            except TribeRequestError as exc:
                if "HTTP 404" in str(exc):
                    raise TribeRequestError(
                        f"job {job_id} is gone: the Space no longer knows it. It restarted after accepting the job (a settings or "
                        "secret change, a crash or an out-of-memory kill) and jobs live in memory. Check the Space's Container "
                        "logs, then run again.") from exc
                raise
            status = report.get("status") if isinstance(report, dict) else None
            if status == "done":
                result = report.get("result")
                if not isinstance(result, dict):
                    raise TribeOutputError("report is done but has no result object")
                return result
            if status == "failed":
                return {"error": str(report.get("error") or "job failed")[:800]}
            if status not in ("queued", "running"):
                raise TribeOutputError(f"unexpected job status {status!r}")
            if self.clock() >= deadline:
                raise TribeRequestError(f"job {job_id} still {status} after {self.cfg.job_wait_s:.0f} s; "
                                        "it keeps running on the Space, check its logs")
            self.sleep(self.cfg.poll_s)


def _explain(status: int, error: str | None) -> str:
    hint = {401: "check HF_TOKEN", 403: "the token has no access to this endpoint",
            404: "check TRIBE_ENDPOINT_URL (endpoint deleted or wrong URL)",
            413: "video too large for the endpoint's request limit"}.get(status)
    return f"{error}" + (f" ({hint})" if hint else "")


# ---------------------------------------------------------------- response validation

def parse_response(resp: Any, cfg: TribeConfig) -> tuple[np.ndarray, dict]:
    """Decode and validate the handler's response. Raises ``TribeOutputError`` on anything unexpected."""
    if not isinstance(resp, dict):
        raise TribeOutputError("response is not a JSON object")
    if resp.get("error"):
        raise TribeOutputError(f"endpoint error: {str(resp['error'])[:500]}")
    missing = [k for k in ("model_repo", "model_revision", "tr_s", "mesh", "shape", "dtype", "preds_b64", "segments")
               if k not in resp]
    if missing:
        raise TribeOutputError(f"response lacks {missing}; is the endpoint running endpoint/tribev2/handler.py?")
    if (resp["model_repo"], resp["model_revision"]) != (cfg.model_repo, cfg.model_revision):
        raise TribeOutputError(f"endpoint serves {resp['model_repo']}@{resp['model_revision']}, "
                               f"expected {cfg.model_repo}@{cfg.model_revision}")
    if resp["mesh"] != "fsaverage5" or resp["dtype"] != "float32":
        raise TribeOutputError(f"unexpected mesh/dtype {resp['mesh']!r}/{resp['dtype']!r}")
    shape = resp["shape"]
    if not (isinstance(shape, list) and len(shape) == 2 and all(isinstance(n, int) and n > 0 for n in shape)):
        raise TribeOutputError(f"shape must be [n_segments, n_vertices], got {shape!r}")
    if shape[1] != FSAVERAGE5_VERTICES:
        raise TribeOutputError(f"{shape[1]} vertices, fsaverage5 has {FSAVERAGE5_VERTICES}")
    tr = resp["tr_s"]
    if not (isinstance(tr, (int, float)) and math.isfinite(tr) and tr > 0):
        raise TribeOutputError(f"invalid tr_s {tr!r}")
    try:
        raw = base64.b64decode(resp["preds_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise TribeOutputError(f"preds_b64 is not base64: {exc}") from exc
    if len(raw) != shape[0] * shape[1] * 4:
        raise TribeOutputError(f"preds has {len(raw)} bytes, shape {shape} needs {shape[0] * shape[1] * 4}")
    preds = np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)
    if not np.isfinite(preds).all():
        raise TribeOutputError("predictions contain NaN or infinite values")
    segs = resp["segments"]
    if not (isinstance(segs, list) and len(segs) == shape[0]):
        raise TribeOutputError(f"{len(segs) if isinstance(segs, list) else 'no'} segments for {shape[0]} prediction rows")
    starts = []
    for s in segs:
        start, dur = (s or {}).get("start_s"), (s or {}).get("duration_s")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, dur)) or start < 0 or dur <= 0:
            raise TribeOutputError(f"invalid segment {s!r}")
        starts.append(float(start))
    if any(b < a for a, b in zip(starts, starts[1:])):
        raise TribeOutputError("segment starts are not in time order")
    return preds, {"starts": starts, "durations": [float(s["duration_s"]) for s in segs]}


# ---------------------------------------------------------------- stage

def _base(ctx: MediaContext, cfg: TribeConfig, key: str) -> dict[str, Any]:
    return {
        "schema_version": TRIBE_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": ctx.video_id, "source_sha256": ctx.source_sha256, "cache_key": key,
        "status": FAILED, "measured": False, "sent": False, "reason": None,
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision, "license": "CC BY-NC 4.0 (non-commercial)"},
        "config": {k: v for k, v in cfg.to_dict().items() if k not in ("approved_ids", "protected_roots")},
        "output_meaning": OUTPUT_MEANING, "note": NOTE, "requests": [], "upload_mb": None,
    }


def extract_tribe(ctx: MediaContext, cfg: TribeConfig, transport_factory: Callable[[str, str], Transport] | None = None,
                  env: dict | None = None, root: Path | None = None,
                  dotenv: Path | None = None) -> tuple[dict | None, np.ndarray | None]:
    """One TRIBE v2 request for the video. Returns (artifact, predictions or None); None, None when disabled.
    Never raises: every problem becomes the artifact's ``reason``."""
    if not cfg.enabled:
        return None, None
    art = _base(ctx, cfg, tribe_key(ctx, cfg))
    if ctx.visual_stream == ABSENT and ctx.audio_stream == ABSENT:
        art.update(status=ABSENT, reason="no video or audio stream; nothing to send")
        return art, None
    blocked = remote_block_reason(ctx, cfg, root)
    if blocked:
        art["reason"] = f"not sent: {blocked}"
        return art, None
    url, token = load_secret("TRIBE_ENDPOINT_URL", env, dotenv), load_secret("HF_TOKEN", env, dotenv)
    if not url or not token:
        art["reason"] = "not sent: " + " and ".join(n for n, v in (("TRIBE_ENDPOINT_URL", url), ("HF_TOKEN", token)) if not v) \
                        + " not set (environment or untracked .env)"
        return art, None
    path = Path(ctx.source_path)
    try:
        video = path.read_bytes()
    except OSError as exc:
        art["reason"] = f"not sent: cannot read {path}: {exc}"
        return art, None
    art["upload_mb"] = round(len(video) / 2**20, 2)
    if len(video) > cfg.max_upload_mb * 2**20:
        art["reason"] = f"not sent: {art['upload_mb']} MB exceeds max_upload_mb={cfg.max_upload_mb}"
        return art, None
    payload = {"inputs": {"video_b64": base64.b64encode(video).decode("ascii"), "filename": path.name},
               "parameters": {"language": cfg.language}}
    transport_cls = JobTransport if cfg.api == "job" else HttpTransport
    transport = (transport_factory or (lambda u, t: transport_cls(u, t, cfg)))(url, token)
    t0 = time.perf_counter()
    try:
        art["sent"] = True
        resp = transport.post(payload)
        preds, segs = parse_response(resp, cfg)
    except (TribeRequestError, TribeOutputError) as exc:
        art["reason"] = str(exc).replace(token, "***")
        art["requests"] = getattr(transport, "requests", [])
        return art, None
    except Exception as exc:  # never let a client bug break the pipeline
        art["reason"] = f"{type(exc).__name__}: {exc}".replace(token, "***")
        art["requests"] = getattr(transport, "requests", [])
        return art, None
    art.update(
        status=AVAILABLE, measured=True, reason=None, requests=getattr(transport, "requests", []),
        seconds=round(time.perf_counter() - t0, 2),
        endpoint={k: resp.get(k) for k in ("handler_version", "language", "timing", "versions", "events")},
        tr_s=float(resp["tr_s"]), timestamps_s=segs["starts"], segment_durations_s=segs["durations"],
        output={"file": PREDS_FILE, "shape": list(preds.shape), "dtype": "float32", "sha256": None},
        summary={"n_segments": int(preds.shape[0]), "first_start_s": segs["starts"][0],
                 "last_end_s": round(segs["starts"][-1] + segs["durations"][-1], 3), "video_duration_s": ctx.duration,
                 "value_min": float(preds.min()), "value_max": float(preds.max())},
    )
    return art, preds


def preds_bytes(preds: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, preds.astype(np.float32), allow_pickle=False)
    return buf.getvalue()


def write_tribe(out_dir: Path, art: dict, preds: np.ndarray | None, write_json: Callable[[Path, dict], None]) -> bool:
    """Write ``tribe.json`` (+ predictions). A failure never replaces an earlier successful result."""
    path = out_dir / ARTIFACT
    if not art.get("measured"):
        try:
            old = json.loads(path.read_text())
        except (OSError, ValueError):
            old = None
        if old and old.get("measured"):
            return False
        write_json(path, art)
        return True
    data = preds_bytes(preds)
    art["output"]["sha256"] = hashlib.sha256(data).hexdigest()
    tmp = out_dir / (PREDS_FILE + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out_dir / PREDS_FILE)
    write_json(path, art)
    return True


def cached_tribe(out_dir: Path, key: str) -> dict | None:
    """Earlier successful artifact with this key whose predictions file is intact."""
    try:
        art = json.loads((out_dir / ARTIFACT).read_text())
    except (OSError, ValueError):
        return None
    if not art.get("measured") or art.get("cache_key") != key:
        return None
    npy = out_dir / PREDS_FILE
    if not npy.is_file() or hashlib.sha256(npy.read_bytes()).hexdigest() != (art.get("output") or {}).get("sha256"):
        return None
    return art
