"""TTCC pretrained audience-retention predictions (opt-in, separate from every heuristic score).

TTCC (``liangyuch/ttcc-sft-qwen25omni-3b-v8``) is Qwen2.5-Omni-3B with a hazard head that predicts
R(t), the fraction of viewers still watching at second t, for t = 1..60. It loads only through the
upstream ``go_viral`` (ttcc-rl) plugin, whose dependencies conflict with both Hookline venvs, so
inference runs in a separate environment: this module validates inputs, builds a request, runs
``scripts/ttcc_infer.py`` as a subprocess, then validates and normalises what comes back. It
imports no model library.

Per video, in ``<artifacts>/features/<video_id>/``:
- ``ttcc_status.json``: the latest attempt, always written when TTCC is enabled: ``success``,
  ``failed`` (input or inference error) or ``unavailable`` (environment, checkpoint or duration
  outside the model's range), with the reason and timings.
- ``ttcc_retention.json``: the last successful prediction (normalised, with provenance). A later
  failure never replaces it; ``ttcc_status.json`` says whether it is current.
- ``ttcc_raw.json``: the runner's raw output for that prediction (all 60 head values).
- ``ttcc_retention.svg``: the predicted curve.

Nothing is interpolated, padded or invented: a video without a successful run has no curve.
The curve keeps TTCC's native 1 s resolution; it is not resampled onto Hookline's 0.5 s segments.
"""

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .config import (
    TTCC_CHECKPOINT_FILES,
    TTCC_UPSTREAM_BRANCH,
    TTCC_UPSTREAM_REPO,
    TTCC_UPSTREAM_VERIFIED_COMMIT,
    TTCC_VERSION,
    TtccConfig,
    file_sha256,
    stable_hash,
)
from .config import TribeConfig
from .media import probe_video
from .paths import PROJECT_ROOT
from .tribe import JobTransport, TribeOutputError, TribeRequestError, load_secret
from .ttcc_windows import AUTO_TIERS, merge_windows, plan_windows, summarize_curve, window_params

SUCCESS, FAILED, UNAVAILABLE, DISABLED = "success", "failed", "unavailable", "disabled"
STATUS_FILE, RETENTION_FILE, RAW_FILE, PLOT_FILE = "ttcc_status.json", "ttcc_retention.json", "ttcc_raw.json", "ttcc_retention.svg"
RAW_REJECTED_FILE = "ttcc_raw_rejected.json"
WINDOWS_DIR = "ttcc_windows"  # per-window raw predictions, cached by key so a rerun only repeats missing windows
RUNNER_SCRIPT = PROJECT_ROOT / "scripts" / "ttcc_infer.py"
T_MIN, T_MAX = 5, 60  # horizon range of the V8 training data (make_v8_from_hf.horizon)
HEAD_OUTPUTS = 60
SHRINK_FACTOR = 0.7  # window length after an out-of-memory failure (attention memory grows with the square of the length)
REMOTE_MAX_JOBS = 16  # windows per upload to the Space (its MAX_JOBS)
MONOTONE_TOLERANCE = 1e-6  # float32 head output; the hazard parameterisation is non-increasing by construction

# Verbatim from go_viral@ttcc-rl examples/custom/qwen2_5_omni_retention/tools/make_v8_from_hf.py
# ("VERBATIM from the trusted V8 rows ... Do not change"); tests compare them with the upstream file.
SYSTEM_PROMPT = (
    "You are an expert in short-form video advertising. You forecast "
    "second-by-second audience retention curves. R(t) is the fraction of "
    "viewers still watching at second t, with R(0) = 1 by convention."
)
USER_PROMPT_TEMPLATE = "This ad is {T} seconds long. Estimate the per-second retention curve."
ASSISTANT_SPAN = "<cot></cot>"  # leak-free "bypass" span used by rl/srcc_eval.py for the published metric
PROMPTS = {"system": SYSTEM_PROMPT, "user_template": USER_PROMPT_TEMPLATE, "assistant": ASSISTANT_SPAN}

CONVENTION = {
    "quantity": "R(t): predicted fraction of viewers still watching at second t",
    "unit": "fraction of viewers (0-1)",
    "time_unit": "s",
    "resolution_s": 1.0,
    "t0": "R(0) = 1 by convention; prepended, not a model output",
    "horizon": ("T = round(duration_s); values for t = 0..T. The head always emits 60 values; those past T are kept in "
                "ttcc_raw.json only. Videos longer than 60 s: the model sees their first 60 s with T = 60, and the rest "
                "of the video has no prediction (see coverage)"),
    "trained_range_s": [T_MIN, T_MAX],
    "monotonicity": "non-increasing by construction (hazard head: R(t) = exp(-cumsum(softplus(z))))",
    "alignment": "native 1 s grid, not resampled onto Hookline's 0.5 s segments",
}
NOTE = ("Prediction of a pretrained model (TTCC v8, trained on TikTok ads), not measured retention. It has not been "
        "validated on these reels (no ground-truth retention is available), and retention is not virality.")
# Library versions that can change a prediction; part of the cache key.
KEY_LIBRARIES = ("torch", "transformers", "qwen-omni-utils", "decord", "eva-decord", "torchcodec", "torchvision", "av",
                 "flash_attn", "safetensors", "ms_swift")


class TtccOutputError(ValueError):
    """The runner's output is missing, malformed or violates the model's output conventions."""


def default_python() -> Path:
    return PROJECT_ROOT / ".venv-ttcc" / "bin" / "python"


def default_repo() -> Path:
    return PROJECT_ROOT / "go_viral"


def horizon(duration_s: float | None) -> int | None:
    """Prompt horizon T, as upstream (``round(duration)``); None outside the trained 5-60 s range."""
    if duration_s is None or not math.isfinite(duration_s):
        return None
    t = round(float(duration_s))
    return t if T_MIN <= t <= T_MAX else None


def plan_window(duration_s: float | None, long_videos: str = "first_60s") -> tuple[int | None, float | None, str | None]:
    """(T, window_end_s, reason). T is the prompt horizon; ``window_end_s`` is set when only the
    start of the video is shown to the model; ``reason`` explains why there is no prediction.

    The head emits R(1..60), so no prediction can reach past 60 s. Videos longer than the trained
    range are, with ``long_videos="first_60s"``, shown to the model as their first 60 s with T=60
    (1 frame/s, exactly like a 60 s training ad); the rest of the video gets no prediction."""
    if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
        return None, None, "duration is unknown"
    t = round(float(duration_s))
    if T_MIN <= t <= T_MAX:
        return t, None, None
    if t < T_MIN:
        return None, None, (f"duration {duration_s:.2f} s is shorter than TTCC's trained range ({T_MIN}-{T_MAX} s after "
                            "rounding); no prediction is made")
    if long_videos == "first_60s":
        return T_MAX, float(T_MAX), None
    return None, None, (f"duration {duration_s:.2f} s is longer than TTCC's trained range ({T_MIN}-{T_MAX} s after rounding) "
                        "and long videos are skipped (--long-videos skip)")


def plan_hook(duration_s: float | None, hook_s: int = 30, prompt_s: int | None = None) -> tuple[int | None, float | None, str | None]:
    """Hook analysis: (prompt T, seconds shown or None for the whole video, reason). One pass over the first min(duration, hook_s) s.

    The model is shown the opening and told the ad is that long (``prompt_s`` overrides what it is told, for sensitivity checks only).
    R(t) is then a native prediction for viewers who start at 0: R(3), R(5), R(10) describe the opening. Nothing is stitched."""
    T, _, why = plan_window(duration_s, "first_60s")
    if T is None:
        return None, None, why
    n = round(float(duration_s))
    shown = min(n, hook_s)
    prompt = prompt_s or shown
    return prompt, (float(shown) if (n > shown or prompt != shown) else None), None


HOOK_SECONDS = (1, 3, 5, 10)
EARLY_S = 10


def hook_metrics(retention: list[float], shown_s: float | None = None) -> dict:
    """R(1), R(3), R(5), R(10) and the steepest one-second drop in the first 10 s, from a start-anchored curve R(0..T).
    A second the model was not shown (or past the curve) is None, never extrapolated."""
    last = len(retention) - 1 if shown_s is None else min(len(retention) - 1, int(shown_s))
    at = {f"R{t}": (float(retention[t]) if t <= last else None) for t in HOOK_SECONDS}
    early = min(EARLY_S, last)
    drops = [(t, retention[t - 1] - retention[t]) for t in range(1, early + 1)]
    steep = max(drops, key=lambda d: d[1]) if drops else None
    return at | {"steepest_early_drop": None if not steep else {"from_s": steep[0] - 1, "to_s": steep[0], "drop": float(steep[1])},
                 "early_window_s": early, "seconds_shown": last,
                 "meaning": "TTCC prediction of the share of viewers still watching (viewers who start at 0); not measured retention"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


# ---------------------------------------------------------------- runner (subprocess boundary)

class Runner(Protocol):
    def probe(self, cfg: TtccConfig) -> dict: ...

    def infer(self, cfg: TtccConfig, request: dict, workdir: Path) -> list[dict]: ...


class SubprocessRunner:
    """Runs ``scripts/ttcc_infer.py`` with the TTCC environment's interpreter."""

    def __init__(self, script: Path = RUNNER_SCRIPT, log=print):
        self.script = script
        self.log = log

    def python(self, cfg: TtccConfig) -> Path:
        return Path(cfg.python).expanduser() if cfg.python else default_python()

    def probe(self, cfg: TtccConfig) -> dict:
        python = self.python(cfg)
        if not python.exists():
            return {"inference_possible": False, "problems": [
                f"TTCC environment not found: {python} (set TTCC_PYTHON or --ttcc-python; setup in docs/ttcc.md)"]}
        repo = Path(cfg.repo_dir).expanduser() if cfg.repo_dir else default_repo()
        try:
            out = subprocess.run([str(python), "-I", str(self.script), "probe", "--repo", str(repo)],
                                 capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"inference_possible": False, "problems": [f"probe could not run: {type(exc).__name__}: {exc}"]}
        try:
            return json.loads(out.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            tail = (out.stderr or out.stdout).strip()[-600:]
            return {"inference_possible": False, "problems": [f"probe failed (exit {out.returncode}): {tail}"]}

    def infer(self, cfg: TtccConfig, request: dict, workdir: Path) -> list[dict]:
        workdir.mkdir(parents=True, exist_ok=True)
        req_path, results_path, log_path = workdir / "request.json", workdir / "results.jsonl", workdir / "runner.log"
        _write_json(req_path, request)
        error = None
        try:
            with log_path.open("w") as log:
                proc = subprocess.run([str(self.python(cfg)), "-I", str(self.script), "infer", "--request", str(req_path),
                                       "--results", str(results_path)], stdout=log, stderr=subprocess.STDOUT,
                                      timeout=cfg.timeout_s)
            if proc.returncode not in (0, 1):
                error = f"runner exited with code {proc.returncode}"
        except subprocess.TimeoutExpired:
            error = f"runner timed out after {cfg.timeout_s:.0f} s"
        except OSError as exc:
            error = f"runner could not start: {exc}"
        results = []
        if results_path.exists():
            for line in results_path.read_text().splitlines():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if error:
            tail = log_path.read_text()[-800:] if log_path.exists() else ""
            done = {r.get("video_id") for r in results}
            results += [{"video_id": j["video_id"], "status": "failed", "stage": "runner",
                         "error": f"{error}; runner log tail: {tail}"} for j in request["jobs"] if j["video_id"] not in done]
        return results


class RemoteRunner:
    """Runs the same upstream runner (``scripts/ttcc_infer.py``) inside a Hugging Face Space
    (``endpoint/ttcc_space/``) and brings the raw outputs back, so nothing heavy runs on this machine.

    ``probe`` reads ``GET /health``; ``infer`` posts each video with ``POST /analyze`` and polls
    ``GET /report/{id}`` (the job API shared with TRIBE). The Space writes the raw output exactly as the
    local runner would, and everything downstream (validation, cache, plot, report) is unchanged."""

    def __init__(self, url: str, token: str, log=print, transport_cls=JobTransport, sleep=time.sleep):
        self.url, self.token, self.log, self.transport_cls, self.sleep = url, token, log, transport_cls, sleep

    def python(self, cfg: TtccConfig) -> str:
        return "remote Space"

    def _transport(self, cfg: TtccConfig):
        net = TribeConfig(api="job", timeout_s=min(cfg.timeout_s, 900.0), job_wait_s=cfg.timeout_s, poll_s=cfg.poll_s,
                          max_retries=6, retry_wait_s=30.0)
        return self.transport_cls(self.url, self.token, net, sleep=self.sleep)

    def probe(self, cfg: TtccConfig) -> dict:
        try:
            health = self._transport(cfg).request("GET", "/health")
        except (TribeRequestError, TribeOutputError) as exc:
            return {"inference_possible": False, "problems": [f"TTCC Space not reachable: {str(exc).replace(self.token, '***')[:300]}"]}
        if not isinstance(health, dict) or "probe" not in health:
            return {"inference_possible": False, "problems": ["the Space did not return a TTCC probe; is it running endpoint/ttcc_space/app.py?"]}
        probe = dict(health["probe"] or {})
        problems = list(probe.get("problems") or [])
        if not health.get("checkpoint_ready"):
            problems.append(health.get("checkpoint_error") or "the Space is still downloading the checkpoint (about 9.4 GB); try again in a few minutes")
            probe["inference_possible"] = False
        probe["problems"] = problems
        probe["inference_possible"] = bool(probe.get("inference_possible")) and not problems
        return probe

    def infer(self, cfg: TtccConfig, request: dict, workdir: Path) -> list[dict]:
        workdir.mkdir(parents=True, exist_ok=True)
        by_video: dict[str, list[dict]] = {}
        for job in request["jobs"]:  # windows of one video travel together: one upload, one model load
            by_video.setdefault(job["video_path"], []).append(job)
        results = []
        for path, jobs in by_video.items():
            for i in range(0, len(jobs), REMOTE_MAX_JOBS):
                results += self._infer_group(cfg, request, jobs[i:i + REMOTE_MAX_JOBS], path)
        return results

    def _infer_group(self, cfg: TtccConfig, request: dict, jobs: list[dict], path: str) -> list[dict]:
        failed = [{"video_id": j["video_id"], "status": "failed", "stage": "remote"} for j in jobs]
        try:
            video = Path(path).read_bytes()
            if len(video) > cfg.max_upload_mb * 2**20:
                raise TribeRequestError(f"{len(video) / 2**20:.1f} MB exceeds max_upload_mb={cfg.max_upload_mb}")
            keys = ("video_id", "T", "video_start", "video_end", "shrink_to_fit")
            specs = [{k: j[k] for k in keys if k in j} for j in jobs]
            meta = {k: request[k] for k in ("settings", "prompts")} | {"checkpoint": {k: request["checkpoint"][k] for k in (
                "repo_id", "revision", "expected_files", "verify_weights_sha256")}}
            payload = {"request": meta | ({"job": specs[0]} if len(specs) == 1 else {"jobs": specs}),
                       "video_b64": base64.b64encode(video).decode("ascii"), "filename": Path(path).name}
            resp = self._transport(cfg).post(payload)
        except (TribeRequestError, TribeOutputError, OSError) as exc:
            return [r | {"error": str(exc).replace(self.token, "***")[:800]} for r in failed]
        if "error" in resp:
            return [r | {"error": str(resp["error"])[:800]} for r in failed]
        items = resp["items"] if isinstance(resp.get("items"), list) else [{"video_id": jobs[0]["video_id"], "result": resp.get("result"), "raw": resp.get("raw")}]
        by_id = {i.get("video_id"): i for i in items if isinstance(i, dict)}
        out = []
        for job, base in zip(jobs, failed):
            item = by_id.get(job["video_id"])
            if not item or not isinstance(item.get("result"), dict) or not isinstance(item.get("raw"), dict):
                out.append(base | {"error": "the Space returned neither a raw output nor an error for this job"})
            elif item["result"].get("status") != "ok":
                out.append(base | {"error": item["result"].get("error"), "stage": item["result"].get("stage") or "inference"})
            else:
                _write_json(Path(job["raw_output"]), item["raw"])
                out.append(item["result"] | {"video_id": job["video_id"], "raw_output": job["raw_output"]})
        return out


def upload_block_reason(video_id: str, path: Path, cfg: TtccConfig, root: Path | None = None) -> str | None:
    """Why this video may not be uploaded to the remote Space, or None if it may."""
    if not cfg.allow_remote:
        return "remote processing not enabled (--ttcc-allow-remote)"
    root = root or PROJECT_ROOT
    src = Path(path).resolve()
    for protected in cfg.protected_roots:
        base = (root / protected).resolve()
        if (src == base or base in src.parents) and video_id not in cfg.approved_ids:
            return f"{video_id} is under protected {protected}/; approve it explicitly (--ttcc-approve {video_id})"
    return None


def remote_runner_or_reason(cfg: TtccConfig, env: dict | None = None, dotenv: Path | None = None, log=print):
    """(runner, None) for remote mode, or (None, reason) when the URL or token is missing."""
    url, token = load_secret("TTCC_ENDPOINT_URL", env, dotenv), load_secret("HF_TOKEN", env, dotenv)
    missing = [n for n, v in (("TTCC_ENDPOINT_URL", url), ("HF_TOKEN", token)) if not v]
    if missing:
        return None, " and ".join(missing) + " not set (environment or untracked .env)"
    return RemoteRunner(url, token, log=log), None


# ---------------------------------------------------------------- validation and normalisation

def validate_video(path: Path) -> tuple[dict, str | None]:
    """Probe the input. Returns (info, problem); ``problem`` is None for a usable video."""
    p = probe_video(path)
    info = {"probe_status": p.status, "duration_s": p.duration_seconds, "has_video": p.has_video,
            "has_audio": p.has_audio, "width": p.width, "height": p.height, "fps": p.fps}
    if p.status != "ok":
        return info, f"input video could not be read: {p.error or p.status}"
    if p.has_video is False:
        return info, "input has no video stream"
    if not p.duration_seconds or p.duration_seconds <= 0:
        return info, "input duration is unknown or zero"
    return info, None


def parse_raw(raw: dict, T: int, head_type: str = "hazard") -> dict:
    """Validate the runner's raw output and return the normalised curve.

    Raises ``TtccOutputError`` unless ``r_pred`` holds exactly 60 finite values in (0, 1] that do not
    increase (the hazard head guarantees all three), and the horizon is in the trained range."""
    if not isinstance(raw, dict):
        raise TtccOutputError("raw output is not a JSON object")
    if not (isinstance(T, int) and T_MIN <= T <= T_MAX):
        raise TtccOutputError(f"horizon T={T!r} outside the trained range {T_MIN}-{T_MAX} s")
    if raw.get("T") != T:
        raise TtccOutputError(f"raw output is for T={raw.get('T')!r}, expected {T}")
    values = raw.get("r_pred")
    if not isinstance(values, list):
        raise TtccOutputError("raw output has no r_pred list")
    if len(values) != HEAD_OUTPUTS:
        raise TtccOutputError(f"r_pred has {len(values)} values, the head emits {HEAD_OUTPUTS}")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise TtccOutputError("r_pred contains missing or non-finite values")
    if not all(0.0 < v <= 1.0 + MONOTONE_TOLERANCE for v in values):
        raise TtccOutputError(f"r_pred outside (0, 1]: min {min(values)}, max {max(values)}")
    if head_type == "hazard":
        bad = [i + 1 for i in range(1, len(values)) if values[i] > values[i - 1] + MONOTONE_TOLERANCE]
        if values[0] > 1.0 + MONOTONE_TOLERANCE or bad:
            raise TtccOutputError(f"hazard-head output increases at t={bad[:5]} s; not a valid hazard-head prediction")
    retention = [1.0] + [float(v) for v in values[:T]]
    timestamps = list(range(T + 1))
    drops = [(t, retention[t - 1] - retention[t]) for t in range(1, T + 1)]
    steepest = max(drops, key=lambda d: d[1])
    return {
        "timestamps_s": timestamps,
        "retention": retention,
        "summary": {
            "retention_at_horizon": retention[-1],
            "steepest_one_second_drop": {"from_s": steepest[0] - 1, "to_s": steepest[0], "drop": steepest[1]},
            "retention_at_s": {str(t): retention[t] for t in (1, 3, 5, 10, 15, 30, 60) if t <= T},
        },
    }


# ---------------------------------------------------------------- runtime (device, attention kernel)

def resolve_runtime(cfg: TtccConfig, probe_info: dict) -> tuple[dict, list[str]]:
    """Concrete ``device_map`` / ``attn_impl`` for this environment, and problems that make the
    requested ones impossible. "detect" picks CUDA, then Apple MPS, then CPU, and flash_attn only
    where FlashAttention 2 runs (CUDA, compute capability >= 8.0, flash_attn installed)."""
    cuda = probe_info.get("cuda") or {}
    libs = probe_info.get("libraries") or {}
    problems: list[str] = []
    device = cfg.device_map
    if device == "detect":
        device = "cuda" if cuda.get("available") else "mps" if cuda.get("mps_available") else "cpu"
    elif device in ("cuda", "auto") and not cuda.get("available"):
        problems.append(f"device_map={device!r} needs a CUDA GPU and none is visible (use --device-map detect, mps or cpu)")
    elif device == "mps" and not cuda.get("mps_available"):
        problems.append("device_map='mps' needs Apple Silicon with MPS available in torch")
    caps = [float(d.get("capability") or 0) for d in cuda.get("devices") or []]
    fa2_ok = device in ("cuda", "auto") and bool(cuda.get("available")) and bool(libs.get("flash_attn")) and bool(caps) and min(caps) >= 8.0
    attn = cfg.attn_impl
    if attn == "detect":
        attn = "flash_attn" if fa2_ok else "sdpa"
    elif attn == "flash_attn" and not fa2_ok:
        problems.append("attn_impl='flash_attn' needs flash_attn installed on a CUDA GPU with compute capability >= 8.0 "
                        "(use --attn-impl detect or sdpa)")
    return {"device_map": device, "attn_impl": attn}, problems


def accelerator(probe_info: dict, device_map: str) -> str:
    """Hardware the prediction ran on (part of the cache key: kernels differ between devices)."""
    cuda = probe_info.get("cuda") or {}
    if device_map in ("cuda", "auto"):
        return "cuda:" + ",".join(d.get("name") or "?" for d in cuda.get("devices") or [])
    return f"{device_map}:{probe_info.get('machine') or probe_info.get('platform') or '?'}"


def deviations(settings: dict, probe_info: dict, window_end: float | None = None, duration: float | None = None,
               hook_s: int | None = None) -> list[str]:
    """Differences from the environment of the checkpoint's published evaluation (CUDA, bf16,
    FlashAttention 2, decord). Recorded with every prediction; they can shift values slightly."""
    out = []
    if window_end is not None:
        why = (f"longer than the analysed opening ({window_end:.0f} s, hook analysis)" if hook_s is not None and abs(window_end - hook_s) < 0.5
               else "longer than the trained 5-60 s range" if (duration or 0) > T_MAX + 0.5
               else f"too many tokens for max_length={settings['max_length']} at full length")
        out.append(f"video is {duration or 0:.1f} s, {why}: the model saw only the first {window_end:.0f} s and was told "
                   f"the ad is {window_end:.0f} s long; nothing after {window_end:.0f} s is predicted")
    if settings["device_map"] not in ("cuda", "auto"):
        out.append(f"device {settings['device_map']} instead of a CUDA GPU")
    if settings["attn_impl"] != "flash_attn":
        out.append(f"attention kernel {settings['attn_impl']} instead of flash_attn")
    if settings["torch_dtype"] != "bfloat16":
        out.append(f"dtype {settings['torch_dtype']} instead of bfloat16")
    reader = probe_info.get("video_reader")
    if reader and reader != "decord":
        out.append(f"video reader {reader} instead of decord")
    if reader == "decord" and (probe_info.get("libraries") or {}).get("eva-decord"):
        out.append("decord provided by a fork build (eva-decord)")
    return out


# ---------------------------------------------------------------- cache identity

def cache_key(source_sha256: str, T: int, cfg: TtccConfig, probe_info: dict, runtime: dict | None = None,
              window_end: float | None = None, window_start: float | None = None) -> str:
    code = probe_info.get("code") or {}
    libs = probe_info.get("libraries") or {}
    runtime = runtime or resolve_runtime(cfg, probe_info)[0]
    return stable_hash(
        "ttcc", TTCC_VERSION, source_sha256, T, cfg.inference_settings(**runtime), PROMPTS,
        {k: list(v) for k, v in TTCC_CHECKPOINT_FILES.items()},
        {k: code.get(k) for k in ("commit", "dirty", "plugin_sha256", "runner_version", "runner_sha256")},
        {k: libs.get(k) for k in KEY_LIBRARIES}, accelerator(probe_info, runtime["device_map"]),
        probe_info.get("video_reader"), window_end, *(() if window_start is None else (window_start,)),
    )


def cached_prediction(out_dir: Path, key: str) -> dict | None:
    """The stored successful prediction if it has this key and its raw output is intact."""
    art = _load_json(out_dir / RETENTION_FILE)
    if not art or art.get("status") != SUCCESS or art.get("cache_key") != key:
        return None
    raw = out_dir / ((art.get("raw_output") or {}).get("path") or RAW_FILE)
    if not raw.is_file() or file_sha256(raw) != (art.get("raw_output") or {}).get("sha256"):
        return None
    return art


# ---------------------------------------------------------------- plot

def retention_svg(art: dict | None, width: int = 640, height: int = 360) -> str | None:
    """SVG of a successful prediction; None for anything else (never draws invented points)."""
    if not art or art.get("status") != SUCCESS:
        return None
    ts, rs = art.get("timestamps_s"), art.get("retention")
    if not (isinstance(ts, list) and isinstance(rs, list) and len(ts) == len(rs) >= 2):
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 + MONOTONE_TOLERANCE for v in rs):
        return None
    left, right, top, bottom = 64, 20, 40, 56
    pw, ph = width - left - right, height - top - bottom
    gap = (art.get("coverage") or {}).get("unpredicted_s")
    t_end = max(ts[-1], math.ceil(gap[1])) if gap and all(isinstance(g, (int, float)) for g in gap) else ts[-1]
    x = lambda t: left + pw * t / t_end  # noqa: E731
    y = lambda r: top + ph * (1.0 - min(r, 1.0))  # noqa: E731
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Helvetica, Arial, sans-serif" font-size="11">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="{left}" y="18" font-size="13" font-weight="bold">{esc(art.get("video_id"))}: TTCC-predicted retention R(t)</text>',
             f'<text x="{left}" y="32" fill="#555">Model prediction (checkpoint {esc((art.get("model") or {}).get("checkpoint_revision", ""))[:12]}), '
             'not measured retention' + ('; stitched from overlapping windows (approximate)' if art.get("stitched")
                                                      else f'; model saw only the first {ts[-1]} s of the video' if art.get("partial_coverage") else '')
             + '</text>']
    for i in range(6):  # y grid every 0.2
        r = i / 5
        parts.append(f'<line x1="{left}" y1="{y(r):.1f}" x2="{left + pw}" y2="{y(r):.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left - 6}" y="{y(r) + 4:.1f}" text-anchor="end">{r:.1f}</text>')
    if t_end > ts[-1]:  # video continues past the model's 60 outputs: shown empty, never filled in
        parts.append(f'<rect x="{x(ts[-1]):.1f}" y="{top}" width="{x(gap[1]) - x(ts[-1]):.1f}" height="{ph}" fill="#f1f1f1"/>')
        parts.append(f'<text x="{(x(ts[-1]) + x(gap[1])) / 2:.1f}" y="{top + ph / 2:.1f}" text-anchor="middle" fill="#777">'
                     f'not predicted</text>')
        parts.append(f'<text x="{(x(ts[-1]) + x(gap[1])) / 2:.1f}" y="{top + ph / 2 + 14:.1f}" text-anchor="middle" '
                     f'fill="#777">({ts[-1]}-{gap[1]:.1f} s)</text>')
    step = 1 if t_end <= 10 else 5 if t_end <= 40 else 10
    for t in range(0, t_end + 1, step):
        parts.append(f'<line x1="{x(t):.1f}" y1="{top + ph}" x2="{x(t):.1f}" y2="{top + ph + 4}" stroke="#333"/>')
        parts.append(f'<text x="{x(t):.1f}" y="{top + ph + 16}" text-anchor="middle">{t}</text>')
    parts.append(f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" stroke="#333"/>')
    parts.append(f'<text x="{left + pw / 2:.1f}" y="{height - 14}" text-anchor="middle">Time (s)</text>')
    parts.append(f'<text transform="translate(16 {top + ph / 2:.1f}) rotate(-90)" text-anchor="middle">'
                 'Predicted retention R(t) (fraction of viewers)</text>')
    pts = " ".join(f"{x(t):.1f},{y(r):.1f}" for t, r in zip(ts, rs))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="1.5" stroke-opacity="0.6"/>')
    parts += [f'<circle cx="{x(t):.1f}" cy="{y(r):.1f}" r="2.5" fill="#2563eb"><title>t={t} s, R={r:.4f}</title></circle>'
              for t, r in zip(ts, rs)]
    parts.append(f'<text x="{left + pw}" y="{top + 12}" text-anchor="end" fill="#555">dots: model outputs at whole seconds; '
                 'R(0)=1 by convention; lines are visual guides</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- orchestration

@dataclass
class TtccJob:
    video_id: str
    source_path: Path
    out_dir: Path
    record: dict = field(default_factory=dict)
    source_sha256: str | None = None
    T: int | None = None
    window_end: float | None = None  # only the first window_end seconds were shown to the model
    key: str | None = None
    plan: list | None = None  # long_videos="windows": [(start_s, end_s), ...] on the original timeline


def _status(job: TtccJob, status: str, reason: str | None, stage: str, **extra) -> dict:
    """Write ``ttcc_status.json`` (latest attempt) and update the run record."""
    current = _load_json(job.out_dir / RETENTION_FILE)
    st = {
        "video_id": job.video_id, "source_path": str(job.source_path), "source_sha256": job.source_sha256,
        "status": status, "stage": stage, "reason": reason, "cache_key": job.key, "horizon_s": job.T,
        "attempted_at": now(), "ttcc_version": TTCC_VERSION,
        "prediction_file": RETENTION_FILE if current and current.get("status") == SUCCESS else None,
        "prediction_is_current": bool(current and job.key and current.get("cache_key") == job.key and status == SUCCESS),
        **extra,
    }
    if job.out_dir.parent.is_dir():
        _write_json(job.out_dir / STATUS_FILE, st)
    job.record.update(status=status, stage=stage, reason=reason, **{k: v for k, v in extra.items() if k != "probe"})
    return st


def _artifact(job: TtccJob, cfg: TtccConfig, raw: dict, curve: dict, raw_sha: str, duration: float | None,
              runtime: dict, probe_info: dict) -> dict:
    settings = cfg.inference_settings(**runtime)
    prov = raw.get("provenance") or {}
    predicted_until = float(job.T)
    coverage = {"predicted_from_s": 0.0, "predicted_until_s": predicted_until, "video_duration_s": duration,
                "model_input_window_s": [0.0, job.window_end if job.window_end is not None else duration],
                "unpredicted_s": [predicted_until, duration] if duration and duration > predicted_until + 0.5 else None}
    return {
        "schema": "hookline.ttcc_retention", "ttcc_version": TTCC_VERSION, "status": SUCCESS,
        "video_id": job.video_id, "source_path": str(job.source_path), "source_sha256": job.source_sha256,
        "duration_s": duration, "horizon_s": job.T, "cache_key": job.key, "generated_at": now(),
        "out_of_trained_range": bool(duration and duration > T_MAX + 0.5), "partial_coverage": job.window_end is not None,
        "coverage": coverage,
        "model": {"name": "TTCC", "checkpoint_repo": cfg.checkpoint_repo, "checkpoint_revision": cfg.checkpoint_revision,
                  "model_type": "qwen2_5_omni_retention", "head_type": cfg.head_type, "base_model": "Qwen/Qwen2.5-Omni-3B",
                  "upstream": f"{TTCC_UPSTREAM_REPO}@{TTCC_UPSTREAM_BRANCH}"},
        "convention": CONVENTION,
        "timestamps_s": curve["timestamps_s"],
        "retention": curve["retention"],
        "summary": curve["summary"],
        "inference": {"settings": settings, "window_end_s": job.window_end, "window_fit": raw.get("window"),
                      "deviations_from_upstream_eval": deviations(settings, probe_info, job.window_end, duration,
                                                                    cfg.hook_s if cfg.long_videos == "hook" else None),
                      "accelerator": accelerator(probe_info, runtime["device_map"]),
                      "video_reader": probe_info.get("video_reader"), "prompts": PROMPTS, "input_tokens": raw.get("input_tokens"),
                      "video_grid_thw": raw.get("video_grid_thw"), "anchor": raw.get("anchor"),
                      "inference_seconds": raw.get("inference_seconds"), "device": prov.get("device"),
                      "cuda_device": prov.get("cuda_device"), "machine": prov.get("machine"),
                      "mps_cpu_fallback": prov.get("mps_cpu_fallback")},
        "provenance": {"runner_version": prov.get("runner_version"), "code": prov.get("code"),
                       "verified_upstream_commit": TTCC_UPSTREAM_VERIFIED_COMMIT, "libraries": prov.get("libraries"),
                       "checkpoint": prov.get("checkpoint"), "retention_head": prov.get("retention_head"),
                       "python": prov.get("python"), "platform": prov.get("platform")},
        "raw_output": {"path": RAW_FILE, "sha256": raw_sha, "values": HEAD_OUTPUTS},
        "note": NOTE,
    }


# ---------------------------------------------------------------- long videos: overlapping windows

CONVENTION_WINDOWED = CONVENTION | {
    "horizon": ("T = round(duration_s); values for t = 0..T, stitched from overlapping windows of the model's output (each window is a "
                "standalone prediction for that part of the video; see windowing). Native single-pass predictions exist only for "
                "windows of at most 60 s"),
    "alignment": "native 1 s grid on the ORIGINAL video timeline (window outputs are shifted by the window start); not resampled onto Hookline's 0.5 s segments",
}
NOTE_WINDOWED = (NOTE + " The curve is stitched from overlapping windows, each predicted as a standalone ad that starts at the window start, "
                 "and blended in hazard space; it is an approximation of one continuous prediction, not a native single-pass result.")


def is_capacity_error(error: str | None) -> bool:
    """True for a failure caused by input size (GPU memory or the token limit), which a shorter window can fix."""
    e = (error or "").lower()
    return "out of memory" in e or "outofmemory" in e or "max_length" in e


def windowed_key(job: TtccJob, cfg: TtccConfig, probe_info: dict, runtime: dict) -> str:
    """Identity of a whole windowed prediction: the video, the model settings and the windowing parameters."""
    return stable_hash("ttcc-windowed", TTCC_VERSION, cache_key(job.source_sha256, job.T, cfg, probe_info, runtime),
                       {k: getattr(cfg, k) for k in ("window_mode", "window_s", "stride_s", "warmup_s", "min_window_s", "auto_shrink_window")},
                       AUTO_TIERS if cfg.window_mode == "auto" else None)


def window_id(video_id: str, start: int, end: int) -> str:
    return f"{video_id}__w{start:05d}_{end:05d}"


def _load_window(out_dir: Path, start: int, end: int, key: str, head_type: str) -> dict | None:
    """A cached window prediction: {"start", "end", "cache_key", "raw", "retention"}, or None if absent, stale or invalid."""
    rec = _load_json(out_dir / WINDOWS_DIR / f"{start:05d}-{end:05d}.json")
    if not rec or rec.get("cache_key") != key:
        return None
    try:
        curve = parse_raw(rec.get("raw"), end - start, head_type)
    except TtccOutputError:
        return None
    return rec | {"retention": curve["retention"]}


def window_not_honoured(raw: dict, start: int, end: int) -> str | None:
    """None if the runner's raw output shows it decoded seconds ``start`` to ``end`` of the file (it echoes the row it built); else why not."""
    kw = ((raw.get("row") or {}).get("chat_template_kwargs")) if isinstance(raw, dict) else None
    want = {"video_start": float(start), "video_end": float(end)}
    if kw == want:
        return None
    return (f"the runner did not confirm decoding seconds {start}-{end} (it reported {kw!r}); the Space or runner is probably outdated: upload "
            "the current scripts/ttcc_infer.py and endpoint/ttcc_space/app.py and restart it")


def run_windowed(job: TtccJob, cfg: TtccConfig, runner: Runner, runtime: dict, probe_info: dict, workdir: Path, force: bool,
                 log, batch: dict, window_s: int | None = None) -> None:
    """Predict a video longer than one window: overlapping windows, merged into one curve on the original timeline.

    Windows already predicted with the same key are reused. A window that does not fit in memory shrinks the window length
    (``cfg.auto_shrink_window``) and the video is replanned; any other failed window is recorded and leaves a gap that ends the
    curve unless the neighbouring windows overlap it. Without a successful first window there is no curve."""
    n, duration = job.T, job.record.get("duration_s")
    W, attempts = window_s, []
    job.key = windowed_key(job, cfg, probe_info, runtime)
    t0 = time.time()
    while True:
        W, stride, warmup = window_params(n, cfg.window_s, cfg.stride_s, cfg.warmup_s, cfg.window_mode, W)
        plan = plan_windows(n, W, stride)
        done: dict[tuple[int, int], dict] = {}
        todo = []
        for s, e in plan:
            key = cache_key(job.source_sha256, e - s, cfg, probe_info, runtime, window_end=float(e), window_start=float(s))
            rec = None if force else _load_window(job.out_dir, s, e, key, cfg.head_type)
            if rec:
                done[(s, e)] = rec
            else:
                todo.append((s, e, key))
        failures: list[dict] = []
        if todo:
            request = {
                "protocol_version": "1",
                "repo": str(Path(cfg.repo_dir).expanduser().resolve() if cfg.repo_dir else default_repo()),
                "checkpoint": {"repo_id": cfg.checkpoint_repo, "revision": cfg.checkpoint_revision, "local_dir": cfg.checkpoint_dir,
                               "allow_download": cfg.allow_download, "expected_files": {k: list(v) for k, v in TTCC_CHECKPOINT_FILES.items()},
                               "verify_weights_sha256": cfg.verify_weights_sha256},
                "settings": cfg.inference_settings(**runtime), "prompts": PROMPTS,
                "jobs": [{"video_id": window_id(job.video_id, s, e), "video_path": str(job.source_path.resolve()), "T": e - s,
                          "video_start": float(s), "video_end": float(e), "shrink_to_fit": False,
                          "raw_output": str((workdir / "raw" / f"{window_id(job.video_id, s, e)}.json").resolve())} for s, e, _ in todo]}
            log(f"ttcc: {job.video_id}: {len(plan)} window(s) of {W} s every {stride} s over {n} s; running {len(todo)} "
                f"({len(done)} cached)")
            results = {r.get("video_id"): r for r in runner.infer(cfg, request, workdir)}
            for (s, e, key), spec in zip(todo, request["jobs"]):
                res, window = results.get(spec["video_id"]) or {}, {"start": s, "end": e}
                if res.get("status") != "ok":
                    failures.append(window | {"stage": res.get("stage") or "runner", "error": str(res.get("error") or "runner returned no result")[:400]})
                    continue
                raw = _load_json(Path(spec["raw_output"]))
                try:
                    parse_raw(raw, e - s, cfg.head_type)
                except TtccOutputError as exc:
                    failures.append(window | {"stage": "validation_of_output", "error": f"invalid prediction: {exc}"})
                    continue
                if (why := window_not_honoured(raw, s, e)):  # an outdated runner would predict the wrong seconds; never cache that
                    failures.append(window | {"stage": "protocol", "error": why})
                    continue
                rec = {"start": s, "end": e, "cache_key": key, "raw": raw}
                _write_json(job.out_dir / WINDOWS_DIR / f"{s:05d}-{e:05d}.json", rec)
                done[(s, e)] = rec | {"retention": parse_raw(raw, e - s, cfg.head_type)["retention"]}
        attempts.append({"window_s": W, "stride_s": stride, "windows": [list(w) for w in plan],
                         "failed": [[f["start"], f["end"]] for f in failures]})
        smaller = int(W * SHRINK_FACTOR)
        if cfg.auto_shrink_window and any(is_capacity_error(f["error"]) for f in failures) and smaller >= cfg.min_window_s:
            log(f"ttcc: {job.video_id}: a {W} s window did not fit; replanning with {smaller} s windows")
            W = smaller
            continue
        break

    good = [{"start": s, "end": e, "retention": rec["retention"]} for (s, e), rec in sorted(done.items())]
    merged = merge_windows(good, n, warmup) if good else None
    if not merged or merged["covered_until"] < 1:
        reason = ("no window starting at 0 s was predicted: " if good else "no window was predicted: ") + "; ".join(
            f"{f['start']}-{f['end']} s: {f['error']}" for f in failures[:3])
        _status(job, FAILED, reason, failures[0]["stage"] if failures else "inference", failed_windows=failures)
        return
    covered = merged["covered_until"]
    ordered = [done[k] for k in sorted(done)]
    bundle = {"schema": "hookline.ttcc_raw_windows", "windows": [{k: r[k] for k in ("start", "end", "cache_key", "raw")} for r in ordered]}
    _write_json(job.out_dir / RAW_FILE, bundle)
    raw_sha = file_sha256(job.out_dir / RAW_FILE)
    curve = {"timestamps_s": list(range(covered + 1)), "retention": merged["retention"], "summary": summarize_curve(merged["retention"])}
    batch["model_load_seconds"] = (ordered[0]["raw"].get("provenance") or {}).get("model_load_seconds")
    art = _artifact(job, cfg, ordered[0]["raw"], curve, raw_sha, duration, runtime, probe_info)
    complete = covered >= n
    art.update(
        horizon_s=covered, partial_coverage=not complete, stitched=True, convention=CONVENTION_WINDOWED, note=NOTE_WINDOWED,
        coverage={"predicted_from_s": 0.0, "predicted_until_s": float(covered), "video_duration_s": duration,
                  "model_input_window_s": [0.0, duration],
                  "unpredicted_s": [float(covered), duration] if duration and duration > covered + 0.5 else None},
        raw_output={"path": RAW_FILE, "sha256": raw_sha, "windows": len(ordered), "values_per_window": HEAD_OUTPUTS},
        windowing={
            "complete": complete and not failures, "video_seconds_rounded": n, "covered_until_s": covered,
            "window_mode": cfg.window_mode, "window_s": W, "stride_s": attempts[-1]["stride_s"], "warmup_s": warmup, "configured_window_s": cfg.window_s,
            "configured_stride_s": cfg.stride_s, "attempts": attempts,
            "windows": [{"start_s": r["start"], "end_s": r["end"], "input_tokens": r["raw"].get("input_tokens"),
                         "inference_seconds": r["raw"].get("inference_seconds"), "cache_key": r["cache_key"]} for r in ordered],
            "failed_windows": failures, "per_second_coverage": merged["coverage"], "merged_hazard": merged["hazard"],
            "diagnostics": merged["diagnostics"],
            "method": ("Each window is a standalone TTCC prediction (prompted as an ad of the window's length). The curve is rebuilt from "
                       "hazards (share of the remaining viewers lost per second) averaged over the windows that cover each second; a later "
                       "window's first warmup_s seconds are down-weighted because they describe a fresh start, not the middle of a video. "
                       "Continuous and non-increasing; seconds no window covers end the curve (nothing is interpolated)."),
            "limitation": ("Approximation: window k predicts viewers who start watching at its start, the merged curve describes viewers who "
                           "started at 0. Not validated against audience data."),
        })
    art["inference"].update(window_end_s=None, window_fit=None, input_tokens=[r["raw"].get("input_tokens") for r in ordered],
                            inference_seconds=round(sum((r["raw"].get("inference_seconds") or 0.0) for r in ordered), 3),
                            deviations_from_upstream_eval=art["inference"]["deviations_from_upstream_eval"] + [
                                f"video is {duration or 0:.1f} s: predicted in {len(ordered)} overlapping {W} s windows and stitched; "
                                "no single pass saw the whole video"])
    _write_json(job.out_dir / RETENTION_FILE, art)
    svg = retention_svg(art)
    if svg:
        (job.out_dir / PLOT_FILE).write_text(svg)
    job.record.update(cached=False, inference_seconds=art["inference"]["inference_seconds"], windows=len(ordered),
                      windows_failed=len(failures), wall_seconds=round(time.time() - t0, 2))
    reason = None if not failures else (f"{len(failures)} window(s) failed; the curve covers {covered} of {n} s" if not complete else
                                         f"{len(failures)} window(s) failed but the overlapping windows cover the whole video") + f" ({failures[0]['error'][:200]})"
    _status(job, SUCCESS, reason, "inference", cached=False, failed_windows=failures)


def run_ttcc(videos: list, root: Path, cfg: TtccConfig, runner: Runner | None = None, force: bool = False,
             log=print) -> dict:
    """TTCC for each ``VideoInput`` (``video_id``, ``path``). Returns {"records": [...], "batch": {...}}.

    One video's failure never stops the others, and never removes an earlier successful prediction."""
    features_root = root / "features"
    jobs = [TtccJob(v.video_id, Path(v.path), features_root / v.video_id,
                    {"video_id": v.video_id, "source_path": str(v.path)}) for v in videos]
    batch: dict[str, Any] = {"enabled": cfg.enabled, "probe": None, "model_load_seconds": None, "runner_seconds": None}
    if not cfg.enabled:
        for job in jobs:
            job.record.update(status=DISABLED, stage="config",
                              reason="TTCC is disabled (ENABLE_TTCC=false); pass --enable-ttcc or set ENABLE_TTCC=true")
        return {"records": [j.record for j in jobs], "batch": batch}

    candidates: list[TtccJob] = []
    for job in jobs:
        t0 = time.time()
        job.out_dir.mkdir(parents=True, exist_ok=True)
        info, problem = validate_video(job.source_path)
        job.record["duration_s"] = info.get("duration_s")
        if problem:
            _status(job, FAILED, problem, "validation", input=info)
            continue
        job.source_sha256 = file_sha256(job.source_path)
        if cfg.long_videos == "windows":  # T is the whole video; windows of window_s seconds are planned when it is longer
            job.T, job.window_end, why_not = plan_window(info["duration_s"], "first_60s")
            if job.T is not None:
                job.T, job.window_end = round(float(info["duration_s"])), None
                W, stride, _ = window_params(job.T, cfg.window_s, cfg.stride_s, cfg.warmup_s, cfg.window_mode)
                job.plan = plan_windows(job.T, W, stride)
        elif cfg.long_videos == "hook":
            job.T, job.window_end, why_not = plan_hook(info["duration_s"], cfg.hook_s, cfg.hook_prompt_s)
        else:
            job.T, job.window_end, why_not = plan_window(info["duration_s"], cfg.long_videos)
        job.record["validation_seconds"] = round(time.time() - t0, 3)
        if job.T is None:
            _status(job, UNAVAILABLE, why_not, "validation", input=info)
            continue
        if job.window_end is not None:
            job.record["window_end_s"] = job.window_end
        candidates.append(job)
    if not candidates:
        return {"records": [j.record for j in jobs], "batch": batch}

    if runner is None and cfg.remote:
        runner, why = remote_runner_or_reason(cfg, log=log)
        if runner is None:
            for job in candidates:
                _status(job, UNAVAILABLE, f"remote TTCC is not configured: {why}", "environment")
            return {"records": [j.record for j in jobs], "batch": batch}
    runner = runner or SubprocessRunner(log=log)
    batch["remote"] = cfg.remote
    probe = runner.probe(cfg)
    batch["probe"] = {k: probe.get(k) for k in ("inference_possible", "problems", "libraries", "cuda", "code", "python")}
    problems = list(probe.get("problems") or [])
    runtime, runtime_problems = resolve_runtime(cfg, probe)
    if probe.get("inference_possible"):
        problems += runtime_problems
    batch["runtime"] = runtime
    if not probe.get("inference_possible") or problems:
        reason = "TTCC inference is unavailable here: " + "; ".join(problems or ["unknown reason"])
        for job in candidates:
            _status(job, UNAVAILABLE, reason, "environment")
        return {"records": [j.record for j in jobs], "batch": batch}

    pending = []
    for job in candidates:
        multi = bool(job.plan and len(job.plan) > 1)
        job.key = windowed_key(job, cfg, probe, runtime) if multi else cache_key(job.source_sha256, job.T, cfg, probe, runtime, job.window_end)
        art = None if force else cached_prediction(job.out_dir, job.key)
        if art is None and not multi and not force and cfg.long_videos == "windows" and cfg.auto_shrink_window and job.plan:
            wkey = windowed_key(job, cfg, probe, runtime)  # a video that did not fit one pass was stored as windows: do not retry the single pass
            stored = cached_prediction(job.out_dir, wkey)
            if stored and stored.get("stitched") and (stored.get("windowing") or {}).get("complete"):
                art, job.key = stored, wkey
        if art and (multi or art.get("stitched")) and not (art.get("windowing") or {}).get("complete"):
            art = None  # an earlier run lost a window: retry it (finished windows come from the per-window cache)
        if art:
            job.record.update(cached=True, inference_seconds=0.0)
            _status(job, SUCCESS, None, "cache", cached=True)
        else:
            pending.append(job)
    if cfg.remote:  # nothing leaves this machine unless allowed (and, for company reels, approved per video)
        allowed = []
        for job in pending:
            blocked = upload_block_reason(job.video_id, job.source_path, cfg)
            if blocked:
                _status(job, UNAVAILABLE, f"not sent: {blocked}", "permission")
            else:
                allowed.append(job)
        pending = allowed
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    workdir = root / "cache" / "_ttcc_runs" / run_id
    for job in [j for j in pending if j.plan and len(j.plan) > 1]:
        run_windowed(job, cfg, runner, runtime, probe, workdir, force, log, batch)
    pending = [j for j in pending if not (j.plan and len(j.plan) > 1)]
    if not pending:
        shutil.rmtree(workdir / "raw", ignore_errors=True)  # window outputs were copied into the video folders
        return {"records": [j.record for j in jobs], "batch": batch}
    request = {
        "protocol_version": "1",
        "repo": str(Path(cfg.repo_dir).expanduser().resolve() if cfg.repo_dir else default_repo()),
        "checkpoint": {"repo_id": cfg.checkpoint_repo, "revision": cfg.checkpoint_revision,
                       "local_dir": cfg.checkpoint_dir, "allow_download": cfg.allow_download,
                       "expected_files": {k: list(v) for k, v in TTCC_CHECKPOINT_FILES.items()},
                       "verify_weights_sha256": cfg.verify_weights_sha256},
        "settings": cfg.inference_settings(**runtime),
        "prompts": PROMPTS,
        "jobs": [{"video_id": j.video_id, "video_path": str(j.source_path.resolve()), "T": j.T, "video_end": j.window_end,
                  "shrink_to_fit": cfg.long_videos == "first_60s" or (cfg.long_videos == "hook" and cfg.hook_prompt_s is None),
                  "raw_output": str((workdir / "raw" / f"{j.video_id}.json").resolve())} for j in pending],
    }
    log(f"ttcc: running {len(pending)} video(s) in {runner.python(cfg) if hasattr(runner, 'python') else 'runner'}")
    t0 = time.time()
    results = {r.get("video_id"): r for r in runner.infer(cfg, request, workdir)}
    batch["runner_seconds"] = round(time.time() - t0, 2)
    batch["runner_workdir"] = str(workdir)

    for job, spec in zip(pending, request["jobs"]):
        res = results.get(job.video_id)
        if not res:
            _status(job, FAILED, "runner returned no result for this video", "runner")
            continue
        if res.get("status") != "ok":
            smaller = int(job.T * SHRINK_FACTOR)
            if cfg.long_videos == "windows" and cfg.auto_shrink_window and is_capacity_error(res.get("error")) and smaller >= cfg.min_window_s:
                log(f"ttcc: {job.video_id}: {job.T} s did not fit in one pass; retrying in windows of {smaller} s")
                run_windowed(job, cfg, runner, runtime, probe, workdir, force, log, batch, window_s=smaller)
                continue
            _status(job, FAILED, res.get("error") or "inference failed", res.get("stage") or "inference")
            continue
        raw_path = Path(spec["raw_output"])
        raw = _load_json(raw_path)
        shrunk = (raw or {}).get("window") if isinstance(raw, dict) else None
        if shrunk and isinstance(shrunk.get("T"), int) and shrunk.get("requested_T") == job.T and shrunk["T"] < job.T:
            job.T, job.window_end = shrunk["T"], float(shrunk["T"])  # runner used a shorter opening window to fit
            job.record["window_end_s"] = job.window_end
        try:
            curve = parse_raw(raw, job.T, cfg.head_type)
        except TtccOutputError as exc:
            if raw_path.is_file():
                shutil.copyfile(raw_path, job.out_dir / RAW_REJECTED_FILE)
            _status(job, FAILED, f"invalid prediction: {exc}", "validation_of_output",
                    rejected_raw_output=RAW_REJECTED_FILE if raw_path.is_file() else None)
            continue
        batch["model_load_seconds"] = (raw.get("provenance") or {}).get("model_load_seconds")
        shutil.copyfile(raw_path, job.out_dir / RAW_FILE)
        raw_sha = file_sha256(job.out_dir / RAW_FILE)
        art = _artifact(job, cfg, raw, curve, raw_sha, job.record.get("duration_s"), runtime, probe)
        art["hook"] = hook_metrics(curve["retention"], job.window_end)
        art["policy"] = cfg.long_videos
        if cfg.long_videos == "hook" and job.window_end is not None and job.T != round(job.window_end):
            art["inference"]["deviations_from_upstream_eval"].append(
                f"sensitivity run: the model was shown {job.window_end:.0f} s but told the ad is {job.T} s long")
        _write_json(job.out_dir / RETENTION_FILE, art)
        svg = retention_svg(art)
        if svg:
            (job.out_dir / PLOT_FILE).write_text(svg)
        job.record.update(cached=False, inference_seconds=raw.get("inference_seconds"))
        _status(job, SUCCESS, None, "inference", cached=False)
    shutil.rmtree(workdir / "raw", ignore_errors=True)  # copied into the video folders; request/log/results stay
    return {"records": [j.record for j in jobs], "batch": batch}


def artifact_paths(out_dir: Path) -> dict[str, str | None]:
    return {name: str(out_dir / name) if (out_dir / name).is_file() else None
            for name in (STATUS_FILE, RETENTION_FILE, RAW_FILE, PLOT_FILE)}
