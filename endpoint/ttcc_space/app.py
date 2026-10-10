"""Hugging Face Space (Docker, GPU) serving the TTCC retention model as a job API for Hookline.

    GET  /health          -> {"probe": <ttcc_infer.py probe output>, "checkpoint_ready", "checkpoint_error", "jobs"}
    POST /analyze         body {"request": {...settings, prompts, checkpoint, job | jobs}, "video_b64", "filename"}
                          (``jobs``: several windows of one long video, each with video_start/video_end; one upload, one model load)
                          -> 202 {"job_id"}; 503 while the checkpoint is still downloading; 400 for a bad request
    GET  /report/{job_id} -> {"status": queued|running|done|failed, "result": {"result", "raw"} | "error"}

Each job runs ``ttcc_infer.py infer`` (the same upstream-protocol runner Hookline uses locally) in a
subprocess, then returns the raw output file unchanged. The model is the pinned checkpoint
liangyuch/ttcc-sft-qwen25omni-3b-v8, loaded through the go_viral ``ttcc-rl`` plugin; the runner verifies the
checkpoint files and the retention head before predicting. Jobs live in memory (a restart forgets them).
Keep the Space private. Output = predicted audience-retention curve R(t) from a pretrained model, not a
measurement and not a guarantee of virality.

No ``from __future__ import annotations``: FastAPI must resolve route annotations at runtime.
"""

import base64
import binascii
import contextlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

CHECKPOINT_REPO = "liangyuch/ttcc-sft-qwen25omni-3b-v8"
CHECKPOINT_REVISION = "1686d5f00408bfd8f6a32be12e71248b6f8ff447"
REPO_DIR = Path(os.environ.get("TTCC_REPO_DIR", "/home/user/go_viral"))
CKPT_DIR = Path(os.environ.get("TTCC_CKPT_DIR", "/home/user/ttcc_ckpt"))
RUNNER = Path(__file__).with_name("ttcc_infer.py")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_JOBS_KEPT = 50
MAX_BODY_MB = 200
JOB_TIMEOUT_S = 1800
JOB_TIMEOUT_PER_JOB_S = 300  # added for each further window of the same upload
MAX_JOBS = 16  # windows per upload (a 60 s video needs 3; an 8-minute reel with 30/15 s windows needs 31 -> sent in groups)


class JobStore:
    """Sequential job runner around ``handle(request) -> response dict`` (same as the TRIBE Space's)."""

    def __init__(self, handle: Callable[[dict], dict]):
        self.handle = handle
        self.jobs: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, request: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[job_id] = {"job_id": job_id, "status": "queued", "submitted_at": time.time(), "request": request}
            self.order.append(job_id)
            for old in [j for j in self.order if self.jobs[j]["status"] in ("done", "failed")][:-MAX_JOBS_KEPT]:
                self.order.remove(old)
                self.jobs.pop(old, None)
        self.queue.put(job_id)
        return job_id

    def report(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return {k: v for k, v in job.items() if k != "request"}

    def counts(self) -> dict:
        with self.lock:
            statuses = [j["status"] for j in self.jobs.values()]
        return {s: statuses.count(s) for s in ("queued", "running", "done", "failed")}

    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            with self.lock:
                job = self.jobs.get(job_id)
                if job is None:
                    continue
                job["status"], job["started_at"] = "running", time.time()
                request = job.pop("request")
            try:
                response = self.handle(request)
                failed = isinstance(response, dict) and "error" in response
                update = {"status": "failed", "error": str(response["error"])[:800]} if failed else \
                         {"status": "done", "result": response}
            except Exception as exc:  # never kill the worker
                update = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:800]}"}
            with self.lock:
                job.update(update, finished_at=time.time())
                job["timing"] = {"queued_s": round(job["started_at"] - job["submitted_at"], 2),
                                 "run_s": round(job["finished_at"] - job["started_at"], 2)}


class TtccService:
    """Checkpoint preparation, the environment probe and one-video inference through ``ttcc_infer.py``."""

    def __init__(self, runner_cmd: list | None = None, repo_dir: Path = REPO_DIR, ckpt_dir: Path = CKPT_DIR):
        self.runner_cmd = runner_cmd or [sys.executable, str(RUNNER)]
        self.repo_dir, self.ckpt_dir = repo_dir, ckpt_dir
        self.ready, self.error = False, None
        self.probe: dict = {"inference_possible": False, "problems": ["the Space is still starting"]}

    def prepare(self, download: Callable[[], str] | None = None) -> None:
        """Download the pinned checkpoint, then probe the environment. Sets ``ready`` or ``error``."""
        try:
            if download is None:
                from huggingface_hub import snapshot_download
                snapshot_download(CHECKPOINT_REPO, revision=CHECKPOINT_REVISION, local_dir=str(self.ckpt_dir))
            else:
                download()
            out = subprocess.run([*self.runner_cmd, "probe", "--repo", str(self.repo_dir)], capture_output=True, text=True, timeout=600)
            self.probe = json.loads(out.stdout.strip().splitlines()[-1])
            self.ready = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc)[:600]}"

    def handle(self, request: dict) -> dict:
        """Validate the request, run the runner on the uploaded video, return {"result", "raw"} or {"error"}."""
        if not self.ready:
            return {"error": self.error or "the Space is still preparing the checkpoint"}
        try:
            meta = request["request"]
            jobs = meta.get("jobs") if meta.get("jobs") is not None else [meta["job"]]  # windows of one long video share one upload
            if not (isinstance(jobs, list) and jobs and len(jobs) <= MAX_JOBS):
                return {"error": f"expected between 1 and {MAX_JOBS} jobs"}
            ckpt = meta["checkpoint"]
            if (ckpt["repo_id"], ckpt["revision"]) != (CHECKPOINT_REPO, CHECKPOINT_REVISION):
                return {"error": f"this Space serves {CHECKPOINT_REPO}@{CHECKPOINT_REVISION[:12]}, not {ckpt['repo_id']}@{ckpt['revision'][:12]}"}
            suffix = Path(str(request.get("filename") or "video.mp4")).suffix.lower()
            if suffix not in VIDEO_SUFFIXES:
                return {"error": f"unsupported video suffix {suffix!r}"}
            video = base64.b64decode(request["video_b64"], validate=True)
            specs = [{"video_id": str(j["video_id"]), "T": int(j["T"]), "video_start": j.get("video_start"), "video_end": j.get("video_end"),
                      "shrink_to_fit": bool(j.get("shrink_to_fit"))} for j in jobs]
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            return {"error": f"bad request: {type(exc).__name__}: {exc}"}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video_path = tmp / f"input{suffix}"
            video_path.write_bytes(video)
            req = {"protocol_version": "1", "repo": str(self.repo_dir),
                   "checkpoint": {"repo_id": ckpt["repo_id"], "revision": ckpt["revision"], "local_dir": str(self.ckpt_dir),
                                  "allow_download": False, "expected_files": ckpt["expected_files"],
                                  "verify_weights_sha256": bool(ckpt.get("verify_weights_sha256"))},
                   "settings": meta["settings"], "prompts": meta["prompts"],
                   "jobs": [spec | {"video_path": str(video_path), "raw_output": str(tmp / f"raw_{i}.json")} for i, spec in enumerate(specs)]}
            (tmp / "request.json").write_text(json.dumps(req))
            try:  # one runner process: the model loads once and every window reuses it
                proc = subprocess.run([*self.runner_cmd, "infer", "--request", str(tmp / "request.json"),
                                       "--results", str(tmp / "results.jsonl")], capture_output=True, text=True,
                                      timeout=JOB_TIMEOUT_S + JOB_TIMEOUT_PER_JOB_S * (len(specs) - 1))
            except subprocess.TimeoutExpired:
                return {"error": f"the runner timed out ({len(specs)} job(s))"}
            lines = (tmp / "results.jsonl").read_text().splitlines() if (tmp / "results.jsonl").exists() else []
            if not lines:
                return {"error": f"the runner produced no result (exit {proc.returncode}): {(proc.stdout + proc.stderr)[-600:]}"}
            by_id = {r.get("video_id"): r for r in map(json.loads, lines)}
            items = []
            for i, spec in enumerate(specs):
                result = by_id.get(spec["video_id"])
                raw_path = tmp / f"raw_{i}.json"
                if result is None:
                    items.append({"video_id": spec["video_id"], "result": {"video_id": spec["video_id"], "status": "failed", "stage": "runner",
                                                                           "error": "the runner returned no result for this job"}, "raw": {}})
                elif result.get("status") != "ok":
                    items.append({"video_id": spec["video_id"], "result": result, "raw": {}})
                else:
                    items.append({"video_id": spec["video_id"], "result": result | {"raw_output": None}, "raw": json.loads(raw_path.read_text())})
            if "jobs" not in meta:  # single-job request: the original response shape
                return {"result": items[0]["result"], "raw": items[0]["raw"]}
            return {"items": items}


def create_app(service: TtccService | None = None, start: bool = True):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    service = service or TtccService()
    store = JobStore(service.handle)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        if start:
            threading.Thread(target=service.prepare, daemon=True).start()  # checkpoint download + probe in the background
        yield

    app = FastAPI(title="hookline-ttcc", docs_url="/docs", lifespan=lifespan)

    @app.get("/")
    def index() -> dict:
        return {"service": "hookline-ttcc", "docs": "/docs", "flow": "POST /analyze (video) -> GET /report/{job_id}", "health": "/health"}

    @app.get("/health")
    def health() -> dict:
        return {"probe": service.probe, "checkpoint_ready": service.ready, "checkpoint_error": service.error, "jobs": store.counts()}

    @app.post("/analyze", status_code=202)
    async def analyze(request: Request) -> JSONResponse:
        if service.error:
            return JSONResponse({"detail": service.error}, status_code=500)
        if not service.ready:
            return JSONResponse({"detail": "the checkpoint is still downloading; retry in a few minutes"}, status_code=503)
        if int(request.headers.get("content-length") or 0) > MAX_BODY_MB * 2**20:
            raise HTTPException(413, f"request larger than {MAX_BODY_MB} MB")
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "body must be JSON") from None
        if not (isinstance(body, dict) and isinstance(body.get("request"), dict) and isinstance(body.get("video_b64"), str)):
            raise HTTPException(400, "expected {'request': {...}, 'video_b64': <base64>, 'filename': <name>}")
        return JSONResponse({"job_id": store.submit(body)}, status_code=202)

    @app.get("/report/{job_id}")
    def report(job_id: str) -> dict:
        out = store.report(job_id)
        if out is None:
            raise HTTPException(404, "unknown job_id (jobs are forgotten when the Space restarts)")
        return out

    return app
