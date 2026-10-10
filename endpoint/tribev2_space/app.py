"""Hugging Face Space (Docker) serving TRIBE v2 as a small job API.

    POST /analyze         body: the handler request JSON, {"inputs": {"video_b64", "filename"}, "parameters": {"language"}}
                          -> 202 {"job_id"}; 503 while the model is still loading; 400 for a bad request
    GET  /report/{job_id} -> {"job_id", "status": queued|running|done|failed, "result"|"error", "timing"}
                          ``result`` is exactly what handler.py returns (predicted fMRI responses on fsaverage5)
    GET  /health          -> {"model_loaded", "load_error", "jobs"}

One worker thread runs the jobs one after another (a single GPU, one model in memory). Jobs live in
memory: a restart of the Space forgets them. Access control is the Space's own privacy: keep it
private, and call it with a Hugging Face token that can read it. The model, licence and output
meaning are those of handler.py (TRIBE v2, CC BY-NC 4.0; predicted brain responses, not engagement).
"""

# No ``from __future__ import annotations`` here: FastAPI must resolve the route annotations (Request) at runtime.
import contextlib
import queue
import threading
import time
import uuid
from typing import Any, Callable

MAX_JOBS_KEPT = 50
MAX_BODY_MB = 200  # request body limit; Hookline refuses uploads over 50 MB by default


class JobStore:
    """Sequential job runner around ``handle(request) -> response dict`` (handler.py's contract)."""

    def __init__(self, handle: Callable[[dict], dict]):
        self.handle = handle
        self.jobs: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self.queue: queue.Queue[str] = queue.Queue()
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

    def report(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            out = {k: v for k, v in job.items() if k != "request"}
        out["queue_position"] = None
        if out["status"] == "queued":
            with self.queue.mutex:
                pending = list(self.queue.queue)
            out["queue_position"] = pending.index(job_id) + 1 if job_id in pending else None
        return out

    def counts(self) -> dict[str, int]:
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


def create_app(handler_factory: Callable[[], Callable[[dict], dict]] | None = None):
    """FastAPI app. ``handler_factory`` builds the request handler (loads the model); default is handler.py."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    if handler_factory is None:
        def handler_factory():  # noqa: E306 - imported lazily so tests can run without the model stack
            from handler import EndpointHandler
            return EndpointHandler()

    state: dict[str, Any] = {"store": None, "load_error": None}

    def load() -> None:
        try:
            state["store"] = JobStore(handler_factory())
        except Exception as exc:  # shown by /health; the Space needs a restart after fixing it
            state["load_error"] = f"{type(exc).__name__}: {str(exc)[:800]}"

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=load, daemon=True).start()  # the model loads in the background; /health shows when
        yield

    app = FastAPI(title="hookline-tribe", docs_url="/docs", lifespan=lifespan)

    @app.get("/")
    def index() -> dict:
        return {"service": "hookline-tribe", "docs": "/docs", "flow": "POST /analyze (video) -> GET /report/{job_id}",
                "health": "/health"}

    @app.get("/health")
    def health() -> dict:
        store = state["store"]
        return {"model_loaded": store is not None, "load_error": state["load_error"], "jobs": store.counts() if store else None}

    @app.post("/analyze", status_code=202)
    async def analyze(request: Request) -> JSONResponse:
        store = state["store"]
        if store is None:
            detail = state["load_error"] or "model is still loading; retry in a minute"
            return JSONResponse({"detail": detail}, status_code=503 if not state["load_error"] else 500)
        if int(request.headers.get("content-length") or 0) > MAX_BODY_MB * 2**20:
            raise HTTPException(413, f"request larger than {MAX_BODY_MB} MB")
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "body must be JSON") from None
        inputs = body.get("inputs") if isinstance(body, dict) else None
        if not isinstance(inputs, dict) or not isinstance(inputs.get("video_b64"), str):
            raise HTTPException(400, "expected {'inputs': {'video_b64': <base64>, 'filename': <name>}}")
        return JSONResponse({"job_id": store.submit(body)}, status_code=202)

    @app.get("/report/{job_id}")
    def report(job_id: str) -> dict:
        store = state["store"]
        if store is None:  # still loading after a restart: the job (if any) was lost, but say so honestly
            raise HTTPException(503, state["load_error"] or "model is still loading after a restart; earlier jobs were forgotten")
        out = store.report(job_id)
        if out is None:
            raise HTTPException(404, "unknown job_id (jobs are forgotten when the Space restarts)")
        return out

    return app
