"""Upload page and job runner for trying Hookline on your own video, on this machine (http://127.0.0.1).

One page: choose a video, watch the progress, open the review page when it is done. Behind it the same
one-command pipeline as ``scripts/run_e2e.py`` (features, speech and text, TTCC on your Hugging Face Space, Gemini, scores, review page).

Safeguards:
- binds to 127.0.0.1 only and checks the Host header (no other machine, no DNS rebinding); uploads need a custom header a web page on another
  origin cannot send, and an Origin header, when present, must be this server;
- a video is sent to your TTCC Space only if "send to my Space" is ticked, and to Google only if "send to Gemini" is ticked (both required), per upload;
- one analysis at a time; an upload size limit; file names are never used as paths (the id is generated); only known video extensions are accepted;
- serves only the upload page, the job status, and each job's own review page and video; lists no directories;
- the page labels everything as model predictions; nothing here says a video will or will not go viral.
"""

from __future__ import annotations

import http.server
import json
import mimetypes
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import unquote
from typing import Callable

CHUNK = 1 << 20
EXTENSIONS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
MAX_UPLOAD_MB = 300
STEP_RE = re.compile(r"^\[(\d+)/(\d+)\] (.+?)(?: \.\.\.|: (ok.*|FAILED.*|skipped.*|not run.*))$")
ID_RE = re.compile(r"^UP[0-9a-z]{4,16}$")


class Job:
    def __init__(self, job_id: str, video: Path, options: dict):
        self.id, self.video, self.options = job_id, video, options
        self.log: list[str] = []
        self.steps: dict[int, dict] = {}
        self.status = "queued"  # queued | running | done | failed
        self.code: int | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.error: str | None = None

    def add(self, line: str) -> None:
        self.log.append(line)
        m = STEP_RE.match(line) if not line.startswith(" ") else None  # indented lines are output of a step, not a step
        if m:
            n, total, name, status = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            self.steps[n] = {"n": n, "of": total, "name": name, "state": "running" if status is None else (
                "ok" if status.startswith("ok") else "skipped" if status.startswith(("skipped", "not run")) else "failed"), "detail": status or ""}

    def view(self) -> dict:
        return {"id": self.id, "status": self.status, "code": self.code, "error": self.error, "options": self.options,
                "steps": [self.steps[k] for k in sorted(self.steps)], "log": self.log[-40:], "seconds": round((self.finished or time.time()) - self.started, 1)}


class App:
    """State shared by the request handlers: where files go and how a job runs (``runner`` is injectable for tests)."""

    def __init__(self, root: Path, runner: Callable[[Job], int], max_upload_mb: int = MAX_UPLOAD_MB, hook_s: int | None = None,
                 whatif_runner: Callable[[str, list[dict]], dict] | None = None):
        self.root, self.runner, self.max_upload_mb, self.hook_s = root, runner, max_upload_mb, hook_s
        self.whatif_runner = whatif_runner
        self.whatif: dict[str, dict] = {}
        self.uploads = root / "uploads"
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.uploads.mkdir(parents=True, exist_ok=True)

    def busy(self) -> bool:
        return any(j.status in ("queued", "running") for j in self.jobs.values()) or any(w["status"] == "running" for w in self.whatif.values())

    def start_whatif(self, job_id: str, trims: list[dict]) -> None:
        state = {"status": "running", "error": None, "result": None, "started": time.time()}
        self.whatif[job_id] = state

        def work() -> None:
            try:
                state["result"] = self.whatif_runner(job_id, trims)
                state["status"] = "done"
            except Exception as exc:  # a failed what-if never takes the server down; the reason is shown on the page
                state["status"], state["error"] = "failed", f"{type(exc).__name__}: {str(exc)[:400]}"
            state["finished"] = time.time()
        threading.Thread(target=work, daemon=True).start()

    def whatif_view(self, job_id: str) -> dict | None:
        st = self.whatif.get(job_id)
        if st is None:
            saved = self.root / "whatif" / job_id / "whatif.json"
            if ID_RE.match(job_id) and saved.is_file():
                return {"status": "done", "error": None, "result": json.loads(saved.read_text()), "seconds": 0}
            return None
        return {"status": st["status"], "error": st["error"], "result": st["result"], "seconds": round((st.get("finished") or time.time()) - st["started"], 1)}

    def new_id(self) -> str:
        return "UP" + format(int(time.time()), "x")[-5:] + secrets.token_hex(2)

    def start(self, job: Job) -> None:
        def work() -> None:
            job.status = "running"
            try:
                job.code = self.runner(job)
                job.status = "done" if job.code == 0 else "failed"
                if job.code != 0:
                    job.error = "the pipeline stopped at a failed step; see the log"
            except Exception as exc:  # the job's own failure never takes the server down
                job.status, job.error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
            job.finished = time.time()
        threading.Thread(target=work, daemon=True).start()

    def review_path(self, job_id: str) -> Path | None:
        p = self.root / "features" / job_id / "review.html"
        return p if ID_RE.match(job_id) and p.is_file() else None

    def video_path(self, job_id: str) -> Path | None:
        if not ID_RE.match(job_id):
            return None
        for ext in EXTENSIONS:
            p = self.uploads / f"{job_id}{ext}"
            if p.is_file():
                return p
        return None

    def recent(self) -> list[dict]:
        out = []
        for p in sorted((self.root / "features").glob("UP*/review.html"), key=lambda q: q.stat().st_mtime, reverse=True)[:12]:
            out.append({"id": p.parent.name, "url": f"/review/{p.parent.name}/"})
        return out


MAX_SHORT_SIDE = 720  # larger uploads are shrunk: a 1080x1920 clip needs more GPU memory than the 22 GB TTCC Space has


def shrink_for_model(path: Path, run=subprocess.run) -> Path:
    """Return a copy of the video whose short side is at most 720 px (an .mp4 next to it), or the original when it is already small
    or ffmpeg fails. The prediction models see the smaller copy; frames are scaled, nothing else changes."""
    try:
        probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                    capture_output=True, text=True, timeout=60)
        w, h = (int(x) for x in probe.stdout.strip().split(",")[:2])
    except (OSError, ValueError, subprocess.SubprocessError):
        return path
    if min(w, h) <= MAX_SHORT_SIDE:
        return path
    out = path.with_name(path.stem + "_small.mp4")
    scale = f"scale={MAX_SHORT_SIDE}:-2" if w <= h else f"scale=-2:{MAX_SHORT_SIDE}"
    try:
        proc = run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", scale, "-c:v", "libx264", "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", str(out)],
                   capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        out.unlink(missing_ok=True)
        return path
    if proc.returncode != 0 or not out.is_file():
        out.unlink(missing_ok=True)
        return path
    final = path.with_suffix(".mp4")
    path.unlink(missing_ok=True)
    out.replace(final)
    return final


def make_handler(app: App, port_holder: dict):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "HooklineApp"

        def log_message(self, *args) -> None:
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in (f"127.0.0.1:{port_holder['port']}", f"localhost:{port_holder['port']}")

        def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain; charset=utf-8", extra: dict | None = None, head: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body and not head:
                self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data).encode(), "application/json; charset=utf-8")

        def _video(self, path: Path, head: bool) -> None:
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end, code = 0, size - 1, 200
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start, end = max(0, size - int(m.group(2))), size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    return self._send(416, b"", extra={"Content-Range": f"bytes */{size}"}, head=head)
                code = 206
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head:
                return
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _get(self, head: bool) -> None:
            if not self._host_ok():
                return self._send(403, b"forbidden host", head=head)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._send(200, INDEX.replace("__MAX_MB__", str(app.max_upload_mb)).replace("__HOOK_S__", str(app.hook_s or "")).encode(), "text/html; charset=utf-8", head=head)
            if path == "/api/recent":
                return self._json(200, {"recent": app.recent(), "busy": app.busy()})
            m = re.fullmatch(r"/api/job/(UP[0-9a-z]+)", path)
            if m:
                job = app.jobs.get(m.group(1))
                if job is None:
                    return self._json(404, {"error": "unknown job"})
                data = job.view()
                data["review_url"] = f"/review/{job.id}/" if app.review_path(job.id) else None
                return self._json(200, data)
            m = re.fullmatch(r"/api/whatif/(UP[0-9a-z]+)", path)
            if m:
                view = app.whatif_view(m.group(1))
                return self._json(200 if view else 404, view or {"error": "no what-if for this analysis yet"})
            m = re.fullmatch(r"/review/(UP[0-9a-z]+)(/|/video)?", path)
            if m:
                jid, tail = m.group(1), m.group(2)
                if tail is None:
                    return self._send(301, b"", extra={"Location": f"/review/{jid}/"}, head=head)
                if tail == "/video":
                    v = app.video_path(jid)
                    return self._video(v, head) if v else self._send(404, b"not found", head=head)
                page = app.review_path(jid)
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8", head=head) if page else self._send(404, b"not found", head=head)
            self._send(404, b"not found", head=head)

        def do_GET(self) -> None:  # noqa: N802
            self._get(False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._get(True)

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._send(403, b"forbidden host")
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{port_holder['port']}", f"http://localhost:{port_holder['port']}"):
                return self._send(403, b"forbidden origin")
            wm = re.fullmatch(r"/api/whatif/(UP[0-9a-z]+)", self.path.split("?", 1)[0])
            if wm:
                return self._post_whatif(wm.group(1))
            if self.path.split("?", 1)[0] != "/api/upload" or self.headers.get("X-Hookline") != "1":
                return self._send(404 if self.path.split("?", 1)[0] != "/api/upload" else 403, b"not allowed")
            try:
                size = int(self.headers.get("Content-Length") or "")
            except ValueError:
                return self._json(411, {"error": "Content-Length is required"})
            if size <= 0 or size > app.max_upload_mb * 2**20:
                return self._json(413, {"error": f"the file must be between 1 byte and {app.max_upload_mb} MB"})
            q = dict(p.split("=", 1) for p in (self.path.split("?", 1) + [""])[1].split("&") if "=" in p)
            name = unquote(self.headers.get("X-Filename") or "video.mp4")
            ext = Path(name).suffix.lower()
            if ext not in EXTENSIONS:
                return self._json(415, {"error": f"unsupported file type {ext!r}; use one of {', '.join(EXTENSIONS)}"})
            with app.lock:
                if app.busy():
                    return self._json(409, {"error": "another analysis is running; wait for it to finish"})
                job_id = app.new_id()
                dest = app.uploads / f"{job_id}{ext}"
                tmp = dest.with_suffix(dest.suffix + ".part")
                left = size
                with tmp.open("wb") as f:
                    while left > 0:
                        chunk = self.rfile.read(min(CHUNK, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                if left:
                    tmp.unlink(missing_ok=True)
                    return self._json(400, {"error": "the upload was cut short"})
                tmp.replace(dest)
                dest = shrink_for_model(dest)
                job = Job(job_id, dest, {"remote": True, "gemini": True, "original_name": Path(name).name[:80]})
                app.jobs[job_id] = job
                app.start(job)
            self._json(200, {"job_id": job_id})

        def _post_whatif(self, job_id: str) -> None:
            if self.headers.get("X-Hookline") != "1":
                return self._send(403, b"not allowed")
            if app.whatif_runner is None or app.video_path(job_id) is None or app.review_path(job_id) is None:
                return self._json(404, {"error": "this analysis cannot be re-predicted"})
            try:
                size = int(self.headers.get("Content-Length") or "")
            except ValueError:
                return self._json(411, {"error": "Content-Length is required"})
            if not 0 < size <= 65536:
                return self._json(413, {"error": "the request is too large"})
            try:
                trims = json.loads(self.rfile.read(size)).get("trims")
            except (ValueError, AttributeError):
                return self._json(400, {"error": "expected JSON {\"trims\": [{\"start_s\": 1, \"end_s\": 2}]}"})
            if not isinstance(trims, list) or not 0 < len(trims) <= 20 or not all(isinstance(t, dict) for t in trims):
                return self._json(400, {"error": "between 1 and 20 trims are required"})
            with app.lock:
                if app.busy():
                    return self._json(409, {"error": "another analysis is running; wait for it to finish"})
                app.start_whatif(job_id, trims)
            self._json(200, {"started": True})

    return Handler


def start_server(app: App, port: int = 0) -> tuple[http.server.ThreadingHTTPServer, int]:
    holder: dict = {"port": port}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, holder))
    holder["port"] = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, holder["port"]


INDEX = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Hookline: try a video</title>
<style>
:root{--bg:#0a0c12;--panel:#10131c;--panel2:#151926;--line:#232838;--line2:#2e3447;--fg:#e8eaf1;--mut:#8c92a6;--g:#5fd38a;--p:#b47cf0;--r:#ff5468;--y:#ffcf47;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 20px 60px}.brand{display:flex;align-items:center;gap:10px;margin-bottom:22px}.mark{display:flex;gap:3px;align-items:flex-end;height:24px}
.mark i{display:block;width:5px;border-radius:2px}.mark i:nth-child(1){height:24px;background:var(--g)}.mark i:nth-child(2){height:17px;background:var(--p)}.mark i:nth-child(3){height:12px;background:var(--r)}.mark i:nth-child(4){height:8px;background:var(--y)}
.brand b{font-size:19px}.by{font:600 11px var(--mono);letter-spacing:2px;margin-left:auto}.by .w{color:var(--g)}.by .l{color:var(--p)}.by .d1{color:var(--r)}.by .d2{color:var(--y)}
h1{font-size:22px;margin:0 0 4px}.mut{color:var(--mut);font-size:12.5px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}
#drop{border:1.5px dashed var(--line2);border-radius:12px;padding:30px;text-align:center;background:var(--panel2);cursor:pointer}#drop:hover,#drop.on{border-color:var(--p)}
#drop b{display:block;font-size:15px;margin-bottom:4px}label.opt{display:flex;gap:10px;align-items:flex-start;margin:12px 0;font-size:13.5px}label.opt input{margin-top:4px;accent-color:var(--p)}
button{font:inherit;padding:9px 18px;border:0;border-radius:9px;background:linear-gradient(135deg,#8b5cf6,var(--p));color:#fff;font-weight:600;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}
.steps{display:flex;flex-direction:column;gap:6px;margin-top:10px}.step{display:grid;grid-template-columns:22px 1fr auto;gap:10px;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:var(--panel2);align-items:center}
.dot{width:10px;height:10px;border-radius:50%;background:var(--line2)}.running .dot{background:var(--y);animation:pulse 1s infinite}.ok .dot{background:var(--g)}.failed .dot{background:var(--r)}.skipped .dot{background:#5b6276}
@keyframes pulse{50%{opacity:.35}}pre{background:#07090e;border:1px solid var(--line);border-radius:10px;padding:10px;max-height:220px;overflow:auto;font:11.5px var(--mono);color:#aab0c4;white-space:pre-wrap}
.note{border-left:3px solid var(--y);padding:8px 12px;background:var(--panel);border-radius:8px;font-size:12.5px;color:#cfd3df;margin:14px 0}a.btn{display:inline-block;padding:9px 18px;border-radius:9px;background:linear-gradient(135deg,#8b5cf6,var(--p));color:#fff;font-weight:600;text-decoration:none}
ul{padding-left:18px;margin:6px 0}.err{color:var(--r)}
</style></head><body><main>
<div class="brand"><div class="mark"><i></i><i></i><i></i><i></i></div><b>Hookline</b><span class="by"><span class="w">W</span><span class="l">L</span><span class="d1">D</span><span class="d2">D</span></span></div>
<h1>Try a video</h1><p class="mut">Upload a reel to see where attention is predicted to drop, what was on screen there, and what to try.</p>
<div class="note">Everything on the result page is a <b>model prediction</b>, not audience data, and none of it says a video will or will not go viral. Each score comes with the measured accuracy of the model behind it.</div>
<section class="panel" id="form">
  <div id="drop"><b id="dropt">Choose a video, or drop it here</b><span class="mut">mp4, mov, m4v, webm or mkv, up to __MAX_MB__ MB.</span></div>
  <input type="file" id="file" accept="video/*,.mp4,.mov,.m4v,.webm,.mkv" hidden>
  <button id="go" disabled>Analyse</button> <span class="mut" id="msg"></span>
</section>
<section class="panel" id="run" style="display:none"><b id="runt">Analysing…</b> <span class="mut" id="runs"></span><div class="steps" id="steps"></div><details id="logbox" style="display:none;margin-top:10px"><summary class="mut">Show details</summary><pre id="log"></pre></details><div id="done"></div></section>
<section class="panel" id="recent" style="display:none"><b>Earlier uploads</b><ul id="recentlist"></ul></section>
<p class="mut">Your video is analysed on Hookline's cloud services and is not shared otherwise. A short clip takes a few minutes.</p>
</main><script>
const $=(i)=>document.getElementById(i);let file=null,poll=null;
const pick=(f)=>{if(!f)return;file=f;$('dropt').textContent=f.name+' ('+(f.size/1048576).toFixed(1)+' MB)';upd();};
const upd=()=>{$('go').disabled=!file;};
$('drop').addEventListener('click',()=>$('file').click());$('file').addEventListener('change',(e)=>pick(e.target.files[0]));
$('drop').addEventListener('dragover',(e)=>{e.preventDefault();$('drop').classList.add('on');});$('drop').addEventListener('dragleave',()=>$('drop').classList.remove('on'));
$('drop').addEventListener('drop',(e)=>{e.preventDefault();$('drop').classList.remove('on');pick(e.dataTransfer.files[0]);});
function show(j){$('run').style.display='';$('runt').textContent=j.status==='done'?'Done':j.status==='failed'?'Stopped':'Analysing…';$('runs').textContent=j.seconds+' s';
const FR=[[/^features/,'Reading the video'],[/^speech/,'Listening and reading on-screen text'],[/^retention/,'Predicting viewer retention'],[/^explanation/,'Understanding the content'],[/^music/,'Analysing music and voice'],[/^popularity/,'Estimating popularity potential'],[/^scores/,'Scoring the hook'],[/^explain/,'Finding strongest and weakest moments'],[/^segments/,'Preparing your review']];
const friendly=(n)=>{for(const[r,t]of FR)if(r.test(n))return t;return 'Working';};
const box=$('steps');box.textContent='';for(const s of j.steps){const d=document.createElement('div');d.className='step '+s.state;const dot=document.createElement('span');dot.className='dot';const n=document.createElement('span');n.textContent=friendly(s.name);const st=document.createElement('span');st.className='mut';st.textContent=s.state==='running'?'in progress':(s.state==='ok'?'done':s.state==='failed'?'failed':'skipped');d.append(dot,n,st);box.append(d);}
$('log').textContent=j.log.join('\n');$('log').scrollTop=1e9;$('logbox').style.display=j.status==='failed'?'':'none';
const done=$('done');done.textContent='';
if(j.status==='done'||j.status==='failed'){clearInterval(poll);$('go').disabled=false;
if(j.review_url){const a=document.createElement('a');a.className='btn';a.href=j.review_url;a.textContent='Open the analysis →';done.append(a);}
if(j.status==='failed'){const p=document.createElement('p');p.className='err';p.textContent='Something went wrong while analysing this video. Try again in a moment; "Show details" has the technical reason.';done.append(p);}}}
async function tick(id){try{const r=await fetch('/api/job/'+id);show(await r.json());}catch(e){$('msg').textContent='lost contact with the server';}}
$('go').addEventListener('click',async()=>{$('go').disabled=true;$('msg').textContent='uploading…';
try{const r=await fetch('/api/upload',{method:'POST',headers:{'X-Hookline':'1','X-Filename':encodeURIComponent(file.name)},body:file});const j=await r.json();
if(!r.ok){$('msg').textContent=j.error||'upload failed';$('go').disabled=false;return;}$('msg').textContent='';clearInterval(poll);poll=setInterval(()=>tick(j.job_id),2000);tick(j.job_id);}
catch(e){$('msg').textContent='upload failed: '+e;$('go').disabled=false;}});
fetch('/api/recent').then(r=>r.json()).then(d=>{if(!d.recent.length)return;$('recent').style.display='';const ul=$('recentlist');for(const x of d.recent){const li=document.createElement('li');const a=document.createElement('a');a.href=x.url;a.style.color='#d9c3fb';a.textContent=x.id;li.append(a);ul.append(li);}});
</script></body></html>
"""
