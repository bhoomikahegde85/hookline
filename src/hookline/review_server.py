"""A tiny local web server for the review page, so the video loads over http://127.0.0.1 instead of file://.

Some browser profiles (managed or extension-restricted) refuse to play a ``file://`` video from a ``file://`` page; plain http on
localhost works everywhere. The server:
- binds to 127.0.0.1 only and checks the Host header (no access from other machines, no DNS rebinding);
- serves exactly two things: ``/`` (the review page) and ``/video`` (the one video file), with HTTP Range support so seeking works;
- lists no directories and serves no other path; nothing is uploaded or fetched from the network.
"""

from __future__ import annotations

import http.server
import mimetypes
import re
import threading
from pathlib import Path

CHUNK = 1 << 20


def make_handler(page: Path, video: Path | None, port_holder: dict):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "HooklineReview"

        def log_message(self, *args) -> None:  # quiet
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            return host in (f"127.0.0.1:{port_holder['port']}", f"localhost:{port_holder['port']}")

        def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain; charset=utf-8", extra: dict | None = None, head: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body and not head:
                self.wfile.write(body)

        def _serve(self, head: bool) -> None:
            if not self._host_ok():
                return self._send(403, b"forbidden host", head=head)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html", "/review.html"):
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8", head=head)
            if path == "/video" and video is not None and video.is_file():
                return self._video(head)
            self._send(404, b"not found", head=head)

        def _video(self, head: bool) -> None:
            size = video.stat().st_size
            ctype = mimetypes.guess_type(video.name)[0] or "application/octet-stream"
            start, end, code = 0, size - 1, 200
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range: the last N bytes
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
                with video.open("rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):  # the browser seeked or closed the stream
                pass

        def do_GET(self) -> None:  # noqa: N802
            self._serve(False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._serve(True)

    return Handler


def start_server(page: Path, video: Path | None, port: int = 0) -> tuple[http.server.ThreadingHTTPServer, int]:
    """Start serving in a background thread; returns (server, port). Call ``server.shutdown()`` to stop."""
    holder: dict = {"port": port}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(page, video, holder))
    holder["port"] = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, holder["port"]
