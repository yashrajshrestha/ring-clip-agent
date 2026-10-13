"""Decider playground: ask Strands Decider 2B any typed question, or replay a
real doorbell event and compare its choice with the rule stub.

    python -m vision.playground [--port 8090] [--samples web/public/samples]

Needs a running `strands-decider serve` at DECIDER_URL (default
http://127.0.0.1:8000). Standard library only. Calls wait up to
PLAYGROUND_TIMEOUT seconds (default 600) because a CPU call is slow.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from . import decider, escalate
from .timeline import _decide

PAGE = Path(__file__).with_name("playground.html")
TIMEOUT_S = float(os.environ.get("PLAYGROUND_TIMEOUT", 600))


def load_events(samples: Path) -> list[dict]:
    """Every event from the timeline files, with the exact state text the
    pipeline would send and the rule stub's answer for comparison."""
    out = []
    for path in sorted(samples.glob("*.timeline.json")):
        clip = path.name.removesuffix(".timeline.json")
        events = json.loads(path.read_text())["events"]
        for n, ev in enumerate(events):
            rule = _decide(ev)
            out.append({
                "clip": clip, "id": ev["id"], "type": ev["type"], "label": ev.get("label"),
                "t_start": ev["t_start"], "t_end": ev["t_end"], "say": ev.get("say"),
                "rule": {"action": rule["action"], "reason": rule["reason"]},
                "request": decider.request(ev, events[:n]),
                "raw": {"ev": ev, "earlier": events[:n]},
            })
    return out


PRIVATE = {"request", "raw"}  # server-side only, never sent in the event list


def timed_ask(body: dict, url: str) -> tuple[int, dict]:
    """Forward one request; returns (status, payload) and never raises."""
    t0 = time.time()
    try:
        resp = decider.ask(body, url, timeout=TIMEOUT_S)
        return 200, {"response": resp, "ms": round((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:2000]}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return 502, {"error": f"{type(e).__name__}: {e}"}


def health(url: str) -> dict:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=3) as resp:
            return {"ok": True, "url": url, **json.load(resp)}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return {"ok": False, "url": url, "error": f"{type(e).__name__}: {e}"}


def make_handler(url: str, samples: Path):
    events = load_events(samples) if samples.is_dir() else []
    by_key = {(e["clip"], e["id"]): e for e in events}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload, ctype: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")

        def _send_video(self, path: Path) -> None:
            """Serve an mp4 with Range support, so <video> can seek."""
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = min(int(b), size - 1) if (a and b) else end
                status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("content-type", "video/mp4")
            self.send_header("accept-ranges", "bytes")
            self.send_header("content-length", str(length))
            if status == 206:
                self.send_header("content-range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    while length > 0:
                        chunk = f.read(min(1 << 16, length))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        length -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the player seeked and dropped this request

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/health":
                self._send(200, health(url))
            elif self.path == "/api/events":
                self._send(200, [{k: v for k, v in e.items() if k not in PRIVATE} for e in events])
            elif self.path.startswith("/samples/"):
                name = unquote(self.path[len("/samples/"):])
                video = samples / name
                if "/" in name or not name.endswith(".mp4") or not video.is_file():
                    return self._send(404, {"error": "no such clip"})
                self._send_video(video)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
            except ValueError:
                return self._send(400, {"error": "body is not JSON"})
            if self.path == "/api/ask":
                self._send(*timed_ask(body, url))
            elif self.path == "/api/event":
                ev = by_key.get((body.get("clip"), body.get("id")))
                if ev is None:
                    return self._send(404, {"error": "no such event"})
                status, out = timed_ask(ev["request"], url)
                self._send(status, {**out, "request": ev["request"], "rule": ev["rule"]})
            elif self.path == "/api/escalate":
                ev = by_key.get((body.get("clip"), body.get("id")))
                if ev is None:
                    return self._send(404, {"error": "no such event"})
                model = os.environ.get("VISION_MODEL")
                if not model:
                    return self._send(503, {"error": "VISION_MODEL is not set on the playground server"})
                t0 = time.time()
                answer = escalate.escalate(samples / f"{ev['clip']}.mp4", ev["raw"]["ev"], ev["raw"]["earlier"])
                if answer is None:
                    return self._send(502, {"error": f"{model} gave no usable answer (is Ollama running with it pulled?)"})
                self._send(200, {"answer": answer, "model": model, "ms": round((time.time() - t0) * 1000)})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--samples", type=Path, default=Path("web/public/samples"))
    args = ap.parse_args()
    url = os.environ.get("DECIDER_URL", "http://127.0.0.1:8000")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(url, args.samples))
    print(f"playground on http://127.0.0.1:{args.port} (decider {url})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
