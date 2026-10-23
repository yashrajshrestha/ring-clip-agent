"""Decider playground: ask Strands Decider 2B any typed question, replay a
real doorbell event and compare its choice with the rule stub, or drop a new
clip in and watch it get analysed and decided.

    python -m vision.playground [--port 8090] [--samples web/public/samples] [--inbox clips/inbox] [--no-inbox]

Needs a running `strands-decider serve` at DECIDER_URL (default
http://127.0.0.1:8000). Standard library only. Calls wait up to
PLAYGROUND_TIMEOUT seconds (default 600) because a CPU call is slow. The
inbox watcher (vision/inbox.py) runs in a thread: a clip dropped into the
inbox folder, or uploaded from the page, is processed with the same settings
(DECIDER_URL, DECIDER_TIMEOUT, VISION_MODEL) and appears in the clip list.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import decider, escalate
from .profiles import PROFILES
from .timeline import VIDEO_EXT, _decide

PAGE = Path(__file__).with_name("playground.html")
TIMEOUT_S = float(os.environ.get("PLAYGROUND_TIMEOUT", 600))
MAX_UPLOAD = 500 << 20   # bytes
PRIVATE = {"request", "raw"}  # server-side only, never sent in the event list


def load_events(samples: Path) -> list[dict]:
    """Every event from the timeline files, with the exact state text the
    pipeline would send and the rule stub's answer for comparison."""
    out = []
    for path in sorted(samples.glob("*.timeline.json")):
        clip = path.name.removesuffix(".timeline.json")
        try:
            tl = json.loads(path.read_text())
            events = tl["events"]
        except (ValueError, KeyError, OSError):
            continue  # a timeline still being written, or the inbox's marker for a broken file
        for n, ev in enumerate(events):
            rule = _decide(ev)
            out.append({
                "clip": clip, "id": ev["id"], "type": ev["type"], "label": ev.get("label"),
                "profile": tl.get("profile", "normal"),
                "t_start": ev["t_start"], "t_end": ev["t_end"], "say": ev.get("say"),
                "decision": ev.get("decision"),
                "pet": ev.get("pet"),            # vision/pets.py: your dog by name, or unknown
                "first_decision": ev.get("first_decision"),
                "decider_decision": ev.get("decider_decision"),
                "vision_decision": ev.get("vision_decision"),   # a second look that could not silence a person
                "rule": {"action": rule["action"], "reason": rule["reason"]},
                "request": decider.request(ev, events[:n]),
                "state": decider.state_text(ev, events[:n]),   # what the decider read, shown without a live call
                "raw": {"ev": ev, "earlier": events[:n]},
            })
    return out


class EventStore:
    """Events from the samples folder, reloaded whenever a timeline file appears or changes."""

    def __init__(self, samples: Path) -> None:
        self.samples = samples
        self.events: list[dict] = []
        self.by_key: dict[tuple, dict] = {}
        self._sig: tuple = ()
        self.refresh()

    def refresh(self) -> None:
        sig = tuple(sorted((p.name, p.stat().st_mtime_ns) for p in self.samples.glob("*.timeline.json"))) \
            if self.samples.is_dir() else ()
        if sig != self._sig:
            self.events = load_events(self.samples) if self.samples.is_dir() else []
            self.by_key = {(e["clip"], e["id"]): e for e in self.events}
            self._sig = sig


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


def safe_clip_name(name: str) -> str | None:
    """A plain file name with a video extension, or None."""
    name = Path(unquote(name)).name
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not stem or ext not in VIDEO_EXT:
        return None
    return f"{stem}{ext}"


def make_handler(url: str, samples: Path, inbox: Path | None = None, uploads: bool = True):
    store = EventStore(samples)

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
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._send(200, health(url))
            elif path == "/api/events":
                store.refresh()
                self._send(200, [{k: v for k, v in e.items() if k not in PRIVATE} for e in store.events])
            elif path == "/api/inbox":
                status_file = samples / "inbox.json"
                status = json.loads(status_file.read_text()) if status_file.exists() else {}
                self._send(200, {"enabled": inbox is not None, "uploads": inbox is not None and uploads,
                                 "folder": str(inbox) if inbox else None, **status})
            elif path.startswith("/samples/"):
                name = unquote(path[len("/samples/"):])
                video = samples / name
                if "/" in name or not name.endswith(".mp4") or not video.is_file():
                    return self._send(404, {"error": "no such clip"})
                self._send_video(video)
            else:
                self._send(404, {"error": "not found"})

        def do_PUT(self):
            parts = urlsplit(self.path)
            if parts.path != "/api/upload":
                return self._send(404, {"error": "not found"})
            if inbox is None:
                return self._send(503, {"error": "the inbox is disabled on this server"})
            if not uploads:   # public demo: the sample clips can be re-run, nothing new comes in
                return self._send(403, {"error": "uploads are turned off on this server; use Re-run on a sample clip"})
            name = safe_clip_name(parse_qs(parts.query).get("name", [""])[0])
            if name is None:
                return self._send(400, {"error": f"name must be a video file ({', '.join(sorted(VIDEO_EXT))})"})
            length = int(self.headers.get("content-length", 0))
            if not 0 < length <= MAX_UPLOAD:
                return self._send(413, {"error": f"upload must be between 1 byte and {MAX_UPLOAD >> 20} MB"})
            inbox.mkdir(parents=True, exist_ok=True)
            tmp = inbox / f".{name}.part"   # hidden while writing, so the watcher ignores it
            with open(tmp, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            tmp.replace(inbox / name)
            self._send(200, {"name": Path(name).stem, "file": name, "bytes": length, "since": time.time()})

        def do_POST(self):
            try:
                body = self._body()
            except ValueError:
                return self._send(400, {"error": "body is not JSON"})
            store.refresh()
            if self.path == "/api/ask":
                self._send(*timed_ask(body, url))
            elif self.path == "/api/event":
                ev = store.by_key.get((body.get("clip"), body.get("id")))
                if ev is None:
                    return self._send(404, {"error": "no such event"})
                status, out = timed_ask(ev["request"], url)
                self._send(status, {**out, "request": ev["request"], "rule": ev["rule"]})
            elif self.path == "/api/rerun":
                # Re-drop the clip: the watcher picks up a touched or copied file like a new one.
                clip = str(body.get("clip", ""))
                profile = str(body.get("profile", "normal"))
                if inbox is None:
                    return self._send(503, {"error": "the inbox is disabled on this server"})
                if not clip or clip != Path(clip).name or not (samples / f"{clip}.timeline.json").exists():
                    return self._send(404, {"error": "no such clip"})
                if profile not in PROFILES:
                    return self._send(400, {"error": f"profile must be one of {', '.join(PROFILES)}"})
                src = next((p for p in inbox.glob(f"{clip}.*") if p.suffix.lower() in VIDEO_EXT), None) \
                    or next((p for p in inbox.parent.rglob(f"{clip}.*")
                             if p.suffix.lower() in VIDEO_EXT and inbox not in p.parents), None)
                inbox.mkdir(parents=True, exist_ok=True)
                if profile != "normal":   # sidecar the watcher reads and removes
                    (inbox / f"{clip}.profile").write_text(profile)
                if src is not None and src.parent == inbox:
                    src.touch()
                else:
                    tmp = inbox / f".{clip}.mp4.part"
                    tmp.write_bytes((src or samples / f"{clip}.mp4").read_bytes())
                    tmp.replace(inbox / f"{clip}.mp4")
                # `since`: the page waits for a finish time later than this, so an old run cannot satisfy it
                self._send(200, {"queued": clip, "profile": profile, "source": str(src or samples / f"{clip}.mp4"),
                                 "since": time.time()})
            elif self.path == "/api/cancel":
                # Ask the watcher to stop the clip it is working on; it checks between steps.
                clip = str(body.get("clip", ""))
                if inbox is None:
                    return self._send(503, {"error": "the inbox is disabled on this server"})
                status_file = samples / "inbox.json"
                current = json.loads(status_file.read_text()).get("processing") if status_file.exists() else None
                if not clip or clip != current:
                    return self._send(409, {"error": f"{clip or 'that clip'} is not being processed right now"})
                inbox.mkdir(parents=True, exist_ok=True)
                (inbox / f"{clip}.cancel").write_text("")
                self._send(200, {"cancelling": clip})
            elif self.path == "/api/escalate":
                ev = store.by_key.get((body.get("clip"), body.get("id")))
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
    ap.add_argument("--inbox", type=Path, default=Path("clips/inbox"))
    ap.add_argument("--no-inbox", action="store_true", help="do not watch or accept new clips")
    ap.add_argument("--no-detect", action="store_true", help="inbox: skip the object detector")
    ap.add_argument("--no-upload", action="store_true",
                    help="public demo: Re-run and Look harder work on the sample clips, uploads are refused")
    args = ap.parse_args()
    url = os.environ.get("DECIDER_URL", "http://127.0.0.1:8000")
    inbox_dir = None if args.no_inbox else args.inbox
    if inbox_dir is not None:
        from .inbox import Inbox
        from .timeline import load_detector
        Inbox(inbox_dir, args.samples, None if args.no_detect else load_detector()).start_thread()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(url, args.samples, inbox_dir, uploads=not args.no_upload))
    print(f"playground on http://127.0.0.1:{args.port} (decider {url}"
          f"{', inbox ' + str(inbox_dir) if inbox_dir else ''})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
