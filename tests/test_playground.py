"""Playground server tests against a fake /v1/systemone server, so they run
offline without the model."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import pytest

from vision import decider, playground

EVENTS = [
    {"id": 0, "type": "motion", "t_start": 1.0, "t_end": 4.0, "peak_area": 0.2, "label": "person",
     "objects": {"person": 3}, "say": "Person at the door."},
    {"id": 1, "type": "scene_change", "t_start": 5.0, "t_end": 5.0, "peak_area": 0.0, "say": "Object left or removed."},
]


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


@pytest.fixture
def fake_decider():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._json(200, {"status": "ok", "model": "fake", "device": "cpu"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append(body)
            if not body["state"]:
                return self._json(422, {"detail": "state must not be empty"})
            name = next(iter(body["questions"]))
            self._json(200, {"model": "fake", "usage": {"input_tokens": 5, "output_tokens": 1},
                             "answers": {name: {"type": "choice", "choice": "ignore",
                                                "probabilities": {"alert": 0.3, "ignore": 0.7},
                                                "confidence": 0.4}}})

        def _json(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


@pytest.fixture
def app(tmp_path, fake_decider):
    (tmp_path / "porch.timeline.json").write_text(json.dumps({"events": EVENTS}))
    url, seen = fake_decider
    srv, base = _serve(playground.make_handler(url, tmp_path))
    yield base, seen
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_events_carry_rule_answer_and_pipeline_request(tmp_path):
    (tmp_path / "porch.timeline.json").write_text(json.dumps({"events": EVENTS}))
    evs = playground.load_events(tmp_path)
    assert [e["rule"]["action"] for e in evs] == ["alert", "alert"]
    assert evs[1]["request"] == decider.request(EVENTS[1], EVENTS[:1])


def test_page_health_and_event_list(app):
    base, _ = app
    status, html = _get(base + "/")
    assert status == 200 and b"Decider Playground" in html
    assert json.loads(_get(base + "/api/health")[1])["ok"] is True
    evs = json.loads(_get(base + "/api/events")[1])
    assert [(e["clip"], e["id"]) for e in evs] == [("porch", 0), ("porch", 1)]
    assert "request" not in evs[0]


def test_ask_forwards_and_times(app):
    base, seen = app
    body = {"state": "hello", "questions": {"answer": {"type": "choice", "instructions": "pick",
                                                       "criteria": {"alert": "a", "ignore": "b"}}}}
    status, out = _post(base + "/api/ask", body)
    assert status == 200 and out["response"]["answers"]["answer"]["choice"] == "ignore"
    assert isinstance(out["ms"], int) and seen[-1] == body


def test_ask_passes_decider_validation_errors_through(app):
    base, _ = app
    status, out = _post(base + "/api/ask", {"state": "", "questions": {"a": {"type": "choice"}}})
    assert status == 422 and "state must not be empty" in out["error"]


def test_event_sends_pipeline_request_and_returns_rule(app):
    base, seen = app
    status, out = _post(base + "/api/event", {"clip": "porch", "id": 1})
    assert status == 200 and out["rule"]["action"] == "alert"
    assert seen[-1] == decider.request(EVENTS[1], EVENTS[:1])
    assert _post(base + "/api/event", {"clip": "porch", "id": 9})[0] == 404


def test_video_is_served_with_range_support(app, tmp_path):
    base, _ = app
    (tmp_path / "porch.mp4").write_bytes(bytes(range(100)))
    status, body = _get(base + "/samples/porch.mp4")
    assert status == 200 and body == bytes(range(100))
    req = urllib.request.Request(base + "/samples/porch.mp4", headers={"Range": "bytes=10-19"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 206 and r.headers["content-range"] == "bytes 10-19/100"
        assert r.read() == bytes(range(10, 20))
    for bad in ("/samples/missing.mp4", "/samples/porch.timeline.json", "/samples/..%2Fporch.mp4"):
        try:
            urllib.request.urlopen(base + bad, timeout=5)
            assert False, bad
        except urllib.error.HTTPError as e:
            assert e.code == 404


def test_unreachable_decider_is_a_502(tmp_path):
    srv, base = _serve(playground.make_handler("http://127.0.0.1:9", tmp_path))
    try:
        status, out = _post(base + "/api/ask", {"state": "x", "questions": {}})
        assert status == 502 and "error" in out
        assert json.loads(_get(base + "/api/health")[1])["ok"] is False
    finally:
        srv.shutdown()
