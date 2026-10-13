"""Decider client tests against a fake /v1/systemone server, so they run
offline without the model."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vision import decider

PERSON = {"type": "motion", "t_start": 1.0, "t_end": 4.0, "peak_area": 0.2, "label": "person",
          "objects": {"person": 1}}
TAKEN = {"type": "scene_change", "t_start": 5.0, "t_end": 5.0, "peak_area": 0.0}


@pytest.fixture
def fake_server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            out = {"answers": {"action": {"type": "choice", "choice": "alert",
                                          "probabilities": {"alert": 0.9, "ignore": 0.1},
                                          "confidence": 0.8}}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def test_off_without_url(monkeypatch):
    monkeypatch.delenv("DECIDER_URL", raising=False)
    assert decider.decide(PERSON, []) is None


def test_unreachable_server_falls_back():
    assert decider.decide(PERSON, [], url="http://127.0.0.1:9") is None


def test_sends_one_choice_question_with_earlier_events(fake_server):
    url, seen = fake_server
    out = decider.decide(TAKEN, [PERSON], url=url)
    assert out == {"action": "alert", "reason": decider.REASONS["alert"], "source": decider.SOURCE,
                   "confidence": 0.8, "probabilities": {"alert": 0.9, "ignore": 0.1}}
    q = seen[0]["questions"]["action"]
    assert q["type"] == "choice" and set(q["criteria"]) == set(decider.CRITERIA)
    assert "scene change" in seen[0]["state"] and "Earlier in this clip: motion" in seen[0]["state"]
