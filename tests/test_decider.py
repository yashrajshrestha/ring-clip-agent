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
    assert 0 <= out.pop("ms") < 5000   # round-trip time, shown in the UI
    assert out == {"action": "alert", "reason": decider.REASONS["alert"], "source": decider.SOURCE,
                   "confidence": 0.8, "probabilities": {"alert": 0.9, "ignore": 0.1}}
    q = seen[0]["questions"]["action"]
    assert q["type"] == "choice" and set(q["criteria"]) == set(decider.CRITERIA)
    state = seen[0]["state"]
    assert "an object was left or taken at 5.0s" in state
    assert "Earlier in this clip: something moved at the door at 1.0s, lasting 3.0s, the detector saw a person" in state


def test_state_text_says_an_animal_is_not_a_person():
    dog = {"type": "motion", "t_start": 2.0, "t_end": 3.0, "peak_area": 0.1, "label": "animal"}
    assert "an animal (a dog, cat or bird), not a person" in decider.state_text(dog, [])
    dark = {"type": "dark", "t_start": 4.3, "t_end": 5.1, "recheck": {"usable": True, "objects": {"animal": 7}}}
    assert "detector saw animal x7 (an animal such as a dog, not a person)" in decider.state_text(dark, [])
    both = {"type": "dark", "t_start": 4.3, "t_end": 5.1, "recheck": {"usable": True, "objects": {"animal": 2, "person": 5}}}
    assert "not a person" not in decider.state_text(both, [])


def test_state_text_spells_out_cuts_and_unlabelled_motion():
    cut = {"type": "cut", "t_start": 12.3, "t_end": 12.3, "peak_area": 0.0}
    assert "an edit in the recording, not something at the door" in decider.state_text(cut, [])
    blob = {"type": "motion", "t_start": 2.0, "t_end": 2.4, "peak_area": 0.01}
    assert "the detector saw no person, animal or vehicle" in decider.state_text(blob, [])
