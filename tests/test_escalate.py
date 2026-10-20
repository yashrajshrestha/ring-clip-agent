"""Escalation tests against a fake Ollama server, so they run without a model."""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import pytest

from vision import decider, escalate

EV = {"id": 0, "type": "dark", "t_start": 0.5, "t_end": 1.5, "peak_area": 0.0,
      "recheck": {"usable": True, "objects": {}, "label": None}}


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 180))
    for _ in range(30):
        w.write(np.full((180, 320, 3), 30, np.uint8))
    w.release()
    return p


@pytest.fixture
def fake_ollama():
    seen, reply = [], {"text": json.dumps({"seen": "A child in a costume is at the door.", "action": "alert"})}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, json.loads(self.rfile.read(int(self.headers["content-length"])))))
            data = json.dumps({"message": {"role": "assistant", "content": reply["text"]}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen, reply
    srv.shutdown()


def test_only_a_low_confidence_decider_answer_escalates():
    assert escalate.should_escalate({"action": "ignore", "confidence": 0.15})
    assert not escalate.should_escalate({"action": "alert", "confidence": 0.83})
    assert not escalate.should_escalate({"action": "recheck", "source": "rule-stub"})


def test_sends_one_enhanced_jpeg_with_the_four_choices(clip, fake_ollama):
    url, seen, _ = fake_ollama
    out = escalate.escalate(clip, EV, [], model="fake-vl", url=url)
    assert isinstance(out.pop("ms"), int) and out.pop("tokens") is None   # Ollama reports no token counts
    assert out == {"action": "alert", "reason": "A child in a costume is at the door.",
                   "source": "vision-llm:fake-vl", "confidence": None}
    path, body = seen[0]
    assert path == "/api/chat" and body["model"] == "fake-vl" and body["format"] == "json"
    msg = body["messages"][0]
    assert all(k in msg["content"] for k in decider.CRITERIA)
    assert "Step 1" in msg["content"] and "dark at 0.5s" in msg["content"]
    assert "nothing recognised" not in msg["content"]  # no detector verdict for the model to parrot
    jpeg = base64.b64decode(msg["images"][0])
    assert jpeg[:2] == b"\xff\xd8"  # JPEG magic
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 640 and img.mean() > 30  # enhanced, not the raw dark frame


class FakeBedrock:
    """Stands in for boto3's bedrock-runtime client: records the Converse call, answers like Nova Lite."""

    def __init__(self, text):
        self.text, self.calls = text, []

    def converse(self, **kw):
        self.calls.append(kw)
        return {"output": {"message": {"role": "assistant", "content": [{"text": self.text}]}},
                "usage": {"inputTokens": 1234, "outputTokens": 28, "totalTokens": 1262}}


def test_bedrock_model_goes_through_converse_with_the_image_inline(clip, monkeypatch):
    fake = FakeBedrock('```json\n{"seen": "A courier is leaving a box.", "action": "alert"}\n```')
    monkeypatch.setattr(escalate, "bedrock_client", lambda region=None: fake)
    out = escalate.escalate(clip, EV, [], model="bedrock:amazon.nova-lite-v1:0")
    assert out["action"] == "alert" and out["reason"] == "A courier is leaving a box."
    assert out["source"] == "vision-llm:bedrock:amazon.nova-lite-v1:0" and out["confidence"] is None
    assert out["tokens"] == {"in": 1234, "out": 28} and isinstance(out["ms"], int)
    call = fake.calls[0]
    assert call["modelId"] == "amazon.nova-lite-v1:0" and call["inferenceConfig"]["temperature"] == 0.0
    image, text = call["messages"][0]["content"]
    assert image["image"]["format"] == "jpeg" and image["image"]["source"]["bytes"][:2] == b"\xff\xd8"
    assert "Step 1" in text["text"] and all(k in text["text"] for k in decider.CRITERIA)


def test_bedrock_failures_leave_the_decider_answer(clip, monkeypatch):
    class Broken:
        def converse(self, **kw):
            raise RuntimeError("Unable to locate credentials")

    monkeypatch.setattr(escalate, "bedrock_client", lambda region=None: Broken())
    assert escalate.escalate(clip, EV, [], model="bedrock:amazon.nova-lite-v1:0") is None


def test_bad_answers_fall_back_to_the_decider(clip, fake_ollama):
    url, _, reply = fake_ollama
    for text in ("not json", json.dumps({"action": "panic"}), json.dumps(["alert"])):
        reply["text"] = text
        assert escalate.escalate(clip, EV, [], model="fake-vl", url=url) is None


def test_timeline_replaces_a_low_confidence_decision_and_keeps_the_deciders(clip, monkeypatch):
    from vision import timeline

    def fake_decide(ev, earlier, url=None):
        return {"action": "ignore", "reason": "x", "source": decider.SOURCE, "confidence": 0.15}

    monkeypatch.setattr(decider, "decide", fake_decide)
    monkeypatch.setattr(escalate, "escalate", lambda video, ev, earlier, model=None, url=None:
                        {"action": "alert", "reason": "Someone is at the door.", "source": "vision-llm:fake", "confidence": None})
    events = timeline.build_timeline(clip)["events"]
    assert events, "the flat dark clip should raise at least a dark event"
    for ev in events:
        assert ev["decision"]["source"] == "vision-llm:fake" and ev["decision"]["action"] == "alert"
        assert ev["decider_decision"]["confidence"] == 0.15


def test_timeline_keeps_a_confident_decision(clip, monkeypatch):
    from vision import timeline

    monkeypatch.setattr(decider, "decide", lambda ev, earlier, url=None:
                        {"action": "log", "reason": "x", "source": decider.SOURCE, "confidence": 0.9})
    monkeypatch.setattr(escalate, "escalate", lambda *a, **k: pytest.fail("escalated a confident decision"))
    for ev in timeline.build_timeline(clip)["events"]:
        assert ev["decision"]["source"] == decider.SOURCE and "decider_decision" not in ev


def test_off_without_a_model_and_on_unreachable_server(clip, monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)
    assert escalate.escalate(clip, EV, []) is None
    assert escalate.escalate(clip, EV, [], model="fake-vl", url="http://127.0.0.1:9") is None


ALERT = {"action": "alert", "reason": "x", "source": decider.SOURCE, "confidence": 0.3}
SAYS_NOTHING = {"action": "ignore", "reason": "Nothing is at the door.", "source": "vision-llm:fake", "confidence": None}


@pytest.mark.parametrize("label, decision, answer, overrides", [
    ("person", ALERT, SAYS_NOTHING, False),                       # one empty frame must not silence a person
    ("figure", ALERT, {**SAYS_NOTHING, "action": "log"}, False),  # person-sized tracked movement counts too
    ("animal", ALERT, {**SAYS_NOTHING, "action": "log"}, True),   # "a dog is at the door": settles a false alarm
    (None, ALERT, SAYS_NOTHING, True),                            # scene changes, dark stretches: the look decides
    ("person", {**ALERT, "action": "ignore"}, {**SAYS_NOTHING, "action": "alert"}, True),  # raising is always allowed
])
def test_the_second_look_may_raise_an_alert_but_not_silence_a_person(label, decision, answer, overrides):
    assert escalate.may_override({"label": label}, decision, answer) is overrides


def test_a_kept_alert_records_the_vision_answer_beside_it():
    ev = {"label": "person", "decision": dict(ALERT)}
    escalate.apply_second_look(ev, dict(SAYS_NOTHING))
    assert ev["decision"]["action"] == "alert" and ev["decision"]["source"] == decider.SOURCE
    assert ev["vision_decision"]["reason"] == "Nothing is at the door."
    assert "decider_decision" not in ev

    ev = {"label": "animal", "decision": dict(ALERT)}
    escalate.apply_second_look(ev, {**SAYS_NOTHING, "action": "log"})
    assert ev["decision"]["source"] == "vision-llm:fake" and ev["decider_decision"] == ALERT
