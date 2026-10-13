"""DeciderGate tests with the decider stubbed out, so they run without
Ollama or the model."""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("strands")

from strands.interventions import Deny, Guide, Proceed  # noqa: E402

from vision import agent, decider  # noqa: E402

PERSON = {"id": 0, "type": "motion", "t_start": 1.0, "t_end": 4.0, "peak_area": 0.2, "label": "person"}
DARK = {"id": 1, "type": "dark", "t_start": 5.0, "t_end": 9.0, "peak_area": 0.0}


def _call(gate, name="notify_owner"):
    return asyncio.run(gate.before_tool_call(SimpleNamespace(tool_use={"name": name, "input": {}})))


def _decision(action, confidence=0.9):
    return {"action": action, "reason": decider.REASONS[action], "source": decider.SOURCE,
            "confidence": confidence}


@pytest.mark.parametrize("action, kind, text", [
    ("alert", Proceed, None),
    ("ignore", Deny, "Do not notify"),
    ("recheck", Guide, "recheck_camera"),
    ("log", Guide, "log_event"),
])
def test_each_decision_maps_to_an_intervention(monkeypatch, action, kind, text):
    monkeypatch.setattr(decider, "decide", lambda ev, earlier, url=None: _decision(action))
    gate = agent.DeciderGate("http://decider")
    gate.event = PERSON
    out = _call(gate)
    assert isinstance(out, kind)
    if text:
        assert text in (out.reason if kind is Deny else out.feedback)
    assert gate.decisions == [_decision(action)]


def test_other_tools_pass_without_asking(monkeypatch):
    monkeypatch.setattr(decider, "decide", lambda *a, **k: pytest.fail("decider asked"))
    gate = agent.DeciderGate("http://decider")
    gate.event = PERSON
    assert isinstance(_call(gate, "log_event"), Proceed) and gate.decisions == []


def test_retried_notify_reuses_the_answer(monkeypatch):
    calls = []
    monkeypatch.setattr(decider, "decide", lambda ev, earlier, url=None: calls.append(ev) or _decision("ignore"))
    gate = agent.DeciderGate("http://decider")
    gate.event = PERSON
    _call(gate), _call(gate)
    assert len(calls) == 1 and len(gate.decisions) == 1


def test_rule_stub_decides_when_decider_is_off(monkeypatch):
    monkeypatch.setattr(decider, "decide", lambda *a, **k: None)
    gate = agent.DeciderGate(None)
    gate.event = DARK
    out = _call(gate)
    assert isinstance(out, Guide) and "recheck_camera" in out.feedback
    assert gate.decisions[0]["source"] == "rule-stub"
