"""Decider-first agent tests with the decider stubbed out, so they run
without Ollama or the model."""
from types import SimpleNamespace

import pytest

pytest.importorskip("strands")

from strands.interventions import Deny, Proceed  # noqa: E402

from vision import agent, decider  # noqa: E402

PERSON = {"id": 0, "type": "motion", "t_start": 1.0, "t_end": 4.0, "peak_area": 0.2, "label": "person"}
DARK = {"id": 1, "type": "dark", "t_start": 5.0, "t_end": 9.0, "peak_area": 0.0}


def _decision(action, source=decider.SOURCE, confidence=0.9):
    return {"action": action, "reason": decider.REASONS[action], "source": source, "confidence": confidence}


class NoModel:
    """Fails the test if the agent ever tries to use it."""

    def __getattr__(self, name):
        pytest.fail(f"language model used: {name}")


def test_stored_decider_decision_is_used_without_a_call(monkeypatch):
    monkeypatch.setattr(decider, "decide", lambda *a, **k: pytest.fail("decider called"))
    ev = {**PERSON, "decision": _decision("alert")}
    assert agent.decide_event(ev, []) == ev["decision"]
    ev = {**PERSON, "decision": {"action": "alert", "reason": "r", "source": "vision-llm:x", "confidence": None}}
    assert agent.decide_event(ev, [])["source"] == "vision-llm:x"


def test_rule_stub_decision_is_replaced_by_a_live_decider_or_the_rules(monkeypatch):
    ev = {**PERSON, "decision": {"action": "ignore", "reason": "r", "source": "rule-stub"}}
    monkeypatch.setattr(decider, "decide", lambda e, earlier, url=None: _decision("alert", confidence=0.7))
    assert agent.decide_event(ev, [], "http://decider")["confidence"] == 0.7
    monkeypatch.setattr(decider, "decide", lambda *a, **k: None)
    out = agent.decide_event(ev, [])
    assert out["source"] == "rule-stub" and out["action"] == "alert"


@pytest.mark.parametrize("action, tool", [("ignore", None), ("log", "log_event"), ("recheck", "recheck_camera")])
def test_non_alert_events_never_wake_the_model(monkeypatch, action, tool):
    monkeypatch.setattr(decider, "decide", lambda e, earlier, url=None: _decision(action))
    rows = list(agent.run_clip([DARK], NoModel(), agent.PolicyGuard()))
    assert len(rows) == 1 and rows[0]["model_ran"] is False and rows[0]["model_ms"] == 0
    assert [t["tool"] for t in rows[0]["tools"]] == ([tool] if tool else [])


def _call(guard, name="notify_owner"):
    return guard.before_tool_call(SimpleNamespace(tool_use={"name": name, "input": {}}))


def test_guard_lets_notify_through_only_on_alert():
    guard = agent.PolicyGuard()
    guard.decision = _decision("alert")
    assert isinstance(_call(guard), Proceed)
    for action in ("ignore", "log", "recheck"):
        guard.decision = _decision(action)
        out = _call(guard)
        assert isinstance(out, Deny) and action in out.reason
    assert guard.denied == 3
    assert isinstance(_call(guard, "other_tool"), Proceed)
