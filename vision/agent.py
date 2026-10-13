"""Strands agent for doorbell events, decider first.

    python -m vision.agent <clip> [--limit N] [--event ID] [--model llama3.2:3b]

For each event in `<samples>/<clip>.timeline.json` the decision comes first
and cheaply: the timeline's own decision when Decider 2B (or the vision
model) made it, otherwise a live Decider 2B call, otherwise the rule stub.
Only an `alert` wakes the language model (a local Ollama model for now,
Bedrock later), and its one job is to write the notification. ignore, log
and recheck never touch it. A PolicyGuard intervention still sits in front of
notify_owner, so even a confused model cannot notify on a non-alert.
Needs `ollama serve`; DECIDER_URL (with DECIDER_TIMEOUT=300 on CPU) for live
decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from strands import Agent, tool
from strands.interventions import Deny, InterventionHandler, Proceed
from strands.models.ollama import OllamaModel

from . import decider
from .timeline import _decide

SYSTEM = (
    "You are a doorbell camera assistant. The decision to notify the home owner has already been made. "
    "Call notify_owner exactly once with one short, friendly sentence telling the owner what happened, "
    "based on the event description you are given. Then reply with one line saying what you sent."
)

MODEL_FREE = {"ignore", "log", "recheck"}   # handled without a language model


def decide_event(ev: dict, earlier: list[dict], url: str | None = None) -> dict:
    """The event's decision, cheapest source first."""
    stored = ev.get("decision")
    if stored and stored.get("source") != "rule-stub":
        return stored
    return decider.decide(ev, earlier, url) or _decide(ev)


class PolicyGuard(InterventionHandler):
    """Guardrail: notify_owner runs only when the decision is alert. Set
    `decision` before each agent call. Cheap: no model call, just a check."""

    name = "policy-guard"

    def __init__(self) -> None:
        self.decision: dict = {}
        self.denied: int = 0

    def before_tool_call(self, event, **kwargs):
        if event.tool_use["name"] == "notify_owner" and self.decision.get("action") != "alert":
            self.denied += 1
            return Deny(reason=f"The decision for this event is {self.decision.get('action')}, not alert. "
                               "Do not notify the owner.")
        return Proceed()


def make_tools(outbox: list[dict]) -> list:
    @tool
    def notify_owner(message: str) -> str:
        """Send a push notification to the home owner's phone.

        Args:
            message: One short sentence for the home owner.
        """
        outbox.append({"tool": "notify_owner", "text": message})
        return "notification sent"

    return [notify_owner]


def handle_without_model(decision: dict, outbox: list[dict]) -> None:
    """ignore does nothing; log and recheck record the decider's own reason."""
    if decision["action"] == "log":
        outbox.append({"tool": "log_event", "text": decision["reason"]})
    elif decision["action"] == "recheck":
        outbox.append({"tool": "recheck_camera", "text": decision["reason"]})


def run_clip(events: list[dict], model, guard: PolicyGuard, limit: int | None = None,
             only: int | None = None, url: str | None = None):
    """Yield one trace row per event: decision, tools that ran, whether the
    model ran, and how long each part took."""
    for n, ev in enumerate(events[:limit]):
        if only is not None and ev["id"] != only:
            continue
        outbox: list[dict] = []
        t0 = time.time()
        decision = decide_event(ev, events[:n], url)
        decide_ms = round((time.time() - t0) * 1000)
        guard.decision = decision
        reply, model_ms = "", 0
        if decision["action"] in MODEL_FREE:
            handle_without_model(decision, outbox)
        else:
            t1 = time.time()
            agent = Agent(model=model, tools=make_tools(outbox), system_prompt=SYSTEM,
                          interventions=[guard], callback_handler=None)
            reply = str(agent(f"Event: {decider.state_text(ev, events[:n])} "
                              f"Decision: alert, because {decision['reason']}.")).strip()
            model_ms = round((time.time() - t1) * 1000)
        yield {
            "id": ev["id"], "t_start": ev["t_start"], "type": ev["type"], "label": ev.get("label"),
            "decision": decision, "tools": outbox, "reply": reply,
            "model_ran": decision["action"] not in MODEL_FREE, "decide_ms": decide_ms, "model_ms": model_ms,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", help="clip name, e.g. theft_7Z4ExUfkUOw")
    ap.add_argument("--samples", type=Path, default=Path("web/public/samples"))
    ap.add_argument("--limit", type=int, help="only the first N events")
    ap.add_argument("--event", type=int, help="only this event id")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"))
    ap.add_argument("--out", type=Path, help="write the trace as JSON here")
    args = ap.parse_args()

    events = json.loads((args.samples / f"{args.clip}.timeline.json").read_text())["events"]
    model = OllamaModel(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"), model_id=args.model)
    url = os.environ.get("DECIDER_URL")
    guard = PolicyGuard()

    trace = []
    for row in run_clip(events, model, guard, args.limit, args.event, url):
        trace.append(row)
        d = row["decision"]
        conf = f" {d['confidence']:.0%}" if d.get("confidence") is not None else ""
        tools = ", ".join(f"{t['tool']}: {t['text']}" for t in row["tools"]) or "nothing sent"
        print(f"{row['t_start']:6.1f}s {row['type']:<12} {d['action']:<7}{conf} [{d['source']}] "
              f"{'model ' + str(row['model_ms'] // 1000) + 's' if row['model_ran'] else 'no model':<12} -> {tools}", flush=True)
    ran = sum(r["model_ran"] for r in trace)
    print(f"\nlanguage model ran for {ran} of {len(trace)} events; guard denied {guard.denied} notify attempts")
    if args.out:
        args.out.write_text(json.dumps(trace, indent=1))


if __name__ == "__main__":
    main()
