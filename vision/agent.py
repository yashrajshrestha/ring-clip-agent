"""Strands agent for doorbell events, with Strands Decider 2B gating its
notifications.

    python -m vision.agent <clip> [--limit N] [--model llama3.2:3b]

For each event in `<samples>/<clip>.timeline.json` the agent (a local Ollama
model for now, Bedrock later) drafts a notification for the home owner. Before
`notify_owner` runs, DeciderGate asks Decider 2B what to do with the event:
alert lets it through, ignore blocks it, recheck and log redirect the agent to
another tool. Needs `ollama serve` and, for real decisions, DECIDER_URL
(with DECIDER_TIMEOUT=300 on CPU); without the decider the rule stub
decides, labelled as such.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from strands import Agent, tool
from strands.interventions import Deny, Guide, InterventionHandler, Proceed
from strands.models.ollama import OllamaModel

from . import decider
from .timeline import _decide

SYSTEM = (
    "You are a doorbell camera assistant. For each camera event you are given, call notify_owner "
    "with one short, friendly sentence telling the home owner what happened. If a tool call is "
    "refused, follow the reason it gives and do not retry the refused call. Finish with one line "
    "saying what you did."
)


def gate_action(d: dict):
    """Decision -> intervention. This mapping plus decider.request() is the reusable pattern."""
    conf = f"{d['confidence']:.0%}" if d.get("confidence") is not None else "rule"
    why = f"Decider chose {d['action']} ({conf}): {d['reason']}."
    if d["action"] == "alert":
        return Proceed(reason=why)
    if d["action"] == "ignore":
        return Deny(reason=f"{why} Do not notify the owner.")
    if d["action"] == "recheck":
        return Guide(feedback=f"{why} Call recheck_camera instead of notifying the owner.")
    return Guide(feedback=f"{why} Call log_event instead of notifying the owner.")


class DeciderGate(InterventionHandler):
    """Asks Decider 2B before every notify_owner call. Set `event` and
    `earlier` before each agent call; every decision is kept in `decisions`."""

    name = "decider-gate"

    def __init__(self, url: str | None = None) -> None:
        self.url = url
        self.event: dict = {}
        self.earlier: list[dict] = []
        self.decisions: list[dict] = []
        self._decided: dict | None = None

    async def before_tool_call(self, event, **kwargs):
        if event.tool_use["name"] != "notify_owner":
            return Proceed()
        if self._decided is not self.event:  # a retried notify reuses the answer
            # The decider call blocks for seconds on CPU; keep it off the agent's event loop.
            d = await asyncio.to_thread(decider.decide, self.event, self.earlier, self.url)
            self.decisions.append(d or _decide(self.event))
            self._decided = self.event
        return gate_action(self.decisions[-1])


def make_tools(outbox: list[dict]) -> list:
    @tool
    def notify_owner(message: str) -> str:
        """Send a push notification to the home owner's phone.

        Args:
            message: One short sentence for the home owner.
        """
        outbox.append({"tool": "notify_owner", "text": message})
        return "notification sent"

    @tool
    def log_event(note: str) -> str:
        """Write a note to the event log without notifying anyone.

        Args:
            note: What happened, in a few words.
        """
        outbox.append({"tool": "log_event", "text": note})
        return "logged"

    @tool
    def recheck_camera(problem: str) -> str:
        """Flag the camera for a re-check because the picture cannot be trusted.

        Args:
            problem: Why the picture is not good enough, in a few words.
        """
        outbox.append({"tool": "recheck_camera", "text": problem})
        return "camera flagged for re-check"

    return [notify_owner, log_event, recheck_camera]


def run_clip(events: list[dict], model, gate: DeciderGate, limit: int | None = None,
             only: int | None = None):
    """Yield one trace row per event: the decision, the tools that ran, the reply.
    `only` runs a single event id, still with the events before it as context."""
    for n, ev in enumerate(events[:limit]):
        if only is not None and ev["id"] != only:
            continue
        outbox: list[dict] = []
        gate.event, gate.earlier = ev, events[:n]
        before = len(gate.decisions)
        agent = Agent(model=model, tools=make_tools(outbox), system_prompt=SYSTEM,
                      interventions=[gate], callback_handler=None)
        reply = str(agent(f"Camera event: {decider.state_text(ev, events[:n])}")).strip()
        yield {
            "id": ev["id"], "t_start": ev["t_start"], "type": ev["type"], "label": ev.get("label"),
            "decision": gate.decisions[before] if len(gate.decisions) > before else None,
            "tools": outbox, "reply": reply,
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
    gate = DeciderGate(os.environ.get("DECIDER_URL"))
    if not gate.url:
        print("DECIDER_URL not set: the rule stub decides (source=rule-stub).", file=sys.stderr)

    trace = []
    for row in run_clip(events, model, gate, args.limit, args.event):
        trace.append(row)
        d = row["decision"]
        gate_txt = (f"{d['action']} {d['confidence']:.0%} [{d['source']}]" if d and d.get("confidence") is not None
                    else f"{d['action']} [{d['source']}]" if d else "not asked")
        tools = ", ".join(f"{t['tool']}: {t['text']}" for t in row["tools"]) or "no tool ran"
        print(f"{row['t_start']:6.1f}s {row['type']:<12} decider: {gate_txt:<32} -> {tools}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(trace, indent=1))


if __name__ == "__main__":
    main()
