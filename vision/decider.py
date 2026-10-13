"""Strands Decider 2B as the event brain: one `choice` question per event,
answered by a local `strands-decider serve` endpoint.

Off unless DECIDER_URL is set (e.g. http://127.0.0.1:8000). Any failure
returns None so the caller falls back to the rule stub. On CPU a call takes
~10 s and the first one far longer, so raise DECIDER_TIMEOUT there.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

TIMEOUT_S = float(os.environ.get("DECIDER_TIMEOUT", 10.0))
SOURCE = "strands-decider-2b"

QUESTION = "What should the doorbell do about this event for the home owner?"
# Option name -> description; the descriptions are what the model reads.
CRITERIA = {
    "alert": "Notify the owner now: a person is at the door, or something was left or taken.",
    "ignore": "Do nothing: an animal, a passing car, shadows, or tiny brief motion.",
    "recheck": "The picture is too blurry or dark to trust; check the camera or look again.",
    "log": "Keep a note only: a camera mode change or a cut in the video, nothing at the door.",
}
# Decider writes no text, so the reason shown in the UI comes from the option.
REASONS = {
    "alert": "Needs your attention",
    "ignore": "Not worth a notification",
    "recheck": "Picture not clear enough to trust",
    "log": "Kept for the record only",
}


def _describe(ev: dict) -> str:
    dur = ev["t_end"] - ev["t_start"]
    parts = [f"{ev['type'].replace('_', ' ')} at {ev['t_start']:.1f}s"]
    if dur > 0:
        parts.append(f"lasting {dur:.1f}s")
    if ev.get("label"):
        parts.append(f"detector saw a {ev['label']}")
    if ev.get("peak_area"):
        parts.append(f"covering {ev['peak_area']:.0%} of the frame")
    return ", ".join(parts)


def state_text(ev: dict, earlier: list[dict]) -> str:
    """Plain-text state: the event itself plus the last few events before it,
    so 'person, then object taken' reads as one story."""
    lines = [f"Doorbell camera event: {_describe(ev)}."]
    if ev.get("objects"):
        seen = ", ".join(f"{k} x{v}" for k, v in sorted(ev["objects"].items()))
        lines.append(f"Objects detected: {seen}.")
    if ev.get("recheck") is not None:
        rc = ev["recheck"]
        if rc["usable"]:
            seen = ", ".join(f"{k} x{v}" for k, v in sorted(rc["objects"].items())) or "nothing recognised"
            lines.append(f"Rechecked after enhancing the picture with OpenCV: now usable, detector saw {seen}.")
        else:
            lines.append("Rechecked after enhancing the picture with OpenCV: still too dark or blurry to trust.")
    if earlier:
        lines.append("Earlier in this clip: " + "; ".join(_describe(e) for e in earlier[-3:]) + ".")
    return " ".join(lines)


def request(ev: dict, earlier: list[dict]) -> dict:
    """The /v1/systemone body for one event: the reusable decision pattern."""
    return {
        "state": state_text(ev, earlier),
        "questions": {"action": {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA}},
    }


def ask(body: dict, url: str, timeout: float = TIMEOUT_S) -> dict:
    """POST a raw request; returns the server's JSON. Raises on any failure."""
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def decide(ev: dict, earlier: list[dict], url: str | None = None) -> dict | None:
    url = url or os.environ.get("DECIDER_URL")
    if not url:
        return None
    try:
        ans = ask(request(ev, earlier), url)["answers"]["action"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError):
        return None
    action = ans.get("choice")
    if action not in CRITERIA:
        return None
    return {
        "action": action,
        "reason": REASONS[action],
        "source": SOURCE,
        "confidence": ans.get("confidence"),
        "probabilities": ans.get("probabilities"),
    }
