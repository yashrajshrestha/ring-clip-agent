"""Strands Decider 2B as the event brain: one `choice` question per event,
answered by a local `strands-decider serve` endpoint.

Off unless DECIDER_URL is set (e.g. http://127.0.0.1:8000). Any failure
returns None so the caller falls back to the rule stub. On CPU a call takes
~10 s and the first one far longer, so raise DECIDER_TIMEOUT there.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

TIMEOUT_S = float(os.environ.get("DECIDER_TIMEOUT", 10.0))
SOURCE = "strands-decider-2b"

QUESTION = "What should the doorbell do about this event for the home owner?"
# Option name -> description; the descriptions are what the model reads.
CRITERIA = {
    "alert": "Notify the owner now: a person is at the door, something was left or taken, an animal has stayed "
             "at the door a long time (it may be trying to get in), or a dangerous wild animal is there.",
    "ignore": "Do nothing: an animal passing by, a passing car, shadows, or tiny brief motion.",
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


# Plain English for each event type. Measured on 70 events: the bare type
# names ("cut at 12.3s", "scene change at 42.8s") left the decider agreeing
# with the rule stub on 30/70 at 47% mean confidence; these words gave 65/70
# at 69%, fixing every alert on a video edit and every package event it had
# downgraded to log.
PLAIN = {
    "motion": "something moved at the door",
    "scene_change": "the doorstep looks different after the visit, so an object was left or taken",
    "cut": "the video cut to a different shot (an edit in the recording, not something at the door)",
    "night_mode": "the camera switched to night mode",
    "day_mode": "the camera switched to day mode",
    "blurry": "the picture was blurry",
    "dark": "the picture was too dark",
}


# What the detector saw, spelled out: "saw a animal" left the decider alerting on dogs (39% sure).
SAW = {
    "person": "the detector saw a person",
    "animal": "the detector saw an animal (a dog, cat or bird), not a person",
    "wild": "the detector saw a wild animal such as a bear, which can be dangerous",
    "vehicle": "the detector saw a vehicle passing, not a person at the door",
    "bag": "the detector saw a bag or parcel",
}


def _seen(objects: dict) -> str:
    """Detector counts in words; when only animals were seen, say outright that it is not a person."""
    if not objects:
        return "nothing recognised"
    text = ", ".join(f"{k} x{v}" for k, v in sorted(objects.items()))
    if objects.get("wild"):
        text += " (a wild animal such as a bear, which can be dangerous)"
    elif objects.get("animal") and not objects.get("person"):
        text += " (an animal such as a dog, not a person)"
    return text


def _describe(ev: dict) -> str:
    dur = ev["t_end"] - ev["t_start"]
    parts = [f"{PLAIN.get(ev['type'], ev['type'].replace('_', ' '))} at {ev['t_start']:.1f}s"]
    if dur > 0:
        parts.append(f"lasting {dur:.1f}s")
    if ev["type"] == "motion":
        if ev.get("label") == "figure":
            parts.append("a person-sized shape moved steadily across the porch, which the detector could not "
                         "identify (likely a person in the dark or partly hidden)")
        elif ev.get("label"):
            parts.append(SAW.get(ev["label"], f"the detector saw a {ev['label']}"))
        if ev.get("lingering"):
            parts.append(f"the animal has stayed at the door for {dur:.0f}s, which could mean it is trying to get in or is stuck")
        else:
            parts.append("the detector saw no person, animal or vehicle")
    if ev.get("peak_area"):
        parts.append(f"covering {ev['peak_area']:.0%} of the frame")
    return ", ".join(parts)


def state_text(ev: dict, earlier: list[dict]) -> str:
    """Plain-text state: the event itself plus the last few events before it,
    so 'person, then object taken' reads as one story."""
    lines = [f"Doorbell camera event: {_describe(ev)}."]
    if ev.get("objects"):
        lines.append(f"Objects detected: {_seen(ev['objects'])}.")
    if ev.get("recheck") is not None:
        rc = ev["recheck"]
        if rc["usable"]:
            lines.append(f"Rechecked after enhancing the picture with OpenCV: now usable, detector saw {_seen(rc['objects'])}.")
        else:
            lines.append("Rechecked after enhancing the picture with OpenCV: still too dark or blurry to trust.")
        if rc.get("second_look"):
            sl = rc["second_look"]
            lines.append(f"Second look, zoomed in over {sl['frames']} more frames: detector saw {_seen(sl['objects'])}.")
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
    started = time.perf_counter()
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
        "ms": round((time.perf_counter() - started) * 1000),   # round trip, so the UI can show how long it took
    }
