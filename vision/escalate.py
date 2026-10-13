"""Escalation: when Decider 2B is unsure, show one enhanced frame to a vision
language model and let it decide the same four-way question.

The decider handles every event it is confident about; only low-confidence
events reach the expensive model, which is the point: measured on the Pexels
clips, 2 events in 6 escalate. Off unless VISION_MODEL is set. Provider is
Ollama (VISION_URL, default http://127.0.0.1:11434); Bedrock plugs in here
later. Any failure returns None so the decider's own answer stands.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from . import decider
from .enhance import enhance, sample_frames

ESCALATE_BELOW = float(os.environ.get("ESCALATE_BELOW", 0.4))   # decider confidence under this escalates
TIMEOUT_S = float(os.environ.get("VISION_TIMEOUT", 600))         # a 3B vision model on CPU is slow
JPEG_QUALITY = 85

PROMPT = (
    "You are a doorbell camera assistant. This is one frame from the camera, brightened for you. "
    "Context from the video analysis: {state}\n"
    "Decide what the doorbell should do for the home owner. Choose exactly one action:\n"
    + "\n".join(f"- {k}: {v}" for k, v in decider.CRITERIA.items())
    + '\nReply with JSON only: {{"action": "<alert|ignore|recheck|log>", "reason": "<one short sentence saying what you see>"}}'
)


def should_escalate(decision: dict, threshold: float = ESCALATE_BELOW) -> bool:
    """Only a real decider answer with low confidence escalates; the rule stub
    has no confidence and never does."""
    conf = decision.get("confidence")
    return conf is not None and conf < threshold


def frame_jpeg(video: str | Path, ev: dict) -> bytes | None:
    """One enhanced frame from the middle of the event, JPEG-encoded."""
    t = (ev["t_start"] + ev["t_end"]) / 2
    frames = sample_frames(video, t, t, n=1)
    if not frames:
        return None
    ok, buf = cv2.imencode(".jpg", enhance(frames[0]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def ask_ollama(prompt: str, jpeg: bytes, model: str, url: str, timeout: float = TIMEOUT_S) -> str:
    body = json.dumps({
        "model": model, "stream": False, "format": "json",
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(jpeg).decode()]}],
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["message"]["content"]


def parse_answer(text: str) -> dict | None:
    try:
        ans = json.loads(text)
    except ValueError:
        return None
    action = ans.get("action") if isinstance(ans, dict) else None
    if action not in decider.CRITERIA:
        return None
    reason = str(ans.get("reason") or decider.REASONS[action]).strip()[:200]
    return {"action": action, "reason": reason}


def escalate(video: str | Path, ev: dict, earlier: list[dict], model: str | None = None,
             url: str | None = None) -> dict | None:
    """Ask the vision model about this event. Returns a decision dict
    (action, reason, source) or None when off or anything fails."""
    model = model or os.environ.get("VISION_MODEL")
    if not model:
        return None
    url = url or os.environ.get("VISION_URL", "http://127.0.0.1:11434")
    jpeg = frame_jpeg(video, ev)
    if jpeg is None:
        return None
    prompt = PROMPT.format(state=decider.state_text(ev, earlier))
    try:
        text = ask_ollama(prompt, jpeg, model, url)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError):
        return None
    ans = parse_answer(text)
    if ans is None:
        return None
    return {**ans, "source": f"vision-llm:{model}", "confidence": None}
