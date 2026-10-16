"""Escalation: when Decider 2B is unsure, show one enhanced frame to a vision
language model and let it decide the same four-way question.

The decider handles every event it is confident about; only low-confidence
events reach the expensive model, which is the point: measured on the Pexels
clips, 2 events in 6 escalate. Off unless VISION_MODEL is set. Two providers:

    VISION_MODEL=qwen2.5vl:3b                      Ollama (VISION_URL, default http://127.0.0.1:11434)
    VISION_MODEL=bedrock:amazon.nova-lite-v1:0     Amazon Bedrock, Converse API (BEDROCK_REGION, default us-east-1;
                                                   credentials from the usual AWS_PROFILE / environment)

Every answer carries `ms` and, for Bedrock, `tokens`, so the report can put a
time and a price on each second look. Any failure returns None so the decider's
own answer stands.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from . import decider
from .enhance import enhance, sample_frames

ESCALATE_BELOW = float(os.environ.get("ESCALATE_BELOW", 0.4))   # decider confidence under this escalates
TIMEOUT_S = float(os.environ.get("VISION_TIMEOUT", 600))         # a 3B vision model on CPU is slow
JPEG_QUALITY = 85
BEDROCK_PREFIX = "bedrock:"
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
BEDROCK_MAX_TOKENS = 200

# The picture comes first and the context stays thin on purpose: a small
# vision model given "detector saw nothing" will repeat it instead of looking.
PROMPT = (
    "You are a doorbell camera assistant looking at one frame from the camera, brightened for you. "
    "When: {state}\n"
    "Step 1: look carefully and say in one sentence what is at the door: any person (how many, what they "
    "are doing), any package, animal or vehicle, or nothing.\n"
    "Step 2: choose exactly one action for the home owner:\n"
    + "\n".join(f"- {k}: {v}" for k, v in decider.CRITERIA.items())
    + '\nReply with JSON only: {{"seen": "<step 1 sentence>", "action": "<alert|ignore|recheck|log>"}}'
)


def context(ev: dict, earlier: list[dict]) -> str:
    """Event timing and what came before, without the detector's verdicts."""
    lines = [decider._describe({**ev, "label": None, "peak_area": None})]
    if earlier:
        lines.append("Earlier in this clip: " + "; ".join(decider._describe(e) for e in earlier[-3:]))
    return ". ".join(lines) + "."


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


def bedrock_client(region: str = BEDROCK_REGION):
    """A bedrock-runtime client; a separate function so tests can swap in a fake."""
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-runtime", region_name=region,
                        config=Config(read_timeout=60, connect_timeout=10, retries={"max_attempts": 2}))


def ask_bedrock(prompt: str, jpeg: bytes, model_id: str, client=None) -> tuple[str, dict]:
    """One Converse call with the image inline. Returns (text, usage) where usage
    holds inputTokens / outputTokens for the cost record."""
    client = client or bedrock_client()
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [
            {"image": {"format": "jpeg", "source": {"bytes": jpeg}}},
            {"text": prompt},
        ]}],
        inferenceConfig={"maxTokens": BEDROCK_MAX_TOKENS, "temperature": 0.0},
    )
    text = "".join(part.get("text", "") for part in resp["output"]["message"]["content"])
    return text, resp.get("usage", {})


def parse_answer(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):  # Nova sometimes fences its JSON
        text = text.strip("`")
        text = text[text.find("{"):text.rfind("}") + 1]
    try:
        ans = json.loads(text)
    except ValueError:
        return None
    action = ans.get("action") if isinstance(ans, dict) else None
    if action not in decider.CRITERIA:
        return None
    reason = str(ans.get("seen") or ans.get("reason") or decider.REASONS[action]).strip()[:200]
    return {"action": action, "reason": reason}


def escalate(video: str | Path, ev: dict, earlier: list[dict], model: str | None = None,
             url: str | None = None) -> dict | None:
    """Ask the vision model about this event. Returns a decision dict
    (action, reason, source, ms, tokens) or None when off or anything fails."""
    model = model or os.environ.get("VISION_MODEL")
    if not model:
        return None
    jpeg = frame_jpeg(video, ev)
    if jpeg is None:
        return None
    prompt = PROMPT.format(state=context(ev, earlier))
    t0 = time.monotonic()
    tokens = None
    try:
        if model.startswith(BEDROCK_PREFIX):
            text, usage = ask_bedrock(prompt, jpeg, model[len(BEDROCK_PREFIX):])
            tokens = {"in": usage.get("inputTokens"), "out": usage.get("outputTokens")}
        else:
            text = ask_ollama(prompt, jpeg, model, url or os.environ.get("VISION_URL", "http://127.0.0.1:11434"))
    except Exception:  # unreachable server, missing credentials, throttling, bad payload: the decider's answer stands
        return None
    ans = parse_answer(text)
    if ans is None:
        return None
    return {**ans, "source": f"vision-llm:{model}", "confidence": None,
            "ms": round((time.monotonic() - t0) * 1000), "tokens": tokens}
