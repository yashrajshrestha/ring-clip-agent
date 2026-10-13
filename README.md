# ring-clip-agent

OpenCV 5 clip quality and event checks for Ring-style doorbell video, feeding a Bedrock agent on AWS.

## Run

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    pytest
    python -m vision.cli path/to/clips > facts.jsonl

## Decider playground

Ask Strands Decider 2B any typed question (`choice`, `noul` or `score`), or replay a real event
from `web/public/samples/*.timeline.json` and compare its choice with the rule stub.

    pip install -r requirements-decider.txt
    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu
    python -m vision.playground        # http://127.0.0.1:8090

`DECIDER_URL` points both the playground and the pipeline at the server (default
`http://127.0.0.1:8000`). On CPU a call takes about 10 s and the first one over a minute, so set
`DECIDER_TIMEOUT=300` before running the pipeline, or it falls back to the rules.

## Recheck loop

When a blurry or dark event gets a `recheck` decision, `vision/enhance.py` improves the frames with
OpenCV (gamma, CLAHE, bilateral denoise), measures them again and runs the detector on the enhanced
picture. The decider then decides a second time on what the better picture shows. The event keeps
`first_decision`, `recheck` (quality before and after, what was seen) and the final `decision`:

If the first look sees nothing, a second look samples more frames and cuts each into five zoomed
crops, so a small or distant figure reaches the detector larger.

| Enhanced picture | Decision |
|---|---|
| person seen, first or second look | alert |
| usable, nothing recognised | log |
| still too dark or blurry | recheck, with a hint: add a light, or clean the lens |

A truly black frame cannot be rescued; the recheck says so instead of guessing.

## Escalation to a vision model

When Decider 2B answers with confidence under `ESCALATE_BELOW` (default 0.4), one enhanced frame from
the event goes to a vision language model with the same four choices, and its answer replaces the
decider's (`vision/escalate.py`). The event keeps `decider_decision` so both are visible. Off unless
`VISION_MODEL` is set; the provider is Ollama for now, Bedrock later:

    ollama pull qwen2.5vl:3b
    VISION_MODEL=qwen2.5vl:3b DECIDER_URL=http://127.0.0.1:8000 DECIDER_TIMEOUT=300 \
      python -m vision.timeline clips/pexels web/public/samples

On the Pexels clips 2 events in 6 escalate: packages left with nobody there, and a child in a dark
costume. The playground has an "Ask the vision model" button on every event.

## Strands agent

`vision/agent.py` is a Strands agent that writes the notification for each event, decider first:
the decision is made before any language model runs (the timeline's Decider 2B or vision-model
decision, else a live Decider 2B call, else the rule stub). Only `alert` wakes the model, whose one
job is to write the message. `ignore`, `log` and `recheck` never touch it, and a `PolicyGuard`
intervention denies `notify_owner` on anything but an alert, so a confused model cannot notify.

| Decision | Language model | Effect |
|---|---|---|
| alert | runs, writes one sentence | notification sent (guard: Proceed) |
| ignore | skipped | nothing |
| log | skipped | note with the decider's reason |
| recheck | skipped | camera flagged with the decider's hint |

The run prints how many events woke the model, the measured saving the Strands hackathon asks for.

It runs on a local Ollama model for now (Bedrock comes later):

    pip install -r requirements-agent.txt
    ollama pull llama3.2:3b
    DECIDER_URL=http://127.0.0.1:8000 DECIDER_TIMEOUT=300 \
      python -m vision.agent theft_7Z4ExUfkUOw --limit 3

Without `DECIDER_URL` the rule stub decides instead, labelled `rule-stub`.

## Labels

`eval/labels.csv` columns: `clip, quality (good|bad), dup_group (int), event (person|package|pet|vehicle|none), night (0|1), notes`.
