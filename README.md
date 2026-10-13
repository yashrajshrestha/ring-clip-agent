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

## Strands agent

`vision/agent.py` is a Strands agent that drafts a notification for each event. Before its
`notify_owner` tool runs, the `DeciderGate` intervention asks Decider 2B what to do:

| Decider | Intervention | Effect |
|---|---|---|
| alert | Proceed | notification sent |
| ignore | Deny | nothing sent |
| recheck | Guide | agent calls `recheck_camera` instead |
| log | Guide | agent calls `log_event` instead |

It runs on a local Ollama model for now (Bedrock comes later):

    pip install -r requirements-agent.txt
    ollama pull llama3.2:3b
    DECIDER_URL=http://127.0.0.1:8000 DECIDER_TIMEOUT=300 \
      python -m vision.agent theft_7Z4ExUfkUOw --limit 3

Without `DECIDER_URL` the rule stub decides instead, labelled `rule-stub`.

## Labels

`eval/labels.csv` columns: `clip, quality (good|bad), dup_group (int), event (person|package|pet|vehicle|none), night (0|1), notes`.
