# ring-clip-agent

OpenCV 5 clip quality and event checks for Ring-style doorbell video, feeding a Bedrock agent on AWS.

## Run

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    pytest
    python -m vision.cli path/to/clips > facts.jsonl

## Labels

`eval/labels.csv` columns: `clip, quality (good|bad), dup_group (int), event (person|package|pet|vehicle|none), night (0|1), notes`.
