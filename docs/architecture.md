# Architecture

Smart alerts for any camera: OpenCV 5 finds what happened and whether the picture can be trusted,
Strands Decider 2B decides what to do about it, a vision model settles the unsure cases, and a Strands
agent writes the notification only when one is needed.

```mermaid
flowchart TD
    V[Camera clip<br/>any mp4: doorbell, old phone, webcam] --> T[vision/timeline.py<br/>OpenCV 5: motion MOG2, cuts, day/night,<br/>blur and dark checks, YOLOX-S via cv2.dnn,<br/>doorstep before/after = object left or taken]
    T --> E{Events}
    E --> D[Strands Decider 2B<br/>one choice question per event<br/>alert / ignore / recheck / log + confidence]
    D -->|recheck| R[vision/enhance.py<br/>gamma, CLAHE, denoise,<br/>zoomed second look over more frames,<br/>detector again]
    R --> D2[Decider 2B again<br/>on what the better picture shows]
    D -->|confidence < 0.4| X[vision/escalate.py<br/>one enhanced frame to a vision LLM<br/>Ollama now, Bedrock Claude on AWS]
    D2 --> A
    X --> A
    D -->|confident| A{Final decision}
    A -->|alert| G[vision/agent.py Strands agent<br/>PolicyGuard intervention<br/>writes one sentence, notify_owner]
    A -->|ignore| N0[nothing]
    A -->|log| N1[event log]
    A -->|recheck| N2[camera flagged: add a light / clean the lens]
    G --> O[Owner's phone]
    T --> P[vision/playground.py<br/>video, decisions, confidences,<br/>ask the decider or the vision model]
    T --> Q[vision/report.py<br/>eval/report.md: hits, misses, failure cases]
```

## AWS deployment (planned, under $5)

| Part | Where | Why |
|---|---|---|
| OpenCV pipeline (`vision/Dockerfile`) | container on a small EC2 instance | the "meaningful component on AWS"; CPU is enough |
| Strands Decider 2B | g6.xlarge GPU instance for test and demo sessions only | ~10 s per decision on a laptop CPU, ~115 ms on a GPU |
| Vision escalation and the agent's language model | Amazon Bedrock (Claude) | required by the Strands hackathon; pay per call, cents for a test |
| Playground / demo endpoint | same instance, port 8090, scheduled screen-share for judges | OpenCV rules: live endpoint or arranged demo |

## Data

- `clips/real`: 18 labelled doorbell clips (`eval/real_clips.csv`), deliveries and thefts, 4 at night.
- `clips/pexels`: 6 free-licence doorstep clips usable in the submission.
- `clips/blurry`: 5 clips with foggy, wet or smudged lenses.
- `web/public/samples/*.timeline.json`: per-frame measurements, events, every decision with its source.
