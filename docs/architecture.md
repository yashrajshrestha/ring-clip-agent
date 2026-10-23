# Architecture

LookTwice: OpenCV 5 finds what happened and whether the picture can be trusted, Strands Decider 2B decides what
to do about it, OpenCV looks again when the picture is poor, Amazon Bedrock settles the unsure cases, and a
Strands agent writes the notification only when one is needed.

## Agent workflow: perception, decision, action

```mermaid
flowchart TD
    V[Camera clip<br/>any mp4] --> T

    subgraph P[Perception: OpenCV 5]
        T[vision/timeline.py<br/>MOG2 motion, cuts, day/night,<br/>blur and dark checks, YOLOX-S via cv2.dnn,<br/>tracking, doorstep before/after = object left or taken]
        R[vision/enhance.py<br/>gamma, CLAHE, bilateral denoise,<br/>zoomed second look over more frames,<br/>detector again]
    end

    subgraph D[Decision]
        D1[Strands Decider 2B<br/>one typed choice per event<br/>alert / ignore / recheck / log + probabilities]
        D2[Decider 2B again<br/>on what the better picture shows]
        X[vision/escalate.py<br/>one enhanced frame to Bedrock Nova Lite<br/>may raise an alert, never silence a person]
    end

    subgraph A[Action]
        G[vision/agent.py Strands agent<br/>PolicyGuard intervention<br/>Nova Micro writes one sentence, notify_owner]
        N0[nothing]
        N1[event log]
        N2[camera flagged:<br/>add a light / clean the lens]
    end

    T --> D1
    D1 -->|recheck| R
    R --> D2
    D1 -->|confidence < 0.4| X
    D2 -->|confidence < 0.4| X
    D1 -->|confident| F{Final decision}
    D2 -->|confident| F
    X --> F
    F -->|alert| G
    F -->|ignore| N0
    F -->|log| N1
    F -->|recheck| N2
    G --> O[Owner's phone]
    T --> W[LookTwice page vision/playground.py<br/>video, decisions, confidence, source, time;<br/>re-run, look harder, ask again, upload]
```

Every step is written to the clip's `*.timeline.json`: `first_decision`, `recheck` (before and after
measurements), `decider_decision` when escalated, `vision_decision` when a second look was overruled, and the
final `decision` with `source`, `confidence`, `ms`, and Bedrock `tokens`.

## AWS deployment

Built by Terraform (`deploy/terraform`): `terraform apply` builds it, `terraform destroy` removes it.

```mermaid
flowchart LR
    U[Visitor's browser] -->|HTTP :80| NG

    subgraph EC2[EC2 t4g.xlarge, Graviton Arm64, Ubuntu 24.04]
        NG[nginx<br/>method allowlist, body caps,<br/>12 live calls/min per visitor]
        PG[LookTwice page + pipeline<br/>OpenCV 5, YOLOX-S<br/>uploads: 20 MB, 3/hour, removed hourly]
        DC[Strands Decider 2B<br/>CPU, localhost:8000]
        CW[CloudWatch agent]
        NG --> PG
        PG --> DC
    end

    PG -->|Converse API| BL[Amazon Bedrock<br/>Nova Lite: second look<br/>Nova Micro: alert text]
    S3[(S3, private, encrypted<br/>published clips + results)] -->|first boot and hourly restore| PG
    CW --> LG[CloudWatch logs and metrics<br/>/ring-clip-agent]
    SSM[Systems Manager<br/>Session Manager] -.->|admin, no SSH| EC2
    IAM[IAM role: Bedrock Nova only,<br/>own bucket, logs] -.-> EC2
```

| Part | Where | Why |
|---|---|---|
| OpenCV 5 pipeline and the LookTwice page | EC2 t4g.xlarge (Graviton) | the vision work runs on Arm CPU; one instance keeps the demo simple |
| Strands Decider 2B | same instance, CPU | about 7 to 8 s a decision, about 5.5 GB of memory; no GPU needed for a drop box |
| Second look and alert text | Amazon Bedrock, Nova Lite and Nova Micro | pay per call, about $0.00007 each; called for about 1 event in 4 |
| Published clips | S3 | only the clips in `deploy/published_clips.txt` |
| Logs and metrics | CloudWatch | setup, decider, page, and nginx streams; memory and CPU |
| Admin | Systems Manager | no SSH port and no keys on the server |

## Data

- `eval/cc_clips.csv` + 2 Pexels clips: the 10 published, shareable clips used for the benchmark and the demo.
- `eval/real_clips.csv`: 18 labelled public doorbell clips used during development (listed, not redistributed).
- `web/public/samples/*.timeline.json`: per-frame measurements, events, and every decision with its source.
