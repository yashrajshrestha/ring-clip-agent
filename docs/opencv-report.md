# LookTwice: technical report

*OpenCV AI Competition 2026, Agentic Vision path. Team: Yash Shrestha (solo).*

| | |
|---|---|
| Live demo | http://184.197.167.91/ (AWS Graviton, us-east-1) |
| Code | https://github.com/yashrajshrestha/ring-clip-agent (MIT) |
| Video (3:36) | https://youtu.be/KZdM_F4J7U4 |
| Architecture and agent workflow diagrams | [docs/architecture.md](architecture.md) |
| Evaluation | [eval/benchmark.md](../eval/benchmark.md), [eval/report.md](../eval/report.md) |

LookTwice turns a doorbell or porch clip into a short list of decisions: alert, ignore, log, or "look again".
OpenCV 5 finds what happened and measures whether the picture can be trusted. A small decision model
(Strands Decider 2B) decides each event. When the picture is poor, OpenCV enhances it and looks again; when the
decider is unsure, Amazon Bedrock takes a second look at one frame. A language model writes a message only when
the answer is "alert".

## 1. Problem

Doorbell cameras fail in two ways:

- **Too many alerts.** Animals, cars, shadows, and lighting changes all trigger "motion". People learn to ignore
  the phone, and then miss the alert that mattered.
- **Silent misses when the picture is poor.** A dark porch, a fogged or wet lens, or a parcel taken while the
  detector never named a "person".

Branded cameras sell smarter alerts as a monthly subscription tied to their own hardware. Sending every event to a
large vision model is the obvious alternative, but it is slow, costs money per event, and (as section 7 shows)
misses real visits when it judges from a single frame.

## 2. Users

- Home owners and renters who want fewer, better alerts from the camera they already have.
- Small landlords and shared buildings with several cameras and no budget for per-camera plans.
- Developers who want a worked example of an agent that acts on vision results instead of narrating them.

## 3. Architecture

Full diagrams (pipeline, agent loop, AWS deployment) are in [docs/architecture.md](architecture.md). In short:

1. **Perception (OpenCV 5):** a clip becomes a list of events (motion, person, animal, vehicle, object left or
   taken, dark, blurry), each with quality measurements.
2. **Decision (Strands Decider 2B):** each event becomes a typed question with four answers (`alert`, `ignore`,
   `recheck`, `log`) and a probability for each.
3. **Action:**
   - `recheck` sends the event back to OpenCV to enhance the frames and look again, then asks the decider again.
   - Confidence under 0.4 sends one enhanced frame to Amazon Bedrock (Nova Lite) for a second look.
   - `alert` wakes a Strands agent that writes one sentence (Bedrock Nova Micro).
   - `ignore` and `log` cost nothing more.
4. **Interface:** the LookTwice page plays the clip, lists every event with its decision, source, confidence and
   time, and lets the visitor re-run the clip, look harder, ask the decider again, ask the vision model, or drop
   in a clip of their own.

## 4. OpenCV 5 implementation

All vision work uses OpenCV 5 (`opencv-python-headless==5.0.0.93`) on CPU, on a laptop and on AWS Graviton.

| Task | OpenCV 5 | Notes |
|---|---|---|
| Motion | `createBackgroundSubtractorMOG2`, morphology, `findContours` | per-frame boxes with on/off hysteresis; background reset on a cut or lighting jump |
| Object labels | `cv2.dnn.readNet` + YOLOX-S ONNX (OpenCV Model Zoo, hash-checked), own letterbox, `NMSBoxesBatched` | runs 2× a second only while something moves; person / animal / vehicle / bag groups |
| Tracking | box centers across frames: net travel vs path length | catches a person-sized shape moving steadily that the detector could not name (dark, partly hidden) |
| Picture quality | Laplacian variance (blur), mean gray (dark), HSV saturation share (infrared night mode) | measured on background pixels so a dark jacket does not flip night mode |
| Video edits | 16×9 color cells vs half a second ago, all four quadrants must change | separates a video cut from a person filling the frame |
| Edited stills | share of pixels unchanged frame to frame | stops title and end cards reading as a blurry camera |
| Object left / taken | doorstep before vs after a visit, masked for the moving region | COCO has no `box` class, so this is the package signal |
| Recheck enhancement | gamma LUT, CLAHE on L (LAB), bilateral filter, zoomed crops, detector again | turns a `recheck` decision into a second look |

Thresholds were tuned on the labelled clips; the measured values are recorded next to each constant in the code.

### Making the OpenCV pass fast on Arm

The server is an Arm (Graviton2) machine, so the OpenCV pass was tuned for it, without changing any result:

| Change | What it does |
|---|---|
| Frames read ahead | a second thread decodes and resizes the next frames while the loop checks this one (`_prefetch`); OpenCV releases the GIL, so the two really overlap |
| Detector alongside | YOLOX runs on its own thread, one forward pass at a time; its answer fills the frames that share it |
| Classic dnn engine | YOLOX loads on OpenCV 5's classic engine (`DETECTOR_ENGINE=classic`, now the default): 1.97× faster than the new engine on Graviton2 |
| Counting in OpenCV | masks and counts use `compare`, `countNonZero` and `mean` with a mask instead of numpy boolean indexing; grayscale once per frame |

Measured on the demo server itself (t4g.xlarge, at first boot, `vision/coolbench.py`, results in
[eval/speed-graviton.md](../eval/speed-graviton.md)): the whole OpenCV pass over the 10 published clips went from
**177.1 s to 113.0 s (1.57×)**, with the same number of events in every clip. On the M2 Pro, where the full event
lists were compared, all 10 clips produce identical events; one frame's brightness rounds differently in the last
decimal (74.6 vs 74.7). `PIPELINE_PARALLEL=0 DETECTOR_ENGINE=new` restores the old path for comparison.
`DETECTOR_FP16=1` is an opt-in half-precision setting: on the M2 Pro it changed one borderline detection in 40
frames, and on Graviton2 OpenCV reports falling back to FP32.

## 5. The agentic loop

The Agentic Vision rules ask that vision results change what the system does next. In LookTwice they do, in four
places:

1. **Poor picture → recheck → enhance → decide again.** OpenCV marks a frame too dark or blurry, the decider
   answers `recheck`, OpenCV enhances and re-measures the frames and runs the detector again, and the decider
   decides a second time on what the better picture shows.
2. **Nothing found → second look.** If the first look finds nothing, a second look samples more frames and five
   zoomed crops of each, so a small or distant figure reaches the detector larger.
3. **Unsure → Bedrock.** Decider confidence under 0.4 sends one enhanced frame to Nova Lite with the same four
   choices.
4. **Only `alert` reaches the language model.** A `PolicyGuard` intervention denies the notify tool on any other
   decision, so a confused model cannot send a message.

**Safety rule:** the Bedrock second look may *raise* an alert or calm a non-person event, but it may never
*silence* an alert about a person (`vision/escalate.py`, `may_override`). Single-frame looks miss people who
have just moved out of the frame (section 7), so the decider's alert stands and Bedrock's answer is kept next to
it for the owner to see.

### A real trace

From the published clip `pexels_dog_doorway_5394386` (a dog in a dark doorway), stored in
`web/public/samples/pexels_dog_doorway_5394386.timeline.json`:

| Step | Who | What happened |
|---|---|---|
| 1 | OpenCV | event 0 at 4.3 s: "Image is too dark", brightness 80.7, blur score 1815 |
| 2 | Decider 2B | `recheck`, 57% (alert 5%, ignore 24%, log 13%), 494 ms |
| 3 | OpenCV recheck | 8 frames enhanced: brightness 80.7 → 103.0, blur score 1815 → 4455; usable; detector finds an animal in 8 of 8 frames |
| 4 | Decider 2B again | `ignore`, but only 33% sure, 478 ms |
| 5 | Bedrock Nova Lite | under 40%, so one enhanced frame goes to Bedrock: `ignore`, "A dog is standing at the door." 1.96 s, 960 tokens in, 19 out |
| 6 | Result | no notification, no language-model message; every step is kept in the event record |

The opposite case, where the safety rule matters: in `ring_vS55Kxp-rZQ` and `ring_RM8AxCNwzcQ` the decider said
`alert` for a person, Bedrock's single frame said "Nothing is at the door", and the alert stood, with
Bedrock's answer recorded as `vision_decision`.

Every event records `first_decision`, `recheck` (before and after measurements, what was seen),
`decider_decision` when escalated, `vision_decision` when a second look was overruled, and the final `decision`
with its `source`, `confidence`, `ms`, and (for Bedrock) `tokens`.

## 6. AWS deployment

The live demo runs on AWS, built by Terraform in `deploy/terraform` (one `terraform apply`, one
`terraform destroy`).

| Part | AWS | Details |
|---|---|---|
| App server | EC2 **t4g.xlarge (Graviton, Arm64)**, Ubuntu 24.04, 40 GB encrypted gp3, Elastic IP | OpenCV 5 pipeline, Decider 2B on CPU, the LookTwice page; systemd services under a non-root user |
| Second look | **Amazon Bedrock**, Nova Lite (vision) | about $0.00007 per look |
| Alert message | **Amazon Bedrock**, Nova Micro | one sentence per alert |
| Published clips and results | **S3**, private, encrypted | only the clips listed in `deploy/published_clips.txt` are ever uploaded |
| Logs and metrics | **CloudWatch** log group `/ring-clip-agent` (14 days) + CloudWatch agent | streams for setup, decider, page, and nginx access; memory and CPU metrics |
| Admin access | **Systems Manager** Session Manager | no SSH port, no keys on the server |
| Permissions | **IAM** instance role | Bedrock Nova models only, read its own bucket, write logs |

The server sets itself up on first boot: it installs Python, OpenCV 5, CPU PyTorch and the decider, downloads the
YOLOX model (hash-checked) and the decider weights, copies the published clips from S3, and starts the services
behind nginx.

**Hardening on a public page:**
- nginx accepts only GET, HEAD, POST, and PUT (PUT only on the upload path).
- Request bodies are capped at 64 KB, uploads at 25 MB.
- Live model calls are limited to 12 per minute per visitor (burst 6).
- Uploads are capped at 20 MB and 3 per visitor per hour, and an hourly job removes every upload and restores the
  published results.
- IMDSv2 only.

**Measured:**
- The OpenCV pass over the 10 published clips: 177.1 s before the Arm tuning, 113.0 s after (1.57×), timed at
  first boot and logged to CloudWatch (section 4).
- A live Decider 2B call takes about 7 to 8 s on the Graviton CPU (0.5 s on an Apple M2 Pro), using about 5.5 GB
  of memory.
- Running cost is about $3.30 a day, mostly the instance; about $0.23 a day when stopped between judging
  sessions.

The Best Use of COOL Award is not claimed: the server runs the standard OpenCV 5 wheel (with KleidiCV) on
Graviton. The Terraform can switch to COOL's Graviton4 image (`ami_id`, `use_cool`); the boot benchmark then also
compares the pip wheel with COOL on the same machine.

**Reproduce:**

    cd deploy/terraform
    terraform init && terraform apply     # prints the page URL; the page answers once setup finishes
    terraform destroy                     # removes everything

## 7. Evaluation

### 7.1 Four ways to decide the same events (shareable clips)

Ten clips anyone can download: 8 Ring doorbell clips published under Creative Commons Attribution
(`eval/cc_clips.csv`) and 2 Pexels dog clips. Five are thefts, two are visits, and three show only animals.
OpenCV ran once, so every method decided the same 35 events. The script is `vision/benchmark.py` and the output
is `eval/benchmark.md`. Decider 2B ran on an Apple M2 Pro; Bedrock ran in us-east-1.

| Method | Visit clips alerted | Pet clips kept quiet | Phone alerts | Bedrock calls | Bedrock cost | Decision time per clip |
|---|---|---|---|---|---|---|
| Hand-written rules | 7 / 7 | 0 / 3 | 20 | 0 | $0 | 0.0 s |
| Decider 2B | 7 / 7 | 2 / 3 | 15 | 0 | $0 | 2.5 s |
| **Decider 2B + Bedrock when unsure (LookTwice)** | **7 / 7** | **3 / 3** | **14** | 9 | $0.00059 | 4.3 s |
| Bedrock on every event | 4 / 7 | 3 / 3 | 4 | 35 | $0.00230 | 13.8 s |

What this shows:
- LookTwice kept every real visit, silenced every pet-only clip, and sent 30% fewer alerts than the rules.
- It asked Bedrock about 9 of 35 events.
- Sending everything to Bedrock was quiet but missed 3 of 7 real visits, because one frame often catches the
  moment after the person has left.
- Before the safety rule in section 5, the combined method also lost person alerts this way. With it, 7 of 7
  visits alert.

### 7.2 Labelled doorbell clips (development set)

18 public doorbell clips (10 deliveries, 8 thefts, 4 at night), labelled in `eval/real_clips.csv`. These are
not redistributed, only listed. Scoring is `python -m vision.report`, which writes `eval/report.md`.

| Measure | Result |
|---|---|
| Clips with at least one alert | 18 / 18 |
| Object left or taken detected | 11 / 18 (3 / 4 at night) |
| Events decided by Decider 2B | 131 / 132 (1 fell back to the rules) |
| Events that would wake a language model (alerts) | 90 / 132 |

During development, measured on the same clips and on stock clips:
- Tracking person-sized movement the detector could not name added 5 alerts, all real visits, no false alarms;
  it recovered the late-night delivery.
- The edited-still check cut false "blurry" events from title cards from 6 to 0.
- The recheck rescued 7 of 8 blurry moments on stock clips; a person was confirmed each time.

### 7.3 Failure cases

- **Package events missed** on 7 of 18 labelled clips: small parcels, night footage, one compilation with three
  visits. Two of them are "thief stopped" clips where nothing may have been taken.
- **Single-frame second looks miss things.** On the bear clip (`ring_k-WvGk7a9Ms`) a live Bedrock look said
  "nothing at the door". In another run Nova Lite called a cat a dog. This is why Bedrock is a second opinion
  that cannot silence a person alert.
- **One theft clip lost a package alert to Bedrock:** in `ring_vS55Kxp-rZQ` the "object left or removed" event
  went from alert to ignore. The safety rule protects person alerts only, and the person alert in that clip
  stood.
- **Decider confidence is often modest.** On the 35 benchmark events the median is 52%, and 21 of 35 sit between
  30% and 60%, which is why the escalation line exists.

## 8. Limitations

- Package detection depends on a before/after comparison; a detector with a parcel class is the next step.
- Decider 2B on CPU takes about 7 to 8 s per decision on Graviton. That is fine for a drop box, not for a live
  feed without a GPU.
- The system assumes a doorstep view. A clip of something else (a person holding an object close to the camera)
  is still described in doorstep terms.
- Most test clips are edited videos. Cuts and title cards are artifacts a real camera never produces; the
  pipeline detects and skips them.
- In true darkness without infrared nothing can be rescued, and the recheck says so ("add a light") rather than
  guessing.

## 9. Responsible use

- **No identification.** The system watches a doorstep the owner already records. It stores decisions and
  measurements, not faces, and it never identifies people.
- **Uncertainty is visible.** Every decision shows its source, confidence, and time. A poor picture ends in
  `recheck` with a hint to the owner, not a guess.
- **Human control.** The owner decides what to do with an alert. Bedrock can never silence a person alert, and
  the page lets anyone ask the decider or the vision model again and compare.
- **Licensed media.** The published clips are Creative Commons Attribution (Ring LLC's channel, credited on the
  page) and the Pexels license. The 18 development clips are listed, not redistributed. Visitor uploads are
  removed every hour.
- **Not affiliated.** LookTwice is an independent project, not affiliated with or endorsed by Ring LLC, and the
  page says so.

## 10. Related work

| Product | What it does | How LookTwice differs |
|---|---|---|
| Ring and Nest smart alerts | person, package, and animal alerts on the vendor's own cameras, behind a subscription | works on any clip; decisions and confidence are visible |
| Frigate (open-source video recorder) | object detection with zones, on local hardware | adds a decision step, picture-quality checks, and "look again" when the picture is poor |
| Home Assistant LLM Vision | sends camera snapshots to a vision language model | asks the large model only when a small decider is unsure (about 1 in 4 events here), and never lets one frame silence a person |

## 11. Build, test, and run

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements-dev.txt          # pinned: OpenCV 5.0.0.93, numpy, boto3, pytest
    pytest                                       # 117 passed, 1 skipped
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements-decider.txt -r requirements-agent.txt
    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu
    VISION_MODEL=bedrock:amazon.nova-lite-v1:0 DECIDER_URL=http://127.0.0.1:8000 DECIDER_TIMEOUT=300 \
      python -m vision.playground                # http://127.0.0.1:8090

To deploy on AWS, see section 6.
