# Technical report: smart alerts for any camera, that know when the picture can't be trusted

*Draft for the OpenCV AI Competition 2026 (Agentic Vision path). Repo: github.com/yashrajshrestha/ring-clip-agent.*

## 1. Problem

Doorbell and porch cameras produce two failures: too many alerts (animals, cars, shadows, video artefacts) and
silent misses when the picture is poor (a dark porch, a fogged or smudged lens, a parcel taken while no
"person" was detected). Branded cameras sell AI alerts for $18–20 a month, tied to their own hardware. People
with an old phone, a webcam or a no-name camera get nothing but raw motion alerts.

## 2. Users

Home owners and renters who want fewer, better alerts from whatever camera they already have, including an
old phone on a windowsill; small landlords and shared buildings; anyone who cannot or will not pay a
per-camera subscription. Secondary: developers who want a worked example of an agent that acts on vision
results instead of narrating them.

## 3. Architecture

See `docs/architecture.md` for the diagram. In one line: OpenCV 5 turns a clip into events and measures
whether each frame can be trusted; Strands Decider 2B decides what to do about each event; a `recheck`
decision makes OpenCV enhance the frames and look again; a low-confidence decision routes one enhanced frame
to a vision language model; a Strands agent writes the notification only when the decision is `alert`.
Everything runs from one command (`python -m vision.playground`) with a drop box: put a clip in a folder or
on the page and it comes out decided.

## 4. OpenCV 5 implementation

All vision work is OpenCV 5 (`opencv-python-headless==5.0.0.93`), on CPU:

| Task | OpenCV 5 | Notes |
|---|---|---|
| Motion | `createBackgroundSubtractorMOG2`, morphology, `findContours` | per-frame boxes with on/off hysteresis; background model reset on a cut or lighting jump |
| Object labels | `cv2.dnn.readNet` + YOLOX-S ONNX (OpenCV Model Zoo), own letterbox and NMS (`NMSBoxesBatched`) | runs 2× a second only while something moves; person / animal / vehicle / bag groups |
| Picture quality | Laplacian variance (blur), mean gray (dark), HSV saturation share (IR night mode) | judged on background pixels so a jacket does not flip night mode |
| Edits | 16×9 colour cells vs half a second ago, all four quadrants must change | separates a video cut from a person filling the frame |
| Edited stills | share of pixels unchanged frame to frame | a title card after a cut is 100% still; live footage flickers (≤99%); stops end cards reading as a blurry camera |
| Object left/taken | doorstep before vs after a visit, masked for the moving region | no `box` class in COCO, so this is the only package signal |
| Recheck enhancement | gamma LUT, CLAHE on L (LAB), bilateral filter, zoomed crops, detector again | turns a `recheck` decision into a second look |

Thresholds were tuned on the labelled clips; measured values are recorded next to each constant in the code.

## 5. The agentic loop (Agentic Vision path)

Vision results change what the system does next:

1. OpenCV flags a frame as too dark → Decider 2B says `recheck` → OpenCV enhances and zooms → the detector
   finds the courier → Decider 2B, told what the better picture shows, says `alert`. (Measured on a real
   delivery clip darkened to brightness 11/255: usable at 93, person in 7 of 8 frames.)
2. The first look sees nothing → a second look samples 12 more frames and 5 zoomed crops each → two children
   the detector missed at full size are found.
3. Decider confidence under 0.4 → one enhanced frame goes to a vision model → its answer replaces the
   decider's; both are kept in the event record.
4. Only `alert` wakes the language model, and a `PolicyGuard` intervention denies notifications on any other
   decision. Human control: every decision carries its source and confidence; `recheck` ends in a hint to the
   owner ("add a light", "clean the lens") rather than a guess.

Traces: every event in `web/public/samples/*.timeline.json` records `first_decision`, `recheck` (before/after
measurements, what was seen, the second look), `decider_decision` when escalated, and the final `decision`.

## 6. AWS deployment

Planned, under a $5 budget: the OpenCV pipeline in its container (`vision/Dockerfile`) on a small EC2 instance;
Decider 2B on a g6.xlarge for test and demo sessions (≈$0.80/h); Amazon Bedrock (Claude) for the vision
escalation and the notification text; the playground on the instance as the demo endpoint, shown to judges in a
scheduled screen-share. *(To be replaced with the actual deployment notes, instance ids and costs.)*

## 7. Evaluation

Labelled set: 18 public doorbell clips (`eval/real_clips.csv`), 10 deliveries and 8 thefts, 4 at night, from
Ring and Wyze cameras. Plus 6 free-licence stock doorstep clips (Pexels) and 5 clips with foggy, wet or smudged
lenses. Scoring: `python -m vision.report` → `eval/report.md`.

Headline, all 132 events decided by Decider 2B with the plain-English state wording:

| Measure | Result |
|---|---|
| Clips with at least one alert | 18 / 18 (the late-night delivery was recovered by tracking person-sized movement; it was 17 / 18 on single-frame detection alone) |
| Object left/taken detected | 11 / 18 (3 / 4 at night) |
| Decider mean confidence; events under the 40% escalation line | 68%; 13 / 132 |
| Events that would wake a language model (alerts) | 92 / 132 |
| Agreement with the hand-written rule baseline | 124 / 132 |
| Tracking (person-sized, steady movement the detector could not name) | fired on 3 of 132 events, all real visits; +5 alerts, no new false alarms |
| False "blurry" events from edited title cards | 6 → 0 after the edited-still check |
| Recheck rescues on 8 blurry moments (stock clips) | 7 rescued, person confirmed each time |
| Decider vs rule baseline, 70 events, plain-English wording | agreement 30 → 65, mean confidence 47% → 69% |
| Language model calls, decider-first agent, real theft clip | 2 of 5 events |

Failure cases are listed by name in `eval/report.md`: seven clips where the package event was not detected
(small parcels, night, one compilation with three visits), two of which are "thief stopped" clips where nothing
may have been taken at all.

## 8. Limitations

- Package detection depends on a before/after comparison; small parcels, night footage and busy backgrounds
  defeat it. A detector with a parcel class is the obvious next step.
- Decider 2B on CPU is ~10 s per decision; the recheck's second look is detector-heavy (60 inferences). Fine
  for a drop box, not for a live feed without a GPU.
- Small vision models can echo the prompt instead of looking; the escalation prompt is written around that.
- The labelled clips are edited YouTube videos: cuts and title cards are artefacts a real camera never produces.
- Old phones have no infrared: in true darkness nothing can be rescued, and the system says so.

## 9. Responsible use

The system watches a doorstep the owner already records. It stores decisions and measurements, not faces; it
never identifies people; it surfaces its own uncertainty ("recheck", confidence, source) rather than asserting;
and the owner, not the agent, decides what to do with an alert. The public clips used for evaluation are not
redistributed; the submission demo uses free-licence and self-recorded footage.
