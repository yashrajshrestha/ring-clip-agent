# A doorbell agent that knows when it can't see: Strands Decider 2B decides, OpenCV looks again, Bedrock only when needed

*Draft for AWS Builder Center. Tags: #strands-decider-2b #policy. Repo: github.com/yashrajshrestha/ring-clip-agent (MIT).*

## The problem

Doorbell cameras either flood you with alerts (a cat, a car, a shadow) or go quiet when it matters (a dark porch,
a smudged lens, a parcel taken while nobody was "detected"). The smart alerts that fix this cost $18–20 a month
and only work with one brand's camera. I wanted alerts that work with **any** camera, including an old phone,
with no subscription, and that are honest when the picture can't be trusted.

## The agent

```
camera clip → OpenCV 5 finds events → Decider 2B decides each one → recheck / escalate when unsure → agent writes the message
```

- **OpenCV 5** turns a clip into events: motion, a person / animal / vehicle (YOLOX-S through `cv2.dnn`), a cut,
  day/night switch, a blurry or dark stretch, and "the doorstep looks different after the visit" (object left or taken).
- **Strands Decider 2B** gets one typed `choice` question per event and returns one of four actions with a
  calibrated confidence: **alert**, **ignore**, **recheck**, **log**. No text is generated. This is the decision
  I handed to Decider 2B, and it fires dozens of times per clip.
- **recheck is an action, not a shrug.** OpenCV brightens and sharpens the frames (gamma, CLAHE, denoise), zooms
  in over more frames, runs the detector again, and the decider decides a second time on what the better picture
  shows. A delivery clip darkened to brightness 11/255 came back at 93, the courier was found in 7 of 8 frames,
  and the decision flipped from recheck to alert. A lens-blur clip stayed recheck, with the hint "clean the lens".
- **Model routing on confidence.** Under 40% confidence, one enhanced frame goes to **Amazon Nova Lite on
  Amazon Bedrock** (Converse API) with the same four options. On six stock doorstep clips the decider was
  82–87% sure on the four easy ones and 28% / 15% on exactly the two hard ones (packages with nobody there;
  a child in a dark costume), and those are the ones Bedrock settled (see "Bedrock in practice" below).
- **Decider first, language model last.** The Strands agent only runs for `alert`, and its one job is to write
  the notification sentence. A `PolicyGuard` intervention denies `notify_owner` on anything but an alert, so a
  confused model can't notify. On a real theft clip the language model ran for 2 of 5 events. (The sentence
  writer runs on a small local model today; in Strands, moving it to Bedrock is a change of model provider.)

## The decision pattern (reusable)

`PATTERN.md` in the repo has the full thing. The short version:

```python
CRITERIA = {
    "alert":   "Notify the owner now: a person is at the door, or something was left or taken.",
    "ignore":  "Do nothing: an animal, a passing car, shadows, or tiny brief motion.",
    "recheck": "The picture is too blurry or dark to trust; check the camera or look again.",
    "log":     "Keep a note only: a camera mode change or a cut in the video, nothing at the door.",
}
```

plus a plain-text state (the event, what the detector saw, the last three events, what a recheck found), the
`PolicyGuard` intervention in front of the expensive tool, and a confidence threshold for routing up.

## What I learned

1. **Wording is the model.** Same decider, same 70 events, only the state text changed. Bare type names
   ("cut at 12.3s", "scene change at 42.8s") agreed with my rule baseline on 30/70 at 47% mean confidence.
   Plain English ("the video cut to a different shot (an edit in the recording, not something at the door)",
   "the doorstep looks different after the visit, so an object was left or taken") agreed on **65/70 at 69%**,
   and fixed every alert on a video edit and every package event the decider had downgraded to "log".
2. **Confidence is the feature.** The decider's low-confidence events were the genuinely hard ones. Routing on
   it means the big model sees only what it needs to: 2 of 6 events on the stock clips.
3. **Small vision models repeat your prompt.** Given "the detector saw nothing", a 3B vision model answered
   "ignore: the detector saw nothing" without looking. Making it describe the picture first, with the detector's
   verdict removed from the prompt, turned that into "alert: a person is standing outside the door".
4. **Latency and cost.** Decider 2B (`strands-decider-2B-qwen3.5-v1-2610`) on a laptop CPU: ~10 s per decision
   when idle, 132 decisions in 24 minutes for the whole labelled set. On an Apple M2 Pro laptop GPU (PyTorch MPS)
   the same decision takes **about 0.5 s**; AWS quotes ~115 ms on a data-centre GPU. A Bedrock second look costs
   **about $0.00007** and takes 1.6–1.8 s once warm. Only alerts wake the language model, and only sub-40%
   confidence wakes Bedrock.
5. **A second look on a sliver can be worse than the first.** Two 0.3-second events, where the detector saw a
   person and the decider said alert at 24%, came back from Bedrock as "nothing is at the door": the frame was
   from the middle of a blink-length event. The clip still alerted on its main 6-second approach, but very short
   events need more than one frame, or no second look at all.

## Bedrock in practice

First runs on Amazon Bedrock (Nova Lite, us-east-1), Decider 2B on the laptop, three clips:

| Clip | Events | Decider alone | Sent to Bedrock | What Bedrock changed | Cost |
|---|---|---|---|---|---|
| Parcel left at the door (10 s) | 14 | 9 | 5 | at 10.7 s the decider said *log* (34%); Nova Lite saw "a package is on the doorstep" → **alert** | ≈ $0.0003 |
| Child in a dark costume at dusk (8 s) | 1 | 0 | 1 | decider: **recheck, 10% sure**; Nova Lite, shown the brightened frame: "a person is standing at the door" → **alert** | ≈ $0.00007 |
| Repeat of the parcel clip | 14 | 9 | 5 | same answers both times (temperature 0) | ≈ $0.0003 |

**11 Bedrock calls in all, about $0.0008.** Each look reads ~1,000 tokens (mostly the image) and writes ~20.
Sending every event to Bedrock would have cost about three times as much and been slower; the decider's
confidence number is what keeps the big model for the cases that need it.

## Evaluation

18 labelled doorbell clips (deliveries and thefts, 4 at night), 131 of 132 events decided by Decider 2B
(one fell back to the rule baseline):
18 of 18 clips raised at least one alert. The last one to fall in line was a late-night delivery where the
detector never labelled the courier: OpenCV now tracks a person-sized blob that moves steadily across the
frame and tells the decider so ("a person-sized shape moved steadily across the porch, which the detector could
not identify"), which turned that clip from "ignore 52%" into "alert 27%", low confidence by design, and so a
candidate for the vision model. The package being left or taken was detected in 11 of 18 (the known weak spot:
the detector has no "box" class, so this relies on comparing the doorstep before and after). Mean confidence
68%; 13 of 132 events fell under the 40% escalation line; 90 of 132 events were alerts, so the language model
would run for about two events in three and never for the rest. Agreement with a hand-written rule baseline: 124 of
132. `eval/report.md` names every failure.

**Compared with Ring's alerts** (simulated from Ring's documented settings on this project's detections, 28
clips: 26 visits that should alert, 2 with only a dog that should not; no Ring device was used):

| System | Phone alerts | Of which noise | Visits caught | Dog-only clips left quiet |
|---|---|---|---|---|
| Ring, basic motion alerts | 120 | 17 | 25 / 26 | 0 / 2 |
| Ring, person and package smart alerts | 102 | 0 | 24 / 26 | 2 / 2 |
| This agent (OpenCV 5 + Decider 2B) | 117 | 2 | **26 / 26** | 2 / 2 |

Ring's own package detector is likely better than the before/after comparison standing in for it here.

## What's next

The notification text on Bedrock too, several frames instead of one for very short events, and the OpenCV
pipeline in its container on AWS. And my own webcam clips, because an old phone really is enough.
