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
- **Model routing on confidence.** Under 40% confidence, one enhanced frame goes to a vision model with the same
  four options (Amazon Bedrock Claude in the AWS build; a local Ollama model while developing). On six stock
  doorstep clips the decider was 82–87% sure on the four easy ones and 28% / 15% on exactly the two hard ones
  (packages with nobody there; a child in a dark costume), which the vision model then settled.
- **Decider first, language model last.** The Strands agent only runs for `alert`, and its one job is to write
  the notification sentence. A `PolicyGuard` intervention denies `notify_owner` on anything but an alert, so a
  confused model can't notify. On a real theft clip the language model ran for 2 of 5 events.

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
4. **Latency and cost.** Decider 2B on a laptop CPU: ~10 s per decision when idle, 132 decisions in 24 minutes
   for the whole labelled set (AWS quotes ~115 ms on a GPU). Only alerts wake the language model, and only
   sub-40% confidence wakes the vision model.

## Evaluation

18 labelled doorbell clips (deliveries and thefts, 4 at night), all 132 events decided by Decider 2B:
17 of 18 clips raised at least one alert (the miss is a late-night delivery where the detector never labelled
the courier and the decider read a 1.3 s person-sized blob as "ignore"); the package being left or taken was
detected in 11 of 18 (the known weak spot: the detector has no "box" class, so this relies on comparing the
doorstep before and after). Mean confidence 68%; 13 of 132 events fell under the 40% escalation line; 87 of
132 events were alerts, so the language model would run for two thirds of events and never for the rest.
Agreement with a hand-written rule baseline: 123 of 132. `eval/report.md` names every failure.

## What's next

Bedrock for the vision escalation and the notification text, the decider on a GPU instance, and the OpenCV
pipeline in its container on AWS. And my own webcam clips, because an old phone really is enough.
