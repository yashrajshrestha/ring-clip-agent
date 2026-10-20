# Decide Cheap, Look Twice: A Doorbell Agent Built on Strands Decider 2B

*Draft for AWS Builder Center. Tags: #strands-decider-2b #policy. Repo: github.com/yashrajshrestha/ring-clip-agent (MIT).*

## The problem

Doorbell cameras either flood you with alerts (a cat, a car, a shadow) or go quiet when it matters (a dark porch,
a smudged lens, a parcel taken while nobody was "detected"). The smart alerts that fix this cost $18–20 a month
and only work with one brand's camera. I wanted alerts that work with **any** camera, including an old phone,
with no subscription, and that are honest when the picture can't be trusted.

## The agent

```
camera clip → OpenCV 5 finds events → Decider 2B decides each one → unsure? Bedrock looks again → alert? Bedrock writes the message
```

| Step | Runs on | Cost |
|---|---|---|
| OpenCV 5 turns the clip into events | the laptop | $0 |
| **Strands Decider 2B** decides every event: alert, ignore, recheck or log | the laptop (Apple M2 Pro), ~0.5 s each | $0 |
| Second look, only when Decider is under 40% sure | **Amazon Bedrock, Nova Lite** (one frame + the same four options) | ~$0.00007 a look |
| The phone message, only for an alert | **Amazon Bedrock, Nova Micro**, inside a Strands agent | ~$0.00007 a message |

- **OpenCV 5** turns a clip into events: motion, a person / animal / vehicle (YOLOX-S through `cv2.dnn`), a cut,
  day/night switch, a blurry or dark stretch, and "the doorstep looks different after the visit" (object left or taken).
- **Strands Decider 2B** gets one typed `choice` question per event and returns one of four actions with a
  calibrated confidence. No text is generated. This is the decision I handed to Decider 2B, and it fires
  dozens of times per clip.
- **recheck is an action, not a shrug.** OpenCV brightens and sharpens the frames (gamma, CLAHE, denoise), zooms
  in over more frames, runs the detector again, and the decider decides a second time on what the better picture
  shows. A delivery clip darkened to brightness 11/255 came back at 93, the courier was found in 7 of 8 frames,
  and the decision flipped from recheck to alert.
- **Model routing on confidence.** Under 40% confidence, one enhanced frame goes to Nova Lite on Bedrock with the
  same four options. Its answer replaces the decider's, with one exception (below): it may raise an alert, but it
  may not silence one where OpenCV saw a person.
- **Decider first, language model last.** A Strands agent runs only for `alert`, and its one job is to write the
  notification on Bedrock. A `PolicyGuard` intervention denies `notify_owner` on anything but an alert, so a
  confused model can't notify. When Bedrock's second look made the call, the message describes what it saw.

## Seeing it work on real doorbell footage

Eight Creative Commons clips from Ring's own YouTube channel, plus two free stock dog clips:

- **Two thieves, read as "an animal".** The detector labelled a struggle at the door "animal ×12". Decider said
  ignore, but only 39% sure, so Bedrock looked: *"Two people are at the door, one is holding a package"* → alert.
  The phone message read: *"Two people are at your door, one holding a package and the other opening it."*
- **A cat on the parcels.** Three animal events were ignored by Decider alone. When the cat stepped off the boxes,
  OpenCV saw "the doorstep looks different" and Decider leaned towards alert at 40%. Bedrock looked and said log.
  No alert. (Nova Lite called the cat a dog; the decision was still right.)
- **A German Shepherd in a dark doorway.** Two "too dark" events: Decider unsure (30–33%), Bedrock saw *"a dog is
  standing at the door"* → ignore.

## The benchmark: four ways to decide the same events

Same 10 clips, same 35 events (OpenCV ran once), four deciders. Seven clips are visits or thefts that should
alert; three show only a pet and should stay quiet.

| Method | Visit clips alerted | Pet clips kept quiet | Phone alerts | Bedrock calls | Bedrock cost | Decision time per clip |
|---|---|---|---|---|---|---|
| Hand-written rules | 7 / 7 | 0 / 3 | 20 | 0 | $0 | – |
| Decider 2B alone | 7 / 7 | 2 / 3 | 15 | 0 | $0 | 2.5 s |
| **Decider 2B + Bedrock when unsure** | **7 / 7** | **3 / 3** | **14** | **9** | **$0.0006** | **4.3 s** |
| Bedrock on every event | 4 / 7 | 3 / 3 | 4 | 35 | $0.0023 | 13.8 s |

Decider-first was the only method to get every clip right, with a quarter of Bedrock's calls and a third of its
time. Sending everything to the bigger model was *worse*: from a single frame it often answered "nothing at the
door", even during a theft. The small model's confidence number is what decides when the big one is worth asking.
(Ten clips is a small test; the costs are fractions of a cent either way, so the ratios are the point.
`vision/benchmark.py` reproduces it.)

## The decision pattern (reusable)

`PATTERN.md` in the repo has the full thing. The short version:

```python
CRITERIA = {
    "alert":   "Notify the owner now: a person is at the door, or something was left or taken.",
    "ignore":  "Do nothing: an animal, a passing car, shadows, or tiny brief motion.",
    "recheck": "The picture is too blurry or dark to trust; check the camera or look again.",
    "log":     "Keep a note only: a camera mode change or a cut in the video, nothing at the door.",
}

def may_override(ev, decision, answer):
    """The second look may raise an alert or settle a doubt, never silence a person."""
    silences = decision["action"] == "alert" and answer["action"] != "alert"
    return not (silences and ev.get("label") in ("person", "figure"))
```

plus a plain-text state (the event, what the detector saw, the last three events, what a recheck found), the
`PolicyGuard` intervention in front of the expensive tool, and a confidence threshold for routing up.

## What I learned

1. **Wording is the model.** Same decider, same 70 events, only the state text changed. Bare type names
   ("cut at 12.3s", "scene change at 42.8s") agreed with my rule baseline on 30/70 at 47% mean confidence.
   Plain English ("the video cut to a different shot (an edit in the recording, not something at the door)",
   "the doorstep looks different after the visit, so an object was left or taken") agreed on **65/70 at 69%**.
2. **Confidence is the feature.** The decider's low-confidence events were the genuinely hard ones: 9 of 35 on
   the Ring and stock clips. Routing on that number means the big model sees only what it needs to.
3. **A second look can add an alarm, but shouldn't remove a person.** The first version let Bedrock overrule
   anything. On the Ring clips, 3 of its 6 looks said "nothing at the door" over a person the detector had seen:
   it gets one frame from the middle of the event, and on a 0.1-second event that frame is often empty. One theft
   clip went unalerted. The fix is four lines (`may_override` above): keep the alert, store Bedrock's answer next
   to it. Re-run: 7 of 7 visit clips alerted, the pet clips stayed quiet, the same 6 Bedrock calls.
4. **Small vision models repeat your prompt.** Given "the detector saw nothing", a 3B vision model answered
   "ignore: the detector saw nothing" without looking. Making it describe the picture first, with the detector's
   verdict removed from the prompt, turned that into "alert: a person is standing outside the door".
5. **Latency and cost.** Decider 2B (`strands-decider-2B-qwen3.5-v1-2610`) takes about 0.5 s per decision on an
   Apple M2 Pro laptop (PyTorch MPS), ~10 s on a laptop CPU, and AWS quotes ~115 ms on a data-centre GPU. A Bedrock
   second look takes 1.6–2.3 s from my laptop. On the ten clips, Bedrock cost **$0.0006 for the second looks and
   $0.0009 for 14 phone messages**; 15 of 35 events never reached any language model at all.

## Evaluation on the full labelled set

18 labelled doorbell clips (deliveries and thefts, 4 at night), 131 of 132 events decided by Decider 2B (one fell
back to the rule baseline). 18 of 18 clips raised at least one alert. The last one to fall in line was a
late-night delivery where the detector never labelled the courier: OpenCV now tracks a person-sized blob that moves
steadily across the frame and tells the decider so, which turned that clip from "ignore 52%" into "alert 27%", low
confidence by design, and so a candidate for the second look. The package being left or taken was detected in 11 of
18 (the known weak spot: the detector has no "box" class, so this relies on comparing the doorstep before and after).
Mean confidence 68%; 13 of 132 events fell under the 40% escalation line; 90 of 132 events were alerts. Agreement
with a hand-written rule baseline: 124 of 132. `eval/report.md` names every failure.

**Compared with Ring's alerts** (simulated from Ring's documented settings on this project's detections, 28 clips:
26 visits that should alert, 2 with only a dog that should not; no Ring device was used):

| System | Phone alerts | Of which noise | Visits caught | Dog-only clips left quiet |
|---|---|---|---|---|
| Ring, basic motion alerts | 120 | 17 | 25 / 26 | 0 / 2 |
| Ring, person and package smart alerts | 102 | 0 | 24 / 26 | 2 / 2 |
| This agent (OpenCV 5 + Decider 2B) | 117 | 2 | **26 / 26** | 2 / 2 |

This is not a claim to beat Ring: its own package detector is likely better than the before/after comparison
standing in for it here, and its smart alerts send fewer notifications. What this agent adds is that it works with
any camera, says when it can't see, and shows its reasoning with a confidence number.

## What's next

Several frames instead of one for very short events, the OpenCV pipeline in its container on AWS, and my own
webcam clips, because an old phone really is enough.

---

*Doorbell clips: RingTV (Ring LLC), Creative Commons Attribution (CC BY 3.0). Stock clips: Pexels.*
