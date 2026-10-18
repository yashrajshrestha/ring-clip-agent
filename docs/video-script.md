# Demo video: script and shot list

**Target:** 2:50 (hard limit 3:00), 1920×1080, uploaded **public** on YouTube.
**Title card:** *Decide Cheap, Look Twice: A Doorbell Agent Built on Strands Decider 2B*
**Voice:** about 330 words (about 2:15 read aloud, leaving room for pauses and demo moments). Rules: no copyrighted music, no third-party logos.

## Footage we may show

Only the free-licence Pexels clips (`eval/stock_clips.csv`), the Creative Commons RingTV clips
(`eval/cc_clips.csv`, saved in `clips/cc/`), or your own recordings. **Never** a clip marked
"standard (test only)" in `eval/real_clips.csv`.

**Credit required for the RingTV clips (CC BY):** put this in the YouTube description and on the end card:
*"Doorbell clips: RingTV (Ring LLC), licensed Creative Commons Attribution (CC BY 3.0), youtube.com/@Ring.
Stock clips: Pexels."* The clips carry Ring's watermark; fine with credit here, but avoid them in any entry
whose rules forbid third-party trademarks.

| Use in the video | Clip | Why |
|---|---|---|
| Hero clip: parcel left | `pexels_leave_package_6667264` | Decider says log at 34%, Bedrock sees the parcel → alert |
| Night clip | `pexels_kids_knock_night_5856456` | Decider says recheck at 10%, Bedrock sees a person → alert |
| Quiet case | `pexels_dog_doorway_5394386` | A dog, nobody there → ignore, no alert |
| Optional: taken | `pexels_girl_pickup_6994935` | Parcel picked up → alert |
| Real night theft | `ring_k-WvGk7a9Ms` (CC BY) | Real doorbell footage at night: present taken |
| Real day theft | `ring_RM8AxCNwzcQ` (CC BY) | Neighbour takes a package |
| Real dog case | `ring_xEqUvf5UDnY` (CC BY) | Dog guarding packages → should not alert for the dog |

## Script

### 1. Hook (0:00–0:15)

**Show:** quick cuts: a phone buzzing with alerts ("Motion detected" ×5), then a dark porch frame.
On-screen text: *"Motion detected." "Motion detected." "Motion detected."*

**Say:**
> My doorbell camera alerts me for everything: a cat, a car, a shadow. And when it really matters, on a dark
> porch at night, it often stays quiet.

### 2. The idea (0:15–0:35)

**Show:** title card, then the one-line flow as a simple diagram:
`camera clip → OpenCV finds events → Decider 2B decides → look again if unsure → Bedrock only when needed`

**Say:**
> So I built an agent that decides cheap and looks twice. OpenCV finds what happened in the clip. Strands
> Decider 2B, a small decision model running on my laptop, decides each event: alert, ignore, recheck, or log.
> And only when it's unsure does it ask Amazon Bedrock for a second look.

### 3. Live demo: the parcel (0:35–1:15)

**Show (screen recording of the playground, http://127.0.0.1:8091):**
1. Drag `pexels_leave_package_6667264.mp4` onto the drop box. Progress bar runs.
2. The event list fills in: person-sized movement, object left, each with a label and "% sure".
3. Click the event at **10.7 s**. The right panel shows Decider's four bars: *log 34%*.
4. Click **Ask the vision model**. Nova Lite answers: *"A package is on the doorstep"* → **alert**.

Zoom in on the bars and the Bedrock answer (Screen Studio does this automatically; in iMovie use Ken Burns).

**Say:**
> Here's a real delivery. I drop the clip in. OpenCV finds the events, and Decider decides each one in about
> half a second, for free. Most are easy. But here, at ten seconds, Decider says "log", and it's only 34% sure.
> That low number is the trigger. One frame goes to Amazon Nova Lite on Bedrock, and it sees the parcel. Alert.

### 4. Live demo: the dark porch (1:15–1:45)

**Show:** select the night clip. The event reads *"Picture too dark"*. Decider: **recheck, 10%**. Show the
brightened frame next to the original, then the Bedrock answer: *"A person is standing at the door"* → alert.

**Say:**
> Now a child in a dark costume at dusk. OpenCV can't recognise anything. Decider doesn't guess: it says
> recheck, and it's only 10% sure. OpenCV brightens the frame, Bedrock looks, and finds the person. The agent
> knew it couldn't see, and said so.

### 5. The quiet case and the message (1:45–2:05)

**Show:** the dog clip → **ignore**, no alert. Then a terminal running the Strands agent:
`python -m vision.agent pexels_leave_package_6667264 --limit 5`, ending on the notification line and
*"language model ran for 2 of 5 events; guard denied 0 notify attempts"*.

**Say:**
> A dog in the doorway? Ignore. No alert, no AI bill. And when it is an alert, a Strands agent writes one short
> message. A policy guard stops it from notifying on anything else.

### 6. The numbers (2:05–2:30)

**Show:** three number cards, one after another (big text, plain background):
- **26 of 26** visits caught · **2** false alarms (basic motion alerts: **17**)
- **11** Bedrock calls · **$0.0008** total
- Decider: **~0.5 s** per decision on a laptop · **$0**

**Say:**
> On 28 test clips it caught all 26 real visits with just 2 false alarms. Basic motion alerts had 17. Bedrock
> was called 11 times in my first runs, for less than a tenth of a cent, because the small model handles the
> easy calls.

### 7. The reusable pattern (2:30–2:45)

**Show:** `PATTERN.md` in the repo: the four options (alert / ignore / recheck / log) and the `PolicyGuard`
class, scrolled slowly.

**Say:**
> The pattern is reusable: four plain-English options, a confidence line for routing to Bedrock, and a guard in
> front of the expensive tool. It's all in the repo.

### 8. Close (2:45–2:55)

**Show:** end card: repo URL, `#strands-decider-2b #policy`, your name. Plus the clip credit line (RingTV CC BY, Pexels).

**Say:**
> Decide cheap, look twice. Works with any camera, even an old phone. Thanks for watching.

## Before recording

- [ ] Run the three Pexels clips through the pipeline with Bedrock on, so the playground has fresh results:
      `VISION_MODEL=bedrock:amazon.nova-lite-v1:0 BEDROCK_REGION=us-east-1 DECIDER_URL=http://127.0.0.1:8000 DECIDER_TIMEOUT=300 python -m vision.playground`
- [ ] Decider server running on the newer checkpoint (`strands-decider-2B-qwen3.5-v1-2610`).
- [ ] Browser: full screen, zoom 125%, dark mode, no other tabs, notifications off (Do Not Disturb).
- [ ] Close private windows and anything with keys, tokens or email on screen.
- [ ] Terminal: large font (18 pt), clean prompt, `clear` before each take.

## Recording and editing

1. Record each section separately (easier retakes): **Cmd + Shift + 5** → *Record Selected Portion*, or
   Screen Studio for automatic zooms.
2. Record the voice separately with a headset or phone mic in a quiet room. Read slowly.
3. Assemble in iMovie (free) or Descript. Put the voice on top, trim the screen clips to fit.
4. Add captions (YouTube auto-captions, then correct them). Many judges watch muted.
5. Check the length is **under 3:00**, then upload to YouTube as **Public**.
