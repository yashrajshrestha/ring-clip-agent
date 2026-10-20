# Demo video: script and shot list

**Target:** 2:50 (hard limit 3:00), 1920×1080, uploaded **public** on YouTube.
**Title card:** *Decide Cheap, Look Twice: A Doorbell Agent Built on Strands Decider 2B*
**Voice:** about 360 words (about 2:25 read aloud, leaving room for demo moments). Rules: no copyrighted music.

## Footage we may show

Only the Creative Commons RingTV clips (`eval/cc_clips.csv`, saved in `clips/cc/`), the free-licence Pexels
clips (`eval/stock_clips.csv`), or your own recordings. **Never** a clip marked "standard (test only)" in
`eval/real_clips.csv`.

**Credit required for the RingTV clips (CC BY):** put this in the YouTube description and on the end card:
*"Doorbell clips: RingTV (Ring LLC), licensed Creative Commons Attribution (CC BY 3.0), youtube.com/@Ring.
Stock clips: Pexels."* The clips carry Ring's watermark; fine with credit here, but avoid them in any entry
whose rules forbid third-party trademarks.

| Scene | Clip | What it shows |
|---|---|---|
| 3. Live drop | `clips/demo/ring_nRV3nmU33oE.mp4` (thieves) | Detector says "animal"; Decider ignore 39%; Bedrock: "Two people are at the door, one is holding a package" → alert |
| 4. The rule | `ring_vS55Kxp-rZQ` (neighbour takes packages), event at 0.8 s | Bedrock said "nothing", but a person was seen, so the alert stands |
| 5. Pets | `ring_xEqUvf5UDnY` (cat on the parcels), `pexels_dog_doorway_5394386` (German Shepherd) | No alerts; Bedrock sees "a dog is standing at the door" in the dark doorway |
| 6. The message | terminal, thieves clip | Bedrock Nova Micro writes the alert; "language model ran for 2 of 6 events" |

## Script

### 1. Hook (0:00–0:15)

**Show:** quick cuts: a phone buzzing with alerts ("Motion detected" ×5), then a dark porch frame.
On-screen text: *"Motion detected." "Motion detected." "Motion detected."*

**Say:**
> My doorbell camera alerts me for everything: a cat, a car, a shadow. And when it really matters, it can
> still get it wrong.

### 2. The idea (0:15–0:35)

**Show:** title card, then the flow as a simple diagram:
`camera clip → OpenCV finds events → Decider 2B decides → unsure? Bedrock looks again → alert? Bedrock writes the message`

**Say:**
> So I built an agent that decides cheap and looks twice. OpenCV finds what happened. Strands Decider 2B, a small
> decision model running on my laptop, decides each event: alert, ignore, recheck, or log. Only when it's unsure
> does Amazon Bedrock take a second look.

### 3. Live demo: the thieves (0:35–1:15)

**Show (Chrome, full screen, http://127.0.0.1:8091):**
1. Drag `ring_nRV3nmU33oE.mp4` from Finder onto the drop box. The progress bar runs (about 20 s: cut most of it).
2. The events appear: *Person at the door → ALERT*, *Animal at the door → ALERT*, then four *ignore*s.
3. Click **Animal at the door (13.1 s)**. Point at the right panel: the detector's text says "animal ×12";
   Decider said **ignore at 39%**; the second look from **bedrock:amazon.nova-lite-v1:0** says **ALERT**,
   "Decider said ignore at 39%; the vision model overrode it."
4. Play the video at 13 s: two people grabbing a package.

**Say:**
> Here's real doorbell footage. I drop the clip in, and in about twenty seconds every event is decided. Decider
> takes half a second each, for free. Now this one: the detector thinks it's an animal, and Decider says ignore,
> but it's only 39% sure. That low number is the trigger. One frame goes to Amazon Nova Lite on Bedrock, and it
> sees two people taking a package. Alert.

### 4. The rule: never silence a person (1:15–1:35)

**Show:** choose `ring_vS55Kxp-rZQ`, click **Person at the door (0.8 s)**. Zoom on the note: *"The vision model
said ignore ("nothing"), but a person was seen, so the alert stands."*

**Say:**
> The second look isn't perfect. It sees one frame, and on a split-second event that frame can be empty. So the
> rule is: it can add an alarm, but it can't silence a person.

### 5. Pets stay quiet (1:35–1:55)

**Show:** choose `ring_xEqUvf5UDnY`: four events, no ALERT (animal → ignore, the last one → log). Then
`pexels_dog_doorway_5394386`: click a *too dark* event, Bedrock: *"A dog is standing at the door"* → ignore.

**Say:**
> A cat sitting on the parcels? Ignored. A dog in a dark doorway? Decider wasn't sure, Bedrock looked, saw the
> dog. No alert, no wake-up.

### 6. The message (1:55–2:10)

**Show:** a terminal (18 pt font) running:
`AWS_PROFILE=personal DECIDER_URL=http://127.0.0.1:8000 python -m vision.agent ring_nRV3nmU33oE --model bedrock:amazon.nova-micro-v1:0`
ending on *"Two people are at your door, one holding a package…"* and *"language model ran for 2 of 6 events"*.

**Say:**
> When it is an alert, a Strands agent on Bedrock writes one short message, using what the second look saw. A
> policy guard stops it from notifying on anything else. Four of six events never touched a language model.

### 7. The numbers (2:10–2:35)

**Show:** the benchmark table from `eval/benchmark.md` as a clean slide (big text):

| | Visits caught | Pets quiet | Bedrock calls | Cost |
|---|---|---|---|---|
| Decider + Bedrock when unsure | **7 / 7** | **3 / 3** | 9 | $0.0006 |
| Bedrock on everything | 4 / 7 | 3 / 3 | 35 | $0.0023 |

**Say:**
> I raced four ways of deciding the same ten clips. Decider first, with Bedrock only when unsure, caught all seven
> visits and kept all three pets quiet. Sending everything to Bedrock caught only four, at four times the cost and
> three times the time.

### 8. The reusable pattern (2:35–2:47)

**Show:** `PATTERN.md` in the repo: the four options and the `may_override` function, scrolled slowly.

**Say:**
> The pattern is reusable: four plain-English options, a confidence line for routing to Bedrock, a guard in front
> of the expensive tool, and a rule that the second look can't silence a person.

### 9. Close (2:47–2:55)

**Show:** end card: repo URL, `#strands-decider-2b #policy`, your name, and the clip credit line.

**Say:**
> Decide cheap, look twice. Works with any camera, even an old phone. Thanks for watching.

## Before recording

The playground is prepared: the 9 analysed clips are in the list, the drawn practice clip is gone, and the
thieves clip waits in `clips/demo/` for the live drop.

- [ ] Decider server running on the newer checkpoint (`strands-decider-2B-qwen3.5-v1-2610`), port 8000.
- [ ] Playground with Bedrock on:
      `AWS_PROFILE=personal BEDROCK_REGION=us-east-1 VISION_MODEL=bedrock:amazon.nova-lite-v1:0 DECIDER_URL=http://127.0.0.1:8000 DECIDER_TIMEOUT=300 python -m vision.playground --port 8091`
- [ ] Chrome: open http://127.0.0.1:8091, full screen, zoom 125%, dark mode, no other tabs, Do Not Disturb on.
- [ ] Finder window with `clips/demo/` open beside Chrome, for the drag.
- [ ] Close private windows and anything with keys, tokens or email on screen.
- [ ] Terminal: 18 pt font, `cd ~/ring-clip-agent && source .venv/bin/activate`, `clear` before the take.
- [ ] To redo the live drop: delete `web/public/samples/ring_nRV3nmU33oE.*` and
      `clips/inbox/ring_nRV3nmU33oE.mp4`, then drag again.

## Recording and editing

1. Record each scene separately (easier retakes): **Cmd + Shift + 5** → *Record Selected Portion*, or Screen
   Studio for automatic zooms.
2. Record the voice separately with a headset or phone mic in a quiet room. Read slowly.
3. Assemble in iMovie (free) or Descript. Put the voice on top, trim the screen clips to fit; cut the 20-second
   wait in scene 3 down to 2–3 seconds.
4. Add captions (YouTube auto-captions, then correct them). Many judges watch muted.
5. Check the length is **under 3:00**, then upload to YouTube as **Public**, with the credit line in the description.
