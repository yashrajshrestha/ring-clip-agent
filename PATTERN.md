# Reusable Decider 2B decision pattern: doorbell event policy

A **policy** decision (`#policy`) with **model routing** on top (`#model-routing`): Strands Decider 2B
classifies every camera event into one of four actions before the agent acts, and only a low-confidence
answer is routed to a larger vision model. Drop it into any agent that reacts to a stream of events.

## 1. The typed question

One `choice` question per event. The option *descriptions* are what the model reads, so they carry the
policy (`vision/decider.py`):

```python
QUESTION = "What should the doorbell do about this event for the home owner?"
CRITERIA = {
    "alert":   "Notify the owner now: a person is at the door, or something was left or taken.",
    "ignore":  "Do nothing: an animal, a passing car, shadows, or tiny brief motion.",
    "recheck": "The picture is too blurry or dark to trust; check the camera or look again.",
    "log":     "Keep a note only: a camera mode change or a cut in the video, nothing at the door.",
}

def request(ev, earlier):
    return {"state": state_text(ev, earlier),
            "questions": {"action": {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA}}}
```

The state is plain text: the event, what the detector saw, the last three events before it (so "person,
then object taken" reads as one story), and what a recheck found. `POST /v1/systemone` returns the choice,
per-option probabilities and a calibrated confidence in one call, no text generation.

## 2. The wiring

**Decider first, model only when needed** (`vision/agent.py`): the decision is made before any language
model runs. `alert` wakes the model to write the notification; `ignore`, `log` and `recheck` never touch it.
A `PolicyGuard` intervention keeps the model honest:

```python
class PolicyGuard(InterventionHandler):
    name = "policy-guard"
    def before_tool_call(self, event, **kwargs):
        if event.tool_use["name"] == "notify_owner" and self.decision.get("action") != "alert":
            return Deny(reason=f"The decision for this event is {self.decision['action']}, not alert.")
        return Proceed()

agent = Agent(model=model, tools=[notify_owner], system_prompt=SYSTEM, interventions=[guard])
```

**Recheck is an action, not a dead end** (`vision/timeline.py`, `vision/enhance.py`): a `recheck` answer
makes OpenCV enhance the frames, zoom in over more frames, run the detector again, and ask the decider a
second time with what the better picture showed.

**Routing on confidence** (`vision/escalate.py`): when `confidence < 0.4`, one enhanced frame goes to a
vision language model (Amazon Bedrock Nova Lite) with the same four options; its answer replaces the
decider's and the decider's is kept alongside as `decider_decision`. One exception: the second look may
raise an alert or settle a doubt, but it may not silence an alert where OpenCV saw a person. One frame from
a short event is often empty, and on the RingTV clips 3 of 6 looks said "nothing" over a person:

```python
def may_override(ev, decision, answer):
    silences = decision["action"] == "alert" and answer["action"] != "alert"
    return not (silences and ev.get("label") in ("person", "figure"))
```

When it does not override, the answer is kept as `vision_decision` beside the alert.

## 3. How to use it in your own agent

1. Write your options as one-line descriptions of *when to pick them*; keep 3 to 6 options.
2. Render your event as a short plain-text state, including a little recent history.
3. Call Decider 2B before the model; act on `ignore`/`log`-type answers without the model.
4. Put a guard in front of the expensive or irreversible tool that checks the decision.
5. Pick a confidence threshold from your own data and route below it to a bigger model.

## 4. What it measured

- **Wording matters more than anything else.** The same decider, the same 70 events, only the state text
  changed: bare type names ("cut at 12.3s", "scene change at 42.8s") agreed with the rule stub on 30/70 at
  47% mean confidence; plain English ("the video cut to a different shot (an edit in the recording, not
  something at the door)", "the doorstep looks different after the visit, so an object was left or taken")
  agreed on 65/70 at 69%. Every alert on a video edit and every package event downgraded to "log" was fixed.
  Write the state the way you would explain it to a person.

- Pexels doorstep clips: 4 of 6 main events decided confidently (82-87%); the 2 low-confidence ones
  (28%, 15%) were exactly the ones a vision model needed to settle.
- Real theft clip, decider-first agent: the language model ran for 2 of 5 events.
- Decider 2B: ~0.5 s per decision on an Apple M2 Pro (MPS), ~10 s on a laptop CPU (115 ms on a GPU per AWS).
- Four-way benchmark on 10 shareable clips (`eval/benchmark.md`): Decider first with a Bedrock second look
  alerted on 7/7 visit clips and kept 3/3 pet clips quiet with 9 Bedrock calls ($0.0006); Bedrock on every
  event caught 4/7 visits with 35 calls ($0.0023), and Decider alone kept 2/3 pet clips quiet.
