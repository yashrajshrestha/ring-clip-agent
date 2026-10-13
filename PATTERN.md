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
vision language model with the same four options; its answer replaces the decider's and the decider's is
kept alongside as `decider_decision`.

## 3. How to use it in your own agent

1. Write your options as one-line descriptions of *when to pick them*; keep 3 to 6 options.
2. Render your event as a short plain-text state, including a little recent history.
3. Call Decider 2B before the model; act on `ignore`/`log`-type answers without the model.
4. Put a guard in front of the expensive or irreversible tool that checks the decision.
5. Pick a confidence threshold from your own data and route below it to a bigger model.

## 4. What it measured

- Pexels doorstep clips: 4 of 6 main events decided confidently (82-87%); the 2 low-confidence ones
  (28%, 15%) were exactly the ones a vision model needed to settle.
- Real theft clip, decider-first agent: the language model ran for 2 of 5 events.
- Decider 2B on a laptop CPU: ~10 s per decision when idle (115 ms on a GPU per AWS).
