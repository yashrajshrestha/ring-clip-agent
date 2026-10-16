# AWS journey: moving the porch agent's "second look" to Amazon Bedrock

A running log of every AWS change made for this project, written for other builders who want to
pair Strands Decider 2B (running on hardware you already own) with Amazon Bedrock. Dates, what was
done, what happened, and what it cost. Account numbers are omitted; the region is us-east-1 throughout.

## Why AWS at all

The agent already works offline: OpenCV reads the clip, Decider 2B picks alert / ignore / recheck / log.
Two things want the cloud:

1. **The second look.** When the Decider's confidence is low (under 0.4) the agent shows the best frame
   to a vision model and asks what it sees. That ran on a 3B model on the laptop; it moves to
   **Amazon Nova Lite on Bedrock**, which reads images and costs about $0.0002 per look.
2. **Honest speed numbers.** Decider 2B answers in ~115 ms on a GPU and ~10 s on this laptop's CPU.
   A few hours on a **g4dn.xlarge** (one T4 GPU) gives real latency figures for the write-up.

Design rule kept from the start: **Bedrock is called only when the Decider is unsure.** Most events
never leave the laptop. That is the cost story: the cheap, fast decision handles the common case and
the generative model is reserved for the hard 10–20%.

## Log

### 2026-10-16: GPU quota request

- New accounts start with **0 vCPUs** for G and VT instances, so no GPU machine can start.
- Service Quotas → Amazon EC2 → *Running On-Demand G and VT instances* (quota code `L-DB2E81BA`) →
  **Request increase at account level** → **4** vCPUs (enough for one g4dn.xlarge).
- Result: "Quota increase requested". Small requests are often auto-approved; AWS says larger ones go to
  Support. Waiting for the email.
- Cost: $0.

### 2026-10-16: first Amazon Bedrock call (Nova Lite)

- Bedrock → Model catalog → **Nova Lite** (`amazon.nova-lite-v1:0`, serverless, text + image + video in,
  text out, 300K-token context). No "request model access" step was needed in this account.
- **Open in playground** → Chat → prompt `Reply with exactly one word: porch`.
- Result: `porch`. **Input 7 tokens, output 2 tokens, latency 290 ms.**
- Cost: under $0.001.
- Lesson: the console playground is the fastest way to confirm a model works and is billable *before*
  writing any code or creating credentials.

### 2026-10-16: the second look learns to call Bedrock

- `vision/escalate.py` gained a Bedrock provider next to the local Ollama one. Setting
  `VISION_MODEL=bedrock:amazon.nova-lite-v1:0` sends the enhanced frame through the **Converse API**
  (`bedrock-runtime`, image inline as JPEG bytes, `temperature 0`, 200 output tokens max).
- Each answer now records `ms` and `tokens` (input/output), so the evaluation report can state what a
  second look costs in time and money next to the Decider's own figure.
- Nova Lite likes to wrap JSON in a code fence; the parser strips it.
- Any failure (no credentials, throttling, timeout) returns None and the Decider's answer stands, so a
  cloud problem can never silence an alert.
- Tested with a fake client (8 escalation tests, 105 in the suite). The real call waits for credentials.
- Cost: $0 (no real calls from code yet).

### 2026-10-16: an identity for the app, and the first real run

- IAM → Users → **porch-agent**, no console access, one managed policy: **AmazonBedrockLimitedAccess**.
  Access key created for "application running outside AWS"; `aws configure --profile personal`
  (region us-east-1). Work profiles on the same laptop are never used for this project.
- Check: `aws sts get-caller-identity --profile personal` → `user/porch-agent`. A one-word Converse call
  from Python: 9 tokens, 1.1 s (first call, includes connection setup).
- Lesson learned the hard way: a pasted terminal transcript can carry the secret key with it. Treat a
  leaked key as burned: deactivate, delete, create a new one.

**First full run, one clip** (Pexels "person leaves a parcel at the door", 10 s, Decider 2B on the laptop CPU):

| | Count | Time per answer | Tokens | Cost per answer |
|---|---|---|---|---|
| Decider 2B decided alone (confidence ≥ 0.4) | 9 of 14 events | 9.7–18.6 s on CPU (~0.1 s on a GPU) | – | $0 |
| Bedrock Nova Lite second look (confidence < 0.4) | 5 of 14 events | 1.7–5.8 s (first call slowest) | ~1,050 in / 17 out | ≈ $0.00007 |

- Whole clip on Bedrock: about **$0.0003**. Sending every event would have cost 3× that and taken longer.
- What Nova Lite got right: at 10.7 s the Decider said *log* (34%); Nova Lite looked and said
  "A package is on the doorstep" → **alert**. That is the clip's point, and the Decider alone missed it.
- What it got wrong: two short events at 2.0–2.8 s where the detector saw a *person* (Decider: alert, 24%)
  came back "Nothing is at the door" → ignore. The frame is from the middle of a 0.3 s event, and the prompt
  hides the detector's verdict on purpose (so the model looks instead of parroting). The main 6-second
  approach at 3.1–9.6 s still alerted, so the clip outcome was right, but this is a real failure case for
  the report: a second look on a sliver of an event can be worse than the first.

**Night clip** (Pexels "child in a dark costume at the door at dusk", 8 s): the hardest case in the set.

| Step | Said | Confidence | Time |
|---|---|---|---|
| OpenCV | one *dark* event at 1.4–3.1 s, nothing recognised even after brightening | – | – |
| Decider 2B | **recheck** ("picture not clear enough to trust") | **10%** | 8.5 s (CPU) |
| Nova Lite, shown the brightened frame | "A person is standing at the door, possibly waiting for someone." → **alert** | – | 3.3 s, 972 in / 24 out tokens, ≈ $0.00007 |

This is the pattern working as designed: the Decider knew it did not know, and said so with a low number;
one cheap second look turned a shrug into the right alert. Clip cost: $0.00007.

**"Blurry" clip** (`dropped_delivery.mp4`): turned out to be the same video as the package clip under
another name (identical 14 events, identical token counts), so this run was a free repeat test. Useful
anyway: the second looks gave the **same answers both times** (temperature 0), and Bedrock latency settled
at 1.6–1.8 s per look once warm. Cost: ≈ $0.0003. Lesson: check for duplicate test clips before counting them.

### Running total after 3 clips

| | |
|---|---|
| Bedrock calls | 11 |
| Bedrock cost | ≈ $0.0008 |
| Decider 2B cost | $0 (laptop) |
| GPU hours | 0 (quota pending) |

### Next

- [ ] A truly blurry clip (clips/blurry: fog, water on the lens) through the same path, after asking.
- [ ] Decide whether very short events (< 0.5 s) should skip the second look or send 3 frames instead of 1.
- [ ] Decide whether very short events (< 0.5 s) should skip the second look or send 3 frames instead of 1.
- [ ] Measure: per-clip count of Decider-only decisions vs Bedrock second looks; cost and latency per path.
- [ ] GPU session: run Decider 2B on g4dn.xlarge for a few hours, record ms per decision, stop the instance.
- [ ] Budget alert at $20 in AWS Billing.

## Costs so far

| Item | Amount |
|---|---|
| Bedrock Nova Lite, 1 call | < $0.001 |
| GPU quota request | $0 |
| **Total** | **≈ $0** |
