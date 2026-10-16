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

### Next

- [ ] Programmatic access for the app (an IAM identity for the personal account; the laptop currently
      only holds unrelated work profiles, which must not be used here).
- [ ] First real `VISION_MODEL=bedrock:...` run on the Pexels clips; record ms and tokens per second look.
- [ ] Measure: per-clip count of Decider-only decisions vs Bedrock second looks; cost and latency per path.
- [ ] GPU session: run Decider 2B on g4dn.xlarge for a few hours, record ms per decision, stop the instance.
- [ ] Budget alert at $20 in AWS Billing.

## Costs so far

| Item | Amount |
|---|---|
| Bedrock Nova Lite, 1 call | < $0.001 |
| GPU quota request | $0 |
| **Total** | **≈ $0** |
