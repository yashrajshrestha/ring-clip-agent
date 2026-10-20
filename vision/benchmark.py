"""Four ways to decide the same doorbell events, side by side.

    AWS_PROFILE=personal VISION_MODEL=bedrock:amazon.nova-lite-v1:0 DECIDER_URL=http://127.0.0.1:8000 \
      python -m vision.benchmark clip1 clip2 ... > eval/benchmark.md

The events come from the clips' saved timelines (OpenCV runs once, so every
method sees exactly the same events). For each event:

1. rules           the hand-written rule baseline (timeline._decide), no model
2. decider         Strands Decider 2B alone
3. decider+bedrock Decider 2B, and a Bedrock second look only under 40% confidence,
                   which may raise an alert but not silence a person (escalate.may_override)
4. bedrock         the Bedrock vision model on every event (the "just use the big model" baseline)

A clip counts as alerted when any of its events ends as `alert`. Clips whose
kind is `animal` should stay quiet; every other clip should alert. Cost uses the
Nova Lite on-demand price in us-east-1; Decider runs on the laptop for $0.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from . import decider, escalate
from .timeline import _decide

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "web" / "public" / "samples"
PRICE_IN, PRICE_OUT = 0.06 / 1e6, 0.24 / 1e6   # Amazon Nova Lite, USD per token, us-east-1 on-demand
METHODS = ("rules", "decider", "decider+bedrock", "bedrock")


def kinds() -> dict[str, str]:
    out = {}
    for name in ("cc_clips.csv", "stock_clips.csv"):
        for row in csv.DictReader(open(ROOT / "eval" / name)):
            out[row["name"]] = row["kind"]
    return out


def cost(tokens: dict | None) -> float:
    return (tokens["in"] or 0) * PRICE_IN + (tokens["out"] or 0) * PRICE_OUT if tokens else 0.0


def run_clip(clip: str) -> dict:
    tl = json.load(open(SAMPLES / f"{clip}.timeline.json"))
    video = SAMPLES / f"{clip}.mp4"
    events = tl["events"]
    res = {m: {"actions": [], "ms": 0, "usd": 0.0, "calls": 0} for m in METHODS}
    for n, ev in enumerate(events):
        earlier = events[:n]
        res["rules"]["actions"].append(_decide(ev)["action"])

        t = time.perf_counter()
        d = decider.decide(ev, earlier) or _decide(ev)
        decider_ms = (time.perf_counter() - t) * 1000
        res["decider"]["ms"] += decider_ms
        res["decider"]["actions"].append(d["action"])

        # Decider first, Bedrock only when unsure: the decider's time plus any second look.
        final, extra_ms = d, 0.0
        if escalate.should_escalate(d):
            t = time.perf_counter()
            ans = escalate.escalate(video, ev, earlier)
            extra_ms = (time.perf_counter() - t) * 1000
            if ans is not None:
                res["decider+bedrock"]["calls"] += 1
                res["decider+bedrock"]["usd"] += cost(ans.get("tokens"))
                probe = {"label": ev.get("label"), "decision": d}
                escalate.apply_second_look(probe, ans)
                final = probe["decision"]
        res["decider+bedrock"]["ms"] += decider_ms + extra_ms
        res["decider+bedrock"]["actions"].append(final["action"])

        t = time.perf_counter()
        ans = escalate.escalate(video, ev, earlier)
        res["bedrock"]["ms"] += (time.perf_counter() - t) * 1000
        if ans is not None:
            res["bedrock"]["calls"] += 1
            res["bedrock"]["usd"] += cost(ans.get("tokens"))
        res["bedrock"]["actions"].append(ans["action"] if ans else "ignore")
    return {"clip": clip, "events": len(events), "opencv_s": tl.get("process_seconds"), "res": res}


def report(rows: list[dict], kind_of: dict[str, str]) -> str:
    lines = ["# Benchmark: four ways to decide the same doorbell events", "",
             f"{len(rows)} clips, {sum(r['events'] for r in rows)} events (OpenCV ran once; every method saw the same events). "
             "Decider 2B on an Apple M2 Pro laptop (MPS); Bedrock Nova Lite in us-east-1 called from the same laptop.", "",
             "| Method | Visit clips alerted | Pet clips kept quiet | Phone alerts | Bedrock calls | Bedrock cost | Decision time per clip |",
             "|---|---|---|---|---|---|---|"]
    visits = [r for r in rows if kind_of.get(r["clip"]) != "animal"]
    pets = [r for r in rows if kind_of.get(r["clip"]) == "animal"]
    for m in METHODS:
        alerted = lambda r: "alert" in r["res"][m]["actions"]
        alerts = sum(r["res"][m]["actions"].count("alert") for r in rows)
        calls = sum(r["res"][m]["calls"] for r in rows)
        usd = sum(r["res"][m]["usd"] for r in rows)
        sec = sum(r["res"][m]["ms"] for r in rows) / 1000 / len(rows)
        lines.append(f"| {m} | {sum(map(alerted, visits))} / {len(visits)} | {sum(not alerted(r) for r in pets)} / {len(pets)} | "
                     f"{alerts} | {calls} | ${usd:.5f} | {sec:.1f} s |")
    lines += ["", "## Per clip", "", "| Clip | Kind | Events | " + " | ".join(METHODS) + " |",
              "|---|---|---|" + "---|" * len(METHODS)]
    for r in rows:
        cells = [" ".join(a[0].upper() if a == "alert" else a[0] for a in r["res"][m]["actions"]) for m in METHODS]
        lines.append(f"| {r['clip']} | {kind_of.get(r['clip'], '?')} | {r['events']} | " + " | ".join(cells) + " |")
    lines += ["", "Per-clip cells list each event's final action in order: **A** = alert, i = ignore, l = log, r = recheck."]
    return "\n".join(lines) + "\n"


def main() -> None:
    clips = sys.argv[1:]
    if not clips:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    rows = []
    for c in clips:
        rows.append(run_clip(c))
        print(f"done {c}", file=sys.stderr)
    print(report(rows, kinds()))
    json.dump(rows, open(ROOT / "eval" / "benchmark.json", "w"), indent=1)


if __name__ == "__main__":
    main()
