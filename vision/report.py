"""Score the timelines against the labelled clips and write a report.

    python -m vision.report [--samples web/public/samples] [--labels eval/real_clips.csv] [--out eval/report.md]

Every labelled clip is a delivery (`left`) or a theft (`taken`), so each one
should raise at least one alert and one object left/taken event. The report
counts hits and misses per clip, alert volume (alert fatigue), quality
events and decision sources, and lists the failures by name: the OpenCV
competition asks for evaluation evidence including failure cases.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

EXPECT_SCENE = {"left", "taken"}


def score_clip(label: dict, tl: dict) -> dict:
    events = tl["events"]
    types = Counter(e["type"] for e in events)
    actions = Counter(e["decision"]["action"] for e in events)
    sources = Counter(e["decision"]["source"] for e in events)
    rechecks = [e for e in events if "recheck" in e]
    return {
        "name": label["name"], "kind": label["kind"], "night": label["night"] == "1",
        "duration": round(tl["duration"], 1), "events": len(events),
        "alerts": actions["alert"], "person_events": sum(1 for e in events if e.get("label") == "person"),
        "scene_changes": types["scene_change"], "quality_events": types["blurry"] + types["dark"],
        "cuts": types["cut"], "rechecks": len(rechecks),
        "rescued": sum(1 for e in rechecks if e["decision"]["action"] != "recheck"),
        "sources": dict(sources),
        "alert_hit": actions["alert"] >= 1,
        "scene_hit": types["scene_change"] >= 1 if label["kind"] in EXPECT_SCENE else None,
        "notes": label.get("notes", ""),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    scene_rows = [r for r in rows if r["scene_hit"] is not None]
    alerts = sorted(r["alerts"] for r in rows)
    return {
        "clips": n, "events": sum(r["events"] for r in rows),
        "alert_events": sum(r["alerts"] for r in rows),  # the only events that wake a language model
        "alert_hits": sum(r["alert_hit"] for r in rows),
        "scene_hits": sum(r["scene_hit"] for r in scene_rows), "scene_expected": len(scene_rows),
        "night_clips": sum(r["night"] for r in rows),
        "night_scene_hits": sum(r["scene_hit"] for r in scene_rows if r["night"]),
        "night_scene_expected": sum(1 for r in scene_rows if r["night"]),
        "alerts_per_clip_median": alerts[n // 2] if n else 0, "alerts_per_clip_max": max(alerts) if alerts else 0,
        "quality_events": sum(r["quality_events"] for r in rows),
        "rechecks": sum(r["rechecks"] for r in rows), "rescued": sum(r["rescued"] for r in rows),
        "sources": dict(sum((Counter(r["sources"]) for r in rows), Counter())),
    }


def load(samples: Path, labels: Path) -> list[dict]:
    rows = []
    for label in csv.DictReader(open(labels, newline="")):
        p = samples / f"{label['name']}.timeline.json"
        if p.exists():
            rows.append(score_clip(label, json.loads(p.read_text())))
    return rows


def markdown(rows: list[dict], s: dict) -> str:
    out = ["# Evaluation report", "",
           f"{s['clips']} labelled clips (every one a delivery or a theft, so each should alert at least once "
           f"and detect an object left or taken).", "",
           "## Summary", "",
           "| Measure | Result |", "|---|---|",
           f"| Clips with at least one alert | {s['alert_hits']} / {s['clips']} |",
           f"| Events that would wake a language model (alerts) | {s['alert_events']} / {s['events']} |",
           f"| Object left/taken detected | {s['scene_hits']} / {s['scene_expected']} |",
           f"| Object left/taken detected at night | {s['night_scene_hits']} / {s['night_scene_expected']} |",
           f"| Alerts per clip, median (max) | {s['alerts_per_clip_median']} ({s['alerts_per_clip_max']}) |",
           f"| Blurry/dark events | {s['quality_events']} |",
           f"| Rechecks, of which rescued by enhancement | {s['rechecks']}, {s['rescued']} |",
           f"| Decision sources | {', '.join(f'{k}: {v}' for k, v in sorted(s['sources'].items()))} |", "",
           "## Per clip", "",
           "| Clip | Kind | Night | Events | Alerts | Person | Left/taken | Quality | Cuts | Result |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        result = ("ok" if r["alert_hit"] and r["scene_hit"] else
                  "no left/taken event" if r["alert_hit"] else "NO ALERT")
        out.append(f"| {r['name']} | {r['kind']} | {'yes' if r['night'] else ''} | {r['events']} | {r['alerts']} | "
                   f"{r['person_events']} | {r['scene_changes']} | {r['quality_events']} | {r['cuts']} | {result} |")
    misses = [r for r in rows if not (r["alert_hit"] and r["scene_hit"])]
    out += ["", "## Failure cases", ""]
    if not misses:
        out.append("None.")
    for r in misses:
        what = "no alert at all" if not r["alert_hit"] else "the package being left or taken was not detected"
        out.append(f"- **{r['name']}** ({r['kind']}{', night' if r['night'] else ''}): {what}. {r['notes']}")
    out += ["", "## Known limitations", "",
            "- The object detector (YOLOX-S, COCO classes) has no `box`/`package` class, so left/taken relies on "
            "comparing the doorstep before and after a visit; small parcels and night footage are the weak cases.",
            "- A person and a parked car can both appear in a frame; the label priority picks the person.",
            "- Alert volume: edited clips with many cuts raise many short person events; a real camera feed does not cut.",
            "- Decider 2B on a laptop CPU takes about 10 s per decision when idle; the rule stub decides when it is off.", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, default=Path("web/public/samples"))
    ap.add_argument("--labels", type=Path, default=Path("eval/real_clips.csv"))
    ap.add_argument("--out", type=Path, default=Path("eval/report.md"))
    args = ap.parse_args()
    rows = load(args.samples, args.labels)
    s = summarize(rows)
    args.out.write_text(markdown(rows, s))
    args.out.with_suffix(".json").write_text(json.dumps({"summary": s, "clips": rows}, indent=1))
    print(f"{s['alert_hits']}/{s['clips']} clips alerted, {s['scene_hits']}/{s['scene_expected']} left/taken detected "
          f"({s['night_scene_hits']}/{s['night_scene_expected']} at night); wrote {args.out}")


if __name__ == "__main__":
    main()
