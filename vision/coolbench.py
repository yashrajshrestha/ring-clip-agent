"""Time the pipeline's own OpenCV work under two OpenCV builds: pip OpenCV 5 and COOL.

    python -m vision.coolbench run --label pip  clip1.mp4 clip2.mp4 ... > pip.json
    PYTHONPATH=/opt/cool/python_3.12/site-packages LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib \
      python -m vision.coolbench run --label cool clip1.mp4 clip2.mp4 ... > cool.json
    python -m vision.coolbench compare pip.json cool.json > eval/cool.md

Run both on the same instance, one after the other, so only the OpenCV build
differs. Each step is the code the pipeline runs (checks, enhance, detector,
the MOG2 motion pass), timed on real frames from the clips; the median of
REPEATS runs is kept. `timeline` is the whole OpenCV pass over each clip
(build_timeline with the detector and no decider), wall clock.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .checks import blur_score, is_ir
from .detector import letterbox
from .enhance import enhance, zoom_crops

REPEATS = 5
FRAMES_PER_CLIP = 30


def _frames(clip: str, n: int = FRAMES_PER_CLIP) -> list[np.ndarray]:
    cap = cv2.VideoCapture(clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * total // n)
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def _motion(frames: list[np.ndarray]) -> None:
    """The timeline's motion pass: MOG2, open, close, contours."""
    sub = cv2.createBackgroundSubtractorMOG2(history=75, varThreshold=32, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    join = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for f in frames:
        m = sub.apply(cv2.GaussianBlur(f, (5, 5), 0))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, join)
        cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def _median_ms(fn, repeats: int = REPEATS) -> float:
    fn()   # warm up (first dnn forward, thread pools)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(times), 2)


def run(label: str, clips: list[str]) -> dict:
    from .timeline import build_timeline, load_detector
    os.environ.pop("DECIDER_URL", None)    # OpenCV only: no decider, no second look
    os.environ.pop("VISION_MODEL", None)
    detector = load_detector()
    frames = [f for c in clips for f in _frames(c)]
    if not frames:
        sys.exit("no frames decoded: does this OpenCV build read mp4?")
    steps = {
        "gray + blur check (Laplacian)": lambda: [blur_score(f) for f in frames],
        "night check (HSV)": lambda: [is_ir(f) for f in frames],
        "motion (MOG2 + morphology + contours)": lambda: _motion(frames),
        "enhance (gamma LUT + CLAHE + bilateral)": lambda: [enhance(f) for f in frames],
        "zoom crops + letterbox": lambda: [letterbox(c) for f in frames[:10] for c in zoom_crops(f)],
    }
    if detector:
        steps["YOLOX-S detector (cv2.dnn), 10 frames"] = lambda: [detector.detect(f) for f in frames[:10]]
    result = {"label": label, "opencv": cv2.__version__, "cv2_file": cv2.__file__,
              "machine": platform.machine(), "cpus": os.cpu_count(), "threads": cv2.getNumThreads(),
              "kleidicv": "kleidicv" in cv2.getBuildInformation().lower(),
              "frames": len(frames), "steps": {}, "timeline": {}}
    for name, fn in steps.items():
        result["steps"][name] = _median_ms(fn)
        print(f"{label} {name}: {result['steps'][name]} ms", file=sys.stderr)
    for c in clips:
        t = time.perf_counter()
        tl = build_timeline(c, detector)
        result["timeline"][Path(c).stem] = {"s": round(time.perf_counter() - t, 2), "events": len(tl["events"])}
        print(f"{label} timeline {Path(c).stem}: {result['timeline'][Path(c).stem]['s']} s", file=sys.stderr)
    return result


def compare(a: dict, b: dict) -> str:
    def speed(x: float, y: float) -> str:
        return f"{x / y:.2f}x" if y else "n/a"
    lines = [f"# OpenCV build comparison: {a['label']} vs {b['label']}", "",
             f"Same machine ({a['machine']}, {a['cpus']} CPUs). {a['label']}: OpenCV {a['opencv']} "
             f"(KleidiCV {'yes' if a['kleidicv'] else 'no'}); {b['label']}: OpenCV {b['opencv']} "
             f"(KleidiCV {'yes' if b['kleidicv'] else 'no'}). {a['frames']} real frames from the clips, "
             f"median of {REPEATS} runs.", "",
             f"| Step | {a['label']} ms | {b['label']} ms | {b['label']} speedup |", "|---|---|---|---|"]
    for name, x in a["steps"].items():
        y = b["steps"].get(name)
        lines.append(f"| {name} | {x} | {y} | {speed(x, y)} |")
    ta = sum(v["s"] for v in a["timeline"].values())
    tb = sum(v["s"] for v in b["timeline"].values())
    lines += ["", f"| Whole OpenCV pass per clip | {a['label']} s | {b['label']} s | {b['label']} speedup | events |",
              "|---|---|---|---|---|"]
    for clip, v in a["timeline"].items():
        w = b["timeline"].get(clip, {})
        lines.append(f"| {clip} | {v['s']} | {w.get('s')} | {speed(v['s'], w.get('s', 0))} | "
                     f"{v['events']} / {w.get('events')} |")
    lines.append(f"| **all clips** | **{ta:.1f}** | **{tb:.1f}** | **{speed(ta, tb)}** | |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "run" and argv[1] == "--label":
        print(json.dumps(run(argv[2], argv[3:]), indent=1))
        return 0
    if len(argv) == 3 and argv[0] == "compare":
        print(compare(json.loads(Path(argv[1]).read_text()), json.loads(Path(argv[2]).read_text())))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
