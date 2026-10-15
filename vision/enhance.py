"""Recheck: when a picture is too dark or blurry, improve it with OpenCV and
look again before bothering the owner.

First look: gamma lifts the dark tones, CLAHE restores local contrast, a
bilateral filter smooths the grain both of those amplify, and the detector
runs on the result. Second look, only when the first saw nothing (or always,
under the `harder` profile): more frames, each cut into five overlapping crops
so small or distant figures reach the detector larger. None of it can invent
light that never reached the sensor: a truly black frame stays unusable, and
the recheck says so.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .checks import BLUR_MIN, DARK_MAX_MEAN, blur_score, inner, mask_overlays

# Read at call time, so vision/profiles.py can raise them for one analysis.
RECHECK_FRAMES = 8        # frames sampled across the event window on the first look
SECOND_LOOK_FRAMES = 12   # more frames on the second look
ALWAYS_SECOND_LOOK = False  # True: zoomed second look even when the first look found something
ZOOM = 0.6                # crop side as a share of the frame; the detector letterboxes to 640, so a figure shows ~1.7x larger
TARGET_MEAN = 110.0       # gamma aims the mean gray here; measured day footage sits at 80-130
GAMMA_MAX = 3.0           # beyond this the frame is noise, not picture
LABEL_PRIORITY = ("person", "wild", "animal", "vehicle")


def enhance(frame: np.ndarray) -> np.ndarray:
    """Brighten and sharpen a dim frame. Returns a new BGR image."""
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    gamma = 1.0
    if 0 < gray_mean < TARGET_MEAN:
        # mean^(1/g) = target, solved on the 0..1 scale
        gamma = min(GAMMA_MAX, float(np.log(gray_mean / 255.0) / np.log(TARGET_MEAN / 255.0)))
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], np.uint8)
    out = cv2.LUT(frame, lut)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(out, 5, 40, 40)


def zoom_crops(frame: np.ndarray) -> list[np.ndarray]:
    """Four corner crops and a centre crop, each ZOOM of the frame on a side."""
    h, w = frame.shape[:2]
    ch, cw = round(h * ZOOM), round(w * ZOOM)
    origins = [(0, 0), (0, w - cw), (h - ch, 0), (h - ch, w - cw), ((h - ch) // 2, (w - cw) // 2)]
    return [frame[y:y + ch, x:x + cw] for y, x in origins]


def _measure(frame: np.ndarray) -> dict:
    core = inner(mask_overlays(frame))
    return {"blur": round(blur_score(core), 1),
            "bright": round(float(cv2.cvtColor(core, cv2.COLOR_BGR2GRAY).mean()), 1)}


def _usable(m: dict) -> bool:
    return m["blur"] >= BLUR_MIN and m["bright"] >= DARK_MAX_MEAN


def _label(counts: dict) -> str | None:
    return next((g for g in LABEL_PRIORITY if counts.get(g)), None)


def sample_frames(video: str | Path, t_start: float, t_end: float, n: int | None = None,
                  width: int = 640) -> list[np.ndarray]:
    """Up to n frames spread across [t_start, t_end], downscaled to `width`."""
    n = RECHECK_FRAMES if n is None else n
    cap = cv2.VideoCapture(str(video))
    out = []
    for t in np.linspace(t_start, max(t_start, t_end), n):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if ok:
            out.append(cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA))
    cap.release()
    return out


def second_look(video: str | Path, event: dict, detector) -> dict:
    """More frames, enhanced, each cut into zoomed crops for the detector.
    A frame counts a group once however many crops show it."""
    frames = sample_frames(video, event["t_start"], event["t_end"], n=SECOND_LOOK_FRAMES)
    counts: dict[str, int] = {}
    for f in frames:
        seen: set[str] = set()
        for crop in zoom_crops(enhance(f)):
            seen |= {o["group"] for o in detector.detect(crop)}
        for g in seen:
            counts[g] = counts.get(g, 0) + 1
    return {"frames": len(frames), "crops": 5, "objects": counts, "label": _label(counts)}


def recheck(video: str | Path, event: dict, detector=None) -> dict:
    """Enhance the frames of a blurry/dark event and look again.

    Returns what changed: picture quality before and after, whether the
    enhanced picture is usable, what the detector saw, and, when it saw
    nothing (or always under the harder profile), what a zoomed second look
    over more frames found."""
    frames = sample_frames(video, event["t_start"], event["t_end"])
    if not frames:
        return {"frames": 0, "before": None, "after": None, "usable": False, "objects": {}, "label": None}
    enhanced = [enhance(f) for f in frames]
    before = [_measure(f) for f in frames]
    after = [_measure(f) for f in enhanced]
    med = lambda ms, k: round(float(np.median([m[k] for m in ms])), 1)
    before_m = {"blur": med(before, "blur"), "bright": med(before, "bright")}
    after_m = {"blur": med(after, "blur"), "bright": med(after, "bright")}
    counts: dict[str, int] = {}
    if detector is not None:
        for f in enhanced:
            for g in {o["group"] for o in detector.detect(f)}:
                counts[g] = counts.get(g, 0) + 1
    out = {
        "frames": len(frames), "before": before_m, "after": after_m,
        "usable": _usable(after_m), "objects": counts, "label": _label(counts),
    }
    if detector is not None and (out["label"] is None or ALWAYS_SECOND_LOOK):
        out["second_look"] = second_look(video, event, detector)
        out["label"] = out["label"] or out["second_look"]["label"]
    return out
