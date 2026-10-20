"""Per-frame timeline for the React player: motion boxes, quality signals and
narrated events, keyed by video time so overlays stay in sync with playback.

    python -m vision.timeline <clip-or-folder> <out-dir> [--no-detect]

Writes a browser-safe H.264 copy of each clip, `<name>.timeline.json`, and
`index.json`. OpenCV runs on the transcoded copy so the boxes match what plays.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import decider, escalate
from .checks import BLUR_MIN, DARK_MAX_MEAN, MOTION_MIN_AREA, blur_score, exposure, inner, is_ir, mask_overlays
from .enhance import recheck
from .pets import MODEL_FILE as PET_MODEL, identify_event, load_pets

ANALYSIS_WIDTH = 640     # frames are downscaled to this before analysis; boxes are fractions
WARMUP_FRAMES = 5        # MOG2 needs a few frames before its foreground means anything
MOTION_ON = 3            # frames of motion before an event starts
MOTION_OFF = 6           # quiet frames before it ends
MERGE_GAP = 1.5          # seconds; MOG2 absorbs someone standing still, so rejoin the halves
QUALITY_RUN = 5          # frames a quality problem must last to be narrated
GLOBAL_AREA = 0.5        # foreground bigger than this is a lighting change, not an object
BRIGHT_JUMP = 25.0       # mean-gray jump that counts as a lighting change
SCENE_DIFF_MIN = 0.003   # share of frame changed after motion = object left or taken
CUT_CELL_DELTA = 30      # a 16x9 cell whose colour moves this much has changed
CUT_QUAD_SHARE = 0.25    # every quadrant must change this much; measured: edits 0.34+, footage 0.06 max
DETECT_HZ = 2            # YOLOX runs this many times a second while motion is on screen
IR_HOLD_S = 5.0         # seconds a day/night reading must hold without a frame jump
STILL_SHARE = 0.998      # share of pixels unchanged frame to frame = a frozen still; measured: end cards 1.0, night footage 0.99 max
REPAINT_GAP_S = 0.5      # a quality run starting this soon after a cut or mode switch may be an edit
PERSON_AREA = 0.03       # rule-stub: peak motion at least this big reads as a person
ZOOM_MOTION = False      # harder profile: also run the detector on a crop around the moving region
ZOOM_GROW = 2.0          # crop is this many times the motion box on each side...
ZOOM_MIN_SHARE = 0.4     # ...and at least this share of the frame, so the crop is not tiny
ZOOM_DARK_BRIGHT = 60.0  # zoomed detection runs on its own below this mean gray...
ZOOM_IR_BRIGHT = 100.0   # ...or in IR night mode below this; measured IR night porches sit at 70-90
TRACK_MIN_S = 1.0        # a person-sized blob that moves steadily this long reads as a figure...
TRACK_MIN_MOVE = 0.08    # ...when its centre travels at least this share of the frame...
ANIMAL_LINGER_S = 12.0   # an animal at the door this long is not passing by: it may be trying to get in or be stuck
FIGURE_MIN_ASPECT = 0.5  # ...and its box is not long and low (upper-quartile height / width over the event).
                         # Measured: a dog crossing a doorway 0.07, a courier bending at night 1.12, a thief 0.92.
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi"}


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_browser_mp4(src: Path, dst: Path) -> None:
    """Re-encode to H.264 yuv420p with the index up front so <video> can play and seek it."""
    subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-movflags", "+faststart", str(dst)],
        check=True,
    )


def zoomed_detect(detector, frame: np.ndarray, box: list[float]) -> list[dict]:
    """Run the detector on a crop around the motion box and map the boxes back
    to frame fractions. The detector letterboxes to 640, so a figure that fills
    a fifth of the frame arrives two to three times larger this way."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = box
    cw, ch = min(1.0, max(bw * ZOOM_GROW, ZOOM_MIN_SHARE)), min(1.0, max(bh * ZOOM_GROW, ZOOM_MIN_SHARE))
    x0 = min(max(bx + bw / 2 - cw / 2, 0.0), 1.0 - cw)
    y0 = min(max(by + bh / 2 - ch / 2, 0.0), 1.0 - ch)
    crop = frame[round(y0 * h):round((y0 + ch) * h), round(x0 * w):round((x0 + cw) * w)]
    if crop.size == 0:
        return []
    out = []
    for o in detector.detect(crop):
        ox, oy, ow, oh = o["box"]
        out.append({**o, "box": [round(x0 + ox * cw, 4), round(y0 + oy * ch, 4), round(ow * cw, 4), round(oh * ch, 4)],
                    "zoomed": True})
    return out


def _box(contour, w: int, h: int) -> list[float]:
    x, y, bw, bh = cv2.boundingRect(contour)
    return [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]


def _decide(event: dict) -> dict:
    """Rule fallback when Strands Decider is off or unreachable: same output
    shape, labelled so it is never mistaken for model output."""
    kind = event["type"]
    label = event.get("label")
    if kind == "motion" and label == "person":
        dur = event["t_end"] - event["t_start"]
        action, reason = "alert", f"Person detected for {dur:.1f}s"
    elif kind == "motion" and label == "figure":
        dur = event["t_end"] - event["t_start"]
        action, reason = "alert", f"Person-sized shape moved steadily for {dur:.1f}s"
    elif kind == "motion" and label == "wild":
        action, reason = "alert", "Wild animal (such as a bear) at the door"
    elif kind == "motion" and label == "animal" and (event.get("pet") or {}).get("pet"):
        name, dur = event["pet"]["pet"], event["t_end"] - event["t_start"]
        if event.get("lingering"):
            action, reason = "alert", f"{name} has been waiting at the door for {dur:.0f}s"
        else:
            action, reason = "log", f"{name}, your dog, is at the door"
    elif kind == "motion" and label == "animal" and event.get("lingering"):
        dur = event["t_end"] - event["t_start"]
        action, reason = "alert", f"An animal has stayed at the door for {dur:.0f}s"
    elif kind == "motion" and label == "animal":
        action, reason = "ignore", "Animal passing, not a visitor"
    elif kind == "motion" and label == "vehicle":
        action, reason = "ignore", "Vehicle passing, nobody at the door"
    elif kind == "motion":
        dur = event["t_end"] - event["t_start"]
        if event["peak_area"] >= PERSON_AREA and dur >= 1.0:
            action, reason = "alert", f"Person-sized motion for {dur:.1f}s"
        else:
            action, reason = "ignore", "Small or brief motion, likely noise"
    elif kind == "scene_change":
        action, reason = "alert", "Something was left or taken near the door"
    elif kind in ("blurry", "dark") and event.get("recheck") is not None:
        rc = event["recheck"]
        if rc["label"] == "person" and rc.get("second_look"):
            action, reason = "alert", "Person found after zooming in over more frames"
        elif rc["usable"] and rc["label"] == "person":
            action, reason = "alert", "Person visible once the picture was enhanced"
        elif rc["usable"]:
            action, reason = "log", "Picture usable after enhancement, nothing recognised"
        else:
            fix = "add a light" if kind == "dark" else "clean or refocus the lens"
            action, reason = "recheck", f"Still too {kind} after enhancement, {fix}"
    elif kind in ("blurry", "dark"):
        action, reason = "recheck", f"Image too {kind} to trust, check the camera"
    elif kind == "cut":
        action, reason = "log", "Video cut to a different view"
    else:
        action, reason = "log", "Camera lighting mode changed"
    return {"action": action, "reason": reason, "source": "rule-stub"}


SAY = {
    "motion": "Motion at the door.",
    "scene_change": "Object left or removed.",
    "night_mode": "Night mode on.",
    "day_mode": "Day mode on.",
    "blurry": "Image is blurry.",
    "dark": "Image is too dark.",
    "cut": "Camera view changed.",
}
SAY_MOTION = {"person": "Person at the door.", "animal": "Animal at the door.", "vehicle": "Vehicle passing.",
              "wild": "Wild animal at the door.",
              "figure": "Person-sized movement at the door."}
LABEL_PRIORITY = ("person", "wild", "animal", "vehicle")  # a person with a car behind them is a person


def _runs(flags: list[bool], min_len: int) -> list[tuple[int, int]]:
    """(start, end) index pairs where flags stay True for at least min_len frames."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def _edited_still(s: int, e: int, stills: list[float], repaints: list[int], fps: float) -> bool:
    """True when frames s..e are an end card or title, not camera footage: the
    run starts right after a cut or mode switch and the picture is frozen.
    Night footage can be 99% still, so both conditions must hold."""
    after_repaint = any(0 <= s - idx <= round(fps * REPAINT_GAP_S) for idx in repaints)
    return after_repaint and float(np.median(stills[s:e + 1])) >= STILL_SHARE


def _mostly_inside(run: tuple[int, int], others: list[tuple[int, int]], share: float = 0.5) -> bool:
    """True when at least `share` of the frames of `run` fall inside `others`."""
    s, e = run
    covered = sum(max(0, min(e, oe) - max(s, os_) + 1) for os_, oe in others)
    return covered / (e - s + 1) >= share


def build_timeline(path: str | Path, detector=None, progress=None) -> dict:
    """`detector` is anything with detect(bgr) -> [{label, group, score, box}],
    normally vision.detector.Detector. None skips object labels."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    report = progress or (lambda stage, done, total: None)   # progress(stage, done, total) for the drop box bar
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # Close small gaps so one person (head, torso, legs) stays one blob.
    join = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def new_sub():
        return cv2.createBackgroundSubtractorMOG2(history=int(fps * 5), varThreshold=32, detectShadows=False)

    sub = new_sub()
    warm_until = WARMUP_FRAMES
    frames: list[dict] = []
    thumbs: list[np.ndarray] = []
    marks: list[tuple[int, str]] = []   # (frame index, night_mode | day_mode | cut)
    stills: list[float] = []            # share of pixels unchanged since the previous frame
    ir_state, run_start, prev_bright, prev_fg, prev_core = None, None, None, None, None
    cell_hist: deque = deque(maxlen=max(1, round(fps / 2)) + 1)
    cut_cooldown = 0
    det_every = max(1, round(fps / DETECT_HZ))
    last_det, last_objs = -det_every, []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if i > 0 and t <= frames[-1]["t"]:
            t = i / fps  # some backends report 0 or repeat; fall back to frame index
        scale = ANALYSIS_WIDTH / frame.shape[1]
        small = cv2.resize(frame, (ANALYSIS_WIDTH, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        masked = mask_overlays(small)

        # Judge lighting on background pixels only, so a jacket or a body close
        # to the lens does not look like a camera mode switch.
        bg_px = None
        if prev_fg is not None:
            keep = cv2.dilate(prev_fg, join) == 0
            bg_px = keep if keep.mean() > 0.3 else None
        ir_raw = is_ir(small, bg_px)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        bright = float((gray[bg_px] if bg_px is not None else gray).mean())
        # Live footage always flickers a little (sensor and compression noise);
        # an end card or title graphic does not change at all.
        core = inner(gray).astype(np.int16)
        stills.append(float((core == prev_core).mean()) if prev_core is not None else 0.0)
        prev_core = core

        # Cut, zoom or dissolve: every quadrant changes against half a second
        # ago. A person at the lens fills one region; an edit replaces them all.
        cells = cv2.resize(masked, (16, 9), interpolation=cv2.INTER_AREA).astype(np.int16)
        cut = False
        if len(cell_hist) == cell_hist.maxlen and i >= cut_cooldown:
            changed = np.abs(cells - cell_hist[0]).max(axis=2) > CUT_CELL_DELTA
            quads = (changed[:4, :8], changed[:4, 8:], changed[5:, :8], changed[5:, 8:])
            cut = min(q.mean() for q in quads) > CUT_QUAD_SHARE
            if cut:
                cut_cooldown = i + cell_hist.maxlen
        cell_hist.append(cells)

        # A real IR switch repaints the frame (brightness jump or full-frame
        # change), so take it at once. Without that evidence the reading must
        # hold for IR_HOLD_S: a coloured jacket, or a person the background
        # model has absorbed, must not flip the mode.
        reset = False
        jump = prev_bright is not None and abs(bright - prev_bright) > BRIGHT_JUMP
        if ir_state is None:
            ir_state = ir_raw
        elif ir_raw != ir_state:
            run_start = i if run_start is None else run_start
            if cut or jump or i - run_start + 1 >= int(fps * IR_HOLD_S):
                ir_state = ir_raw
                marks.append((i if (cut or jump) else run_start, "night_mode" if ir_state else "day_mode"))
                run_start, reset = None, True
        else:
            run_start = None
        # Any of these repaint the frame; restart the background model
        # instead of reporting one huge moving object.
        if not reset and (cut or jump):
            reset = True
            if cut:
                marks.append((i, "cut"))
        if reset:
            sub = new_sub()
            warm_until = i + WARMUP_FRAMES
        prev_bright = bright
        ir = ir_state

        fg = sub.apply(masked)
        prev_fg = fg if i >= warm_until else None
        box, area = None, 0.0
        if i >= warm_until:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, join)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c) / (sw * sh)
                if MOTION_MIN_AREA <= area <= GLOBAL_AREA:
                    box = _box(c, sw, sh)
                else:
                    area = 0.0 if area < MOTION_MIN_AREA else area

        # YOLOX only while something moves, and only DETECT_HZ times a second:
        # it costs ~0.3 s a frame on CPU. Frames in between reuse the last result.
        objs: list[dict] = []
        if detector is not None and box is not None:
            if i - last_det >= det_every:
                last_objs, last_det = detector.detect(small), i
                # Small figures in the dark need the zoom; under the harder profile every motion gets it.
                if ZOOM_MOTION or bright < ZOOM_DARK_BRIGHT or (ir and bright < ZOOM_IR_BRIGHT):
                    last_objs = last_objs + zoomed_detect(detector, small, box)
            objs = last_objs
        else:
            last_det = -det_every

        frames.append({
            "t": round(t, 3), "box": box, "area": round(area, 4),
            "blur": round(blur_score(inner(small)), 1), "bright": round(bright, 1), "ir": ir,
            "objs": objs,
        })
        thumbs.append(cv2.resize(masked, (160, round(160 * sh / sw)), interpolation=cv2.INTER_AREA))
        i += 1
        if i % 15 == 0:
            report("analyse", i, max(total_frames, i))
    cap.release()
    report("analyse", i, i)

    events = _motion_events(frames, thumbs, fps, [0] + [idx for idx, _ in marks])
    for idx, kind in marks:
        events.append({"type": kind, "t_start": frames[idx]["t"], "t_end": frames[idx]["t"], "box": None, "peak_area": 0.0})
    repaints = [idx for idx, _ in marks]
    dark_runs = [r for r in _runs([f["bright"] < DARK_MAX_MEAN for f in frames], QUALITY_RUN)
                 if not _edited_still(*r, stills, repaints, fps)]
    blur_runs = [r for r in _runs([f["blur"] < BLUR_MIN for f in frames], QUALITY_RUN)
                 if not _edited_still(*r, stills, repaints, fps)]
    # Darkness flattens every edge, so a dark stretch scores blurry as well:
    # one problem, one event, and the recheck brightens it once.
    blur_runs = [r for r in blur_runs if not _mostly_inside(r, dark_runs)]
    for kind, runs in (("dark", dark_runs), ("blurry", blur_runs)):
        for s, e in runs:
            events.append({"type": kind, "t_start": frames[s]["t"], "t_end": frames[e]["t"], "box": None, "peak_area": 0.0})

    events.sort(key=lambda ev: ev["t_start"])
    known_pets = load_pets() if detector is not None and PET_MODEL.exists() else {}   # vision/pets.py
    for n, ev in enumerate(events):
        report("decide", n, len(events))
        ev["id"] = n
        if ev["type"] == "motion":
            ev["objects"], ev["label"] = _label(ev, frames)
            ev["track"] = _track(ev, frames)
            ev["aspect"] = _aspect(ev, frames, height / width if width else 1.0)
            if ev["label"] is None and _is_figure(ev):
                ev["label"] = "figure"
            ev["lingering"] = ev["label"] == "animal" and ev["t_end"] - ev["t_start"] >= ANIMAL_LINGER_S
            if ev["label"] == "animal" and detector is not None and known_pets:
                ev["pet"] = identify_event(path, ev, detector, known_pets)
            ev["say"] = SAY_MOTION.get(ev["label"], SAY["motion"])
        else:
            ev["say"] = SAY[ev["type"]]
        ev["decision"] = decider.decide(ev, events[:n]) or _decide(ev)
        # Perception-decision-action loop: a recheck improves the picture with
        # OpenCV and decides again on what the better picture shows.
        if ev["decision"]["action"] == "recheck" and ev["type"] in ("blurry", "dark"):
            ev["recheck"] = recheck(path, ev, detector)
            ev["first_decision"] = ev["decision"]
            ev["decision"] = decider.decide(ev, events[:n]) or _decide(ev)
        # Model routing: a low-confidence decider answer sends one enhanced
        # frame to a vision model, whose answer replaces it, except that it
        # cannot silence an alert where a person was seen (escalate.may_override).
        if escalate.should_escalate(ev["decision"]):
            answer = escalate.escalate(path, ev, events[:n])
            if answer is not None:
                escalate.apply_second_look(ev, answer)

    return {
        "width": width, "height": height, "fps": round(fps, 3),
        "duration": frames[-1]["t"] if frames else 0.0,
        "thresholds": {"blur_min": BLUR_MIN, "dark_max": DARK_MAX_MEAN},
        "detector": None if detector is None else {"model": "yolox_s_2022nov", "hz": DETECT_HZ},
        "frames": frames, "events": events,
    }


def _overlaps(obj_box: list[float], motion_box: list[float] | None, group: str = "person",
              min_share: float = 0.3, min_fill: float = 0.25) -> bool:
    """True when the boxes share at least min_share of the smaller one. Smaller,
    because a waving arm is a small motion box inside a large person box.
    Vehicles and animals move as a whole, so they must also fill min_fill of
    the motion box; a parked car inside a huge close-up blob does not count."""
    if motion_box is None:
        return False
    ox, oy, ow, oh = obj_box
    mx, my, mw, mh = motion_box
    iw = max(0.0, min(ox + ow, mx + mw) - max(ox, mx))
    ih = max(0.0, min(oy + oh, my + mh) - max(oy, my))
    smaller = min(ow * oh, mw * mh)
    if smaller <= 0 or iw * ih / smaller < min_share:
        return False
    return group == "person" or ow * oh >= min_fill * mw * mh


def _track(event: dict, frames: list[dict]) -> dict:
    """How the motion box moved during the event: net displacement of its
    centre and the length of its path, as shares of the frame."""
    centres = [(f["box"][0] + f["box"][2] / 2, f["box"][1] + f["box"][3] / 2)
               for f in frames if f["box"] is not None and event["t_start"] <= f["t"] <= event["t_end"]]
    if len(centres) < 2:
        return {"frames": len(centres), "net": 0.0, "path": 0.0}
    net = ((centres[-1][0] - centres[0][0]) ** 2 + (centres[-1][1] - centres[0][1]) ** 2) ** 0.5
    path = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(centres, centres[1:]))
    return {"frames": len(centres), "net": round(net, 4), "path": round(path, 4)}


def _aspect(event: dict, frames: list[dict], frame_ratio: float) -> float:
    """Upper-quartile height/width (in pixels) of the motion box over the event. One frame can
    mislead (a person bending down is wide for a moment); a dog stays long and low throughout."""
    asp = [f["box"][3] * frame_ratio / f["box"][2] for f in frames
           if f["box"] is not None and f["box"][2] > 0 and event["t_start"] <= f["t"] <= event["t_end"]]
    return round(float(np.percentile(asp, 75)), 2) if asp else 0.0


def _is_figure(event: dict) -> bool:
    """A person-sized blob that lasts and travels steadily: the detector could
    not name it (small, dark, blurred, behind a railing), but the motion says
    person. Flicker and shadows fail the duration or the travel test."""
    dur = event["t_end"] - event["t_start"]
    return (event["peak_area"] >= PERSON_AREA and dur >= TRACK_MIN_S
            and event["track"]["net"] >= TRACK_MIN_MOVE
            and event.get("aspect", 0.0) >= FIGURE_MIN_ASPECT)


def _label(event: dict, frames: list[dict]) -> tuple[dict, str | None]:
    """Count detector groups seen in the moving region during an event and pick
    one by priority. A parked car behind the visitor must not label the event."""
    counts: dict[str, int] = {}
    for f in frames:
        if event["t_start"] <= f["t"] <= event["t_end"]:
            for o in {o["group"] for o in f["objs"] if _overlaps(o["box"], f["box"], o["group"])}:
                counts[o] = counts.get(o, 0) + 1
    return counts, next((g for g in LABEL_PRIORITY if counts.get(g)), None)


def _motion_events(frames: list[dict], thumbs: list[np.ndarray], fps: float, starts: list[int]) -> list[dict]:
    """Group per-frame boxes into motion events with on/off hysteresis, then
    compare each segment's quiet start with its quiet end to catch objects
    left or taken. `starts` are segment start indices (cuts, mode switches)."""
    bounds = sorted(set(starts)) + [len(frames)]
    seg_of = np.searchsorted(bounds, np.arange(len(frames)), side="right") - 1
    events, on_count, off_count, cur = [], 0, 0, None
    for i, f in enumerate(frames):
        if i in bounds and cur is not None:  # never let an event span a cut
            events.append(cur)
            cur, on_count, off_count = None, 0, 0
        if f["box"] is not None:
            on_count, off_count = on_count + 1, 0
            if cur is None and on_count >= MOTION_ON:
                start = i - MOTION_ON + 1
                cur = {"s": start, "e": i, "peak": f["area"], "box": f["box"]}
            elif cur is not None:
                cur["e"] = i
                if f["area"] > cur["peak"]:
                    cur["peak"], cur["box"] = f["area"], f["box"]
        else:
            on_count = 0
            off_count += 1
            if cur is not None and off_count >= MOTION_OFF:
                events.append(cur)
                cur = None
    if cur is not None:
        events.append(cur)

    merged: list[dict] = []
    for ev in events:
        if (merged and seg_of[ev["s"]] == seg_of[merged[-1]["e"]]
                and frames[ev["s"]]["t"] - frames[merged[-1]["e"]]["t"] <= MERGE_GAP):
            last = merged[-1]
            last["e"] = ev["e"]
            if ev["peak"] > last["peak"]:
                last["peak"], last["box"] = ev["peak"], ev["box"]
        else:
            merged.append(ev)
    events = merged

    out = [{"type": "motion", "t_start": frames[ev["s"]]["t"], "t_end": frames[ev["e"]]["t"],
            "box": ev["box"], "peak_area": round(ev["peak"], 4)} for ev in events]

    # One diff per segment, first quiet window vs last quiet window. Someone
    # who stood still mid-visit has left by the end; a package has not.
    win, min_quiet = max(3, int(fps)), max(3, int(fps / 2))
    for s0, s1 in zip(bounds, bounds[1:]):
        seg = [ev for ev in events if s0 <= ev["s"] < s1]
        if not seg:
            continue
        b1, a0 = seg[0]["s"], seg[-1]["e"] + MOTION_OFF + 1
        n_before, n_after = min(win, b1 - (s0 + WARMUP_FRAMES)), min(win, s1 - a0)
        # The clip starting mid-visit (Ring wake lag) leaves no empty-porch
        # view to compare with; say nothing rather than guess.
        if n_before < min_quiet or n_after < min_quiet:
            continue
        if abs(frames[b1 - 1]["bright"] - frames[s1 - 1]["bright"]) > BRIGHT_JUMP:
            continue  # exposure drifted; a diff would only see that
        before = np.median(np.stack(thumbs[b1 - n_before:b1]), axis=0).astype(np.uint8)
        after = np.median(np.stack(thumbs[s1 - n_after:s1]), axis=0).astype(np.uint8)
        # Max over colour channels: a brown box on a gray porch can be close in
        # brightness but far apart in colour.
        delta = cv2.absdiff(before, after).max(axis=2)
        diff = cv2.threshold(cv2.GaussianBlur(delta, (5, 5), 0), 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        th, tw = diff.shape
        if cv2.contourArea(c) / (tw * th) >= SCENE_DIFF_MIN:
            t_left = frames[seg[-1]["e"]]["t"]
            out.append({"type": "scene_change", "t_start": t_left, "t_end": t_left,
                        "box": _box(c, tw, th), "peak_area": 0.0})
    return out


def load_detector():
    """The YOLOX detector when its model file is present, else None (no object labels)."""
    from .detector import Detector, default_model_path
    if default_model_path().exists():
        return Detector()
    print(f"no model at {default_model_path()}, running without object labels", file=sys.stderr)
    return None


def process_clip(clip: Path, out_dir: Path, detector=None, profile: str = "normal", progress=None) -> dict:
    """Transcode one clip, build and write its timeline, update index.json.
    Returns the index row. Used by the CLI and by the inbox watcher.
    `profile` picks how hard OpenCV looks (see vision/profiles.py)."""
    from .profiles import active
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{clip.stem}.mp4"
    if progress:
        progress("convert", 0, 1)
    to_browser_mp4(clip, video)
    started = time.perf_counter()
    with active(profile, detector):
        tl = build_timeline(video, detector, progress)
    tl["clip"], tl["video"], tl["profile"] = clip.name, video.name, profile
    tl["process_seconds"] = round(time.perf_counter() - started, 2)
    (out_dir / f"{clip.stem}.timeline.json").write_text(json.dumps(tl))
    row = {"name": clip.stem, "timeline": f"{clip.stem}.timeline.json",
           "events": len(tl["events"]), "duration": tl["duration"]}
    # Merge with the existing index so runs on different folders add up.
    index_path = out_dir / "index.json"
    index = [r for r in (json.loads(index_path.read_text()) if index_path.exists() else []) if r["name"] != clip.stem]
    index.append(row)
    index_path.write_text(json.dumps(sorted(index, key=lambda r: r["name"]), indent=2))
    print(f"{clip.name}: {len(tl['frames'])} frames, {len(tl['events'])} events, {tl['process_seconds']}s",
          file=sys.stderr)
    return row


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, out_dir = Path(args[0]), Path(args[1])
    detector = None if "--no-detect" in argv else load_detector()
    clips = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT) if src.is_dir() else [src]
    for clip in clips:
        process_clip(clip, out_dir, detector)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
