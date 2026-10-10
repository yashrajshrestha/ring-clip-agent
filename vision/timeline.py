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

from .checks import BLUR_MIN, DARK_MAX_MEAN, MOTION_MIN_AREA, blur_score, exposure, inner, is_ir, mask_overlays

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
PERSON_AREA = 0.03       # rule-stub: peak motion at least this big reads as a person
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


def _box(contour, w: int, h: int) -> list[float]:
    x, y, bw, bh = cv2.boundingRect(contour)
    return [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]


def _decide(event: dict) -> dict:
    """Placeholder for the Bedrock agent: same output shape, labelled so it is
    never mistaken for agent output."""
    kind = event["type"]
    label = event.get("label")
    if kind == "motion" and label == "person":
        dur = event["t_end"] - event["t_start"]
        action, reason = "alert", f"Person detected for {dur:.1f}s"
    elif kind == "motion" and label == "animal":
        action, reason = "ignore", "Animal, not a visitor"
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
SAY_MOTION = {"person": "Person at the door.", "animal": "Animal at the door.", "vehicle": "Vehicle passing."}
LABEL_PRIORITY = ("person", "animal", "vehicle")  # a person with a car behind them is a person


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


def build_timeline(path: str | Path, detector=None) -> dict:
    """`detector` is anything with detect(bgr) -> [{label, group, score, box}],
    normally vision.detector.Detector. None skips object labels."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
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
    ir_state, run_start, prev_bright, prev_fg = None, None, None, None
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
    cap.release()

    events = _motion_events(frames, thumbs, fps, [0] + [idx for idx, _ in marks])
    for idx, kind in marks:
        events.append({"type": kind, "t_start": frames[idx]["t"], "t_end": frames[idx]["t"], "box": None, "peak_area": 0.0})
    for kind, flags in (("blurry", [f["blur"] < BLUR_MIN for f in frames]),
                        ("dark", [f["bright"] < DARK_MAX_MEAN for f in frames])):
        for s, e in _runs(flags, QUALITY_RUN):
            events.append({"type": kind, "t_start": frames[s]["t"], "t_end": frames[e]["t"], "box": None, "peak_area": 0.0})

    events.sort(key=lambda ev: ev["t_start"])
    for n, ev in enumerate(events):
        ev["id"] = n
        if ev["type"] == "motion":
            ev["objects"], ev["label"] = _label(ev, frames)
            ev["say"] = SAY_MOTION.get(ev["label"], SAY["motion"])
        else:
            ev["say"] = SAY[ev["type"]]
        ev["decision"] = _decide(ev)

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


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, out_dir = Path(args[0]), Path(args[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    if "--no-detect" not in argv:
        from .detector import Detector, default_model_path
        if default_model_path().exists():
            detector = Detector()
        else:
            print(f"no model at {default_model_path()}, running without object labels", file=sys.stderr)
    clips = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXT) if src.is_dir() else [src]

    # Merge with any existing index so runs on different folders add up.
    index_path = out_dir / "index.json"
    old = json.loads(index_path.read_text()) if index_path.exists() else []
    names = {c.stem for c in clips}
    index = [row for row in old if row["name"] not in names]
    for clip in clips:
        video = out_dir / f"{clip.stem}.mp4"
        to_browser_mp4(clip, video)
        started = time.perf_counter()
        tl = build_timeline(video, detector)
        tl["clip"], tl["video"] = clip.name, video.name
        tl["process_seconds"] = round(time.perf_counter() - started, 2)
        (out_dir / f"{clip.stem}.timeline.json").write_text(json.dumps(tl))
        index.append({"name": clip.stem, "timeline": f"{clip.stem}.timeline.json",
                      "events": len(tl["events"]), "duration": tl["duration"]})
        print(f"{clip.name}: {len(tl['frames'])} frames, {len(tl['events'])} events, "
              f"{tl['process_seconds']}s", file=sys.stderr)
    index_path.write_text(json.dumps(sorted(index, key=lambda r: r["name"]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
