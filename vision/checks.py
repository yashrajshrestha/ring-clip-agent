"""OpenCV 5 quality and event checks for Ring-style doorbell clips.

Every check takes frames already loaded by `load_clip` and returns plain
numbers or small dicts, so the agent layer never touches pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

# Thresholds start as guesses; tune them on the 100 labelled Ring clips.
BLUR_MIN = 60.0          # Laplacian variance below this = blurry
DARK_MAX_MEAN = 40.0     # mean gray below this = too dark
IR_SAT_MAX = 12.0        # mean HSV saturation below this = IR grayscale
MOTION_MIN_AREA = 0.004  # motion blob smaller than this share of frame = noise
DUP_HASH_DIST = 10       # Hamming distance on 64-bit pHash; lower = same scene
OVERLAY_BAND = 0.06      # share of height at top/bottom masked for timestamp/logo


def load_clip(path: str | Path, max_frames: int = 90, stride: int = 3) -> list[np.ndarray]:
    """Read every `stride`-th frame, up to `max_frames`. Uses frame timestamps,
    not a fixed fps, because Ring clips drop frames on weak Wi-Fi."""
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    i = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


def mask_overlays(frame: np.ndarray, extra_boxes: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """Black out the timestamp/logo bands and any privacy-zone rectangles so
    they do not count as motion or edges."""
    out = frame.copy()
    h = out.shape[0]
    band = int(h * OVERLAY_BAND)
    out[:band, :] = 0
    out[h - band:, :] = 0
    for x, y, w, bh in extra_boxes or []:
        out[y:y + bh, x:x + w] = 0
    return out


def undistort(frame: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Approximate wide-angle correction without a calibration run.
    `strength` is k1 in a simple radial model; calibrate per camera if time allows."""
    h, w = frame.shape[:2]
    f = w * 0.8
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    D = np.array([-strength, 0.05, 0, 0], dtype=np.float64)
    return cv2.undistort(frame, K, D)


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def blur_score(frame: np.ndarray) -> float:
    """Variance of the Laplacian. Low = soft or out of focus."""
    return float(cv2.Laplacian(to_gray(frame), cv2.CV_64F).var())


def exposure(frame: np.ndarray) -> dict:
    """Mean brightness plus share of clipped black/white pixels."""
    g = to_gray(frame)
    return {
        "mean": float(g.mean()),
        "black_pct": float((g < 10).mean()),
        "white_pct": float((g > 245).mean()),
    }


def is_ir(frame: np.ndarray) -> bool:
    """Night IR frames come back as 3 identical channels, so saturation ~0."""
    if frame.ndim != 3:
        return True
    s = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(s.mean()) < IR_SAT_MAX


def motion_region(frames: list[np.ndarray]) -> dict:
    """MOG2 background subtraction across the clip. Returns the largest
    moving blob as a share of frame area plus its bounding box."""
    sub = cv2.createBackgroundSubtractorMOG2(history=len(frames), varThreshold=32, detectShadows=False)
    best_area, best_box = 0.0, (0, 0, 0, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # MOG2 marks everything as foreground until it has a background model,
    # so the first few frames only warm it up and are not scored.
    warmup = max(1, min(5, len(frames) // 4))
    for i, f in enumerate(frames):
        m = sub.apply(mask_overlays(f))
        if i < warmup:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c) / (f.shape[0] * f.shape[1])
        if area > best_area:
            best_area, best_box = area, cv2.boundingRect(c)
    return {"area": float(best_area), "box": [int(v) for v in best_box], "has_motion": best_area >= MOTION_MIN_AREA}


def phash(frame: np.ndarray) -> int:
    """64-bit perceptual hash via DCT. Same scene under slight noise = close hash."""
    g = cv2.resize(to_gray(frame), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8]
    med = np.median(d[1:, 1:])
    bits = (d > med).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def is_duplicate(hashes_a: list[int], hashes_b: list[int]) -> bool:
    """Two clips are duplicates when their mid-clip hashes are within DUP_HASH_DIST."""
    if not hashes_a or not hashes_b:
        return False
    return hash_distance(hashes_a[len(hashes_a) // 2], hashes_b[len(hashes_b) // 2]) <= DUP_HASH_DIST


@dataclass
class ClipFacts:
    path: str
    frames: int
    blur: float
    mean_brightness: float
    black_pct: float
    white_pct: float
    ir: bool
    motion_area: float
    motion_box: list[int]
    has_motion: bool
    hashes: list[int]
    flags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(path: str | Path, undistort_frames: bool = True) -> ClipFacts:
    """Run every check on one clip and return the facts the agent reads."""
    frames = load_clip(path)
    if not frames:
        return ClipFacts(str(path), 0, 0.0, 0.0, 1.0, 0.0, False, 0.0, [0, 0, 0, 0], False, [], ["unreadable"])
    if undistort_frames:
        frames = [undistort(f) for f in frames]
    mid = frames[len(frames) // 2]
    exp = exposure(mid)
    blur = blur_score(mask_overlays(mid))
    ir = is_ir(mid)
    motion = motion_region(frames)
    hashes = [phash(frames[i]) for i in (0, len(frames) // 2, len(frames) - 1)]

    flags = []
    if blur < BLUR_MIN:
        flags.append("blurry")
    if exp["mean"] < DARK_MAX_MEAN:
        flags.append("dark")
    if exp["white_pct"] > 0.2:
        flags.append("overexposed")
    if ir:
        flags.append("ir")
    # A clip with fewer than 2 scored frames cannot show motion either way,
    # so mark it short instead of claiming nothing moved.
    if len(frames) < 3:
        flags.append("too_short")
    elif not motion["has_motion"]:
        flags.append("no_motion")

    return ClipFacts(
        path=str(path), frames=len(frames), blur=blur,
        mean_brightness=exp["mean"], black_pct=exp["black_pct"], white_pct=exp["white_pct"],
        ir=ir, motion_area=motion["area"], motion_box=motion["box"], has_motion=motion["has_motion"],
        hashes=hashes, flags=flags,
    )
