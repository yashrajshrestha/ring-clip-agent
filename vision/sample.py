"""Synthetic porch clip for demos and tests until real Ring clips exist.

    python -m vision.sample clips/sample_porch.mp4

20 s at 15 fps, 1280x720: a person walks in, leaves a package, walks out;
the camera switches to night IR at 12 s, smears at 15-17 s, and a second
person crosses at night. Timestamp and logo overlays sit in the masked bands.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 1280, 720, 15, 20


def _background(rng: np.random.Generator) -> np.ndarray:
    bg = np.zeros((H, W, 3), np.float32)
    bg[:] = (120, 125, 130)
    bg[int(H * 0.72):] = (150, 155, 160)                    # porch floor
    cv2.rectangle(bg, (500, 110), (760, int(H * 0.72)), (60, 85, 125), -1)   # door
    cv2.rectangle(bg, (300, 200), (420, 330), (90, 110, 120), -1)            # window
    # Fine texture (brick, wood grain) so a sharp frame scores like a real one.
    tex = cv2.GaussianBlur(rng.normal(0, 40, (H, W)).astype(np.float32), (0, 0), 0.8)
    return np.clip(bg + tex[:, :, None], 0, 255)


def _person_x(t: float, t0: float, t_stop: float, t_go: float, t1: float) -> float | None:
    if t < t0 or t > t1:
        return None
    if t < t_stop:
        return -150 + (520 + 150) * (t - t0) / (t_stop - t0)
    if t < t_go:
        return 520
    return 520 + (1400 - 520) * (t - t_go) / (t1 - t_go)


def _draw_person(img: np.ndarray, x: float, colour) -> None:
    x = int(x)
    cv2.rectangle(img, (x, 260), (x + 120, 590), colour, -1)
    cv2.circle(img, (x + 60, 215), 42, colour, -1)


def render(out: Path) -> None:
    rng = np.random.default_rng(7)
    bg = _background(rng)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        t = i / FPS
        img = bg.copy()
        if t >= 6.0:
            cv2.rectangle(img, (600, 545), (730, 625), (70, 130, 190), -1)   # package stays
        x = _person_x(t, 2.0, 5.0, 6.0, 9.0)
        if x is not None:
            _draw_person(img, x, (50, 50, 150))
        x2 = _person_x(t, 17.5, 18.3, 18.3, 19.5)
        if x2 is not None:
            _draw_person(img, x2, (40, 40, 40))
        img = np.clip(img + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)

        if t >= 12.0:                                   # night IR: gray, darker
            g = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 0.75).astype(np.uint8)
            img = cv2.merge([g, g, g])
        if 15.0 <= t < 17.0:                            # rain smear on the lens
            img = cv2.GaussianBlur(img, (0, 0), 6)

        cv2.putText(img, f"2026-10-09 07:00:{int(t):02d}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, "RING", (W - 110, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(img)
    writer.release()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    render(Path(sys.argv[1]))
