"""Analysis profiles: how hard OpenCV looks at a clip.

`normal` is the everyday pass. `harder` is the second opinion the playground's
"Look harder" button asks for: a less strict detector, twice the detector
rate, stronger brightening, more frames per recheck, and the zoomed second
look on every recheck, not only when the first look saw nothing. It costs
roughly 2-4x the time and finds small, dim or partly hidden figures the normal
pass misses; it also admits a few more false positives, which is why it is a
button and not the default.
"""
from __future__ import annotations

from contextlib import contextmanager

PROFILES: dict[str, dict] = {
    "normal": {},
    "harder": {
        "detector_conf": 0.35,              # the zoo default; normal uses 0.5 (bags scored "person 0.41")
        "timeline.DETECT_HZ": 4,
        "timeline.ZOOM_MOTION": True,       # detector also runs on a crop around the moving region
        "enhance.TARGET_MEAN": 130.0,
        "enhance.GAMMA_MAX": 4.0,
        "enhance.RECHECK_FRAMES": 16,
        "enhance.SECOND_LOOK_FRAMES": 24,
        "enhance.ALWAYS_SECOND_LOOK": True,
    },
}


@contextmanager
def active(name: str, detector=None):
    """Apply a profile's settings for the duration of one analysis, then restore them."""
    spec = PROFILES[name]  # KeyError for an unknown profile, on purpose
    from . import enhance, timeline  # local: timeline imports profiles at call time, avoid a cycle
    modules = {"timeline": timeline, "enhance": enhance}
    saved = []
    for key, value in spec.items():
        if key == "detector_conf":
            if detector is not None:
                saved.append((detector, "conf", detector.conf))
                detector.conf = value
            continue
        mod, attr = key.split(".")
        saved.append((modules[mod], attr, getattr(modules[mod], attr)))
        setattr(modules[mod], attr, value)
    try:
        yield spec
    finally:
        for obj, attr, old in reversed(saved):
            setattr(obj, attr, old)
