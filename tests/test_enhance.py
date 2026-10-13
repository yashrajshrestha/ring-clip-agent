"""Recheck tests on synthetic clips: a dim scene OpenCV can rescue, a black
one it cannot, and the loop from first decision to second decision."""
import cv2
import numpy as np
import pytest

from vision import decider, enhance, timeline
from vision.checks import BLUR_MIN, DARK_MAX_MEAN

FPS, W, H = 15, 640, 360


def _scene(rng):
    """A textured porch with a door frame and a person-sized block."""
    g = np.clip(130 + cv2.GaussianBlur(rng.normal(0, 40, (H, W)), (0, 0), 0.8), 0, 255).astype(np.uint8)
    f = cv2.merge([g, g, g])
    cv2.rectangle(f, (200, 40), (440, 330), (60, 60, 60), 6)
    cv2.rectangle(f, (300, 90), (360, 300), (30, 30, 160), -1)
    return f


def _write_clip(path, level, seconds=4):
    """Scene multiplied by `level`: 1.0 is daylight, 0.12 is dim, 0.0 is black."""
    rng = np.random.default_rng(7)
    base = _scene(rng).astype(np.float32)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * seconds):
        noise = rng.normal(0, 3, (H, W, 3)).astype(np.float32)
        w.write(np.clip(base * level + noise, 0, 255).astype(np.uint8))
    w.release()


@pytest.fixture
def dim_clip(tmp_path):
    p = tmp_path / "dim.mp4"
    _write_clip(p, 0.12)
    return p


@pytest.fixture
def black_clip(tmp_path):
    p = tmp_path / "black.mp4"
    _write_clip(p, 0.0)
    return p


def test_enhance_brightens_a_dim_frame_and_leaves_daylight_alone():
    rng = np.random.default_rng(1)
    day = _scene(rng)
    dim = (day * 0.12).astype(np.uint8)
    out = enhance.enhance(dim)
    assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean() > 3 * cv2.cvtColor(dim, cv2.COLOR_BGR2GRAY).mean()
    day_out = enhance.enhance(day)
    assert abs(float(day_out.mean()) - float(day.mean())) < 25


def test_recheck_rescues_a_dim_clip(dim_clip):
    ev = {"type": "dark", "t_start": 0.5, "t_end": 3.5}
    rc = enhance.recheck(dim_clip, ev)
    assert rc["frames"] > 0
    assert rc["before"]["bright"] < DARK_MAX_MEAN
    assert rc["after"]["bright"] >= DARK_MAX_MEAN and rc["after"]["blur"] >= BLUR_MIN
    assert rc["usable"] is True and rc["label"] is None


def test_recheck_cannot_rescue_a_black_clip(black_clip):
    rc = enhance.recheck(black_clip, {"type": "dark", "t_start": 0.5, "t_end": 3.5})
    assert rc["usable"] is False
    assert rc["after"]["bright"] < DARK_MAX_MEAN


def test_timeline_rechecks_then_decides_again(dim_clip, monkeypatch):
    monkeypatch.delenv("DECIDER_URL", raising=False)
    dark = [e for e in timeline.build_timeline(dim_clip)["events"] if e["type"] == "dark"]
    assert len(dark) == 1
    ev = dark[0]
    assert ev["first_decision"]["action"] == "recheck"
    assert ev["recheck"]["usable"] is True
    assert ev["decision"]["action"] == "log"
    assert "enhancement" in ev["decision"]["reason"]


def test_dark_stretch_is_one_event_not_dark_plus_blurry(dim_clip, monkeypatch):
    monkeypatch.delenv("DECIDER_URL", raising=False)
    types = [e["type"] for e in timeline.build_timeline(dim_clip)["events"]]
    assert types.count("dark") == 1
    assert "blurry" not in types


def test_black_clip_stays_recheck_with_a_light_hint(black_clip, monkeypatch):
    monkeypatch.delenv("DECIDER_URL", raising=False)
    dark = [e for e in timeline.build_timeline(black_clip)["events"] if e["type"] == "dark"]
    assert dark and dark[0]["decision"]["action"] == "recheck"
    assert "add a light" in dark[0]["decision"]["reason"]


class CropOnlyDetector:
    """Fake detector: sees a person only in a zoomed crop, never in a full frame,
    like a real one with a figure too small to detect at full size."""

    def __init__(self):
        self.full, self.crops = 0, 0

    def detect(self, bgr):
        if bgr.shape[1] < 640:
            self.crops += 1
            return [{"label": "person", "group": "person", "score": 0.8, "box": [0.4, 0.2, 0.2, 0.6]}]
        self.full += 1
        return []


class AlwaysPerson:
    def detect(self, bgr):
        return [{"label": "person", "group": "person", "score": 0.9, "box": [0.4, 0.2, 0.2, 0.6]}]


def test_second_look_zooms_in_when_the_first_look_sees_nothing(dim_clip):
    det = CropOnlyDetector()
    rc = enhance.recheck(dim_clip, {"type": "dark", "t_start": 0.5, "t_end": 3.5}, det)
    assert rc["objects"] == {} and rc["label"] == "person"
    sl = rc["second_look"]
    assert sl["frames"] == enhance.SECOND_LOOK_FRAMES and sl["crops"] == 5
    assert sl["objects"] == {"person": enhance.SECOND_LOOK_FRAMES}
    assert det.full == enhance.RECHECK_FRAMES and det.crops == 5 * enhance.SECOND_LOOK_FRAMES
    assert timeline._decide({"type": "dark", "t_start": 0.5, "t_end": 3.5, "recheck": rc})["action"] == "alert"
    assert "zoomed in over 12 more frames: detector saw person x12" in decider.state_text({"type": "dark", "t_start": 0.5, "t_end": 3.5, "recheck": rc}, [])


def test_no_second_look_when_the_first_look_finds_a_person(dim_clip):
    rc = enhance.recheck(dim_clip, {"type": "dark", "t_start": 0.5, "t_end": 3.5}, AlwaysPerson())
    assert rc["label"] == "person" and "second_look" not in rc


def test_no_second_look_without_a_detector(dim_clip):
    rc = enhance.recheck(dim_clip, {"type": "dark", "t_start": 0.5, "t_end": 3.5})
    assert "second_look" not in rc


def test_rule_alerts_when_enhanced_picture_shows_a_person():
    ev = {"type": "dark", "t_start": 1.0, "t_end": 2.0,
          "recheck": {"usable": True, "label": "person", "objects": {"person": 4}}}
    assert timeline._decide(ev)["action"] == "alert"


def test_state_text_tells_the_decider_what_the_recheck_found():
    ev = {"type": "dark", "t_start": 1.0, "t_end": 2.0, "recheck": {"usable": True, "objects": {"person": 4}}}
    assert "Rechecked after enhancing the picture with OpenCV: now usable, detector saw person x4." in decider.state_text(ev, [])
    ev["recheck"] = {"usable": False, "objects": {}}
    assert "still too dark or blurry" in decider.state_text(ev, [])
