"""Detector tests: box decoding by hand, timeline wiring with a fake detector,
and one model smoke test that is skipped when the ONNX file is not present."""
import json
import math

import numpy as np
import pytest

from tests.test_timeline import _write_clip
from vision import detector, timeline


def test_decode_maps_one_anchor_back_to_image_fractions():
    raw = np.zeros((8400, 85), np.float32)
    k = 20 * 80 + 10                      # stride-8 grid cell x=10, y=20
    raw[k, :2] = 0.5                      # centre (10.5, 20.5) * 8 = (84, 164) in 640 space
    raw[k, 2:4] = math.log(4)             # 4 * 8 = 32 px square
    raw[k, 4] = 0.9                       # objectness
    raw[k, 5] = 0.9                       # class 0 = person
    out = detector.decode(raw, ratio=0.5, img_w=1280, img_h=720, conf=0.5, nms=0.5)
    assert len(out) == 1
    o = out[0]
    assert o["label"] == "person" and o["group"] == "person"
    assert o["score"] == pytest.approx(0.81, abs=1e-3)
    # 640-space box (68, 148, 32, 32) / ratio 0.5 = (136, 296, 64, 64) on 1280x720
    assert o["box"] == pytest.approx([136 / 1280, 296 / 720, 64 / 1280, 64 / 720], abs=1e-3)
    json.dumps(out)  # the timeline is written as JSON; numpy scalars would crash it


def test_decode_drops_low_scores():
    raw = np.zeros((8400, 85), np.float32)
    raw[0, 4], raw[0, 5] = 0.6, 0.6       # 0.36 < 0.5
    assert detector.decode(raw, 1.0, 640, 640, conf=0.5, nms=0.5) == []


class FakeDetector:
    def __init__(self, group="person"):
        self.calls, self.group = 0, group

    def detect(self, bgr):
        self.calls += 1
        return [{"label": self.group, "group": self.group, "score": 0.9, "box": [0.1, 0.2, 0.1, 0.6]}]


def test_motion_event_takes_detector_label_and_decision(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_clip(p)
    fake = FakeDetector("person")
    tl = timeline.build_timeline(p, fake)
    motion = [e for e in tl["events"] if e["type"] == "motion"]
    assert len(motion) == 1
    assert motion[0]["label"] == "person"
    assert motion[0]["say"] == "Person at the door."
    assert motion[0]["decision"]["action"] == "alert"
    # ~3 s of motion at DETECT_HZ, not one call per frame (45 frames)
    assert 4 <= fake.calls <= 3 * timeline.DETECT_HZ + 2


class ParkedCarDetector:
    """Always sees a car in the top-right corner, far from where things move."""
    def detect(self, bgr):
        return [{"label": "car", "group": "vehicle", "score": 0.9, "box": [0.85, 0.02, 0.1, 0.08]}]


def test_background_object_does_not_label_the_event(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_clip(p)
    motion = [e for e in timeline.build_timeline(p, ParkedCarDetector())["events"] if e["type"] == "motion"]
    assert motion[0]["label"] != "vehicle"            # the parked car never labels the event...
    assert motion[0]["label"] == "figure"             # ...the steady person-sized movement does
    assert motion[0]["objects"] == {}


def test_small_motion_inside_a_person_still_counts():
    person, arm = [0.4, 0.2, 0.2, 0.7], [0.55, 0.4, 0.04, 0.05]
    assert timeline._overlaps(person, arm)
    assert not timeline._overlaps([0.85, 0.02, 0.1, 0.08], arm)


def test_parked_car_inside_a_huge_close_up_blob_is_not_the_mover():
    close_up = [0.3, 0.0, 0.7, 1.0]          # box held right at the lens
    car = [0.4, 0.45, 0.12, 0.08]
    assert not timeline._overlaps(car, close_up, "vehicle")
    assert timeline._overlaps(car, [0.38, 0.42, 0.16, 0.12], "vehicle")   # car that is the motion


def test_animal_is_ignored(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_clip(p)
    motion = [e for e in timeline.build_timeline(p, FakeDetector("animal"))["events"] if e["type"] == "motion"]
    assert motion[0]["decision"]["action"] == "ignore"


@pytest.mark.skipif(not detector.default_model_path().exists(), reason="YOLOX model not downloaded")
def test_real_model_loads_and_finds_nothing_in_a_blank_frame():
    d = detector.Detector()
    assert d.detect(np.full((360, 640, 3), 128, np.uint8)) == []
