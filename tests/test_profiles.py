"""Profiles apply for one analysis and restore afterwards."""
from types import SimpleNamespace

import pytest

from vision import enhance, profiles, timeline


def test_harder_raises_the_settings_then_restores_them():
    before = (enhance.RECHECK_FRAMES, enhance.SECOND_LOOK_FRAMES, enhance.ALWAYS_SECOND_LOOK,
              enhance.TARGET_MEAN, enhance.GAMMA_MAX, timeline.DETECT_HZ)
    det = SimpleNamespace(conf=0.5)
    with profiles.active("harder", det) as spec:
        assert spec["detector_conf"] == 0.35 and det.conf == 0.35
        assert enhance.RECHECK_FRAMES > before[0] and enhance.SECOND_LOOK_FRAMES > before[1]
        assert enhance.ALWAYS_SECOND_LOOK is True
        assert enhance.TARGET_MEAN > before[3] and enhance.GAMMA_MAX > before[4]
        assert timeline.DETECT_HZ > before[5]
    assert det.conf == 0.5
    assert (enhance.RECHECK_FRAMES, enhance.SECOND_LOOK_FRAMES, enhance.ALWAYS_SECOND_LOOK,
            enhance.TARGET_MEAN, enhance.GAMMA_MAX, timeline.DETECT_HZ) == before


class CropOnlyDetector:
    """Sees a person only in a crop narrower than the full 640 px frame."""

    conf = 0.5

    def detect(self, bgr):
        if bgr.shape[1] < 640:
            return [{"label": "person", "group": "person", "score": 0.8, "box": [0.3, 0.1, 0.3, 0.8]}]
        return []


def test_harder_profile_finds_a_person_by_zooming_on_the_motion(tmp_path, monkeypatch):
    from tests.test_timeline import _write_clip
    monkeypatch.delenv("DECIDER_URL", raising=False)
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    normal = [e for e in timeline.build_timeline(clip, CropOnlyDetector())["events"] if e["type"] == "motion"]
    assert normal and normal[0]["label"] != "person"      # full-frame detector sees nothing (tracking may say "figure")
    with profiles.active("harder", CropOnlyDetector()):
        harder = [e for e in timeline.build_timeline(clip, CropOnlyDetector())["events"] if e["type"] == "motion"]
    assert harder and harder[0]["label"] == "person"       # the zoomed crop does
    assert timeline.ZOOM_MOTION is False                   # restored after the harder run


def test_zoomed_boxes_map_back_to_frame_fractions():
    import numpy as np

    class Echo:
        def detect(self, bgr):
            return [{"label": "person", "group": "person", "score": 0.9, "box": [0.0, 0.0, 1.0, 1.0]}]

    frame = np.zeros((360, 640, 3), np.uint8)
    out = timeline.zoomed_detect(Echo(), frame, [0.4, 0.5, 0.1, 0.2])
    assert len(out) == 1 and out[0]["zoomed"] is True
    x, y, w, h = out[0]["box"]
    assert w == 0.4 and h == 0.4                         # crop is the 40% minimum share
    assert abs((x + w / 2) - 0.45) < 1e-6 and abs((y + h / 2) - 0.6) < 1e-6   # centred on the motion box


def test_normal_changes_nothing_and_unknown_is_an_error():
    det = SimpleNamespace(conf=0.5)
    with profiles.active("normal", det):
        assert det.conf == 0.5 and enhance.ALWAYS_SECOND_LOOK is False
    with pytest.raises(KeyError):
        with profiles.active("hardest"):
            pass


def test_restores_even_when_the_analysis_fails():
    det = SimpleNamespace(conf=0.5)
    with pytest.raises(RuntimeError):
        with profiles.active("harder", det):
            raise RuntimeError("boom")
    assert det.conf == 0.5 and enhance.ALWAYS_SECOND_LOOK is False
