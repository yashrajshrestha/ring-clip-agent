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
