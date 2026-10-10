"""Synthetic-frame tests so the checks are verified before any real clip exists."""
import cv2
import numpy as np

from vision import checks


def _flat(val=128, color=True):
    shape = (360, 640, 3) if color else (360, 640)
    return np.full(shape, val, dtype=np.uint8)


def test_blur_score_drops_when_blurred():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    soft = cv2.GaussianBlur(sharp, (21, 21), 0)
    assert checks.blur_score(sharp) > checks.blur_score(soft) * 5


def test_blur_score_ignores_overlay_bands():
    rng = np.random.default_rng(2)
    soft = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (21, 21), 0)
    assert checks.blur_score(checks.inner(soft)) < checks.BLUR_MIN
    assert checks.blur_score(checks.mask_overlays(soft)) > checks.BLUR_MIN


def test_dark_frame_flagged():
    assert checks.exposure(_flat(5))["mean"] < checks.DARK_MAX_MEAN


def test_ir_detected_on_gray_channels():
    assert checks.is_ir(_flat(120))
    colour = _flat(0)
    colour[:, :, 2] = 200
    assert not checks.is_ir(colour)


def test_motion_region_finds_moving_box():
    frames = []
    for i in range(30):
        f = _flat(90)
        x = 50 + i * 10
        cv2.rectangle(f, (x, 150), (x + 80, 250), (255, 255, 255), -1)
        frames.append(f)
    m = checks.motion_region(frames)
    assert m["has_motion"]
    assert m["area"] > checks.MOTION_MIN_AREA


def test_static_clip_has_no_motion():
    frames = [_flat(90) for _ in range(30)]
    assert not checks.motion_region(frames)["has_motion"]


def test_phash_close_for_noisy_copy_far_for_different():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (31, 31), 0)
    noisy = cv2.add(base, rng.integers(0, 8, base.shape, dtype=np.uint8))
    other = cv2.GaussianBlur(rng.integers(0, 255, base.shape, dtype=np.uint8), (31, 31), 0)
    assert checks.hash_distance(checks.phash(base), checks.phash(noisy)) <= checks.DUP_HASH_DIST
    assert checks.hash_distance(checks.phash(base), checks.phash(other)) > checks.DUP_HASH_DIST


def test_single_frame_clip_is_flagged_short_not_no_motion(tmp_path):
    out = tmp_path / "one.mp4"
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    w.write(_flat(90))
    w.release()
    facts = checks.analyze(out)
    assert "too_short" in facts.flags
    assert "no_motion" not in facts.flags


def test_mask_overlays_zeroes_bands_and_boxes():
    f = _flat(200)
    out = checks.mask_overlays(f, [(10, 100, 50, 50)])
    assert out[0, 0].max() == 0 and out[-1, -1].max() == 0
    assert out[120, 30].max() == 0
    assert out[180, 320].max() == 200
