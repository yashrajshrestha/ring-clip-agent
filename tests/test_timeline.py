"""Timeline tests on a small synthetic clip: timing, warm-up, events, scene diff."""
import cv2
import numpy as np
import pytest

from vision import timeline

FPS, W, H = 15, 640, 360


def _x_with_pause(t, box_from, box_to, pause):
    """Walk in, stand still for `pause` seconds mid-frame, walk out."""
    walk = (box_to - box_from - pause) / 2
    if t < box_from + walk:
        return int(40 + (t - box_from) / walk * 260)
    if t < box_from + walk + pause:
        return 300
    return int(300 + (t - box_from - walk - pause) / walk * 300)


def _write_clip(path, seconds=6, box_from=1.0, box_to=4.0, leave_package=False, pause=0.0, cut_at=None):
    rng = np.random.default_rng(3)
    bg = np.clip(130 + cv2.GaussianBlur(rng.normal(0, 40, (H, W)), (0, 0), 0.8), 0, 255).astype(np.uint8)
    bg = cv2.merge([bg, bg, bg])
    # A different gray scene for the cut: same mode, new layout.
    other = bg.astype(np.int16)
    other[:, : W // 2] -= 70
    other[:, W // 2:] += 50
    other = np.clip(other, 0, 255).astype(np.uint8)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * seconds):
        t = i / FPS
        f = (other if cut_at is not None and t >= cut_at else bg).copy()
        if leave_package and t >= 2.5:
            cv2.rectangle(f, (300, 270), (370, 310), (40, 120, 200), -1)
        if box_from <= t < box_to:
            x = _x_with_pause(t, box_from, box_to, pause) if pause else int(40 + (t - box_from) * 150)
            cv2.rectangle(f, (x, 80), (x + 60, 300), (30, 30, 160), -1)
        w.write(f)
    w.release()


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_clip(p)
    return p


def test_timestamps_increase_and_match_frame_index(clip):
    frames = timeline.build_timeline(clip)["frames"]
    ts = [f["t"] for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    for i, t in enumerate(ts):
        assert abs(t - i / FPS) <= 1 / FPS


def test_no_boxes_during_warmup_and_boxes_are_fractions(clip):
    frames = timeline.build_timeline(clip)["frames"]
    assert all(f["box"] is None for f in frames[:timeline.WARMUP_FRAMES])
    boxes = [f["box"] for f in frames if f["box"]]
    assert boxes
    assert all(0 <= v <= 1 for b in boxes for v in b)


def test_one_motion_event_for_one_moving_box(clip):
    events = [e for e in timeline.build_timeline(clip)["events"] if e["type"] == "motion"]
    assert len(events) == 1
    assert abs(events[0]["t_start"] - 1.0) <= 0.5
    assert abs(events[0]["t_end"] - 4.0) <= 0.5
    assert events[0]["decision"]["source"] == "rule-stub"


def test_package_left_is_a_scene_change(tmp_path):
    p = tmp_path / "pkg.mp4"
    _write_clip(p, seconds=8, box_from=1.0, box_to=3.5, leave_package=True)
    events = timeline.build_timeline(p)["events"]
    scene = [e for e in events if e["type"] == "scene_change"]
    assert len(scene) == 1
    x, y, w, h = scene[0]["box"]
    assert abs(x - 300 / W) < 0.05 and abs(y - 270 / H) < 0.05


def test_person_standing_still_is_not_left_or_taken(tmp_path):
    # Real Ring bug: MOG2 absorbs a driver who stands still, the event ends,
    # and a before/after diff then flags the driver as an object left behind.
    p = tmp_path / "pause.mp4"
    _write_clip(p, seconds=10, box_from=1.0, box_to=7.0, pause=3.0)
    events = timeline.build_timeline(p)["events"]
    assert [e for e in events if e["type"] == "scene_change"] == []


def test_package_left_during_a_pause_is_found_once(tmp_path):
    p = tmp_path / "pause_pkg.mp4"
    _write_clip(p, seconds=10, box_from=1.0, box_to=7.0, pause=3.0, leave_package=True)
    scene = [e for e in timeline.build_timeline(p)["events"] if e["type"] == "scene_change"]
    assert len(scene) == 1
    x, y, w, h = scene[0]["box"]
    assert abs(x - 300 / W) < 0.05 and abs(y - 270 / H) < 0.05


def test_hard_cut_splits_motion_and_is_reported(tmp_path):
    p = tmp_path / "cut.mp4"
    _write_clip(p, seconds=8, box_from=1.0, box_to=6.0, cut_at=3.0)
    events = timeline.build_timeline(p)["events"]
    cuts = [e for e in events if e["type"] == "cut"]
    assert len(cuts) == 1 and abs(cuts[0]["t_start"] - 3.0) <= 0.1
    motion = [e for e in events if e["type"] == "motion"]
    assert all(not (m["t_start"] < 3.0 < m["t_end"]) for m in motion)
    assert not any(e["type"] == "scene_change" for e in events)


def _write_quality_clip(path, card_at=None, blur_from=None, frozen=False, seconds=6):
    """Live gray footage with fresh sensor noise every frame. From `card_at` a
    flat coloured end card follows a hard cut; from `blur_from` the footage is
    defocused. `frozen` reuses one noise frame, so nothing flickers."""
    rng = np.random.default_rng(5)
    base = np.clip(130 + cv2.GaussianBlur(rng.normal(0, 40, (H, W)), (0, 0), 0.8), 0, 255)
    still_noise = rng.normal(0, 4, (H, W))
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * seconds):
        t = i / FPS
        if card_at is not None and t >= card_at:
            w.write(np.full((H, W, 3), (40, 160, 230), np.uint8))
            continue
        g = base + (still_noise if frozen else rng.normal(0, 4, (H, W)))
        if blur_from is not None and t >= blur_from:
            g = cv2.GaussianBlur(g, (0, 0), 8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        w.write(cv2.merge([g, g, g]))
    w.release()


def _types(path):
    return [e["type"] for e in timeline.build_timeline(path)["events"]]


def test_end_card_after_a_cut_is_not_reported_as_blurry(tmp_path):
    p = tmp_path / "card.mp4"
    _write_quality_clip(p, card_at=3.0)
    types = _types(p)
    assert "blurry" not in types
    assert {"cut", "day_mode"} & set(types)  # the edit itself is still on the timeline


def test_real_blur_on_live_footage_is_reported(tmp_path):
    p = tmp_path / "blur.mp4"
    _write_quality_clip(p, blur_from=3.0)
    assert "blurry" in _types(p)


def test_frozen_blur_without_an_edit_is_still_reported(tmp_path):
    p = tmp_path / "stuck.mp4"
    _write_quality_clip(p, blur_from=0.0, frozen=True)
    assert "blurry" in _types(p)


def test_browser_copy_is_h264(clip, tmp_path):
    out = tmp_path / "out.mp4"
    timeline.to_browser_mp4(clip, out)
    cap = cv2.VideoCapture(str(out))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    cap.release()
    assert fourcc in ("avc1", "h264")
