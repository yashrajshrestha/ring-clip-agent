"""Inbox watcher tests on a small synthetic clip, no detector, rule stub."""
import json

import cv2
import numpy as np
import pytest

from vision import inbox


def _write_clip(path, seconds=3):
    rng = np.random.default_rng(2)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 180))
    for i in range(15 * seconds):
        g = np.clip(120 + rng.normal(0, 30, (180, 320)), 0, 255).astype(np.uint8)
        f = cv2.merge([g, g, g])
        if 1.0 <= i / 15 < 2.5:
            cv2.rectangle(f, (40 + i * 4, 40), (80 + i * 4, 160), (30, 30, 160), -1)
        w.write(f)
    w.release()


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.delenv("DECIDER_URL", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    return inbox.Inbox(tmp_path / "in", tmp_path / "out", detector=None, interval=0)


def test_waits_until_the_file_stops_growing_then_processes_it(box):
    _write_clip(box.inbox / "porch.mp4")
    assert box.poll() == []                      # first sight: size recorded, not processed yet
    status = json.loads((box.out_dir / "inbox.json").read_text())
    assert status["queue"] == ["porch"] and status["processing"] is None
    assert box.poll() == ["porch"]               # same size on the next poll: stable, processed
    tl = json.loads((box.out_dir / "porch.timeline.json").read_text())
    assert tl["events"] and all("decision" in e for e in tl["events"])
    assert (box.out_dir / "porch.mp4").exists()
    assert [r["name"] for r in json.loads((box.out_dir / "index.json").read_text())] == ["porch"]
    status = json.loads((box.out_dir / "inbox.json").read_text())
    assert status["done"] == ["porch"] and status["queue"] == [] and status["errors"] == {}
    assert status["finished"]["porch"] <= status["now"]
    assert box.poll() == []                      # nothing new


def test_a_touched_file_is_processed_again(box):
    import os
    import time
    _write_clip(box.inbox / "porch.mp4")
    box.poll(), box.poll()
    first = json.loads((box.out_dir / "inbox.json").read_text())["finished"]["porch"]
    later = time.time() + 5
    os.utime(box.inbox / "porch.mp4", (later, later))   # what the Re-run button does
    assert box.poll() == ["porch"]
    status = json.loads((box.out_dir / "inbox.json").read_text())
    assert status["finished"]["porch"] > first and status["done"] == ["porch"]


def test_ignores_non_videos_and_hidden_files(box):
    (box.inbox / "notes.txt").write_text("x")
    (box.inbox / ".porch.mp4.part").write_bytes(b"xx")
    assert box.pending() == [] and box.poll() == []


def test_a_broken_file_is_reported_once_and_not_retried(box):
    (box.inbox / "broken.mp4").write_bytes(b"not a video at all")
    box.poll()
    assert box.poll() == []
    status = json.loads((box.out_dir / "inbox.json").read_text())
    assert "broken" in status["errors"] and status["processing"] is None
    assert box.pending() == []
