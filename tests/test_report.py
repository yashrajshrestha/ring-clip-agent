"""Report tests on hand-made timelines."""
import json

from vision import report


def _tl(*events):
    return {"duration": 20.0, "events": [{"id": i, "type": t, "t_start": float(i), "t_end": float(i) + 1,
                                           "label": lab, "decision": {"action": a, "source": src}}
                                          for i, (t, lab, a, src) in enumerate(events)]}


def test_scores_hits_misses_and_failures(tmp_path):
    samples, labels = tmp_path / "s", tmp_path / "labels.csv"
    samples.mkdir()
    labels.write_text("name,youtube_id,channel,kind,night,notes\n"
                      "good,x,c,left,0,courier\nnoscene,x,c,taken,1,thief\nsilent,x,c,left,0,quiet\n")
    (samples / "good.timeline.json").write_text(json.dumps(_tl(
        ("motion", "person", "alert", "strands-decider-2b"), ("scene_change", None, "alert", "strands-decider-2b"),
        ("blurry", None, "log", "rule-stub"))))
    (samples / "noscene.timeline.json").write_text(json.dumps(_tl(("motion", "person", "alert", "rule-stub"))))
    (samples / "silent.timeline.json").write_text(json.dumps(_tl(("motion", None, "ignore", "rule-stub"))))
    rows = report.load(samples, labels)
    s = report.summarize(rows)
    assert [r["name"] for r in rows] == ["good", "noscene", "silent"]
    assert s["alert_hits"] == 2 and s["scene_hits"] == 1 and s["scene_expected"] == 3
    assert s["events"] == 5 and s["alert_events"] == 3
    assert "| Events that would wake a language model (alerts) | 3 / 5 |" in report.markdown(rows, s)
    assert s["night_scene_hits"] == 0 and s["night_scene_expected"] == 1
    assert s["sources"] == {"strands-decider-2b": 2, "rule-stub": 3}
    md = report.markdown(rows, s)
    assert "| good | left |" in md and "| ok |" in md
    assert "**noscene** (taken, night): the package being left or taken was not detected" in md
    assert "**silent** (left): no alert at all" in md


def test_missing_timelines_are_skipped(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("name,youtube_id,channel,kind,night,notes\nabsent,x,c,left,0,\n")
    assert report.load(tmp_path, labels) == []
    assert report.summarize([])["clips"] == 0
