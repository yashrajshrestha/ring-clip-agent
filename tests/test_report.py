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


def test_ring_comparison_counts_alerts_noise_catches_and_quiet_dogs(tmp_path):
    def tl(*events):
        return {"events": [{"type": t, "label": lab, "peak_area": area, "decision": {"action": a}}
                           for t, lab, area, a in events]}

    samples = tmp_path / "s"
    samples.mkdir()
    (samples / "visit.timeline.json").write_text(json.dumps(tl(
        ("motion", "person", 0.1, "alert"), ("cut", None, 0.0, "log"), ("motion", None, 0.005, "ignore"))))
    (samples / "dog.timeline.json").write_text(json.dumps(tl(
        ("motion", "animal", 0.1, "ignore"), ("motion", None, 0.06, "ignore"))))
    labels = tmp_path / "l.csv"
    labels.write_text("name,source,kind,night,notes\nvisit,x,visit,0,\ndog,x,animal,0,\n")
    c = report.ring_comparison(samples, [labels, tmp_path / "missing.csv"])
    assert c["clips"] == 2 and c["should_alert"] == 1 and c["animal_only"] == 1
    basic, smart, ours = c["systems"].values()
    assert basic == {"alerts": 4, "noise_alerts": 2, "caught": 1, "quiet_animals": 0}   # every motion buzzes
    assert smart == {"alerts": 1, "noise_alerts": 0, "caught": 1, "quiet_animals": 1}
    assert ours == {"alerts": 1, "noise_alerts": 0, "caught": 1, "quiet_animals": 1}
    md = report.ring_markdown(c)
    assert "| Ring, basic motion alerts (simulated) | 4 | 2 | 1 / 1 | 0 / 1 |" in md


def test_missing_timelines_are_skipped(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("name,youtube_id,channel,kind,night,notes\nabsent,x,c,left,0,\n")
    assert report.load(tmp_path, labels) == []
    assert report.summarize([])["clips"] == 0
