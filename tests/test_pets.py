"""Pet recognition: fingerprint comparison, per-visit judgement, rules and decider text.
The model-backed tests skip when the classifier is not downloaded."""
import numpy as np
import pytest

from vision import decider, pets, timeline


def _fp(breed, colour):
    return {"breed": list(breed), "colour": list(colour), "top_breed": "x", "breed_score": 0.5, "dogness": 0.9}


def test_similarity_is_one_for_the_same_dog_and_low_for_a_different_one():
    a = _fp([0.8, 0.1, 0.1], [0.5, 0.5, 0.0])
    b = _fp([0.0, 0.1, 0.9], [0.0, 0.1, 0.9])
    assert pets.similarity(a, a) == pytest.approx(1.0)
    assert pets.similarity(a, b) < 0.3


def test_identify_event_judges_the_whole_visit_by_the_median(monkeypatch, tmp_path):
    class Dog:   # a detector that sees a dog filling most of the frame
        def detect(self, f):
            return [{"label": "dog", "group": "animal", "score": 0.9, "box": [0.1, 0.1, 0.8, 0.8]}]

    frames = [np.full((360, 640, 3), 100, np.uint8)] * 7
    windows = []
    monkeypatch.setattr("vision.enhance.sample_frames", lambda v, t0, t1, n=None: windows.append((t0, t1)) or frames)
    # a look-alike: a few frames match, most do not (measured: 4 of 11, median 0.61)
    scores = iter([0.86, 0.80, 0.55, 0.52, 0.61, 0.58, 0.75])
    monkeypatch.setattr(pets, "identify", lambda c, store: {"closest": "Max", "score": next(scores)})
    out = pets.identify_event("clip.mp4", {"t_start": 1.0, "t_end": 3.0}, Dog(), {"Max": []})
    assert out == {"pet": None, "score": 0.61, "views": 7}
    assert windows == [(0.0, 5.0)]   # 2 s either side of the 1-3 s event, clipped at the start


def test_rules_for_own_and_unknown_dogs():
    base = {"type": "motion", "t_start": 1.0, "t_end": 4.0, "peak_area": 0.1, "label": "animal"}
    own = {**base, "pet": {"pet": "Max", "score": 0.88, "views": 6}}
    assert timeline._decide(own) == {"action": "log", "reason": "Max, your dog, is at the door", "source": "rule-stub"}
    waiting = {**own, "t_end": 21.0, "lingering": True}
    assert timeline._decide(waiting)["reason"] == "Max has been waiting at the door for 20s"
    unknown = {**base, "pet": {"pet": None, "score": 0.33, "views": 5}}
    assert timeline._decide(unknown)["action"] == "ignore"


def test_decider_is_told_whose_dog_it_is():
    base = {"type": "motion", "t_start": 1.0, "t_end": 4.0, "peak_area": 0.1, "label": "animal"}
    own = decider.state_text({**base, "pet": {"pet": "Max", "score": 0.88, "views": 6}}, [])
    assert "the owner's own dog, Max (recognised, 88% match), not a stranger's" in own
    unknown = decider.state_text({**base, "pet": {"pet": None, "score": 0.33, "views": 5}}, [])
    assert "not one of the owner's pets: an unknown dog" in unknown
    assert "own pet passing by" in decider.CRITERIA["log"]


@pytest.mark.skipif(not pets.MODEL_FILE.exists(), reason="classifier not downloaded")
def test_fingerprint_shape_and_labels():
    crop = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    fp = pets.fingerprint(crop)
    assert len(fp["breed"]) == len(pets.DOG_CLASSES) == 118
    assert len(fp["colour"]) == 72 and 0.0 <= fp["dogness"] <= 1.0
    assert pets.labels()[151] == "Chihuahua" and pets.labels()[235].startswith("German shepherd")
