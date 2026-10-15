"""Is this my dog? Recognise the owner's pets among the animals at the door.

    python -m vision.pets enrol <name> <image-or-clip> [...]    # 3-5 photos or a short clip of your pet
    python -m vision.pets who <image-or-clip>                    # which pet, or unknown, and the breed

A dog's fingerprint is two things OpenCV computes from its crop: the breed
scores of MobileNetV2 (ImageNet, run through cv2.dnn) over its 118 dog classes,
and an HSV colour histogram of the coat. Two fingerprints are compared with
cosine similarity and histogram correlation. Enrolled pets are kept in
pets/pets.json. Limits: dogs of the same breed and colour look alike to it,
and night (IR) footage has no colour to compare.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODELS / "image_classification_mobilenetv2_2022apr.onnx"
LABELS_FILE = MODELS / "labelsimagenet1k.h"
PETS_FILE = Path(__file__).resolve().parent.parent / "pets" / "pets.json"
DOG_CLASSES = range(151, 269)   # ImageNet: Chihuahua (151) .. Mexican hairless (268)
MATCH = 0.70                     # combined similarity at or above this = the same pet (tuned on the Pexels dogs)
BREED_WEIGHT = 0.6               # the rest is coat colour
CONTEXT_S = 2.0                  # seconds either side of a dog event to look for clear views
VIEWS = 10                       # frames sampled for one visit
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@lru_cache(maxsize=1)
def _net():
    return cv2.dnn.readNet(str(MODEL_FILE))


@lru_cache(maxsize=1)
def labels() -> list[str]:
    return re.findall(r'^\s*"([^"]*)"', LABELS_FILE.read_text(), re.M)


def _scores(crop: np.ndarray) -> np.ndarray:
    """ImageNet class probabilities for a BGR crop (MobileNetV2, 224x224 centre)."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)[16:240, 16:240]
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    net = _net()
    net.setInput(cv2.dnn.blobFromImage(x))
    logits = net.forward().reshape(-1).astype(np.float64)
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _colour(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256]).astype(np.float32)
    cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
    return h.reshape(-1)


def fingerprint(crop: np.ndarray) -> dict:
    """Breed-score vector over the dog classes, coat histogram, and the top breed."""
    p = _scores(crop)
    dogs = p[list(DOG_CLASSES)]
    top = int(np.argmax(dogs)) + DOG_CLASSES.start
    return {"breed": dogs.tolist(), "colour": _colour(crop).tolist(),
            "top_breed": labels()[top].split(",")[0], "breed_score": round(float(p[top]), 3),
            "dogness": round(float(dogs.sum()), 3)}


def similarity(a: dict, b: dict) -> float:
    """0..1: weighted breed-score cosine and coat-colour correlation."""
    va, vb = np.asarray(a["breed"]), np.asarray(b["breed"])
    breed = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    colour = float(cv2.compareHist(np.asarray(a["colour"], np.float32), np.asarray(b["colour"], np.float32),
                                   cv2.HISTCMP_CORREL))
    return BREED_WEIGHT * breed + (1 - BREED_WEIGHT) * max(0.0, colour)


def load_pets(path: Path = PETS_FILE) -> dict[str, list[dict]]:
    return json.loads(path.read_text()) if path.exists() else {}


def save_pets(pets: dict, path: Path = PETS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pets))


def identify(crop: np.ndarray, pets: dict[str, list[dict]] | None = None) -> dict:
    """Which enrolled pet this dog is, if any. Returns name (or None), score, top breed."""
    pets = load_pets() if pets is None else pets
    fp = fingerprint(crop)
    best, best_score = None, 0.0
    for name, prints in pets.items():
        score = max((similarity(fp, p) for p in prints), default=0.0)
        if score > best_score:
            best, best_score = name, score
    return {"pet": best if best_score >= MATCH else None, "score": round(best_score, 3),
            "closest": best, "breed": fp["top_breed"], "dogness": fp["dogness"]}


# ---------------------------------------------------------------- getting dog crops

def dog_crops(source: str | Path, detector, n: int = 12, pad: float = 0.1) -> list[np.ndarray]:
    """Dog crops from an image, or from n frames spread across a clip, using the YOLOX detector."""
    source = Path(source)
    if source.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        frames = [cv2.imread(str(source))]
    else:
        cap = cv2.VideoCapture(str(source))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frames = []
        for idx in np.linspace(0, total - 1, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
    crops = []
    for f in frames:
        if f is None:
            continue
        h, w = f.shape[:2]
        for o in detector.detect(f):
            if o["label"] != "dog":
                continue
            x, y, bw, bh = o["box"]
            x0, y0 = max(0, int((x - pad * bw) * w)), max(0, int((y - pad * bh) * h))
            x1, y1 = min(w, int((x + bw * (1 + pad)) * w)), min(h, int((y + bh * (1 + pad)) * h))
            if x1 - x0 >= 32 and y1 - y0 >= 32:
                crops.append(f[y0:y1, x0:x1])
    return crops


def identify_event(video: str | Path, event: dict, detector, pets: dict[str, list[dict]]) -> dict | None:
    """Which pet a dog event shows, judged over the whole visit: the median score over up to 8
    frames. One frame can mislead (a look-alike dog matched 4 frames in 11); the median did not."""
    from .enhance import sample_frames
    crops = []
    # Look a little either side of the event: a dog moving through a short event is often caught
    # in one blurred frame (measured: half a second, one detection), while it stands clear just before or after.
    t0, t1 = max(0.0, event["t_start"] - CONTEXT_S), event["t_end"] + CONTEXT_S
    for f in sample_frames(video, t0, t1, n=VIEWS):
        h, w = f.shape[:2]
        for o in detector.detect(f):
            if o["label"] == "dog":
                x, y, bw, bh = o["box"]
                c = f[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
                if c.shape[0] >= 32 and c.shape[1] >= 32:
                    crops.append(c)
    if not crops:
        return None
    rs = [identify(c, pets) for c in crops]
    by_pet = {}
    for r in rs:
        by_pet.setdefault(r["closest"], []).append(r["score"])
    closest, scores = max(by_pet.items(), key=lambda kv: float(np.median(kv[1])))
    score = float(np.median(scores))
    return {"pet": closest if score >= MATCH else None, "score": round(score, 3), "views": len(rs)}


def main() -> None:
    from .detector import Detector
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enrol")
    e.add_argument("name")
    e.add_argument("sources", nargs="+")
    w = sub.add_parser("who")
    w.add_argument("source")
    args = ap.parse_args()
    det = Detector()
    if args.cmd == "enrol":
        prints = [fingerprint(c) for s in args.sources for c in dog_crops(s, det)]
        if not prints:
            sys.exit("no dog found in those images")
        pets = load_pets()
        pets[args.name] = prints
        save_pets(pets)
        print(f"enrolled {args.name}: {len(prints)} views, looks like a {prints[0]['top_breed']}")
    else:
        crops = dog_crops(args.source, det)
        if not crops:
            sys.exit("no dog found")
        for r in (identify(c) for c in crops):
            print(f"{r['pet'] or 'unknown dog'}  score={r['score']}  closest={r['closest']}  breed={r['breed']}")


if __name__ == "__main__":
    main()
