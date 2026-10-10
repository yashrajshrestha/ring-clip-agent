"""YOLOX-S object detector run through cv2.dnn, so OpenCV does the inference.

Model: OpenCV Model Zoo `object_detection_yolox_2022nov.onnx` (Apache 2.0),
trained on COCO. COCO has no "package" class; packages still come from the
before/after scene diff. Pre/post-processing follows the zoo's reference code.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

MODEL_FILE = "object_detection_yolox_2022nov.onnx"
MODEL_URL = f"https://huggingface.co/opencv/object_detection_yolox/resolve/main/{MODEL_FILE}"
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
INPUT = 640
STRIDES = (8, 16, 32)

COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)
# Classes that matter at a front door, grouped for the decider.
GROUP = {
    "person": "person",
    "dog": "animal", "cat": "animal", "bird": "animal", "bear": "animal", "horse": "animal",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
    "backpack": "bag", "handbag": "bag", "suitcase": "bag",
}


def default_model_path() -> Path:
    env = os.environ.get("YOLOX_MODEL")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models" / MODEL_FILE


def _anchors() -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


GRIDS, EXP_STRIDES = _anchors()


def letterbox(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize to fit 640x640 keeping aspect, pad bottom/right with 114, RGB, NCHW float."""
    ratio = min(INPUT / bgr.shape[0], INPUT / bgr.shape[1])
    h, w = int(bgr.shape[0] * ratio), int(bgr.shape[1] * ratio)
    padded = np.full((INPUT, INPUT, 3), 114.0, np.float32)
    padded[:h, :w] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_LINEAR)
    return padded.transpose(2, 0, 1)[None], ratio


def decode(raw: np.ndarray, ratio: float, img_w: int, img_h: int, conf: float, nms: float) -> list[dict]:
    """Grid-decode YOLOX output [8400, 85], NMS per class, map back to 0-1 boxes."""
    dets = raw.copy()
    dets[:, :2] = (dets[:, :2] + GRIDS) * EXP_STRIDES
    dets[:, 2:4] = np.exp(dets[:, 2:4]) * EXP_STRIDES
    scores = dets[:, 4:5] * dets[:, 5:]
    cls = scores.argmax(axis=1)
    best = scores.max(axis=1)
    xywh = np.stack([dets[:, 0] - dets[:, 2] / 2, dets[:, 1] - dets[:, 3] / 2, dets[:, 2], dets[:, 3]], 1) / ratio
    keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), best.tolist(), cls.tolist(), conf, nms)
    out = []
    for k in np.array(keep).reshape(-1):
        x, y, w, h = (float(v) for v in xywh[k])  # plain floats so the result is JSON-safe
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        label = COCO[int(cls[k])]
        out.append({
            "label": label, "group": GROUP.get(label, "other"), "score": round(float(best[k]), 3),
            "box": [round(x0 / img_w, 4), round(y0 / img_h, 4), round((x1 - x0) / img_w, 4), round((y1 - y0) / img_h, 4)],
        })
    return out


class Detector:
    # conf 0.5, not the zoo's 0.35: on real porch clips a pile of bags scored
    # "person 0.41" while real people scored 0.73 to 0.90.
    def __init__(self, path: str | Path | None = None, conf: float = 0.5, nms: float = 0.5):
        self.net = cv2.dnn.readNet(str(path or default_model_path()))
        self.conf, self.nms = conf, nms

    def detect(self, bgr: np.ndarray) -> list[dict]:
        blob, ratio = letterbox(bgr)
        self.net.setInput(blob)
        raw = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        return decode(raw, ratio, bgr.shape[1], bgr.shape[0], self.conf, self.nms)
