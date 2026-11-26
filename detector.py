"""Object detection interface producing pseudo-labelled regions.

Detector-assigned regions are *pseudo-labels*: no human annotated which box a caption word
refers to, so downstream code treats them with a precision-first filter.
"""
from __future__ import annotations

import abc
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detected region."""
    label: str
    score: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixels

    def to_dict(self) -> dict:
        return asdict(self)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_problem(bbox: Sequence[float], image_size: Optional[Tuple[int, int]] = None) -> Optional[str]:
    """Why ``bbox`` ([x1, y1, x2, y2] pixels) is geometrically invalid, or None if it is valid.

    Checks: four finite numbers, positive width and height, non-negative origin and, when ``image_size`` (w, h)
    is known, containment in the image (a 1-pixel tolerance absorbs rounding by the detector's post-processing).
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return "box is not four numbers"
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return "non-finite coordinate"
    if x2 <= x1 or y2 <= y1:
        return "non-positive width or height"
    if x1 < -1 or y1 < -1:
        return "negative origin"
    if image_size is not None and (x2 > image_size[0] + 1 or y2 > image_size[1] + 1):
        return "box extends beyond the image"
    return None


def nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """Per-label non-maximum suppression (highest score first).

    Fully deterministic: ties in score are broken by (label, bbox), so exact duplicate boxes collapse to one and the
    result does not depend on the input order.
    """
    kept: List[Detection] = []
    for d in sorted(detections, key=lambda x: (-x.score, x.label, tuple(x.bbox))):
        if all(not (k.label == d.label and iou(k.bbox, d.bbox) > iou_threshold) for k in kept):
            kept.append(d)
    return kept


class BaseDetector(abc.ABC):
    """Detector interface."""

    @abc.abstractmethod
    def detect(self, image: Image.Image, threshold: float = 0.3, key: Optional[object] = None) -> List[Detection]:
        """Return detections with score >= threshold (``key`` identifies the image for replaying detectors)."""

    def provenance(self) -> Dict[str, object]:
        """Identity of the detector for cache provenance (checkpoint, revision, configuration)."""
        return {"name": type(self).__name__}


class StaticDetector(BaseDetector):
    """Returns fixed detections; used for tests and for replaying cached detections."""

    def __init__(self, detections: Dict[object, List[Detection]] | List[Detection]):
        self.detections = detections

    def detect(self, image: Image.Image, threshold: float = 0.3, key: Optional[object] = None) -> List[Detection]:
        dets = self.detections.get(key, []) if isinstance(self.detections, dict) else self.detections
        return [d for d in dets if d.score >= threshold]


class PrecomputedDetector(BaseDetector):
    """Reads detections from a JSON file ``{image_key: [{label, score, bbox}, ...]}``."""

    def __init__(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.data = {k: [Detection(**d) for d in v] for k, v in raw.items()}

    def detect(self, image: Image.Image, threshold: float = 0.3, key: Optional[object] = None) -> List[Detection]:
        return [d for d in self.data.get(str(key), []) if d.score >= threshold]


class GroundingDinoDetector(BaseDetector):
    """Open-vocabulary Grounding DINO (Hugging Face implementation).

    The text prompt lists the COCO object vocabulary (the vocabulary CHAIR scores against), so
    labels returned by the detector can be matched to caption nouns through synonym tables.
    """

    def __init__(self, checkpoint: str = "IDEA-Research/grounding-dino-tiny", labels: Optional[Sequence[str]] = None,
                 device: Optional[torch.device] = None, text_threshold: float = 0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        from src.grounding.region_matching import COCO_CATEGORIES

        self.labels = list(labels or COCO_CATEGORIES)
        self.checkpoint = checkpoint
        self.device = device or torch.device("cpu")
        self.text_threshold = text_threshold
        try:
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(self.device).eval()
        except OSError as exc:
            raise RuntimeError(f"Could not load detector '{checkpoint}': {exc}") from exc
        for p in self.model.parameters():
            p.requires_grad = False
        self.prompt = " ".join(f"{l}." for l in self.labels)

    def provenance(self) -> Dict[str, object]:
        import hashlib

        cfg = self.model.config
        return {"name": "GroundingDinoDetector", "checkpoint": self.checkpoint, "revision": str(getattr(cfg, "_commit_hash", None) or "unknown"),
                "text_threshold": self.text_threshold, "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest()}

    @torch.no_grad()
    def detect(self, image: Image.Image, threshold: float = 0.3, key: Optional[object] = None) -> List[Detection]:
        from src.grounding.region_matching import word_to_category

        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target = [image.size[::-1]]
        post = self.processor.post_process_grounded_object_detection
        try:
            res = post(outputs, inputs.input_ids, threshold=threshold, text_threshold=self.text_threshold, target_sizes=target)[0]
        except TypeError:  # older transformers versions use box_threshold
            res = post(outputs, inputs.input_ids, box_threshold=threshold, text_threshold=self.text_threshold, target_sizes=target)[0]
        names = res.get("text_labels", res.get("labels"))
        dets: List[Detection] = []
        for box, score, name in zip(res["boxes"].tolist(), res["scores"].tolist(), names):
            cat = word_to_category(str(name).strip())
            if cat is not None:
                dets.append(Detection(label=cat, score=float(score), bbox=[float(x) for x in box]))
        return dets
