"""Run B1 / B2 / A on one image with SHARED grounding computation and per-model status (R13/R14).

Flow::

    image -> validation -> detector + OpenCLIP region embeddings (computed ONCE, cached by image + provenance)
          -> B1 | B2 | A  (each generates its own caption; evidence is assessed against the shared grounding)

The shared grounding is keyed by (image content hash, provenance fingerprint) in a small bounded LRU, so results are
never reused across different detector/OpenCLIP checkpoints or thresholds, and concurrent requests cannot see each
other's entries because the key contains the image content. Each model's caption path is untouched by the sharing.
A model whose checkpoint is missing returns ``{"status": "unavailable"}``; no caption is ever fabricated.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image

from src.config.loader import load_config
from src.grounding.cache import ImageGrounding, compute_grounding, image_fingerprint, make_provenance
from src.inference.predictor import CaptionPredictor
from src.inference.reliability import ReliabilityConfig, assess_caption, overall_flag
from src.visualization.heatmaps import overlay_attention, to_base64_png

logger = logging.getLogger(__name__)
SYSTEMS = ("B1", "B2", "A")
DESCRIPTIONS = {"B1": "Frozen pretrained BLIP", "B2": "LoRA fine-tune, caption loss only", "A": "LoRA + consistency objective"}
RESPONSE_SCHEMA_VERSION = 2


class SharedGroundingProvider:
    """Thread-safe bounded LRU of :class:`ImageGrounding`, computed once per (image, provenance)."""

    def __init__(self, detector, encoder, box_threshold: float = 0.3, nms_iou: float = 0.5, max_entries: int = 8):
        self.detector, self.encoder, self.box_threshold, self.nms_iou = detector, encoder, box_threshold, nms_iou
        self.max_entries = max_entries
        self._lru: "OrderedDict[tuple, ImageGrounding]" = OrderedDict()
        self._lock = threading.Lock()
        self._compute_lock = threading.Lock()   # one detector forward at a time (GPU memory)
        self.compute_calls = 0
        self.provenance = make_provenance(detector, encoder, box_threshold, nms_iou)

    def get(self, image: Image.Image) -> ImageGrounding:
        key = (image_fingerprint(image), self.provenance.fingerprint)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
        with self._compute_lock:
            with self._lock:
                if key in self._lru:
                    return self._lru[key]
            g = compute_grounding(0, image, self.detector, self.encoder, self.box_threshold, self.nms_iou, key[0])
            self.compute_calls += 1
        with self._lock:
            self._lru[key] = g
            while len(self._lru) > self.max_entries:
                self._lru.popitem(last=False)
        return g

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()


def analyse_with(predictor: CaptionPredictor, image: Image.Image, grounding: ImageGrounding, rel_cfg: ReliabilityConfig) -> Dict[str, Any]:
    """Full analysis of one system on one image against an already computed grounding (JSON-serialisable)."""
    gen = predictor.analyse(image)
    z = predictor.project(gen.states)
    toks = assess_caption(gen.words, grounding, z, rel_cfg)
    flag = overall_flag(toks)
    assessed = [i for i, t in enumerate(toks) if t.status != "not_assessed"]
    mean_map = np.mean([gen.attention[i] for i in (assessed or range(len(gen.attention)))], axis=0) if gen.attention else np.zeros((1, 1))
    return {
        "status": "ok", "caption": gen.caption, "tokens": gen.words,
        "reliability": [t.status for t in toks],
        "token_details": [{"word": t.word, "status": t.status, "claim_type": t.claim_type, "similarity": t.q, "category": t.category,
                           "rule": t.rule, "note": t.note, "limitation": t.limitation, "evidence_region_id": t.region_id,
                           "detector_label": t.region.label if t.region else None, "detector_confidence": t.region.score if t.region else None,
                           "bbox": t.region.bbox if t.region else None} for t in toks],
        "attention_map": mean_map.tolist(),
        "heatmap_png": to_base64_png(overlay_attention(image, mean_map)),
        "image_size": list(image.size),
        "overall": flag, "source": "model", "has_token_projection": z is not None,
        "evidence_scope": "object-level only; attributes and relations are not assessed",
    }


class ModelRegistry:
    """Lazy, thread-safe loader of the three systems from ``configs/application.yaml``."""

    def __init__(self, app_cfg: Optional[dict] = None, model_cfg: Optional[dict] = None, det_cfg: Optional[dict] = None,
                 grounding_factory: Optional[Callable[[], SharedGroundingProvider]] = None):
        self.app = app_cfg or load_config("configs/application.yaml")
        self.model_cfg = model_cfg or load_config("configs/model.yaml")
        self.det_cfg = det_cfg or load_config("configs/detector.yaml")
        self.predictors: Dict[str, CaptionPredictor] = {}
        self.grounding: Optional[SharedGroundingProvider] = None
        self._grounding_factory = grounding_factory
        self._load_lock = threading.RLock()
        self._infer_locks: Dict[str, threading.Lock] = {n: threading.Lock() for n in SYSTEMS}
        self.rel_cfg = ReliabilityConfig(**{k: v for k, v in self.app.get("reliability", {}).items() if k in ("confidence_threshold", "consistency_threshold")})

    @property
    def ready(self) -> bool:
        """True once the grounding tools are loaded and at least one model is loaded."""
        return self.grounding is not None and bool(self.predictors)

    def available(self) -> Dict[str, Dict[str, Any]]:
        from src.utils.paths import resolve

        out = {}
        for name in SYSTEMS:
            path = self.app["model_paths"].get(name)
            out[name] = {"checkpoint": path, "exists": True if path is None else resolve(path).is_file(), "loaded": name in self.predictors,
                         "description": DESCRIPTIONS[name]}
        return out

    def ensure_grounding(self) -> SharedGroundingProvider:
        with self._load_lock:
            if self.grounding is None:
                if self._grounding_factory is not None:
                    self.grounding = self._grounding_factory()
                else:
                    from src.grounding.detector import GroundingDinoDetector
                    from src.grounding.region_features import RegionEncoder
                    from src.utils.device import get_device

                    dev = get_device()
                    det = GroundingDinoDetector(self.det_cfg["detector_checkpoint"], device=dev)
                    enc = RegionEncoder(self.model_cfg["clip_model_name"], self.model_cfg.get("clip_pretrained"), dev)
                    self.grounding = SharedGroundingProvider(det, enc, self.det_cfg.get("box_threshold", 0.3), self.det_cfg.get("nms_iou", 0.5))
            return self.grounding

    def get(self, name: str) -> CaptionPredictor:
        if name not in SYSTEMS:
            raise KeyError(f"Unknown model '{name}'. Choose from {SYSTEMS}")
        with self._load_lock:
            if name in self.predictors:
                return self.predictors[name]
            from src.models.blip_model import load_blip
            from src.models.checkpoint import load_trainable_checkpoint
            from src.models.lora import apply_lora
            from src.models.token_projection import TokenProjection
            from src.utils.device import get_device
            from src.utils.paths import resolve

            dev = get_device()
            path = self.app["model_paths"].get(name)
            if path is not None and not resolve(path).is_file():
                raise FileNotFoundError(f"checkpoint for {name} not found")      # never fall back to another model
            model, proc = load_blip(self.model_cfg["blip_model_name"], dev)
            proj = None
            if path is not None:
                apply_lora(model, self.model_cfg["lora_rank"], self.model_cfg["lora_alpha"], self.model_cfg["lora_dropout"], self.model_cfg.get("lora_target_regex"))
                model.to(dev)
                if name == "A":
                    proj = TokenProjection(int(model.config.text_config.hidden_size), int(self.model_cfg.get("clip_dim", 512)),
                                           int(self.model_cfg.get("projection_layers", 1)), int(self.model_cfg.get("projection_hidden", 512))).to(dev).eval()
                load_trainable_checkpoint(resolve(path), model, proj, dev)
            self.predictors[name] = CaptionPredictor(model, proc, dev, proj, self.app.get("decoding"), name)
            return self.predictors[name]

    def caption(self, name: str, image: Image.Image, grounding: Optional[ImageGrounding] = None) -> Dict[str, Any]:
        grounding = grounding or self.ensure_grounding().get(image)
        pred = self.get(name)
        with self._infer_locks[name]:                 # a shared nn.Module is not safe for concurrent generate()
            return analyse_with(pred, image, grounding, self.rel_cfg)

    def compare(self, image: Image.Image) -> Dict[str, Dict[str, Any]]:
        """All systems on one image; the detector/OpenCLIP run once. Per-model ``status``: ok | unavailable | error."""
        out: Dict[str, Dict[str, Any]] = {}
        grounding: Optional[ImageGrounding] = None
        for n in SYSTEMS:
            info = self.available()[n]
            if not info["exists"]:
                out[n] = {"status": "unavailable", "error": f"checkpoint not found: {info['checkpoint']}"}
                continue
            try:
                if grounding is None:
                    grounding = self.ensure_grounding().get(image)
                out[n] = self.caption(n, image, grounding)
            except FileNotFoundError as exc:
                out[n] = {"status": "unavailable", "error": str(exc)}
            except (RuntimeError, ValueError, OSError) as exc:   # incl. CUDA OOM (a RuntimeError subclass)
                logger.exception("system %s failed", n)
                self._free_gpu()
                out[n] = {"status": "error", "error": f"{type(exc).__name__}: inference failed (see server log)"}
        return out

    @staticmethod
    def _free_gpu() -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
