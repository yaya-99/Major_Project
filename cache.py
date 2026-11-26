"""Per-image grounding cache: detections plus frozen region embeddings, with full provenance (R07).

Each ``<image_id>.pt`` entry stores the schema version, the fingerprint of the producing configuration
(detector checkpoint/revision/thresholds, OpenCLIP model + weight hash, preprocessing, embedding dim/dtype),
the SHA-256 of the source image bytes and the image size. On every read the entry is validated: a stale or
corrupt entry is REJECTED with an explicit reason, never silently reused. With a detector available the
provider recomputes and overwrites it; otherwise the caller sees a miss (counted in ``provider.rejected``).
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from src.data.manifest import canonical_hash, sha256_file
from src.grounding.detector import BaseDetector, Detection, box_problem, nms

logger = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 2


class StaleCacheError(RuntimeError):
    """A cache entry does not match the requested provenance / image, or is corrupt."""


@dataclass
class ImageGrounding:
    """Detections of one image and their frozen embeddings (row i <-> detections[i])."""
    image_id: int
    detections: List[Detection]
    embeddings: torch.Tensor  # (N, D), L2-normalised
    image_size: Optional[Tuple[int, int]] = None   # (width, height) in pixels, for geometry validation
    image_sha256: str = ""


@dataclass(frozen=True)
class GroundingProvenance:
    """Everything that determines the content of a cache entry (besides the image itself)."""
    detector: Dict[str, Any]
    encoder: Dict[str, Any]
    box_threshold: float
    nms_iou: float
    dtype: str = "float16"
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {"detector": self.detector, "encoder": self.encoder, "box_threshold": self.box_threshold, "nms_iou": self.nms_iou,
                "dtype": self.dtype, "schema_version": self.schema_version}

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def embedding_dim(self) -> int:
        return int(self.encoder.get("embedding_dim", -1))


def make_provenance(detector: Optional[BaseDetector], encoder: Any, box_threshold: float, nms_iou: float,
                    detector_info: Optional[Dict[str, Any]] = None, encoder_info: Optional[Dict[str, Any]] = None) -> GroundingProvenance:
    """Build the provenance from live objects (or explicit info dicts when loading from cache only)."""
    det = detector_info if detector_info is not None else (detector.provenance() if detector is not None else {"name": "none"})
    enc = encoder_info if encoder_info is not None else (encoder.provenance() if encoder is not None and hasattr(encoder, "provenance")
                                                         else {"name": type(encoder).__name__, "embedding_dim": int(getattr(encoder, "dim", -1))})
    return GroundingProvenance(det, enc, float(box_threshold), float(nms_iou))


def image_fingerprint(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels + size (for images that exist only in memory)."""
    im = image.convert("RGB")
    return hashlib.sha256(f"{im.size}".encode() + im.tobytes()).hexdigest()


def compute_grounding(image_id: int, image: Image.Image, detector: BaseDetector, encoder, box_threshold: float = 0.3,
                      nms_iou: float = 0.5, image_sha256: str = "") -> ImageGrounding:
    """Run detector + geometry validation + NMS + region encoder for one image (invalid boxes are dropped and logged)."""
    size = image.size
    raw = detector.detect(image, box_threshold, key=image_id)
    dets: List[Detection] = []
    for d in raw:
        why = box_problem(d.bbox, size)
        if why is not None:
            logger.warning("image %s: dropping %s box %s (%s)", image_id, d.label, d.bbox, why)
            continue
        dets.append(d)
    dets = nms(dets, nms_iou)
    emb = encoder.encode(image, [d.bbox for d in dets])
    return ImageGrounding(int(image_id), dets, emb, size, image_sha256 or image_fingerprint(image))


class GroundingCache:
    """Disk cache (+ in-memory layer) keyed by image id and validated against a :class:`GroundingProvenance`."""

    def __init__(self, cache_dir: str | Path, provenance: GroundingProvenance, memory: bool = True):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        self.last_rejection: Optional[str] = None   # reason the last get() rejected a stale/corrupt entry
        self.memory: Optional[Dict[int, ImageGrounding]] = {} if memory else None

    def _path(self, image_id: int) -> Path:
        return self.dir / f"{int(image_id)}.pt"

    def has(self, image_id: int) -> bool:
        return (self.memory is not None and int(image_id) in self.memory) or self._path(image_id).exists()

    def put(self, g: ImageGrounding) -> None:
        self._validate_tensors(g.embeddings, len(g.detections), f"image {g.image_id}")
        torch.save({"schema_version": CACHE_SCHEMA_VERSION, "provenance_fingerprint": self.provenance.fingerprint,
                    "provenance": self.provenance.to_dict(), "image_id": g.image_id, "image_sha256": g.image_sha256,
                    "image_size": list(g.image_size) if g.image_size else None, "detections": [d.to_dict() for d in g.detections],
                    "embeddings": g.embeddings.half()}, self._path(g.image_id))
        if self.memory is not None:
            self.memory[g.image_id] = g

    def _validate_tensors(self, emb: torch.Tensor, n_det: int, where: str) -> None:
        dim = self.provenance.embedding_dim
        if emb.ndim != 2 or emb.shape[0] != n_det:
            raise StaleCacheError(f"{where}: embeddings shape {tuple(emb.shape)} does not match {n_det} detections")
        if dim > 0 and emb.shape[1] != dim:
            raise StaleCacheError(f"{where}: embedding dimension {emb.shape[1]} != expected {dim}")
        if not bool(torch.isfinite(emb.float()).all()):
            raise StaleCacheError(f"{where}: non-finite values in embeddings")

    def get(self, image_id: int, image_path: Optional[str | Path] = None, strict: bool = False) -> Optional[ImageGrounding]:
        """Return a validated entry, ``None`` on a plain miss, and ``None`` (or raise if ``strict``) for a stale/corrupt entry.

        If ``image_path`` is given, the entry's image fingerprint must equal the SHA-256 of that file, so a replaced image
        with an unchanged file name is detected.
        """
        image_id = int(image_id)
        self.last_rejection = None
        sha_now = sha256_file(image_path) if image_path is not None else None
        if self.memory is not None and image_id in self.memory:
            g = self.memory[image_id]
            if sha_now is None or not g.image_sha256 or g.image_sha256 == sha_now:
                return g
            del self.memory[image_id]
        p = self._path(image_id)
        if not p.exists():
            return None
        try:
            g = self._read(p, image_id, sha_now)
        except StaleCacheError as exc:
            if strict:
                raise
            logger.warning("Rejecting cache entry %s: %s", p.name, exc)
            self.last_rejection = str(exc)
            return None
        if self.memory is not None:
            self.memory[image_id] = g
        return g

    def _read(self, p: Path, image_id: int, sha_now: Optional[str]) -> ImageGrounding:
        try:
            raw = torch.load(p, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError, EOFError, OSError, ValueError, KeyError) as exc:  # corrupt / truncated / unsafe file
            raise StaleCacheError(f"unreadable or corrupt cache file ({type(exc).__name__}: {exc})") from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise StaleCacheError(f"schema version {raw.get('schema_version') if isinstance(raw, dict) else '?'} != {CACHE_SCHEMA_VERSION}; re-run precompute_grounding")
        if raw.get("provenance_fingerprint") != self.provenance.fingerprint:
            raise StaleCacheError("produced with a different detector / OpenCLIP checkpoint / threshold / preprocessing configuration")
        if int(raw.get("image_id", -1)) != image_id:
            raise StaleCacheError(f"entry belongs to image {raw.get('image_id')}")
        if sha_now is not None and raw.get("image_sha256") != sha_now:
            raise StaleCacheError("source image content changed since the entry was computed")
        dets = [Detection(**d) for d in raw["detections"]]
        emb = raw["embeddings"].float()
        self._validate_tensors(emb, len(dets), p.name)
        size = tuple(raw["image_size"]) if raw.get("image_size") else None
        return ImageGrounding(image_id, dets, emb, size, raw.get("image_sha256", ""))


class GroundingProvider:
    """Cache-first access to grounding; computes (and stores) missing/stale entries when a detector is set."""

    def __init__(self, cache: GroundingCache, detector: Optional[BaseDetector] = None, encoder=None,
                 box_threshold: float = 0.3, nms_iou: float = 0.5):
        self.cache, self.detector, self.encoder = cache, detector, encoder
        self.box_threshold, self.nms_iou = box_threshold, nms_iou
        self.misses = 0
        self.rejected = 0

    def get(self, image_id: int, image_path: Optional[str] = None) -> Optional[ImageGrounding]:
        g = self.cache.get(image_id, image_path)
        if g is not None:
            return g
        if self.cache.last_rejection is not None:
            self.rejected += 1
        if self.detector is None or self.encoder is None or image_path is None:
            self.misses += 1
            return None
        sha = sha256_file(image_path)
        with Image.open(image_path) as im:
            g = compute_grounding(image_id, im.convert("RGB"), self.detector, self.encoder, self.box_threshold, self.nms_iou, sha)
        self.cache.put(g)
        return g
