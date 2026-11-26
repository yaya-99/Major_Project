"""Frozen OpenCLIP region encoder f_v (ViT-B/32, LAION-2B by default)."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.manifest import canonical_hash
from src.models.frozen import lock_eval

logger = logging.getLogger(__name__)


class RegionEncoder:
    """Embeds image crops with a frozen, locked OpenCLIP model.

    ``f_v`` is the composition of the image tower and its fixed projection; its parameters never
    receive gradients, which removes the visual-side freedom needed for representation collapse.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: Optional[str] = "laion2b_s34b_b79k",
                 device: Optional[torch.device] = None, batch_size: int = 32):
        import open_clip

        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        except (RuntimeError, OSError, ValueError, KeyError) as exc:   # download / checkpoint / unknown-tag failures
            raise RuntimeError(f"Could not load OpenCLIP '{model_name}' ({pretrained}): {exc}") from exc
        self.model = lock_eval(model.to(self.device))  # frozen AND permanently in eval mode (R03)
        self.preprocess = preprocess
        self.model_name, self.pretrained = model_name, pretrained
        self.dim = int(self.model.visual.output_dim)

    def provenance(self) -> dict:
        """Identity of the frozen visual target f_v: model, pretrained tag, weight fingerprint, preprocessing, dimension."""
        import hashlib

        import open_clip

        h = hashlib.sha256()
        for name, t in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
        pre = {"transforms": [repr(t) for t in getattr(self.preprocess, "transforms", [])]}
        return {"model": self.model_name, "pretrained": self.pretrained, "open_clip_version": getattr(open_clip, "__version__", "unknown"),
                "weights_sha256": h.hexdigest(), "preprocess_sha256": canonical_hash(pre), "embedding_dim": self.dim}

    @staticmethod
    def crop(image: Image.Image, bbox: Sequence[float]) -> Image.Image:
        """Crop a box, clamped to the image and at least 1x1 pixel."""
        w, h = image.size
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, max(int(bbox[2]), x1 + 1)), min(h, max(int(bbox[3]), y1 + 1))
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        return image.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))

    @torch.no_grad()
    def encode(self, image: Image.Image, boxes: Sequence[Sequence[float]]) -> torch.Tensor:
        """Return L2-normalised embeddings of shape (N, D) (float32, CPU)."""
        if len(boxes) == 0:
            return torch.zeros(0, self.dim)
        outs: List[torch.Tensor] = []
        crops = [self.preprocess(self.crop(image, b)) for b in boxes]
        for i in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[i:i + self.batch_size]).to(self.device)
            feats = self.model.encode_image(batch).float()
            outs.append(F.normalize(feats, dim=-1).cpu())
        return torch.cat(outs, 0)
