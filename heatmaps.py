"""Attention heatmap overlays."""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from matplotlib import colormaps
from PIL import Image

from src.visualization.attention import resize_map


def overlay_attention(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap over ``image`` (map is resized to the image size)."""
    image = image.convert("RGB")
    m = resize_map(heatmap, image.size)
    m = (m - m.min()) / max(m.max() - m.min(), 1e-12)
    rgb = (colormaps["jet"](m)[..., :3] * 255).astype(np.uint8)
    weight = (alpha * m)[..., None]  # transparent where attention is low
    out = np.asarray(image, np.float32) * (1 - weight) + rgb.astype(np.float32) * weight
    return Image.fromarray(out.astype(np.uint8))


def to_base64_png(image: Image.Image, max_side: Optional[int] = 640) -> str:
    """PNG data (base64, no URI prefix); optionally downscaled for transport."""
    im = image.copy()
    if max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
