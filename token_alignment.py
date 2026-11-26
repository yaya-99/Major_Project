"""Draw detector regions linked to caption tokens."""
from __future__ import annotations

from typing import Dict, Sequence

from PIL import Image, ImageDraw

COLORS = {"supported_by_detected_region": (34, 160, 90), "possible_misidentification": (230, 140, 20), "possible_hallucination": (210, 50, 50),
          "insufficient_visual_evidence": (150, 100, 200), "not_assessed": (120, 120, 120)}


def draw_token_boxes(image: Image.Image, items: Sequence[Dict]) -> Image.Image:
    """Draw ``items`` = [{"word", "bbox", "status"}] on a copy of the image."""
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for it in items:
        if not it.get("bbox"):
            continue
        c = COLORS.get(it.get("status", "not_assessed"), COLORS["not_assessed"])
        x1, y1, x2, y2 = it["bbox"]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        d.text((x1 + 3, y1 + 2), it["word"], fill=c)
    return im
