"""Token-to-region association from a word's cross-attention map (pure NumPy, no model needed).

BLIP's image processor resizes every image to a square (384x384 for the base checkpoint) WITHOUT cropping or padding,
so a patch grid cell (i, j) of a g x g map covers the normalised image rectangle
x in [j/g, (j+1)/g], y in [i/g, (i+1)/g] of the ORIGINAL image. A region box [x1, y1, x2, y2] in original pixels is
normalised by the image size and each cell contributes in proportion to its fractional overlap with the box:

    A(t, r) = sum_p  A_t(p) * overlap(p, r) / area(p)

A_t sums to 1 over the grid (src.visualization.attention.attention_to_grid), so A(t, r) is the share of the word's
attention that falls inside region r. ``concentration`` = A(t, r) / (area of r as a fraction of the image) is the lift
over uniform attention (1.0 = no more attention than a region of that size would get by chance).

The association is CLASS-AGNOSTIC on purpose: the region a word is linked to is chosen from attention only, never from
whether the word agrees with the detector label, so a wrong word ("cat") can still be linked to the region it actually
attended (a dog) and the disagreement can then be measured separately (no circular reasoning).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _axis_overlap(g: int, lo: float, hi: float) -> np.ndarray:
    """Fraction of each of the g equal cells of [0, 1] covered by [lo, hi]."""
    edges = np.arange(g + 1) / g
    left, right = edges[:-1], edges[1:]
    cover = np.clip(np.minimum(right, hi) - np.maximum(left, lo), 0.0, None)
    return cover * g


def region_attention_mass(grid: np.ndarray, bbox: Sequence[float], image_size: Tuple[int, int]) -> float:
    """Share of the attention distribution ``grid`` (g x g, sums to 1) inside ``bbox`` (original pixels)."""
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError("image size must be positive")
    g_h, g_w = grid.shape
    x1, y1, x2, y2 = (float(v) for v in bbox)
    fx = _axis_overlap(g_w, max(0.0, x1 / w), min(1.0, x2 / w))
    fy = _axis_overlap(g_h, max(0.0, y1 / h), min(1.0, y2 / h))
    return float((grid * np.outer(fy, fx)).sum())


def box_area_fraction(bbox: Sequence[float], image_size: Tuple[int, int]) -> float:
    w, h = image_size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return max(0.0, min(x2, w) - max(x1, 0.0)) * max(0.0, min(y2, h) - max(y1, 0.0)) / float(w * h)


def associate(grid: np.ndarray, boxes: Sequence[Sequence[float]], image_size: Tuple[int, int],
              min_mass: float, min_margin: float) -> Dict[str, object]:
    """Rank all regions by attention mass and decide the association.

    Returns ``{"candidates": [{region_id, mass, concentration}] (best first), "selected_region_id", "top_mass", "margin",
    "status", "reason"}`` where status is ``associated`` | ``ambiguous`` | ``insufficient`` | ``no_regions``.
    """
    if len(boxes) == 0:
        return {"candidates": [], "selected_region_id": None, "top_mass": None, "margin": None, "status": "no_regions",
                "reason": "the detector returned no region for this image"}
    cands: List[Dict[str, float]] = []
    for i, b in enumerate(boxes):
        m = region_attention_mass(grid, b, image_size)
        a = box_area_fraction(b, image_size)
        cands.append({"region_id": i, "mass": m, "area_fraction": a, "concentration": m / a if a > 0 else None})
    cands.sort(key=lambda c: (-c["mass"], c["region_id"]))
    top = cands[0]["mass"]
    second = cands[1]["mass"] if len(cands) > 1 else 0.0
    margin = top - second
    if top < min_mass:
        status, reason = "insufficient", f"best region holds only {top:.2f} of the word's attention (< {min_mass})"
    elif len(cands) > 1 and margin < min_margin:
        status, reason = "ambiguous", f"top-2 regions are within {margin:.3f} attention mass (< {min_margin})"
    else:
        status, reason = "associated", f"region holds {top:.2f} of the attention, {margin:.2f} more than the runner-up"
    return {"candidates": cands, "selected_region_id": cands[0]["region_id"], "top_mass": top, "margin": margin,
            "status": status, "reason": reason}
