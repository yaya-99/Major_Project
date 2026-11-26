"""Token-region pairing and same-image hard-negative mining (R05/R06).

Hard negatives (v6 specification, four steps): (1) take every other detector region of the SAME image;
(2) drop regions overlapping r+ beyond an IoU limit and regions with the same label as r+ (a second dog is
never a negative for the first, and the positive itself is never a negative); (3) rank the rest by cosine
similarity to u+ under the frozen encoder and keep the top K, ties broken by region index (deterministic, no
sampling); (4) if nothing is left, skip the token - the declared no-negative policy. Negatives are never
fabricated and never come from another image, because they are indexed out of this image's own grounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from src.grounding.cache import ImageGrounding
from src.grounding.detector import Detection, iou
from src.grounding.eligibility import NO_VALID_NEGATIVES, TokenRecord


@dataclass
class TokenRegionPair:
    """An eligible (token, positive region) pair with its hard negatives."""
    record: TokenRecord
    positive: torch.Tensor            # (D,)
    negatives: torch.Tensor           # (K', D) with K' <= K
    positive_index: int
    negative_indices: List[int]


def find_region_index(grounding: ImageGrounding, region: Detection) -> int:
    """Index of ``region`` in ``grounding.detections`` (matched by box and label)."""
    for i, d in enumerate(grounding.detections):
        if d.label == region.label and d.bbox == region.bbox:
            return i
    raise ValueError("Selected region is not part of the image grounding")


def mine_hard_negatives(grounding: ImageGrounding, pos_index: int, k: int, iou_limit: float = 0.5) -> List[int]:
    """Indices of up to ``k`` same-image hard negatives for the region at ``pos_index``."""
    pos = grounding.detections[pos_index]
    u_pos = grounding.embeddings[pos_index]
    cand = []
    for j, d in enumerate(grounding.detections):
        if j == pos_index or d.label == pos.label or iou(d.bbox, pos.bbox) > iou_limit:
            continue
        cand.append((float(torch.dot(grounding.embeddings[j], u_pos)), j))
    cand.sort(key=lambda t: (-t[0], t[1]))   # most similar first; ties -> lower region index
    return [j for _, j in cand[:k]]


def build_pairs(records: Sequence[TokenRecord], grounding: Optional[ImageGrounding], num_negatives: int,
                iou_limit: float = 0.5) -> List[TokenRegionPair]:
    """Pairs for all eligible tokens that have at least one hard negative.

    A token without any valid negative is skipped and its record is re-labelled ``NO_VALID_NEGATIVES`` (is_eligible=False)
    so the exclusion accounting stays truthful.
    """
    if grounding is None or len(grounding.detections) == 0:
        return []
    pairs: List[TokenRegionPair] = []
    for r in records:
        if not r.is_eligible or r.selected_region is None:
            continue
        pi = find_region_index(grounding, r.selected_region)
        neg = mine_hard_negatives(grounding, pi, num_negatives, iou_limit)
        if not neg:  # step 4: skip the token
            r.is_eligible, r.exclusion_reason, r.rule = False, NO_VALID_NEGATIVES, "no same-image region left after label/IoU exclusion"
            continue
        pairs.append(TokenRegionPair(r, grounding.embeddings[pi], grounding.embeddings[neg], pi, neg))
    return pairs
