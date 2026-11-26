"""Cross-attention extraction.

Following the Minor Project: the FINAL decoder layer's cross-attention, averaged over heads,
with the [CLS] image position dropped and the patch grid reshaped (24x24 for 384px inputs,
derived from the tensor so any image size works), then normalised to a distribution.
Attention is read at the *prediction position* of a token (position p-1 for the token at p).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def attention_to_grid(attn_row: torch.Tensor | np.ndarray) -> np.ndarray:
    """(heads, patches+1) or (patches+1,) -> normalised (g, g) map; drops the CLS column."""
    a = attn_row.detach().cpu().numpy() if torch.is_tensor(attn_row) else np.asarray(attn_row)
    if a.ndim == 2:
        a = a.mean(0)
    a = a[1:]
    g = int(round(np.sqrt(a.shape[0])))
    if g * g != a.shape[0]:
        raise ValueError(f"Cannot reshape {a.shape[0]} patches into a square grid")
    a = a.reshape(g, g).astype(np.float64)
    a = a - a.min() if a.min() < 0 else a
    s = a.sum()
    return a / s if s > 0 else np.full((g, g), 1.0 / (g * g))


def token_attention_map(cross_attentions: Sequence[torch.Tensor], position: int, batch_index: int = 0) -> np.ndarray:
    """Attention map of the sequence position ``position`` from the last layer's cross-attention."""
    last = cross_attentions[-1]            # (B, heads, T, patches+1)
    return attention_to_grid(last[batch_index, :, position, :])


def resize_map(grid: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize a map to ``size=(width, height)``."""
    img = Image.fromarray((grid / max(grid.max(), 1e-12) * 255).astype(np.uint8))
    return np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


@torch.no_grad()
def extract_cross_attention(model, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            token_index: int) -> np.ndarray:
    """Teacher-forced pass over ``input_ids``; returns the attention grid read at ``token_index``'s prediction position."""
    from src.training.teacher_forcing import teacher_forcing_step

    out = teacher_forcing_step(model, {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask},
                               output_attentions=True, compute_loss=False)
    if out.cross_attentions is None:
        raise RuntimeError("Model returned no cross-attentions")
    return token_attention_map(out.cross_attentions, max(token_index - 1, 0))
