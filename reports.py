"""Composite report figures (image, caption, attention overlay)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

from src.visualization.heatmaps import overlay_attention  # noqa: E402


def caption_panel(image: Image.Image, caption: str, heatmap: Optional[np.ndarray] = None, out_path: Optional[str | Path] = None,
                  title: str = "") -> Figure:
    """Side-by-side panel: original image and attention overlay with the caption as a title."""
    n = 2 if heatmap is not None else 1
    fig = Figure(figsize=(5 * n, 5.4))
    axes = fig.subplots(1, n, squeeze=False)[0]
    axes[0].imshow(image)
    axes[0].set_axis_off()
    if heatmap is not None:
        axes[1].imshow(overlay_attention(image, heatmap))
        axes[1].set_axis_off()
    fig.suptitle(f"{title}\n{caption}" if title else caption, fontsize=10, wrap=True)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig
