"""Shared FastAPI dependencies. Tests override :func:`get_registry`."""
from __future__ import annotations

import io
import time
from functools import lru_cache
from typing import Any

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.config.loader import load_config

START_TIME = time.time()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 40_000_000
MIN_SIDE = 16
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@lru_cache(maxsize=1)
def get_app_config() -> dict:
    """Application configuration (configs/application.yaml)."""
    return load_config("configs/application.yaml")


@lru_cache(maxsize=1)
def get_registry() -> Any:
    """Process-wide lazy model registry (nothing heavy is loaded until the first request)."""
    from src.inference.comparison import ModelRegistry

    return ModelRegistry(get_app_config())


async def read_image(file: UploadFile) -> Image.Image:
    """Decode an upload with explicit size/format/dimension limits and clear JSON errors (no local paths or tracebacks)."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 12 MB")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, (w, h) = probe.format, probe.size
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(status_code=415, detail=f"Unsupported image format {fmt}; use JPEG, PNG or WEBP")
        if w * h > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image has too many pixels")
        if min(w, h) < MIN_SIDE:
            raise HTTPException(status_code=400, detail=f"Image is smaller than {MIN_SIDE}px on one side")
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="File is not a valid image")
