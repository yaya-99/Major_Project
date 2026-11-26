"""Curated demo images and their manifest (R17).

``examples/demo_images/manifest.json`` lists each curated image with its source, licence and purpose. Images are never
bundled unless their licence permits redistribution; an empty or invalid manifest yields an explicit empty state, never a
fabricated image or output. Anything served from this set is DISCLOSED as a curated fallback, not a live upload.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, UnidentifiedImageError

REQUIRED = ("file", "source", "license", "purpose")


class DemoManifestError(ValueError):
    """Raised for a malformed demo manifest."""


def load_demo_manifest(directory: str | Path) -> Dict[str, Any]:
    """Validate the manifest; returns ``{"images": [valid entries], "problems": [...], "empty": bool}``.

    A missing manifest is an empty state (not an error). Entries with a missing file, undecodable image or missing
    metadata are excluded and reported in ``problems``.
    """
    d = Path(directory)
    mf = d / "manifest.json"
    if not mf.is_file():
        return {"images": [], "problems": [f"no manifest at {mf}"], "empty": True}
    try:
        raw = json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoManifestError(f"{mf}: invalid JSON ({exc})") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("images"), list):
        raise DemoManifestError(f"{mf}: expected an object with an 'images' list")
    valid: List[Dict[str, Any]] = []
    problems: List[str] = []
    for i, e in enumerate(raw["images"]):
        missing = [k for k in REQUIRED if not isinstance(e, dict) or not str(e.get(k, "")).strip()]
        if missing:
            problems.append(f"entry {i}: missing {missing}")
            continue
        name = Path(e["file"]).name
        if name != e["file"]:
            problems.append(f"entry {i}: file must be a plain file name, got {e['file']!r}")
            continue
        p = d / name
        if not p.is_file():
            problems.append(f"entry {i}: file not found: {name}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except (UnidentifiedImageError, OSError):
            problems.append(f"entry {i}: {name} is not a valid image")
            continue
        valid.append({**e, "file": name, "path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    valid.sort(key=lambda e: e["file"])
    return {"images": valid, "problems": problems, "empty": not valid}


def pick_fallback(images: List[Dict[str, Any]], key: str = "") -> Optional[Dict[str, Any]]:
    """Deterministic fallback choice: the same key always selects the same image (sorted by file name)."""
    if not images:
        return None
    ordered = sorted(images, key=lambda e: e["file"])
    return ordered[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(ordered)]
