#!/usr/bin/env python
"""Precompute example outputs (cached demo mode) for the curated images in examples/demo_images/manifest.json.

Every output is labelled ``cached_example``; an empty or invalid manifest is an explicit failure (exit 1), nothing is fabricated.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.config.loader import load_config  # noqa: E402
from src.inference.comparison import ModelRegistry
from src.inference.demo import load_demo_manifest  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.paths import resolve  # noqa: E402

logger = logging.getLogger("prepare_demo")


def main() -> int:
    setup_logging()
    cfg = load_config("configs/application.yaml")
    reg = ModelRegistry(cfg)
    out_dir = resolve("frontend/public/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_demo_manifest(resolve(cfg.get("demo_images_dir", "examples/demo_images")))
    for problem in manifest["problems"]:
        logger.warning("demo manifest: %s", problem)
    done = 0
    for entry in manifest["images"]:
        p = Path(entry["path"])
        res = reg.compare(Image.open(p).convert("RGB"))
        for r in res.values():
            if r.get("status", "ok") == "ok":
                r["source"] = "cached_example"
                r["curated"] = {k: entry[k] for k in ("source", "license", "purpose")}
        (out_dir / f"{p.stem}.json").write_text(json.dumps(res))
        done += 1
    logger.info("Cached %d example(s) in %s", done, out_dir)
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
