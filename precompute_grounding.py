#!/usr/bin/env python
"""Precompute detector boxes and frozen OpenCLIP region embeddings for a split (one validated .pt per image).

Also pins the provenance (detector checkpoint/revision, OpenCLIP weight hash, preprocessing, thresholds) to
``data/grounding_provenance.json`` so that training with ``--no-compute-grounding`` can validate the cache it reads.
Stale or incompatible entries are recomputed; nothing is reused silently.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loader import load_config  # noqa: E402
from src.data.annotations import merge_caption_annotations  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.grounding.cache import GroundingCache, GroundingProvider, make_provenance  # noqa: E402
from src.grounding.detector import GroundingDinoDetector  # noqa: E402
from src.grounding.region_features import RegionEncoder  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.paths import resolve  # noqa: E402

logger = logging.getLogger("precompute_grounding")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    data, model, det = (load_config(f"configs/{n}.yaml") for n in ("data", "model", "detector"))
    ids = load_split(resolve(data[{"train": "train_ids", "dev": "val_ids", "test": "test_ids"}[args.split]]))[:args.limit]
    ann = merge_caption_annotations([resolve(a) for a in data["caption_annotations"]])
    info = {im["id"]: im["file_name"] for im in ann["images"]}
    device = get_device()
    detector = GroundingDinoDetector(det["detector_checkpoint"], device=device, text_threshold=det.get("text_threshold", 0.25))
    encoder = RegionEncoder(model["clip_model_name"], model.get("clip_pretrained"), device)
    prov = make_provenance(detector, encoder, det.get("box_threshold", 0.3), det.get("nms_iou", 0.5))
    pin = resolve(det.get("provenance_file", "data/grounding_provenance.json"))
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps({"detector": prov.detector, "encoder": prov.encoder, "fingerprint": prov.fingerprint}, indent=2))
    provider = GroundingProvider(GroundingCache(resolve(data["grounding_cache_dir"]), prov, memory=False), detector, encoder,
                                 det.get("box_threshold", 0.3), det.get("nms_iou", 0.5))
    done = 0
    for n, iid in enumerate(ids):
        path = next((Path(resolve(d)) / info[iid] for d in data["image_dirs"] if (Path(resolve(d)) / info[iid]).exists()), None)
        if path is None:
            logger.warning("image %s missing, skipped", iid)
            continue
        if provider.get(iid, str(path)) is not None:
            done += 1
        if n % 100 == 0:
            logger.info("%d/%d images grounded (rejected stale entries: %d)", n, len(ids), provider.rejected)
    logger.info("Grounded %d/%d images; provenance pinned to %s", done, len(ids), pin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
