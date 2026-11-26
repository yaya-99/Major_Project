"""Run the REAL pipeline on curated images and save each analysis for offline cached replay.

Usage (repository root):  python -m demo_app.prepare_cases
Reads demo_app/images/manifest.json:  {"images": [{"file", "source", "license", "purpose"}]}  (same format as
examples/demo_images/manifest.json, validated by src.inference.demo.load_demo_manifest). Writes demo_app/cases/<stem>.json.
Nothing is written for an image that fails; no output is ever fabricated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_app.engine import DemoEngine  # noqa: E402
from src.config.loader import load_config  # noqa: E402
from src.inference.demo import load_demo_manifest  # noqa: E402
from src.utils.paths import resolve  # noqa: E402


def main() -> int:
    cfg = load_config("configs/demo.yaml")
    man = load_demo_manifest(resolve(cfg["images_dir"]))
    for p in man["problems"]:
        print("skipped:", p)
    if man["empty"]:
        print("No valid images. Add images and demo_app/images/manifest.json first.")
        return 1
    eng = DemoEngine(cfg, load_config("configs/model.yaml"), load_config("configs/detector.yaml"))
    eng.load()
    if not eng.ready:
        print("Model loading failed:", eng.load_error)
        return 2
    out_dir = resolve(cfg["cases_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in man["images"]:
        with Image.open(e["path"]) as im:
            image = im.convert("RGB")
        st = eng.analyse(image, mode="live", meta={"file": e["file"], "sha256": e["sha256"]})
        stem = Path(e["file"]).stem
        rec = eng.case_record(st, stem, e["source"], e["license"], e["purpose"])
        (out_dir / f"{stem}.json").write_text(json.dumps(rec), encoding="utf-8")
        counts = rec["summary"]["diagnostic_counts"]
        print(f"{e['file']}: '{rec['caption']}'  eligible={rec['summary']['eligible_words']}  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
