"""Download/verify everything the demo needs and run one real smoke inference.

Usage (repository root, internet needed the first time):  python -m demo_app.preflight
Run it once after installing, and again on presentation day (it then works offline from the Hugging Face cache).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from demo_app.engine import DemoEngine  # noqa: E402
from src.config.loader import load_config  # noqa: E402
from src.utils.paths import resolve  # noqa: E402


def main() -> int:
    ok = True
    print("1/4 static files:", "ok" if (ROOT / "demo_app/static/index.html").is_file() else "MISSING")
    eng = DemoEngine(load_config("configs/demo.yaml"), load_config("configs/model.yaml"), load_config("configs/detector.yaml"))
    print("2/4 loading BLIP, Grounding DINO, OpenCLIP and NLTK data (first run downloads about 2.3 GB)...")
    eng.load()
    if not eng.ready:
        print("    FAILED:", eng.load_error)
        return 2
    info = eng.info()
    print(f"    ok in {info['load_seconds']} s on {info['device']}; POS tagger: {info['pos_tagger']['tagger']}")
    if info["pos_tagger"]["tagger"] != "nltk.pos_tag (averaged_perceptron_tagger_eng)":
        print("    WARNING: NLTK data could not be downloaded; the conservative COCO-lexicon fallback tagger is in use.")
        ok = False
    print("3/4 smoke inference on a synthetic image...")
    y, x = np.mgrid[0:240, 0:320]
    img = Image.fromarray(np.stack([x % 256, y % 256, (x + y) % 256], -1).astype("uint8"))
    t = time.time()
    st = eng.analyse(img)
    print(f"    caption: '{st.public['caption']}'  ({time.time() - t:.1f} s, {len(st.public['regions'])} regions)")
    cases = sorted(resolve(load_config("configs/demo.yaml")["cases_dir"]).glob("*.json"))
    print(f"4/4 saved cases for offline replay: {len(cases)}" + ("" if cases else "  (optional: python -m demo_app.prepare_cases)"))
    print("\nPREFLIGHT", "PASSED" if ok else "PASSED WITH WARNINGS", "- start the demo with:  python -m demo_app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
