"""``python -m demo_app`` -> start the demo server (host/port from configs/demo.yaml)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from src.config.loader import load_config  # noqa: E402

cfg = load_config("configs/demo.yaml")
print(f"\n  Caption Reliability Inspector demo -> http://{cfg['host']}:{cfg['port']}\n"
      "  (first start downloads BLIP, Grounding DINO and OpenCLIP, about 2.3 GB; the page shows when models are ready)\n")
uvicorn.run("demo_app.server:app", host=cfg["host"], port=int(cfg["port"]), workers=1)
