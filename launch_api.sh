#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=$(python -c "from src.config.loader import load_config as l; print(l('configs/application.yaml')['host'])")
PORT=$(python -c "from src.config.loader import load_config as l; print(l('configs/application.yaml')['port'])")
exec uvicorn api.main:app --host "$HOST" --port "$PORT" --workers 1
