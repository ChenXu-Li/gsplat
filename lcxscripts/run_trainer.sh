#!/bin/bash

set -e

# Go to gsplat project root
cd "$(dirname "$0")/.."

CONFIG_PATH="lcxscripts/trainer_config.yaml"

echo "[LCX Trainer] Using config: ${CONFIG_PATH}"
echo "[LCX Trainer] Extra args will be forwarded to Python (e.g. --steps_scaler 0.25)"

PYTHONPATH="$(pwd):${PYTHONPATH}" python lcxscripts/trainer.py --config "${CONFIG_PATH}" "$@"

