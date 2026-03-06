#!/bin/bash

set -e

# Go to gsplat project root
cd "$(dirname "$0")/.."

CONFIG_PATH="lcxscripts/pandepth_config.yaml"

echo "[LCX Pandepth Exporter] Using config: ${CONFIG_PATH}"
echo "[LCX Pandepth Exporter] You can override config with: --config /path/to/config.yaml"

PYTHONPATH="$(pwd):${PYTHONPATH}" python -m lcxscripts.pandepth_exporter --config "${CONFIG_PATH}" "$@"
