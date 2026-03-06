#!/bin/bash

set -e

# Go to gsplat project root
cd "$(dirname "$0")/.."

CONFIG_PATH="lcxscripts/viewer_config.yaml"

if [ $# -ge 1 ]; then
  CKPT_PATH="$1"
  shift
  echo "[LCX Viewer] Using config: ${CONFIG_PATH}"
  echo "[LCX Viewer] Using checkpoint (from CLI): ${CKPT_PATH}"
  echo "[LCX Viewer] Extra args will be forwarded to Python (e.g. --enable_viewer)"

  PYTHONPATH="$(pwd):${PYTHONPATH}" python -m lcxscripts.viewer \
    --config "${CONFIG_PATH}" \
    --ckpt "${CKPT_PATH}" \
    "$@"
else
  echo "[LCX Viewer] Using config: ${CONFIG_PATH}"
  echo "[LCX Viewer] No CKPT_PATH given on CLI, will use viewer_ckpt from YAML (if set)."
  echo "[LCX Viewer] Extra args will be forwarded to Python (e.g. --enable_viewer)"

  PYTHONPATH="$(pwd):${PYTHONPATH}" python -m lcxscripts.viewer \
    --config "${CONFIG_PATH}" \
    "$@"
fi


