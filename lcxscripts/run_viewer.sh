#!/bin/bash

set -e

# Go to gsplat project root
cd "$(dirname "$0")/.."

CONFIG_PATH="lcxscripts/viewer_config.yaml"

# Parse arguments
ENABLE_VIEWER=""
CKPT_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --enable_viewer)
      ENABLE_VIEWER="--enable_viewer"
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --*)
      # Other flags passed through
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      # First non-flag argument is treated as checkpoint path
      if [ -z "$CKPT_PATH" ]; then
        CKPT_PATH="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

# Print status
echo "[LCX Viewer] Using config: ${CONFIG_PATH}"

if [ -n "$CKPT_PATH" ]; then
  echo "[LCX Viewer] Using checkpoint (from CLI): ${CKPT_PATH}"
else
  echo "[LCX Viewer] No CKPT_PATH given on CLI, will use viewer_ckpt from YAML (if set)."
fi

if [ -n "$ENABLE_VIEWER" ]; then
  echo "[LCX Viewer] Interactive viewer ENABLED (will open in browser)"
  echo "[LCX Viewer] URL: http://localhost:8080 (default)"
else
  echo "[LCX Viewer] Running in OFFLINE mode (renders only)"
  echo "[LCX Viewer] Tip: Add --enable_viewer to launch interactive browser viewer"
fi

# Export PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Build command arguments
ARGS=("--config" "${CONFIG_PATH}")

if [ -n "$CKPT_PATH" ]; then
  ARGS+=("--ckpt" "${CKPT_PATH}")
fi

if [ -n "$ENABLE_VIEWER" ]; then
  ARGS+=(${ENABLE_VIEWER})
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  ARGS+=(${EXTRA_ARGS[@]})
fi

# Run
python -m lcxscripts.viewer "${ARGS[@]}"
