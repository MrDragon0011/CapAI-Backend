#!/usr/bin/env bash
set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
HOLISTIC_TASK="$BACKEND_DIR/holistic_landmarker.task"

if [ -f "$HOLISTIC_TASK" ]; then
    echo "[startup] holistic_landmarker.task already present"
    exit 0
fi

echo "[startup] Downloading holistic_landmarker.task from MediaPipe CDN..."
curl -fSL \
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" \
    -o "$HOLISTIC_TASK"
echo "[startup] holistic_landmarker.task downloaded ($(du -sh "$HOLISTIC_TASK" | cut -f1))"
