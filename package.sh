#!/bin/bash
# Package submission ZIP
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
SUB="$DIR/submission"
OUT="$DIR/submission.zip"

echo "=== Checking submission files ==="
for f in run.py detector.onnx classifier.safetensors ref_embeddings.npy ref_labels.json transform_config.json; do
    if [ -f "$SUB/$f" ]; then
        SIZE=$(ls -lh "$SUB/$f" | awk '{print $5}')
        echo "  ✓ $f ($SIZE)"
    else
        echo "  ✗ MISSING: $f"
        exit 1
    fi
done

echo ""
echo "=== Checking blocked imports ==="
if grep -rn "^import os\b\|^from os \|^import sys\b\|^import subprocess\|^import pickle\|^import yaml\|^import socket\|^import threading\|^import multiprocessing" "$SUB"/*.py; then
    echo "  ✗ BLOCKED IMPORTS FOUND!"
    exit 1
else
    echo "  ✓ No blocked imports"
fi

echo ""
echo "=== Creating ZIP ==="
rm -f "$OUT"
cd "$SUB"
zip -r "$OUT" . -x ".*" "__MACOSX/*" "*.fp16.*"
cd "$DIR"

echo ""
echo "=== Verifying ZIP ==="
unzip -l "$OUT" | head -15

echo ""
TOTAL=$(unzip -l "$OUT" | tail -1 | awk '{print $1}')
echo "Total uncompressed: $(echo "$TOTAL / 1024 / 1024" | bc -l | head -c6) MB (limit: 420 MB)"

WEIGHT_COUNT=$(unzip -l "$OUT" | grep -cE "\.(pt|pth|onnx|safetensors|npy)$")
echo "Weight files: $WEIGHT_COUNT (limit: 3)"

echo ""
echo "=== Ready to submit: $OUT ==="
