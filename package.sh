#!/bin/bash
# Package submission ZIP — v4 (all-data classifier + flip TTA detection)
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
SUB="$DIR/submission"
OUT="$DIR/submission_4.zip"

echo "=== Checking submission files ==="
for f in run.py detector.onnx classifier.safetensors; do
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
echo "=== Checking blocked calls ==="
if grep -rn "eval()\|exec()\|compile()\|__import__" "$SUB"/*.py; then
    echo "  ✗ BLOCKED CALLS FOUND!"
    exit 1
else
    echo "  ✓ No blocked calls"
fi

echo ""
echo "=== Creating ZIP ==="
rm -f "$OUT"
cd "$SUB"
zip -r "$OUT" . -x ".*" "__MACOSX/*" "*.fp16.*" "ref_embeddings.npy" "ref_labels.json" "transform_config.json"
cd "$DIR"

echo ""
echo "=== Verifying ZIP ==="
unzip -l "$OUT" | head -15

echo ""
TOTAL=$(unzip -l "$OUT" | tail -1 | awk '{print $1}')
echo "Total uncompressed: $(echo "$TOTAL / 1024 / 1024" | bc -l | head -c6) MB (limit: 420 MB)"

WEIGHT_COUNT=$(unzip -l "$OUT" | grep -cE "\.(pt|pth|onnx|safetensors|npy)$")
echo "Weight files: $WEIGHT_COUNT (limit: 3)"

PY_COUNT=$(unzip -l "$OUT" | grep -cE "\.py$")
echo "Python files: $PY_COUNT (limit: 10)"

echo ""
echo "=== Ready to submit: $OUT ==="
