#!/bin/bash
# NM i AI 2026 — DataCrunch H100 Server Setup
# Run: bash setup_server.sh
set -euo pipefail

echo "============================================"
echo " NM i AI 2026 — H100 Server Setup"
echo "============================================"
echo ""

# --------------------------------------------------
# 1. System packages
# --------------------------------------------------
echo ">>> Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev gcc g++ zip unzip curl git \
    libgl1-mesa-glx libglib2.0-0 2>/dev/null
echo "    Done."
echo ""

# --------------------------------------------------
# 2. Directory structure
# --------------------------------------------------
echo ">>> Creating directory structure..."
mkdir -p /clade/ng/submission
mkdir -p /clade/venv
mkdir -p /clade/tripletex
mkdir -p /clade/astar
echo "    /clade/ng/submission"
echo "    /clade/venv"
echo "    /clade/tripletex"
echo "    /clade/astar"
echo ""

# --------------------------------------------------
# 3. Python virtual environment
# --------------------------------------------------
VENV=/clade/venv
if [ ! -f "$VENV/bin/activate" ]; then
    echo ">>> Creating Python venv at $VENV..."
    python3 -m venv "$VENV"
else
    echo ">>> Venv already exists at $VENV, reusing."
fi

source "$VENV/bin/activate"
echo "    Python: $(python --version)"
echo "    pip: $(pip --version | cut -d' ' -f1-2)"
echo ""

# Upgrade pip
pip install --quiet --upgrade pip

# --------------------------------------------------
# 4. Install Python packages
# --------------------------------------------------
echo ">>> Installing PyTorch (cu124)..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo ">>> Installing ML packages..."
pip install --quiet \
    ultralytics==8.1.0 \
    timm==0.9.12 \
    safetensors==0.4.2 \
    onnxruntime-gpu==1.20.0 \
    onnx \
    onnxscript \
    pycocotools \
    pillow \
    opencv-python-headless

echo "    Installed packages:"
pip list 2>/dev/null | grep -iE "torch|timm|safetensors|onnx|ultralytics|pycocotools|pillow" | sed 's/^/    /'
echo ""

# --------------------------------------------------
# 5. Verify CUDA
# --------------------------------------------------
echo ">>> Checking GPU & CUDA..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'
else
    echo "    WARNING: nvidia-smi not found"
fi

python -c "
import torch
print(f'    PyTorch: {torch.__version__}')
print(f'    CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1073741824:.0f} GB)')
else:
    print('    WARNING: No CUDA GPUs detected by PyTorch')
"
echo ""

# --------------------------------------------------
# 6. Check SFS volume mount
# --------------------------------------------------
SFS_PATH=/mnt/SFS-qZE4t9Aw/data
echo ">>> Checking SFS volume at $SFS_PATH..."
if [ -d "$SFS_PATH" ]; then
    echo "    MOUNTED"
    echo "    Contents:"
    ls -1 "$SFS_PATH" 2>/dev/null | sed 's/^/      /'

    # Check expected data dirs
    for subdir in coco_dataset/train/images coco_dataset/train/annotations.json yolo/train yolo/val product_images; do
        if [ -e "$SFS_PATH/$subdir" ]; then
            echo "    [OK] $subdir"
        else
            echo "    [MISSING] $subdir"
        fi
    done

    # Create symlink from /clade/ng/data -> NFS
    if [ ! -e /clade/ng/data ]; then
        ln -s "$SFS_PATH" /clade/ng/data
        echo "    Symlink: /clade/ng/data -> $SFS_PATH"
    fi
else
    echo "    NOT MOUNTED — $SFS_PATH does not exist"
    echo "    Attach the SFS volume in DataCrunch dashboard, then re-run this script."
fi
echo ""

# --------------------------------------------------
# 7. Download YOLO weights (if missing)
# --------------------------------------------------
echo ">>> Checking pretrained weights..."
for model in yolov8m yolov8l yolov8x; do
    if [ -f "/clade/ng/${model}.pt" ]; then
        echo "    [OK] ${model}.pt"
    else
        echo "    Downloading ${model}.pt..."
        curl -sL "https://github.com/ultralytics/assets/releases/download/v8.1.0/${model}.pt" \
            -o "/clade/ng/${model}.pt"
        echo "    [OK] ${model}.pt ($(du -h /clade/ng/${model}.pt | cut -f1))"
    fi
done
echo ""

# --------------------------------------------------
# 8. Summary
# --------------------------------------------------
echo "============================================"
echo " Setup Complete"
echo "============================================"
echo ""
echo " Activate:  source /clade/venv/bin/activate"
echo " Data:      /clade/ng/data/ (-> $SFS_PATH)"
echo " Submit:    /clade/ng/submission/"
echo " Train:     cd /clade/ng && python train_h100.py"
echo ""
echo " Quick GPU check:  nvidia-smi"
echo " Quick env check:  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
