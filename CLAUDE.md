# NM i AI 2026 - Competition Repository

## Context
Norwegian AI Championship. March 19 18:00 - March 22 15:00 CET (69 hours). Prize: 1,000,000 NOK.
Platform: https://app.ainm.no

## Directive
Every submission is ONE SHOT. No room for error.
- Ground every assumption in reproducible testing
- Validate internally before submitting — verify, then verify again
- Excellence is the only acceptable standard
- Must beat current #1 score of **0.7802 mAP** significantly

## Three Tasks (33% each)

| Task | Type | Submission | Status |
|------|------|------------|--------|
| NorgesGruppen Data | Object detection + classification | ZIP upload | **IN PROGRESS** — YOLOv8x training on H100 |
| Tripletex | AI accounting agent | HTTPS `/solve` endpoint | NOT STARTED |
| Astar Island | Norse world prediction | REST API | NOT STARTED |

## Documentation
All competition docs in `docs/` — see `docs/README.md` for index.
MCP server: `claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp`

## Key Constraints
- Code open-sourced (MIT) in public repo before deadline
- Vipps verification for prizes
- Deploy in `europe-north1` for lowest latency

---

## Compute Server — 2x H100 80GB

### Connection
- **Host**: 86.38.238.86
- **User**: root
- **SSH**: `ssh root@86.38.238.86`
- **Provider**: DataCrunch (Finland)

### Hardware
| Resource | Spec |
|----------|------|
| GPU | 2x NVIDIA H100 80GB HBM3 |
| RAM | 363 GB |
| CPU | 80 vCPU |
| OS | Ubuntu 24.04 (Linux 6.8.0) |
| Python | 3.12.3 |

### Storage
| Mount | Size | Used | Purpose |
|-------|------|------|---------|
| `/` (root) | 48 GB | 16 GB | System + venv |
| `/mnt/SFS-qZE4t9Aw` | 200 GB NFS | ~2 GB | Data + training runs |

### Environment
- **Venv**: `/clade/venv` — `source /clade/venv/bin/activate`
- **PyTorch**: 2.6.0+cu124, CUDA confirmed on both GPUs
- **Packages** (matching sandbox versions):
  - ultralytics 8.1.0, timm 0.9.12, safetensors 0.4.2, onnxruntime-gpu 1.20.0
  - pycocotools, onnx, onnxscript, opencv-python-headless

### Directory Structure
```
/clade/
├── venv/                    # Python virtual environment
├── ng/                      # NorgesGruppen task
│   ├── data -> /mnt/SFS-qZE4t9Aw/data  # Symlink to NFS
│   ├── runs/                # Training output
│   ├── submission/          # Submission files (mirror of local)
│   ├── train_h100.py        # YOLOv8x training (100 epochs, 1280px, batch=16)
│   ├── train_h100_l.py      # YOLOv8l training (backup, 80 epochs)
│   ├── train_detect.py      # Original YOLOv8m training
│   ├── build_embeddings.py  # Reference embedding builder
│   ├── validate.py          # Local validation
│   ├── prepare_data.py      # COCO→YOLO converter
│   ├── package.sh           # ZIP packager
│   ├── yolov8m.pt           # Pretrained weights
│   ├── yolov8l.pt           # Pretrained weights
│   └── yolov8x.pt           # Pretrained weights
├── tripletex/               # Tripletex task (empty)
└── astar/                   # Astar Island task (empty)
```

### Data on NFS (`/mnt/SFS-qZE4t9Aw/data/`)
```
data/
├── coco_dataset/train/      # 248 images + annotations.json
├── product_images/          # 328 product reference folders
├── yolo/                    # YOLO format splits
│   ├── data_single.yaml     # Single-class config
│   ├── data_multi.yaml      # Multi-class config
│   ├── train/               # 198 images (symlinks to coco_dataset)
│   └── val/                 # 50 images (symlinks to coco_dataset)
├── NM_NGD_coco_dataset.zip  # Original archive
└── NM_NGD_product_images.zip
```

### Quick Reference
```bash
# SSH in
ssh root@86.38.238.86

# Activate env
source /clade/venv/bin/activate

# Check GPU
nvidia-smi

# Check training progress
tail -f /clade/ng/train_x.log

# Monitor training metrics
cat /clade/ng/runs/detect_x_1280/results.csv | tail -5
```

### Compatibility Patches Required
All training scripts must include these patches before importing ultralytics:
```python
import numpy as np
import torch
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched
```

---

## NorgesGruppen Data — Critical Findings

### Architecture: Two-Stage Pipeline
1. **Detection**: YOLOv8x single-class → ONNX (finds products on shelves)
2. **Classification**: timm EVA-02 base → safetensors → kNN cosine similarity (identifies product category)
3. **Score formula**: `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`

### Dataset
- 248 images (198 train / 50 val), 22,731 annotations, 356 product categories + 1 unknown
- ~92 annotations per image average (very dense shelves)
- 84/356 categories have ≤5 training examples (long-tail distribution)
- Product reference images available for 328/356 categories

### Sandbox Environment (Competition Eval)
- Python 3.11, PyTorch 2.6.0+cu124, NVIDIA L4 GPU (24GB VRAM)
- 8 GB RAM, 4 vCPU, 300s timeout, NO network
- Pre-installed: ultralytics 8.1.0, timm 0.9.12, onnxruntime-gpu 1.20.0, safetensors 0.4.2
- **Blocked imports**: os, sys, subprocess, pickle, yaml, socket, threading, multiprocessing, etc.
- **Blocked calls**: eval(), exec(), compile(), __import__()
- Use `pathlib` not `os`, use `json` not `yaml`, use `model.train(False)` not `model.eval()`

### Submission Constraints
- Max 420 MB uncompressed, max 3 weight files, max 10 .py files
- Allowed weight types: .pt, .pth, .onnx, .safetensors, .npy
- `run.py` MUST be at ZIP root (not in subfolder)
- Command: `python run.py --input /data/images --output /output/predictions.json`
- Output: `[{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}]`

### Weight Budget (420 MB limit)
| File | Size | Format |
|------|------|--------|
| detector.onnx | ~130 MB | YOLOv8x FP32 ONNX (est. at 640px) |
| classifier.safetensors | 164 MB | EVA-02 base FP16 |
| ref_embeddings.npy | 10 MB | 6606×768 FP16 |
| **Total** | **~304 MB** | Within limit |

### Critical Bugs Found & Fixed
1. **torch 2.10 `weights_only=True` default** — ultralytics 8.1.0 can't load .pt files. Fixed: patch `torch.load` to default `weights_only=False`
2. **numpy 2.0+ removed `np.trapz`** — ultralytics 8.1.0 uses it. Fixed: `np.trapz = np.trapezoid`
3. **MPS TAL bug** — `RuntimeError: shape mismatch` in Task-Aligned Learning on Apple GPU. UNFIXABLE. Must train on CPU or CUDA.
4. **ONNX FP16 type errors** — `onnxconverter_common` FP16 breaks EVA-02 Cast nodes. Fixed: use safetensors FP16 + timm native loading instead
5. **timm transform mismatch** — build_embeddings used `Resize(256, bicubic) → CenterCrop(224)` but run.py used `Resize(224, bilinear)`. Fixed: match exact timm pipeline in run.py
6. **PyTorch ONNX dynamo exporter** creates external .data files (not allowed in submission). Fixed: use legacy exporter with `dynamo=False`
7. **Box clamping bug** — negative coords after unpadding not handled correctly. Fixed: xyxy clamp then back to xywh
8. **YOLO data symlinks on server** — rsync preserves Mac symlinks pointing to `/Users/xc/...`. Fixed: regenerated symlinks pointing to NFS paths

### Training Progress
**YOLOv8m (local CPU, 40 epochs)** — completed epoch 18, best mAP50=0.935
**YOLOv8x (H100 CUDA, 100 epochs, 1280px)** — TRAINING NOW (PID 8857)
- Training at 1280px resolution, batch=16, with advanced augmentation
- Export will be at 640px for sandbox inference
- Monitor: `ssh root@86.38.238.86 "tail -f /clade/ng/train_x.log"`

### Classification Pipeline
- EVA-02 base ViT (86M params) pretrained on CLIP data
- 6,606 reference embeddings across all 356 categories
- 5,029 training crop embeddings + 1,577 product image embeddings
- Weighted k-NN (k=5) with cosine similarity
- Expected classification mAP50: ~0.35-0.50

### Score Estimate (with YOLOv8x)
- Detection: 0.95 × 0.7 = 0.665
- Classification: 0.45 × 0.3 = 0.135
- **Estimated total: ~0.80** (target: >0.78)

### Confidence Rating: 6/10
- Detection upgrade to YOLOv8x + 1280px should push detection mAP well above 0.93
- Classification is still the weak link (long-tail categories, visual similarity)
- MUST validate full pipeline end-to-end on server before submission
- Need to verify ONNX export at 640px matches PT model

### Local Files
| File | Purpose |
|------|---------|
| `train_detect.py` | YOLOv8m training with torch/numpy patches |
| `build_embeddings.py` | Compute reference embeddings (streaming, memory-efficient) |
| `validate.py` | Local validation using pycocotools |
| `package.sh` | ZIP packaging with verification |
| `submission/run.py` | Inference pipeline for sandbox |
| `submission/classifier.safetensors` | EVA-02 FP16 weights |
| `submission/ref_embeddings.npy` | Reference embeddings |
| `submission/ref_labels.json` | Category labels for references |
| `submission/transform_config.json` | Preprocessing params |
| `prepare_data.py` | COCO→YOLO format converter |

### Server Files (H100)
| File | Purpose |
|------|---------|
| `/clade/ng/train_h100.py` | YOLOv8x training (100 epochs, 1280px) |
| `/clade/ng/train_h100_l.py` | YOLOv8l backup training |
| `/clade/ng/train_x.log` | Training output log |

### Leaderboard
- Current #1: **0.7802 mAP**
- Checker: `./lbchecker.sh ng`

---

## Tripletex — Not Started
- AI accounting agent that processes accounting tasks
- Sandbox API credentials available (see memory files)
- Deployment target: nm.j6x.com (204.168.177.62)
- Submission: HTTPS endpoint at `/solve`
- **Server workspace**: `/clade/tripletex/` on 86.38.238.86

## Astar Island — Not Started
- Norse world prediction task
- REST API-based predictions
- See `docs/astar-island/` for details
- **Server workspace**: `/clade/astar/` on 86.38.238.86
