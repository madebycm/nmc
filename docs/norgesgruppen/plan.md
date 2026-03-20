# NorgesGruppen Data - Object Detection

## To Future Self (Context Transfer Protocol)

You are competing in **NM i AI 2026** - Norwegian AI Championship. Prize pool: 1,000,000 NOK.
Competition: March 19 18:00 CET - March 22 15:00 CET (69 hours). **ONE SHOT** submissions.

This plan was written at competition start. Read `docs/` for full competition docs.
Read `CLAUDE.md` at repo root for project-level directives.
Data lives in `data/coco_dataset/` and `data/product_images/`.

**Three tasks, 33% each of overall score (normalized to 0-100 by dividing by top scorer).**
**NorgesGruppen leaderboard is EMPTY - first strong submission dominates.**

---

## 1. PROBLEM STATEMENT

**Goal:** Detect and classify grocery products on store shelves.
**Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5**
**Submission:** ZIP file with `run.py` at root. Runs in sandboxed Docker.

## 2. DATASET (VERIFIED)

| Fact | Value |
|------|-------|
| Images | 248 (114 unique resolutions, dominant 4032x3024) |
| Annotations | 22,731 (mean 92/image, min 14, max 235) |
| Categories | 356 (IDs 0-355), cat 355 = `unknown_product` |
| Bbox sizes | 84% large, 16% medium, 0.1% small |
| Classes with <=5 examples | **84 (24%)** — long-tail problem |
| Classes with 100+ examples | 91 |
| Product reference folders | 344, with 1,599 images (avg 4.6/product) |
| Name match (categories↔refs) | 323/356 matched, 33 unmatched |
| Annotations have product_code? | **NO** — docs showed idealized example |
| iscrowd | 0 (clean) |

## 3. ARCHITECTURE (FINAL)

**Two-stage pipeline: Detect → Embed → kNN classify**

### Stage 1: YOLOv8l single-class detection
- All 356 categories collapsed to single "product" class
- Fine-tuned from COCO pretrained, imgsz=1280
- Exported to **ONNX FP16** (eliminates torch version risk)
- Expected detection mAP: 0.85-0.95

### Stage 2: Embedding-based classification
- **timm model** (EVA-CLIP or DINOv2) for feature extraction
- Exported to **ONNX FP16** (no network needed in sandbox)
- Pre-computed reference embeddings from:
  - Product reference images (327 products × multi-angle)
  - **Training annotation crops** (ALL 356 categories covered)
- kNN cosine similarity classification
- Expected classification boost: +15-25% of the 30% classification weight

### Why two-stage beats end-to-end
- Detection quality maximized (no class confusion at detect time)
- 84 rare classes (<5 examples) handled by embedding similarity, not training
- Training crops as references = ALL 356 categories have embeddings
- Research confirms: two-stage is SOTA for long-tail distributions

## 4. SUBMISSION ZIP

```
submission.zip (target ~180 MB)
├── run.py              # Entry point
├── detector.onnx       # YOLOv8l FP16 (~85 MB) [weight 1/3]
├── classifier.onnx     # timm model FP16 (~85 MB) [weight 2/3]
├── ref_embeddings.npy  # Pre-computed embeddings (~2 MB) [weight 3/3]
├── ref_labels.json     # category_id mapping (<1 MB)
└── utils.py            # Helper functions
```

## 5. SANDBOX CONSTRAINTS

| Constraint | Value | Our compliance |
|------------|-------|----------------|
| Python | 3.11 | Compatible |
| PyTorch | 2.6.0+cu124 | ONNX = version-proof |
| onnxruntime-gpu | 1.20.0 | Primary inference engine |
| ultralytics | 8.1.0 | Only for .pt fallback |
| GPU | NVIDIA L4, 24GB | ONNX + CUDA provider |
| Timeout | 300s | ~20s estimated |
| Network | NONE | All weights bundled |
| Blocked imports | os, sys, subprocess, pickle... | pathlib + json only |
| Max weights | 420 MB, 3 files | ~172 MB, 3 files |

## 6. TRAINING ENVIRONMENT

**Machine:** Apple M5, 32GB RAM, MPS GPU
**Approach:** Train locally with ultralytics, export ONNX

```bash
# Venv with pinned versions
python3 -m venv ~/www/nm/.venv
source ~/www/nm/.venv/bin/activate
pip install ultralytics==8.1.0 timm==0.9.12 pycocotools onnx onnxruntime
```

## 7. IMPLEMENTATION STEPS

### Step 1: Setup & Data Prep ← CURRENT
- [x] Download and extract data
- [x] Analyze dataset statistics
- [ ] Create venv with pinned packages
- [ ] Convert COCO to YOLO format
- [ ] Create 80/20 train/val split
- [ ] Build category↔product_code↔reference mapping

### Step 2: Train Detection
- [ ] Train YOLOv8l single-class, imgsz=1280
- [ ] Validate detection mAP on val set
- [ ] Export to ONNX FP16 (opset 17)
- [ ] Test ONNX inference locally

### Step 3: Build Classification
- [ ] Download/save timm model weights
- [ ] Export timm model to ONNX FP16
- [ ] Crop all training annotations as reference images
- [ ] Compute embeddings for all references (product images + crops)
- [ ] Save ref_embeddings.npy + ref_labels.json
- [ ] Test kNN classification on val set crops

### Step 4: Wire Pipeline
- [ ] Write run.py (ONNX detection + ONNX classification + kNN)
- [ ] Write utils.py (preprocessing, kNN logic)
- [ ] Test locally: `python run.py --input test_images --output pred.json`
- [ ] Compute mAP locally with pycocotools

### Step 5: Validate & Submit
- [ ] Verify no blocked imports
- [ ] Verify ZIP structure (run.py at root)
- [ ] Verify total size < 420 MB
- [ ] Dry-run with exact sandbox contract
- [ ] Submit to app.ainm.no
- [ ] Check leaderboard

## 8. CONFIDENCE RATINGS

| Component | Rating | Notes |
|-----------|--------|-------|
| Detection | 9/10 | YOLOv8l on large objects, proven |
| Classification | 7.5/10 | Embedding kNN + training crops covers all 356 classes |
| Pipeline mechanics | 9/10 | ONNX = version-proof, well within size/time |
| Submission compliance | 9.5/10 | Contract crystal clear |
| **Overall** | **8.5/10** | |

## 9. RISK MATRIX

| Risk | Impact | Mitigation |
|------|--------|------------|
| ONNX export fails | High | Fallback: ultralytics .pt (pin 8.1.0) |
| timm model too large | Medium | Use smaller variant (ViT-S vs ViT-B) |
| kNN misclassifies similar products | Medium | Use weighted kNN, tune k, add training crops |
| Timeout >300s | Fatal | ONNX is 2-3x faster than PyTorch, ~20s total |
| Wrong bbox format | Fatal | Convert xyxy→xywh explicitly, test |
| MPS training issues | Low | CPU fallback (~6hrs), still feasible |
