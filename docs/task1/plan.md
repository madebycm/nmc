# TASK 1: NorgesGruppen Data - Object Detection

## To Future Self (Context Transfer Protocol)

You are competing in **NM i AI 2026** - Norwegian AI Championship. Prize pool: 1,000,000 NOK.
Competition: March 19 18:00 CET - March 22 15:00 CET (69 hours). **ONE SHOT** submissions.

This plan was written at competition start. Read `docs/` for full competition docs.
Read `CLAUDE.md` at repo root for project-level directives.

**Three tasks, 33% each of overall score (normalized to 0-100 by dividing by top scorer).**

This file covers **NorgesGruppen Data** (object detection on grocery shelves).
Tripletex and Astar Island have separate plans (see `docs/task2/` and `docs/task3/`).

---

## 1. PROBLEM STATEMENT

**Goal:** Detect and classify grocery products on store shelves.
**Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5**
**Submission:** ZIP file with `run.py` at root. Runs in sandboxed Docker.

### Critical Numbers

| Constraint | Value | Implication |
|------------|-------|-------------|
| Training images | 248 | Small dataset - augmentation critical |
| Annotations | ~22,700 | ~91 products per image (dense shelves) |
| Categories | 356 (IDs 0-355) | Many classes, sparse per-class examples (~64 avg) |
| Product ref images | 327 products, multi-angle | Can augment classification |
| Max weight size | 420 MB total, max 3 files | YOLOv8x FP16 fits (~130MB) |
| GPU | NVIDIA L4, 24GB VRAM | YOLOv8x runs comfortably |
| Timeout | 300 seconds | ~3s/image budget if 100 test images |
| Network | NONE (offline) | All weights must be bundled |
| Daily submissions | 3 | ~9 total over competition. Each one precious |
| Python | 3.11 | |
| PyTorch | 2.6.0+cu124 | |
| ultralytics | **8.1.0** (MUST pin this version for training) |

### Scoring Breakdown

- **Detection-only** (all category_id=0): Max **70%** of score
- **With correct classification**: Unlocks remaining **30%**
- Strategy: Get detection right FIRST, then layer classification

### Security Sandbox

**BLOCKED:** `os`, `sys`, `subprocess`, `pickle`, `yaml`, `threading`, `multiprocessing`, `requests`, `socket`
**USE:** `pathlib` (not `os`), `json` (not `yaml`), `torch.no_grad()` during inference

---

## 2. TRAINING DATA ANALYSIS

### COCO Dataset (~864 MB)
- File: `NM_NGD_coco_dataset.zip` (download from app.ainm.no, login required)
- 248 images from 4 store sections: **Egg, Frokost, Knekkebrod, Varmedrikker**
- Standard COCO format: `annotations.json` with images, categories, annotations
- bbox format: `[x, y, width, height]` in pixels
- Images likely ~2000x1500px (high resolution shelf photos)

### Product Reference Images (~60 MB)
- File: `NM_NGD_product_images.zip`
- 327 products organized by barcode
- Multi-angle: main, front, back, left, right, top, bottom
- `metadata.json` maps products to names and annotation counts
- **Use case:** Few-shot classification, crop-paste augmentation, feature matching

### Data Questions to Investigate (after download)
1. Class distribution - are categories balanced or heavily skewed?
2. Image resolution distribution
3. Bbox size distribution (small objects = harder)
4. How many categories have <5 annotations? (long tail problem)
5. Overlap between product reference images and training categories
6. Are there categories in training data that don't have reference images?

---

## 3. STRATEGY

### Phase 1: Detection Baseline (Target: 60-70% overall, ~85-95% detection mAP)

**Model:** YOLOv8l (or YOLOv8x if weights fit in 420MB)
**Approach:** Fine-tune from COCO pretrained weights on competition data
**Classification:** Set all category_id=0 (detection-only mode)
**Expected score:** Up to 0.70 (70%)

**Training recipe:**
```bash
# MUST use ultralytics==8.1.0 to match sandbox
pip install ultralytics==8.1.0
pip install torch==2.6.0 torchvision==0.21.0

# Convert COCO annotations to YOLO format
# (ultralytics expects YOLO txt format for training)

# Train - detection only (single class)
yolo detect train \
  data=data.yaml \
  model=yolov8l.pt \
  epochs=100 \
  imgsz=1280 \
  batch=4 \
  device=0 \
  patience=20 \
  augment=True \
  mosaic=1.0 \
  mixup=0.1 \
  copy_paste=0.1 \
  single_cls=True  # All products = one class for detection
```

**Why YOLOv8l:**
- 43.7M params, ~170MB FP16 = fits in 420MB limit
- Excellent detection accuracy
- Well below L4 inference capacity
- Proven architecture for dense object detection

**Why imgsz=1280:**
- Shelf images are ~2000x1500. Products can be small
- 1280 preserves more detail than default 640
- L4 with 24GB handles this easily for inference
- Check timeout: if 100 images × ~200ms = 20s (safe)

### Phase 2: Classification (Target: 80-90% overall)

**Option A: End-to-end YOLOv8 with nc=357**
```bash
yolo detect train \
  data=data.yaml \
  model=yolov8l.pt \
  epochs=150 \
  imgsz=1280 \
  batch=4 \
  device=0 \
  patience=30 \
  nc=357  # All 356 categories + unknown
```

Pros: Simple, single model, single inference pass
Cons: 356 classes with ~64 examples each is sparse. Long tail will be weak.

**Option B: Two-stage (detect + classify)**
1. YOLOv8l single-class detection (from Phase 1)
2. Separate classifier on cropped detections
   - timm backbone (EfficientNet-B3 or ConvNeXt-T)
   - Trained on crops from training annotations + product reference images
   - ~50MB additional weight file

Pros: Classification model sees only product crops = cleaner signal
Cons: Two inference passes, more complex pipeline, timeout risk

**Option C: Detect + Feature Matching**
1. YOLOv8l detection
2. Extract features from each crop using timm backbone
3. Match against pre-computed features from product reference images
4. Nearest-neighbor classification

Pros: Uses reference images directly, no training needed for classification
Cons: Feature quality depends on backbone, may not generalize

### Recommended: Option A first, Option B if time allows

Option A is simplest and leverages YOLOv8's built-in classification.
With 248 images × ~91 annotations = 22,700 training examples across 357 classes,
the model should learn at least the common classes well.

### Phase 3: Optimizations (if time)

1. **Test-Time Augmentation (TTA):** Multi-scale + flip inference
2. **Ensemble:** YOLOv8l + YOLOv8x predictions merged with ensemble-boxes (WBF)
3. **FP16 inference:** Faster, smaller weights
4. **Conf threshold tuning:** Optimize detection threshold on val set
5. **Tile inference:** Split high-res images into tiles for small object detection

---

## 4. IMPLEMENTATION PLAN

### Step 1: Download & Analyze Data
```
1. Download NM_NGD_coco_dataset.zip and NM_NGD_product_images.zip
2. Extract and analyze:
   - Class distribution histogram
   - Image dimensions
   - Annotation density per image
   - Product reference coverage
3. Create train/val split (80/20 stratified by image)
```

### Step 2: Prepare YOLO Format Data
```
1. Convert COCO annotations to YOLO txt format
2. Create data.yaml for ultralytics
3. Verify conversion with visualization
```

### Step 3: Train Detection Model
```
1. Train YOLOv8l single-class (detection only)
2. Validate mAP on val set
3. If mAP > 0.85 on val: proceed to Phase 2
4. If not: try YOLOv8x, larger imgsz, more augmentation
```

### Step 4: Train Classification Model
```
1. Train YOLOv8l with nc=357 (all classes)
2. Compare val mAP with detection-only
3. If classification helps: use multi-class model
4. If not: fall back to detection-only (still 70% max)
```

### Step 5: Build run.py
```
1. Write run.py following exact contract
2. Test locally with sample images
3. Verify: no blocked imports, pathlib only, json output format
4. Verify: GPU auto-detection, torch.no_grad()
```

### Step 6: Package & Submit
```
1. Export best model to .pt (or ONNX for safety)
2. Create submission ZIP (run.py at root!)
3. Verify ZIP structure
4. Submit to app.ainm.no
5. Check score on leaderboard
```

---

## 5. run.py TEMPLATE

```python
import argparse
import json
import torch
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned model
    model_path = Path(__file__).parent / "best.pt"
    model = YOLO(str(model_path))

    predictions = []
    img_dir = Path(args.input)

    with torch.no_grad():
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            image_id = int(img_path.stem.split("_")[-1])

            results = model(
                str(img_path),
                device=device,
                imgsz=1280,
                conf=0.25,
                iou=0.5,
                verbose=False,
            )

            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(r.boxes.cls[i].item()),
                        "bbox": [
                            round(x1, 1),
                            round(y1, 1),
                            round(x2 - x1, 1),
                            round(y2 - y1, 1),
                        ],
                        "score": round(float(r.boxes.conf[i].item()), 3),
                    })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
```

### Checklist Before Submission
- [ ] `run.py` at ZIP root (NOT in subfolder)
- [ ] No blocked imports (os, sys, subprocess, pickle, yaml, etc.)
- [ ] Uses `pathlib` for all file operations
- [ ] Uses `json` for output
- [ ] GPU auto-detection via `torch.cuda.is_available()`
- [ ] `torch.no_grad()` during inference
- [ ] Output format: `[{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}]`
- [ ] bbox is COCO format `[x, y, width, height]` (NOT xyxy!)
- [ ] image_id extracted correctly from filename
- [ ] Model weights included in ZIP
- [ ] Total uncompressed < 420 MB
- [ ] Model trained with `ultralytics==8.1.0`
- [ ] Tested locally before upload
- [ ] ZIP created with: `cd submission/ && zip -r ../submission.zip . -x ".*" "__MACOSX/*"`

---

## 6. RISK MATRIX

| Risk | Impact | Mitigation |
|------|--------|------------|
| ultralytics version mismatch | Model won't load | Pin `ultralytics==8.1.0` during training |
| Weight too large (>420MB) | Submission rejected | Use FP16 export, or smaller model |
| Timeout (>300s) | Score = 0 | Test inference time locally, use batch=1 |
| OOM (>8GB RAM) | Exit 137 | Process one image at a time |
| Wrong bbox format (xyxy vs xywh) | Low detection score | COCO format is [x,y,w,h], convert from xyxy |
| category_id mismatch | Low classification score | Verify category IDs match training data exactly |
| run.py in subfolder | "run.py not found" | Verify ZIP with `unzip -l` |
| Sparse classes (few examples) | Low per-class AP | Focus on detection first (70% score), augment |

---

## 7. TRIPLETEX CONNECTION (Verified)

Sandbox credentials (verified working at competition start):
- **Web UI:** `https://kkpqfuj-amager.tripletex.dev`
- **API URL:** `https://kkpqfuj-amager.tripletex.dev/v2`
- **Login email:** `christian.meinhold@gmail.com`
- **Auth:** Basic `("0", session_token)`
- **Session token:** stored in platform

### API Status (tested):
| Endpoint | Status | Records |
|----------|--------|---------|
| /employee | 200 | 1 (Christian Meinhold) |
| /customer | 200 | 0 |
| /product | 200 | 0 |
| /invoice | 200 (needs date params) | 0 |
| /department | 200 | 1 |
| /project | 200 | 0 |
| /ledger/account | 200 | 6 (standard Norwegian chart) |
| /travelExpense | 200 | 0 |

Deployment target for Tripletex + Astar Island endpoints: `nm.j6x.com` (204.168.177.62)

---

## 8. OVERALL COMPETITION TIMELINE

| Window | Focus | Expected Score Impact |
|--------|-------|-----------------------|
| Hours 0-6 | Download data, analyze, train detection-only | Foundation |
| Hours 6-12 | First submission (detection-only YOLOv8l) | ~60-70% NGD |
| Hours 12-24 | Train with classification, iterate | ~75-85% NGD |
| Hours 12-24 | Set up Tripletex endpoint at nm.j6x.com | Foundation |
| Hours 24-48 | Optimize all tasks, Astar Island | All tasks active |
| Hours 48-69 | Final submissions, ensemble, optimize | Peak scores |

---

## 9. COMMANDS CHEATSHEET

```bash
# Training (local machine with GPU)
pip install ultralytics==8.1.0 torch==2.6.0 torchvision==0.21.0

# Convert COCO to YOLO format
# (script needed - see Step 2)

# Train detection-only
yolo detect train data=data.yaml model=yolov8l.pt epochs=100 imgsz=1280 batch=4 single_cls=True

# Train with classification
yolo detect train data=data.yaml model=yolov8l.pt epochs=150 imgsz=1280 batch=4 nc=357

# Export FP16
yolo export model=best.pt format=pt half=True
# OR export ONNX
yolo export model=best.pt format=onnx imgsz=1280 opset=17 half=True

# Package submission
cd submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
unzip -l ../submission.zip  # verify structure

# Test locally
python run.py --input ./test_images --output ./test_predictions.json
```

---

## 10. DECISION LOG

| Decision | Rationale | Confidence |
|----------|-----------|------------|
| YOLOv8l over YOLOv8x | Better size/accuracy tradeoff, fits weight limit with room | High |
| imgsz=1280 over 640 | Shelf images are high-res, small products need resolution | High |
| Detection-first strategy | 70% of score is detection, safer first submission | Very High |
| Pin ultralytics==8.1.0 | Sandbox has exactly this version, any other = broken | Absolute |
| COCO [x,y,w,h] output | Docs explicitly state COCO format | Absolute |
| pathlib not os | os is blocked in sandbox | Absolute |
