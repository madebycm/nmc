# Milestone: Submission 4.1 — Score 0.9034

## Leaderboard
- **Our score: 0.9034** | **Leader: 0.9200** | **Gap: 0.0166**
- Previous score: 0.7743 (v3) → +0.1291 improvement
- Runtime: 79.0s / 300s (221s headroom)
- Size: 271.6 MB / 420 MB (148 MB headroom)
- **Submissions remaining: 1**

## Current Pipeline (v4.1)

```
Image → Letterbox 1280 → YOLOv8x ONNX (orig + flip TTA, NMS 0.5)
     → conf_thresh 0.001 → top-200/image by det_score
     → Crop → EVA-02 FP16 (no flip TTA) → 356-class softmax
     → Score = 0.7*det + 0.3*cls → JSON output
```

**Files**: run.py (9.4KB), detector.onnx (130MB), classifier.safetensors (164MB)

## What v4.1 Includes

| Change | Impact |
|--------|--------|
| All-data classifier (E40, 40 epochs, all 248 images) | Major — full distribution |
| Detection flip TTA (orig + hflip, merge with NMS) | +0.0095 det_mAP |
| No classifier flip TTA | +0.0016 cls_mAP (grocery text) |
| conf_thresh 0.001 | More recall for COCO eval |
| NMS IoU 0.5 | Tighter suppression |
| Linear fusion 0.7*det + 0.3*cls | Mirrors scoring formula |
| Pre-classify cap M=200/image | Safety + speed |
| FP16 classifier | 2x faster |

## Gap Analysis: 0.9034 → 0.9200 = +0.0166 needed

Score = `0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5`

| Path | det gain needed | cls gain needed | Feasibility |
|------|----------------|----------------|-------------|
| Pure detection | +0.024 | — | Hard without retraining |
| Pure classification | — | +0.055 | Very hard |
| **Balanced** | **+0.010** | **+0.022** | **Best bet** |

## Frontier: ALL Options Ranked

### TIER 1 — High confidence, low risk

**A. Hybrid kNN Retrieval** (cls +0.01-0.03)
- STATUS: Fine-tuned embeddings READY on server (6606×768, 10MB)
- When softmax confidence < threshold, boost with cosine similarity to reference embeddings
- Targets 84 categories with ≤5 training examples — exactly where supervised head is weakest
- Size: +10MB (314 MB total). Runtime: +5-10s. Fits easily.
- Implementation: extract `model.forward_features()` for query embedding, compare to ref bank
- **Risk: LOW** — only modifies uncertain predictions, supervised head still primary

**B. Increase pre-classify cap M=200 → M=300** (cls +0.005-0.01)
- Val showed M=300 gives blend 0.8339 vs M=200 at 0.8332
- More crops classified = more correct category assignments for COCO eval
- Runtime: +10-15s (from ~80s to ~95s). Easily fits.
- **Risk: VERY LOW** — strictly more information

**C. Multi-crop classifier TTA** (cls +0.005-0.015)
- Instead of single center-crop, use center + 4 corners (5-crop TTA)
- Average softmax across 5 views of each detection crop
- Runtime: ~5x classification time. 200 crops × 5 = 1000 per image.
- Current ~40s classification → ~200s. Total ~240s. Fits in 300s if test set ≈ val size.
- **Risk: MEDIUM** — runtime dependent on test set size, could timeout

### TIER 2 — Moderate confidence, moderate risk

**D. SAHI Tiled Inference (2×1 split)** (det +0.01-0.03)
- Previously REJECTED at 2×2 (flooded maxDets=100)
- Revisit with MINIMAL tiling: split image into top/bottom halves only
- 2 tiles + 1 global + flip TTA = 6 ONNX passes vs current 2
- Runtime: ~3x detection time. ~40s → ~120s. Total ~160s. Fits.
- Need WBF (not NMS) to merge tile predictions properly
- **Risk: HIGH** — previously hurt badly. But old test used NMS, not WBF.
- MUST validate on val before committing.

**E. Ensemble Checkpoints via Sequential Loading** (cls +0.005-0.015)
- Load E30 weights, run softmax. Load E40 weights, run softmax. Average.
- Can't ship both files (328MB classifiers + 130MB detector = 458MB > 420MB)
- WORKAROUND: ship E40 only, reconstruct E30 from... no, can't.
- Alternative: ship avg of E30+E40 weights as single file (like checkpoint averaging, but only 2)
- We tested avg5 and it hurt. But avg2 (E30+E40 only) might be different.
- **Risk: MEDIUM** — need to test on server

**F. Temperature Scaling** (cls +0.002-0.005)
- Softmax temperature T>1 spreads probability mass, potentially better for rare classes
- Simple: `probs = softmax(logits / T)` with T=1.5 or T=2.0
- Changes confidence distribution, affects fusion score ranking
- **Risk: LOW** but small impact

### TIER 3 — Low confidence, high risk

**G. Different Classifier Architecture** — NO TIME to retrain
**H. Better Detector** — NO TIME, ONNX is fixed
**I. Data Augmentation Changes** — NO TIME to retrain

## Recommended v4.2 Plan (FINAL SUBMISSION)

Execute in this order, validate each on server before committing:

1. **Hybrid kNN retrieval** — implement & test (est. 1-2h)
2. **Increase M=200 → M=300** — trivial change, test (est. 5min)
3. **2-checkpoint weight average (E30+E40 only)** — test on server (est. 30min)
4. **Temperature scaling sweep** — test T=1.0,1.5,2.0 (est. 15min)
5. **Multi-crop TTA** — test 5-crop, monitor runtime (est. 30min)
6. **SAHI 2×1** — ONLY if above gains insufficient (est. 1h)

**Stop criterion**: if cumulative val improvement > +0.02 blend, package and submit.
**Abort criterion**: if any change causes regression > 0.005, revert immediately.

## Assets on Server (135.181.8.209)

- `/clade/ng/submission/classifier.safetensors` — E40 fine-tuned model
- `/clade/ng/submission/detector.onnx` — YOLOv8x 1280px
- `/clade/ng/submission/ref_embeddings_finetuned.npy` — 6606×768 FP16 ✅ READY
- `/clade/ng/submission/ref_labels.json` — 6606 category IDs ✅ READY
- `/clade/ng/checkpoints/` — E03-E40 checkpoints available
- Val images at `/mnt/SFS-qZE4t9Aw/data/yolo/val/images/`

## Key Insight

Val blend was 0.8332 but test scored 0.9034. Val is HARDER than test (contaminated training makes val cls inflated but val detection seems harder). **Do not over-optimize for val metrics.** Prioritize theoretically sound improvements over val number chasing.
