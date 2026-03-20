# Friday Close-Off — March 20, 2026

## Scoreboard
| Version | Config | Val Blend | Test Score | Status |
|---------|--------|-----------|------------|--------|
| v4.1 | M=200, no kNN, fused score | 0.8332 | **0.9034** | Scored |
| v4.2 | M=300, no kNN, fused score | 0.8339 | Pending | Submitted ~16:00 |
| v4.4 | M=300, kNN, **det_score ranking**, top-100 cap | — | — | **SUBMITTING NOW** |

**Leader: 0.9200 | Gap: 0.0166 | Submissions remaining: 0 tonight, 3 tomorrow**

---

## What Changed in v4.4 (tonight's submission)
1. **Score = pure det_score** (was `0.7*det + 0.3*cls`)
   - Codex identified that fused score lets bad classifications demote correct detections below maxDets=100 cap
   - Detection is 70% of total score — optimizing det ranking is mathematically correct
   - On images with 92 avg objects (max 235), top-100 ranking quality is everything
2. **Explicit top-100 per image** — clean cutoff matching COCO maxDets=100
3. **M=300** — retained from v4.2/v4.3
4. **kNN hybrid** — retained from v4.3 (class-mean, T=0.15, route conf<0.5)

## What We Killed Today (with data)
| Change | Val Δ | Why Dead |
|--------|-------|----------|
| T=1.5 temperature scaling | −0.0028 | Doesn't change argmax |
| 5% crop expansion | −0.0022 | Distribution shift from tight-crop training |
| Multi-scale ONNX (1024/1280/1536) | N/A | Fixed 1280×1280 ONNX |
| Global kNN blend (any α) | −0.0004 to −0.0013 | Many-shot bias in aggregator |
| Count-penalty (−β·log n_c) | −0.0014 to −0.0024 | Over-penalizes common classes |
| SAHI tiled detection | −0.045 (prior) | Floods maxDets=100 cap |
| amax aggregator | −0.0008+ | Many-shot bias (20 refs vs 2 refs) |
| Checkpoint averaging | −0.005 | Dilutes converged model |
| WBF (flip TTA merge) | −0.005 | Averaging misaligned boxes |
| Classifier flip TTA | −0.002 | Flipped grocery text confuses model |

## Key Insights (Friday)

### 1. The gap is in detection/ranking, not classification (Codex)
- Closing 0.0166 via detection: +0.024 det_mAP needed (70% weight)
- Closing via classification: +0.055 cls_mAP needed (30% weight) — 2.3x harder
- Avg 92 objects/image, max 235 → maxDets=100 makes ranking king

### 2. Fused score was a point leak
- `0.7*det + 0.3*cls` lets uncertain classifications tank detection ranking
- Routed kNN branch produces scores that look confident but aren't rank-calibrated

### 3. ArcFace not clearly better than CE-only
OOF fold-1 results (honest eval, uncontaminated):

| Model | Best Val Acc | Best Rare Acc | Notes |
|-------|-------------|---------------|-------|
| CE-only (resumed) | **0.8699** | **44.3%** (E14) | Winner so far |
| ArcFace (resumed) | 0.8656 | 41.4% (E1,E5) | No clear advantage |
| ArcFace (clean) | 0.8560 | 35.7% (E8) | Still climbing |
| CE-only (clean) | 0.8418 | 35.7% (E5) | Still climbing |

### 4. Val is contaminated — honest OOF shows different picture
- Contaminated val: 0.9060 cls accuracy (model memorized val)
- Honest OOF: 0.8699 best (23% lower) — this is reality on test

### 5. Sweep harness ≠ shipped pipeline (Codex bug find)
- Sweep used conf_thresh=0.05, NMS=0.7, no M cap
- Ship uses conf_thresh=0.001, NMS=0.5, M=300
- Tuning conclusions may not transfer

---

## Training Jobs Running (H100 @ 86.38.238.168)
4 parallel jobs, 52/80 GB VRAM:
- ArcFace (resumed from finetuned) — E22/30, ~15 min remaining
- CE-only (resumed from finetuned) — E17/30, ~20 min remaining
- ArcFace (clean from pretrained) — E9/40, ~45 min remaining
- CE-only (clean from pretrained) — E6/40, ~50 min remaining

---

## Assets
| File | Size | Status |
|------|------|--------|
| submission_4.4.zip | 304 MB | **READY** |
| detector.onnx | 130 MB | Unchanged since v1 |
| classifier.safetensors | 164 MB | E40 all-data finetuned |
| ref_embeddings_finetuned.npy | 10 MB | E40 finetuned embeddings |
| ref_labels.json | 31 KB | 6606 labels |

## Compute
- **H100 80GB** @ 86.38.238.168 (on-demand, stable)
- A100 80GB @ 135.181.8.209 (alive, idle)

---

## Tomorrow's Plan (Saturday — 3 submissions)

### Priority 1: Score/ranking calibration
- Validate v4.4 score fix on test
- If det_score ranking helped: iterate on ranking shaping
- If not: revert to fused and investigate detection side

### Priority 2: New classifier weights
- Use best model from tonight's training runs
- Extract embeddings from best checkpoint
- ProxyAnchor training if ArcFace didn't help (Codex recommendation)

### Priority 3: Detection improvements
- Soft-NMS (decay instead of suppress overlapping boxes)
- NMS IoU threshold sweep (0.5 → 0.6, 0.65)
- Detector conf threshold sweep (0.001 → 0.01, 0.05)

---

## Confidence Rating
| Outcome | Probability |
|---------|------------|
| Improve over 0.9034 | 70% |
| Break 0.910 | 40% |
| Beat leader at 0.920 | 15% |

---

## Friday Test Score Update
> *To be filled after submission scores arrive*

| Version | Test Score | Delta vs v4.1 | Notes |
|---------|-----------|---------------|-------|
| v4.2 (M=300) | TBD | | |
| v4.4 (det_score ranking) | TBD | | |
