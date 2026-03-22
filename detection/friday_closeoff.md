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

### 1. CRITICAL: maxDets=100 is PER CATEGORY, not global (verified in pycocotools)
- `COCOeval.evaluateImg()` with `useCats=1` groups predictions by (image, category) before truncating to maxDets
- With 356 categories, the cap is almost never hit for cls eval (~0.26 predictions per category per image)
- **EXCEPTION: det eval maps ALL predictions to category_id=1** → maxDets=100 IS a global cap for detection
- v4.4's explicit top-100 global cap was **self-sabotage** for cls eval — removed
- Score ranking still matters for det eval (single-class, global top-100)

### 2. The gap is in detection/ranking, not classification (Codex)
- Closing 0.0166 via detection: +0.024 det_mAP needed (70% weight)
- Closing via classification: +0.055 cls_mAP needed (30% weight) — 2.3x harder
- Det eval has real maxDets=100 pressure; cls eval does not
- **Optimal score = det_score** — optimizes the only eval with a binding cap

### 3. Fused score was a point leak (for det eval only)
- `0.7*det + 0.3*cls` lets uncertain classifications demote correct detections in the single-class det ranking
- For cls eval, score barely matters — argmax (category_id) is what counts, and per-category ranking is permissive

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

## Training Jobs Running (H100 @ XXx--xx-H100)
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
- **H100 80GB** @ XXx--xx-H100 (on-demand, stable)
- A100 80GB @ XXx--xx-A100 (alive, idle)

---

## Overnight Execution Plan (23:45 → 08:45, 9 hours)

Submissions closed. H100 on-demand overnight. Goal: arrive Saturday morning with **honest evaluation**, **optimized detection ranking**, and **3 ranked candidate ZIPs** ready to submit.

### Hour 0–2: Evaluation Infrastructure (THE PRIORITY)

Everything before tonight was optimizing blind on contaminated val. One hour of honest eval is worth more than all the sweeps we've done.

**Step 1: Cache detector outputs (fold-independent)**
- Run YOLOv8x ONNX + flip TTA on all 248 images
- Save per-image: boxes (xywh), det_scores, crops
- Detector is fixed ONNX — these outputs are reusable across all classifier experiments

**Step 2: Build OOF eval pipeline**
- 5-fold image split (same seed as training folds)
- For each fold: load fold-trained classifier → classify cached crops → produce predictions.json
- Compute **real** det_mAP@0.5 + cls_mAP@0.5 via pycocotools with maxDets=100
- Compute blended score: 0.7×det + 0.3×cls
- Output per-fold AND averaged metrics

**Step 3: Baseline measurement**
- Run current pipeline (v4.1 config) through OOF → honest baseline
- Run v4.4 config (det_score ranking) through OOF → honest comparison
- This tells us immediately whether the score fix helps

**Why this matters:** Our contaminated val showed 0.8332 blend → test was 0.9034. We have NO idea which direction changes go on test. OOF gives us an honest proxy that correlates with test. Every optimization after this point can be validated before submission.

### Hour 2–4: Detection/Ranking Sweep

With honest OOF eval, each sweep point takes seconds (just re-scoring cached detections):

| Parameter | Values to test | Why |
|-----------|---------------|-----|
| NMS IoU | 0.45, 0.50, 0.55, 0.60, 0.65 | Dense shelves — aggressive NMS may suppress real products |
| Conf threshold | 0.001, 0.005, 0.01, 0.05, 0.1 | Balance noise vs recall |
| Score formula | det_only, 0.9\*det+0.1\*cls, det\*cls, det\*cls^0.3, det^0.7\*cls^0.3 | Find optimal ranking |
| Soft-NMS | σ=0.3, 0.5, 0.7 vs hard NMS | May recover suppressed adjacent products |
| M cap | 100, 200, 300, 500, unlimited | Are we truncating correct predictions? |
| Output volume | no cap, top-100 det-only, top-200 det-only | Det eval caps at 100 (single-class); cls eval caps per-category (non-binding) |

Expected: 5×5×5×4×5×4 = 10,000 configs. Each is a fast numpy rescore. Full sweep in ~30 min.

**Key question this answers:** Is the gap in detection ranking (NMS/threshold/scoring) or in classification quality? If we can close most of the gap by re-scoring the same detections, the leader's advantage is ranking, not models.

### Hour 4–6: Classifier Retraining

Depends on findings from Hour 2–4.

**If ranking is the main lever (detection sweep closed >50% of gap):**
- Keep current classifier
- Focus on score calibration
- Maybe train a dedicated "re-ranker" that predicts box quality

**If classification quality is the main lever:**
- **ProxyAnchor training** (Codex recommendation — matches our prototype retrieval geometry)
  - Learnable proxy per class, pull positive embeddings close, push negatives away
  - Inherently count-invariant (one proxy = one class, like our class-mean inference)
  - Better aligned than ArcFace for our kNN retrieval branch
- **Copy-paste augmentation** (CTO recommendation)
  - Paste product reference images onto random shelf crop backgrounds
  - 20-50 synthetic examples per rare class
  - Train on augmented data → more robust rare-class features
- **Balanced focal loss** alternative to label smoothing
  - Down-weight easy/common classes, up-weight hard/rare
  - More targeted than uniform oversampling

Training time per experiment: ~40 min on H100. Can run 3 experiments in this window.

### Hour 6–8: Integration + Candidate ZIPs

Combine best findings into 3 submission candidates:

| Candidate | Detection Config | Classifier | Score Formula | Hypothesis |
|-----------|-----------------|------------|---------------|------------|
| **v5.0** | Best NMS/conf from sweep | Current weights | Best score formula | Pure ranking optimization |
| **v5.1** | Same as v5.0 | New ProxyAnchor weights | Same | Better embeddings + ranking |
| **v5.2** | Same as v5.0 | New weights + kNN tuned | Routed suppression | Full stack improvement |

Each candidate validated end-to-end on OOF before packaging.

**Submission strategy for tomorrow's 3 bullets:**
1. Submit v5.0 first (lowest risk — same weights, better ranking)
2. Wait for feedback → adjust v5.1 if needed
3. Submit v5.1 or v5.2 depending on v5.0 result
4. Final bullet: best remaining candidate or iteration on winner

### Hour 8–9: Buffer + Documentation

- Update this doc with all OOF results
- Clear decision matrix for morning submissions
- Document every config tested with OOF scores
- Prepare submission scripts for quick packaging

---

## Codex Review Summary (Critical Findings)

### The Narrative Shift
> "The gap probably does not live primarily in the 84 rare classes. Your reference bank already softens the tail a lot, while the benchmark weighting and dataset density both point much more strongly at detector/ranking quality as the highest-leverage frontier."

### 5 Key Findings
1. **Gap is likely detection/ranking, not classification** — +0.024 det_mAP (70% weight) vs +0.055 cls_mAP needed
2. **ProxyAnchor > ArcFace > SupCon** for our prototype retrieval geometry
3. **Top 3 untried high-EV changes:** honest OOF eval, top-100 score calibration, proxy-aligned metric training
4. **Point leak in score fusion** — fused score demotes correct detections when cls is uncertain
5. **Sweep harness ≠ shipped pipeline** — conf_thresh, NMS, M cap all differ between sweep and submission

---

## Dataset Statistics (for reference)
| Metric | Value |
|--------|-------|
| Images | 248 (198 train / 50 val in YOLO split) |
| Annotations | 22,731 |
| Categories | 356 (84 rare with ≤5 examples) |
| Objects/image | min 14, median 84, p90 152, max 235, mean 92 |
| Ref embeddings | 6,606 (min 1/class, median 22, max 28) |
| Ref classes with ≤5 embeddings | 29 |

---

## Confidence Rating
| Outcome | Probability | Notes |
|---------|------------|-------|
| Improve over 0.9034 | 80% | With honest eval + ranking sweep, very likely |
| Break 0.910 | 50% | Detection ranking gains should stack |
| Break 0.915 | 30% | Needs ranking + classifier improvement |
| Beat leader at 0.920 | 15–20% | Possible if gap is mostly ranking |

---

## Test Score Tracker
> *Updated as scores arrive*

| Version | Config | Test Score | Delta vs v4.1 | Notes |
|---------|--------|-----------|---------------|-------|
| v4.1 | M=200, no kNN, fused 0.7/0.3 | **0.9034** | — | Baseline |
| v4.2 | M=300, no kNN, fused 0.7/0.3 | TBD | | Pending |
| v4.4 | M=300, kNN, det_score, top-100 | TBD | | Ready to submit |
| v5.0 | Best detection config | TBD | | Tomorrow bullet 1 |
| v5.1 | + new classifier | TBD | | Tomorrow bullet 2 |
| v5.2 | Full stack | TBD | | Tomorrow bullet 3 |
