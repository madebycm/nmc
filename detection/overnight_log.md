# Overnight Autonomous Run — March 21, 2026

## STATUS: ACTIVE PUSH — 2 bullets remaining, gap 0.0115
## v5.4_cos20 test: 0.9140 (#25) | Leader: 0.9255 (14 subs) | Gap: 0.0115
## Best OOF: 0.7598 (MixUp E3 cos/20 + TTA + K=15)

### CRITICAL INCIDENT — March 21 ~13:00
- **Bullet #1 BURNED**: submission_5.4_cos20.zip failed — predictions.json 18MB, exceeds undocumented 10MB platform limit
- **Root cause**: K=15 × ~300 det × ~50 images = ~225K predictions, no output size cap
- **Fix**: Added `score < 0.01` early exit + 500/image cap → ~2MB for 50 images
- **Verified**: `submission_5.4_cos20_safe.zip` tested on H100 (248 images → 12MB, est 50 → 2MB)
- **2 bullets remain** (resets midnight UTC)

---

## Test Score History (GROUND TRUTH)
| Version | Config | Test Score | Delta | Key Change |
|---------|--------|-----------|-------|------------|
| v4.1 | M=200, no kNN, fused 0.7d+0.3c | **0.9034** | baseline | Original |
| v4.4 | M=300, kNN, pure det_score | **0.9076** | +0.0042 | M+100, kNN, det-only score |
| v5.0 | M=300, NO kNN, 0.7d+0.3c | **0.9104** | +0.0070 | Removed kNN, additive fusion |
| **v5.1** | **K=10, T=0.9, α=0.4, decay=0.7** | **PENDING** | est +0.01 | **Top-K categories** |
| **v5.2** | **v5.1 + 2-view cls TTA (max-logit)** | **PENDING** | est +0.001 | **Classification TTA** |
| **v5.3** | **v5.2 + MixUp warm-start classifier** | **PENDING** | est +0.005 | **MixUp/CutMix fine-tune** |
| **Leader** | unknown | **0.9255** | +0.0221 | — |

---

## Classification TTA Results (March 21, ~04:00)

### 2-View TTA Comparison (all with K=10, T=0.9, α=0.4, d=0.7)
| TTA Combo | Blend | Cls mAP | Delta vs v5.1 |
|-----------|-------|---------|---------------|
| orig only (v5.1 baseline) | 0.7533 | 0.6963 | — |
| orig + flip | 0.7538 | 0.6979 | +0.0005 |
| **orig + scale224** | **0.7543** | **0.6989** | **+0.0010** |
| orig + scale320 | 0.7530 | 0.6956 | −0.0003 |
| orig + flip + s224 | 0.7542 | 0.6988 | +0.0009 |
| orig + flip + s224 + s320 | 0.7544 | 0.6999 | +0.0011 |
| 5-view (all) | 0.7545 | 0.7004 | +0.0012 |

### Aggregation Methods (orig + scale224)
| Method | Blend | Delta |
|--------|-------|-------|
| Mean logits (equal weight) | 0.7543 | +0.0010 |
| Mean logits (0.6/0.4 orig) | 0.7540 | +0.0007 |
| Mean logits (0.4/0.6 s224) | 0.7542 | +0.0009 |
| **Max logits** | **0.7545** | **+0.0012** |

### Key Finding
- scale224 (resize short_edge=224) is the best single TTA view to add
- scale320 HURTS slightly (too much context, less product detail)
- Max-logit aggregation beats mean by +0.0002
- 5-view gives only +0.0002 more than 2-view — NOT worth 5× compute
- **v5.2 plan: 2-view TTA (orig + s224) with max-logit, ~2× cls time**

### MixUp/CutMix Warm-Start Training — BREAKTHROUGH (March 21, ~05:30)

**Key insight**: Cold-start MixUp training was failing (val_acc=0.8537 at E10/60). Switched to **warm-start**: load CE-only best checkpoint (val_acc=0.8705) and fine-tune with MixUp at low LR (2e-5).

**Fold-1 training progress** (warm-start from CE-only):
| Epoch | Val Acc | Rare Acc | Notes |
|-------|---------|----------|-------|
| E1 | 0.8691 | 0.3000 | Below CE-only baseline |
| E2 | 0.8693 | 0.3286 | Improving |
| **E3** | **0.8716** | **0.3571** | **Exceeds CE-only 0.8705!** |
| E7 | 0.8716 | 0.3571 | Plateauing |
| E8 | 0.8705 | 0.3571 | Slight decline |

**End-to-end OOF evaluation** (E3 checkpoint, K=10, T=0.9, a=0.4, d=0.7):
| Config | Blend | Cls mAP | Delta vs CE |
|--------|-------|---------|-------------|
| CE orig | 0.7533 | 0.6963 | — |
| CE TTA (max-logit) | 0.7545 | 0.6996 | — |
| **MX orig** | **0.7581** | **0.7094** | **+0.0048** |
| **MX TTA (max-logit)** | **0.7598** | **0.7159** | **+0.0053** |

**This is the largest single improvement found**: +0.0053 blend, +0.0163 cls mAP.
MixUp produces better-calibrated probabilities that dramatically improve classification.

**All-data MixUp retraining started**: 5 epochs from CE-only all-data checkpoint.

---

## BREAKTHROUGH: Top-K Category Predictions (+0.012 OOF)

### The Insight
Instead of submitting 1 category per detection box, submit top-K categories with decayed scores. This exploits COCO evaluation's per-category maxDets:
- **Detection eval** (single class): top-100 by score per image → unchanged (rank-0 predictions dominate)
- **Classification eval** (356 classes): maxDets=100 per category → more TP candidates per category → higher recall

### Results (CE-only classifier, fold-1, 49 val images)
| Config | Blend | Det mAP | Cls mAP | Delta |
|--------|-------|---------|---------|-------|
| v5.0 (K=1, T=1.0, α=0.3) | 0.7416 | 0.7778 | 0.6572 | baseline |
| K=3, decay=0.5 | 0.7516 | 0.7778 | 0.6904 | +0.0100 |
| K=5, T=0.9, decay=0.7 | 0.7527 | 0.7779 | 0.6939 | +0.0111 |
| K=7, T=0.9, α=0.4, decay=0.7 | 0.7532 | 0.7777 | 0.6959 | +0.0116 |
| **K=10, T=0.9, α=0.4, decay=0.7** | **0.7533** | **0.7777** | **0.6963** | **+0.0117** |

### Why It Works
- Classification errors are often top-1 wrong but top-K contains the correct category
- Adding top-K predictions at decayed scores increases recall for each category without hurting detection
- Zero extra inference time — just `.topk()` on existing softmax output
- Score decay ensures rank-0 predictions dominate the detection eval's maxDets=100

### v5.1 Config
```
TOP_K = 10
TEMPERATURE = 0.9
ALPHA = 0.4 (was 0.3)
SCORE_DECAY = 0.7
Score formula: (1-α)*det + α*cls, decayed by 0.7^rank for rank>0
```

---

## What Didn't Work (Exhaustive Sweep Results)

### 1. M Increase: NO EFFECT
Post-NMS only ~100 boxes/image (max 132). M cap is non-binding.
| M | Blend | Delta |
|---|-------|-------|
| 200 | 0.7416 | 0 |
| 300 | 0.7416 | 0 |
| 500 | 0.7416 | 0 |
| 1000 | 0.7416 | 0 |

### 2. Hyperparameter Sweep: FULLY SATURATED
3040 multiplicative configs tested (ArcFace + CE-only). Proxy plateaued at config 200.
504 additive configs tested. Best = v5.0 baseline.

| Sweep | Best Proxy | Best Full Eval | Delta vs v5.0 |
|-------|-----------|---------------|---------------|
| ArcFace mul (3040 configs) | 0.8278 | — | — |
| CE-only mul (3040 configs) | 0.8284 | — | — |
| CE-only additive (504 configs) | 0.8285 | 0.7414 | -0.0002 |

### 3. Temperature Scaling: MARGINAL
Best: T=0.9 gives blend=0.7419 (+0.0003). Within noise alone, but stacks with top-K.

### 4. Classifier Ensemble (CE + ArcFace): HURTS
| Config | Blend | Delta |
|--------|-------|-------|
| CE-only alone | 0.7416 | — |
| Average logits | 0.7392 | -0.0024 |
| 0.6 CE + 0.4 Arc | 0.7400 | -0.0016 |
| 0.8 CE + 0.2 Arc (softmax) | 0.7413 | -0.0003 |
ArcFace is consistently worse (cls=0.6497 vs 0.6572). Ensemble drags down.

### 5. Score Formula: NEGLIGIBLE
Additive and multiplicative give nearly identical results on OOF.

---

## Honest OOF Results (fold-1, 49 val images)

### Detector Ceiling (pre-NMS raw boxes, IoU > 0.5)
| Top-K | Recall | Matched/Total |
|-------|--------|---------------|
| 50 | 0.1663 | 780/4689 |
| 100 | 0.2485 | 1165/4689 |
| 200 | 0.3594 | 1685/4689 |
| 300 | 0.4404 | 2065/4689 |
| 500 | 0.5605 | 2628/4689 |
| 1000 | 0.7451 | 3494/4689 |

### Post-NMS Box Counts (CRITICAL FINDING)
NMS reduces to ~100 boxes/image regardless of settings:
| Conf | NMS IoU | Mean Boxes | Max |
|------|---------|-----------|-----|
| 0.001 | 0.5 | 101 | 132 |
| 0.003 | 0.55 | 101 | 145 |
| 0.001 | 0.65 | 109 | 190 |

### Oracle Classification (perfect cls on detected boxes)
- det=0.7775, cls=0.8370, **blend=0.7953**
- Gap from v5.1: 0.7953 − 0.7533 = **0.0420** (still room in cls)

### OOF ↔ Test Calibration (CRITICAL WARNING)
| Version | OOF (CE-f1) | Test | Δ OOF | Δ Test |
|---------|-------------|------|-------|--------|
| v4.1 | 0.7416 | 0.9034 | — | — |
| v4.4 | 0.7380 | 0.9076 | −0.0036 | +0.0042 |
| v5.0 | 0.7416 | 0.9104 | +0.0036 | +0.0028 |
| **v5.1** | **0.7533** | **???** | **+0.0117** | **???** |

v5.1 has the LARGEST OOF improvement we've seen (+0.0117). Previous OOF→test transfers:
- v5.0: +0.0036 OOF → +0.0028 test (OOF overpredicted)
- v4.4: -0.0036 OOF → +0.0042 test (OOF anti-predicted)
- v5.1 improvement is 3x larger than any previous change → more likely to transfer

---

## Key Insights (Updated)

### 1. Post-NMS ~100 boxes/image makes M irrelevant
NMS at IoU=0.5 keeps only ~100 boxes. M=200 already captures all. The recall ceiling at M=300 (44%) is set by NMS, not M.

### 2. Classification quality is the ONLY lever
Oracle cls_mAP = 0.8370. Our best = 0.7159 (MixUp + TTA). Still 0.1211 gap.
MixUp warm-start improved cls_mAP from 0.6963 → 0.7159 (+0.0196).

### 3. Top-K categories exploit COCO eval structure
Not a hack — legitimate strategy. COCO eval handles multiple predictions per image per category natively. Higher K = more recall per category at moderate precision thresholds.

### 4. Hyperparameter tuning is fully exhausted
6500+ configs tested (mul + additive). No improvement over v5.0 baseline from hyperparams alone.

### 5. MixUp warm-start is the single biggest improvement
- Cold-start MixUp FAILED (never reaches CE-only quality)
- Warm-start from CE-only checkpoint + low LR (2e-5) + 3 epochs = sweet spot
- Higher val_acc doesn't mean better mAP — E3 (0.8716) > E5 (0.8723) on blend
- MixUp likely improves calibration of softmax probabilities for top-K scoring

### 6. OOF improvement progression
| Version | OOF Blend | Delta | Key Change |
|---------|-----------|-------|------------|
| v5.0 | 0.7416 | — | Baseline |
| v5.1 | 0.7533 | +0.0117 | Top-K categories |
| v5.2 | 0.7545 | +0.0012 | Classification TTA |
| **v5.3** | **0.7598** | **+0.0053** | **MixUp classifier** |
| **Total** | | **+0.0182** | |

---

## Training Status
| Variant | Progress | Best Val Acc | Best Rare Acc | Status |
|---------|----------|-------------|---------------|--------|
| CE-only resumed | **30/30** | **0.8705** | **35.7%** | **DONE** |
| MixUp warm-start (fold-1, cos/20) | **20/20** | **0.8742** | **35.7%** | **DONE** |
| MixUp save-all (fold-1, cos/5) | **5/5** | 0.8708 | N/A | **DONE** |
| Label smoothing (fold-1) | **5/5** | 0.8684 | N/A | **DONE** — worse than CE |
| MixUp all-data v1 (lr=1e-5, 5ep) | **5/5** | N/A | N/A | **DONE** |
| MixUp all-data v2 (lr=2e-5, 3ep) | **3/3** | N/A | N/A | **DONE** |
| **MixUp all-data 1ep (lr=2e-5)** | **1/1** | N/A | N/A | **DONE** |

### Key Training Findings

**1. MixUp warm-start >> cold-start and label smoothing**
- Cold-start MixUp: val_acc=0.8537 at E10/60 (failed)
- Label smoothing (ε=0.1): val_acc=0.8684 (worse than CE-only 0.8705)
- Warm-start MixUp: val_acc=0.8716 at E3, **blend=0.7598** (+0.0053 over CE TTA)

**2. Higher val_acc ≠ better blend**
- E3 (cos/20): val_acc=0.8716, **blend=0.7598** (BEST)
- E5 (cos/20): val_acc=0.8723, blend=0.7557 (worse)
- E14 (cos/20): val_acc=0.8742, not evaluated (likely even worse)

**3. Per-epoch evaluation (cos/5 schedule)**

| Epoch | Val Acc | TTA Default | TTA BEST | Best K |
|-------|---------|-------------|----------|--------|
| **E1** | 0.8697 | 0.7579 | **0.7584** | K=15, T=0.9, a=0.4, d=0.7 |
| E2 | 0.8704 | 0.7567 | 0.7572 | K=15 |
| E3 | 0.8708 | 0.7572 | 0.7575 | K=15 |
| E4 | 0.8699 | 0.7568 | 0.7576 | K=15 |
| E5 | 0.8703 | 0.7575 | 0.7579 | K=15 |

- **E1 is optimal** for cos/5 schedule (barely trained = similar to E3 of cos/20)
- **K=15 consistently beats K=10** across all epochs (+0.003-0.005)
- For all-data submission: **1 epoch** at lr=2e-5 is optimal

---

## Fine Grid Sweep Results (CE-only, complete)
| TTA Config | Best Blend | Best Params |
|-----------|-----------|-------------|
| orig+s224 (mean) | 0.7545 | K=20, T=1.0, a=0.4, d=0.7 |
| orig+s224+flip (3-view) | 0.7545 | K=15, T=1.0, a=0.45, d=0.7 |
| softmax-then-avg | 0.7540 | K=10, T=0.9, a=0.4, d=0.7 |
| **max-logit (orig+s224)** | **0.7545** | **K=10, T=0.9, a=0.4, d=0.7** |

All CE configurations saturate at **0.7545**. MixUp is the only path beyond this.

---

## Assets — READY FOR SUBMISSION
| File | Size | OOF Blend | Status | Notes |
|------|------|-----------|--------|-------|
| **submission_5.4_cos20.zip** | **295 MB** | **~0.7598** | **PRIMARY** | MixUp cos/20 E3 all-data + K=15 + TTA |
| **submission_5.4.zip** | **295 MB** | **~0.7584** | **BACKUP 1** | MixUp cos/1 E1 all-data + K=15 + TTA |
| **submission_5.3c_k15.zip** | **295 MB** | **~0.7575** | **BACKUP 2** | MixUp cos/3 E3 all-data + K=15 |
| submission_5.2a.zip | 272 MB | 0.7545 | SAFETY | CE-only + TTA + K=10 |
| submission_5.1.zip | 272 MB | 0.7533 | FALLBACK | CE-only + K=10 only |
| submission_5.0.zip | 272 MB | 0.7416 | Submitted | Scored **0.9104** |

### Recommended submission order:
1. **v5.4_cos20** — MixUp with cos/20 schedule (exactly matches validated fold-1 E3 recipe)
2. **v5.4** — MixUp 1-epoch (best on cos/5 eval, different all-data recipe)
3. **v5.2a** — CE-only safety baseline if MixUp doesn't transfer

### LR Schedule Analysis (why cos/20 is best)
| Schedule | Best Fold-1 Blend | LR at Optimal Epoch |
|----------|------------------|---------------------|
| **cos/20, 3 ep** | **0.7598** | **~1.95e-5 (barely decayed)** |
| cos/5, 1 ep | 0.7584 | 2.0e-5 (no decay within E1) |
| const lr=2e-5 | 0.7572 | 2.0e-5 (no decay) |
| cos/5, 3 ep | 0.7575 | ~5.1e-6 (too much decay) |
| cos/3, 3 ep | 0.7575 | ~5.1e-6 (too much decay) |

---

## v5.4_cos20 TEST RESULT — March 21, ~13:30

**Score: 0.9140 → Rank #25 (leader 0.9255, 14 submissions)**

### Test Score Progression (GROUND TRUTH)
| Version | Test Score | Delta | Key Change |
|---------|-----------|-------|------------|
| v5.0 | 0.9104 | — | Baseline: CE-only, K=1, no TTA |
| v5.4_cos20 | **0.9140** | **+0.0036** | MixUp cos/20 + K=15 + 2-view TTA |
| Leader | **0.9255** | +0.0151 | Unknown (14 submissions) |

### Gap Analysis
```
Score = 0.7 × det_mAP + 0.3 × cls_mAP

If det_mAP=0.96: our cls=0.807, leader cls=0.845, cls gap=0.038
If det_mAP=0.95: our cls=0.830, leader cls=0.868, cls gap=0.038

To close 0.0115 gap:
  - Detection only:  +0.0164 det_mAP needed (hard, near ceiling)
  - Classification only: +0.0383 cls_mAP needed (very hard)
  - Split 50/50: +0.008 det + 0.019 cls

OOF→test translation: only 20% (0.0182 OOF → 0.0036 test)
```

### Grounded Strategy — Closing the 0.0115 Gap

**REALITY CHECK**: Our overnight work (MixUp + TTA + top-K) gave +0.0182 OOF but only +0.0036 test. We need 3x that gain. Incremental parameter tuning won't close this — we need a structural improvement.

#### ~~LEVER 1: Soft-NMS~~ — TESTED & ELIMINATED (March 21, 14:00)
- **Result**: Single-cat +0.0018 (noise), **Top-K -0.0019 to -0.0056 (hurts)**
- Duplicate boxes from Soft-NMS dilute Top-K predictions
- det_mAP unchanged across all variants (0.776-0.778) — detection is at ceiling
- Hard NMS at IoU=0.5 is correct for side-by-side shelf products
- **Verdict: DEAD LEVER. Do not revisit.**

#### LIVE LEVER 1: FP16 Detector + Dual-Classifier Ensemble
- FP16 ONNX (68MB) + CE classifier (164MB) + MX classifier (164MB) = 396MB < 420MB
- MixUp-heavy blending (0.7 MX + 0.3 CE), NOT equal weight
- Must verify FP16 parity offline first
- **Status: PREPARING — needs FP16 export + OOF eval tonight**

#### LIVE LEVER 2: Selective Top-K Emission
- Current K=15 for all boxes is blunt
- Smart rule: high-confidence → K=1-3, ambiguous → K=10-15
- Targets eval asymmetry (det=global maxDets, cls=per-category recall)
- **Status: NOT YET IMPLEMENTED — code change + OOF eval needed**

### Revised Plan → see detection/saturday.md
All bullet allocation moved to saturday.md for clean tracking.

### What the Leader Likely Has
- 14 submissions = lots of test-set feedback for tuning
- Possibly: larger classifier (EVA-02-L), better training data, WBF
- Their det/cls split is unknown — could be stronger on either component
