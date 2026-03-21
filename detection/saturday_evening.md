# Saturday Evening — March 21, 19:35 CET

## Scoreboard
| Metric | Value |
|--------|-------|
| Our best test | **0.9161** (Bullet 1 newrecipe E1) |
| Leader | **0.9255** (14 submissions) |
| Gap | **0.0094** |
| Bullets used today | 1 of 2 pre-midnight |
| Bullets remaining | **1 pre-midnight + 6 post-midnight = 7** |
| Competition end | March 22, 15:00 CET (~19.5 hours) |

## Test Score History (ground truth)
| Version | Test Score | Delta | Key Change |
|---------|-----------|-------|------------|
| v4.1 | 0.9034 | — | Original baseline |
| v4.4 | 0.9076 | +0.0042 | M+100, kNN, det-only score |
| v5.0 | 0.9104 | +0.0028 | Removed kNN, additive fusion |
| v5.4_cos20 | 0.9140 | +0.0036 | MixUp cos/20 + K=15 + TTA |
| **Bullet 1** | **0.9161** | **+0.0021** | **New recipe: strong aug + LS 0.05 + cos/60** |

## What 0.9161 Confirmed
1. **Classifier training changes move the live board** — post-process tricks are dead
2. New recipe's regularization (LS + strong aug) helped generalization
3. Remaining edge is in the classifier family, not detection plumbing

## What 0.9161 Does NOT Confirm
- OOF→test "near-perfect translation" was a **misattributed comparison** (fold delta was vs CE anchor, live delta was vs v5.4_cos20 — different baselines)
- Historical OOF numbers are inconsistent across evaluator versions — **only trust paired deltas from the current evaluator**

---

## Bullet 2: Detector-Crop Adaptation — IN FLIGHT

### The Structural Argument
Classifier was trained on GT bounding box crops but deployed on detector-produced crops (noisy, misaligned). Training on 50% GT + 50% detector crops should close this distribution mismatch.

### Paired OOF Eval (March 21, 18:25 CET)
**Single evaluator, same detector cache, same fold-1 split, same inference stack.**

| Checkpoint | det_mAP | cls_mAP | blend | vs baseline |
|---|---|---|---|---|
| **newrecipe E1 (baseline)** | **0.7781** | **0.7067** | **0.7567** | — |
| detcrop E1 | 0.7779 | 0.7036 | 0.7556 | -0.0011 |
| detcrop E2 | 0.7767 | 0.7051 | 0.7552 | -0.0015 |
| detcrop E3 | 0.7771 | 0.7074 | 0.7562 | -0.0005 |
| **detcrop E4** | **0.7774** | **0.7136** | **0.7582** | **+0.0015** |
| detcrop E5 | 0.7770 | 0.7030 | 0.7548 | -0.0019 |

### Signal Assessment
- **Blend**: +0.0015 (meets minimum ship threshold)
- **cls_mAP**: +0.0069 (meets strong-ship ≥+0.006 threshold)
- **det_mAP**: flat (-0.0007, noise)
- **Nature**: narrow peak at E4 — sharp, fragile effect, not a robust family yet
- **Decision**: SHIP as high-EV probe with the expiring bullet

### Ship Thresholds Used
| Condition | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| blend ≥ +0.0015 | +0.0015 | +0.0015 | YES |
| cls_mAP ≥ +0.006 (strong) | +0.006 | +0.0069 | YES |
| det_mAP flat | within noise | -0.0007 | YES |

### All-Data Training — RUNNING
- **Started**: 18:35 UTC (20:35 CET)
- **Script**: `train_detcrop_alldata.py` (same recipe: MixUp + LS 0.05 + cos/60 + lr=2e-5)
- **Dataset**: 42,552 samples (all 248 images, GT + detector crops + products)
- **Saving every epoch**: E1-E5
- **ETA per epoch**: ~1000s (~17 min)
- **ETA all done**: ~21:10 CET

### Epoch-to-Submit Mapping
Fold optimum at E4 with 199 train images → all-data (248 images) has ~25% more data per epoch.
Step-matched all-data optimum is ~E3.2. **Package E3 and E4. Submit E4 primary.**

### Packaging Plan
| Time (CET) | Action |
|-------------|--------|
| ~21:10 | Training done, all 5 checkpoints saved |
| 21:15 | Package E3 + E4 into submission ZIPs |
| 21:20 | E2E verify both (10-image dry run) |
| 21:30 | **Submit E4 as Bullet 2** |
| | Hold E3 as first post-midnight candidate |

---

## Fallback: If Detcrop Flat on Live Board

If Bullet 2 (detcrop E4) scores ≤ 0.9161 on the live board:

**Do NOT resurrect dead detection ideas.**

Use post-midnight bullets on narrow retune of Bullet 1 family:
- K ∈ {10, 15, 20}
- T ∈ {0.8, 0.9, 1.0}
- decay ∈ {0.65, 0.7, 0.75}

Or submit detcrop E3 for information value (structurally different → useful signal for remaining bullets).

---

## What Is Explicitly Dead
| Lever | Status |
|-------|--------|
| Soft-NMS | HURTS (-0.002 to -0.006) |
| WBF | CATASTROPHIC |
| CE+MX ensemble | Doesn't fit budget |
| Selective top-K | No improvement K=8-25 |
| NMS IoU sweep | Noise |
| T/alpha/K broad sweeps | Saturated |
| FP16/INT8 detector | Already FP16, can't shrink |
| val_acc-based selection | Proven unreliable vs blend |

## What Is NOT Being Changed Tonight
- Detector weights (frozen)
- Hard NMS (frozen)
- TTA (frozen: 2-view, max-logit)
- K=15, T=0.9, alpha=0.4, decay=0.7 (frozen)
- Score threshold < 0.01 + 500/image cap (frozen)
- Output format (frozen)

**Only the classifier weights change. Attribution is clean.**

---

## Honest Probability
- Bullet 2 moves board: **40-55%** (structural gain, cls signal real)
- Bullet 2 > 0.9161: **35-45%**
- Close gap to < 0.005: **10-15%**
- Beat 0.9255 tonight: **< 5%** (but opens a family for tomorrow)
