# Saturday Pivot — March 21, 17:10 CET (RESULTS IN)

## Hard Numbers
| Metric | Value |
|--------|-------|
| Our best test | **0.9140** (v5.4_cos20, rank #25) |
| Leader | **0.9255** (14 submissions) |
| Gap | **0.0115** |
| Bullets remaining | 2 today + 6 after midnight = **8 total** |
| OOF→test translation | **20%** (0.0182 OOF → 0.0036 test) |
| Competition end | March 22, 15:00 CET (~22 hours) |

## FOLD-1 MIRROR RESULTS — GO SIGNAL CONFIRMED

### OOF Comparison (fold-1, 49 val images, frozen v5.4 stack)
| Checkpoint | det_mAP | cls_mAP | blend | vs CE | vs orig MixUp |
|---|---|---|---|---|---|
| CE-only (anchor) | 0.7780 | 0.6996 | 0.7545 | — | — |
| Original MixUp fold-1 | 0.7753 | 0.6807 | 0.7469 | -0.0076 | — |
| **New recipe E1** | **0.7781** | **0.7067** | **0.7567** | **+0.0022** | **+0.0098** |
| New recipe E2 | 0.7783 | 0.7058 | 0.7566 | +0.0021 | +0.0097 |

### Key Findings
1. **Detection flat** (~0.778) — confirms only classifier quality matters
2. **Original MixUp fold-1 WORSE than CE-only** — surprising
3. **New recipe E1 is best-ever fold-1 OOF** (+0.0098 blend vs orig MixUp)
4. **E2 already declining** → E1 is peak
5. **cls_mAP +0.026** vs original MixUp — clears +0.015 bar
6. Go/No-Go: **+0.0098 > +0.006 blend bar → GO**

### What Changed in New Recipe
- Stronger augmentation: ColorJitter(0.4), rotation ±15°, Gaussian blur, random erasing 40%
- Label smoothing 0.05 (vs none in original MixUp)
- cos/60 schedule (vs cos/20 or cos/5 in original)
- Same MixUp/CutMix/lr/optimizer

## Current State (17:10 CET)

### H100 GPU
- **All-data new recipe training running** (PID 70264, E1 in progress)
- Rogue background agent processes killed + scripts disabled
- ETA: E1 checkpoint ~17:37, E2 ~17:47

### Bullet 1 Timeline
| Time (CET) | Action |
|-------------|--------|
| 17:37 | E1 checkpoint saved |
| 17:40 | Package into submission ZIP |
| 17:50 | E2E verification |
| **17:55** | **SUBMIT Bullet 1** |

### Bullet 2: Detector-Crop Adaptation
Train classifier on 50% GT crops + 50% detector-produced crops.
Fixes the structural train/infer mismatch.

| Time (CET) | Action |
|-------------|--------|
| 17:50 | Start detector-crop fold-1 mirror (GPU free after packaging) |
| 18:10 | OOF eval |
| 18:20 | Go/no-go |
| 18:30 | All-data version if positive |
| **19:00** | **SUBMIT Bullet 2** |

## Every Post-Processing Lever Tested — All Dead

| Lever | OOF Result | Verdict |
|-------|------------|---------|
| Soft-NMS | -0.002 to -0.006 on Top-K | HURTS |
| FP16 detector | Already FP16 (136.5MB) | CAN'T SHRINK |
| CE+MX ensemble | +0.0004, doesn't fit 420MB | DEAD |
| Mean vs Max TTA | Max wins by +0.0005 | ALREADY OPTIMAL |
| Selective top-K | K=8 through K=25 identical | DEAD |
| WBF | 0.5561 vs hard NMS 0.7776 | CATASTROPHIC |
| NMS IoU sweep | All within noise | DEAD |
| T, alpha, K sweeps | Current params optimal | DEAD |

**Detection is at ceiling. Classification quality is the only live lever.**

## Honest Probability
- Close 0.0115 gap with 2 bullets: 15-25%
- Top-15 climb: plausible
- Cross 0.920: possible
- Beat 0.9255: 10-15%
