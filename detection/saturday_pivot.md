# Saturday Pivot — March 21, 19:15 CET (BULLET 1 LANDED)

## Hard Numbers
| Metric | Value |
|--------|-------|
| Our best test | **0.9161** (Bullet 1 newrecipe, +0.0021) |
| Previous best | 0.9140 (v5.4_cos20, rank #25) |
| Leader | **0.9255** (14 submissions) |
| Gap | **0.0094** (was 0.0115) |
| Bullets remaining | 1 before midnight + 6 after = **7 total** |
| OOF→test (newrecipe) | **~100%** (+0.0022 OOF → +0.0021 test) |
| Competition end | March 22, 15:00 CET (~20 hours) |

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

## Bullet 1 Result — SUBMITTED 0.9161
- New recipe E1 all-data (strong aug + LS 0.05 + cos/60)
- +0.0021 over v5.4_cos20 (0.9140)
- OOF→test translation near 1:1 (regularization helped generalization)

## Current State (19:15 CET)

### Bullet 2: Detector-Crop Adaptation — OOF EVAL PENDING
Train classifier on 50% GT crops + 50% detector-produced crops.
Fixes the structural train/infer mismatch.

- **Fold-1 training DONE**: 5 epochs, E4 peak (val_acc=0.8927)
- Checkpoints: `/root/ng/output_detcrop_fold1/detcrop_e1-e5.safetensors`
- GPU free, ready for OOF eval

| Time (CET) | Action |
|-------------|--------|
| ~~18:50~~ 19:15 | OOF eval on E1-E5 |
| 19:30 | Go/no-go decision |
| 19:40 | All-data training if positive |
| **20:30** | **SUBMIT Bullet 2** |

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

## Honest Probability (updated post-Bullet 1)
- Close remaining 0.0094 gap with 7 bullets: 20-30%
- Top-15 climb: likely if detcrop works
- Cross 0.920: realistic (needs +0.004 = ~+0.02 OOF)
- Beat 0.9255: 10-15% (needs +0.0094 test = structural breakthrough)
