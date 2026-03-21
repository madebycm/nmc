# Saturday Pivot — March 21, 16:15 CET (Updated)

## Hard Numbers
| Metric | Value |
|--------|-------|
| Our best test | **0.9140** (v5.4_cos20, rank #25) |
| Leader | **0.9255** (14 submissions) |
| Gap | **0.0115** |
| Bullets remaining | 2 today + 6 after midnight = **8 total** |
| OOF→test translation | **20%** (0.0182 OOF → 0.0036 test) |
| Competition end | March 22, 15:00 CET (~23 hours) |

## Every Post-Processing Lever Tested — All Dead

| Lever | OOF Result | Verdict |
|-------|------------|---------|
| Soft-NMS | -0.002 to -0.006 on Top-K | HURTS |
| FP16 detector | Already FP16 (136.5MB) | CAN'T SHRINK |
| CE+MX ensemble | +0.0004, doesn't fit 420MB | DEAD |
| Mean vs Max TTA | Max wins by +0.0005 | ALREADY OPTIMAL |
| Selective top-K | K=8 through K=25 identical | DEAD (threshold already adaptive) |
| WBF | 0.5561 vs hard NMS 0.7776 | CATASTROPHIC |
| NMS IoU sweep | All within noise | DEAD |
| T, alpha, K sweeps | Current params optimal | DEAD |

**Detection is at ceiling. Classification quality is the only live lever.**

## Critical Correction: Contamination Trap

The all-data training (15 epochs, running now on H100) produces **packaging candidates only**. Those checkpoints CANNOT be OOF-evaluated — fold-1 val images are in the training set. Any eval would be contaminated.

**The fix:** Run a fold-1 mirror of the same recipe (train on 199 images, eval on 49) to get honest signal. Only then decide if the recipe is bullet-worthy.

## Current State (16:15)

### H100 (1 GPU, 86.38.238.168)
- All-data training on E4/15, acc climbing (0.53→0.63→0.59→0.67)
- E1-E3 checkpoints saved (packaging candidates)
- **ACTION: Kill after E4 saves. Start fold-1 mirror.**

## Execution Plan

### Step 1: Fold-1 Mirror (NOW)
- Kill all-data training (E1-E4 saved = enough packaging candidates)
- Start fold-1 mirror: train on 199 images, val on 49, 5 epochs
- Same recipe: MixUp + label smoothing 0.05, cos/60, lr=2e-5, strong aug
- ETA: ~25 min for 5 epochs

### Step 2: Honest OOF Eval (~16:45)
- Evaluate E1-E5 fold-1 checkpoints with frozen v5.4 inference stack
- Same detector cache, same 2-view TTA, same K=15/T=0.9/alpha=0.4
- det_mAP should be flat (only classifier changed) — if it shifts, eval drifted

### Step 3: Go/No-Go Decision (~16:50)
**Bullet-worthy bar (harsh):**
- At least **+0.006 blend** or **+0.015 cls_mAP** over current MixUp recipe
- Anything smaller is noise / not worth a bullet

**If YES:** Package best all-data epoch (E1-E3), E2E verify, fire bullet
**If NO:** Move to Path B (INT8 detector → larger/dual classifier)

### Step 4: Tiny Calibration (~16:55, only if Step 3 = YES)
- K ∈ {10, 15, 20}, T ∈ {0.8, 0.9, 1.0}
- No big sweeps. Post-processing is saturated.

## Remaining Paths (if fold-1 mirror fails)

### Path B: INT8 Detector → Larger/Dual Classifier
- INT8 ONNX (~34MB) frees ~100MB budget
- Enables: EVA-02 Large (304MB FP16) or dual EVA-02 Base ensemble
- Risk: INT8 detection regression
- **Next in line if current training doesn't clear the bar**

### Path C: Different Classification Signal
- CLIP zero-shot with product reference images
- kNN with MixUp features at test time
- Category-specific calibration
- **Speculative, untested**

## Honest Probability (unchanged)
- Top-15 climb: plausible
- Cross 0.920: possible with one positive probe
- Beat 0.9255: 10-20%
