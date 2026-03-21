# Overnight Sprint v2 — March 21 22:00 → March 22 15:00 CET
## REVISED after Codex review + advisor correction

## Scoreboard
| Metric | Value |
|--------|-------|
| Our best | **0.9161** (Bullet 1 newrecipe) |
| Leader | **0.9255** (14 submissions) |
| Gap | **0.0094** |
| Bullets | **7** (1 pre-midnight + 6 post-midnight) |
| Realistic ceiling | **0.919 - 0.923** |
| #1 probability | **10-15%** (with corrected strategy) |

## Core Principle
**Live gains have come from classifier recipe changes, not detector/post-process folklore.**
Only trust paired deltas. Only combine confirmed live winners.

---

## TONIGHT: Bullet 2 — Detcrop E4

Ship detcrop E4 as-is. Same inference stack as Bullet 1. Only classifier weights change.

- Paired OOF: +0.0015 blend, +0.0069 cls_mAP, det flat
- Realistic EV: +0.0005 to +0.0025
- Bullet expires at midnight

---

## POST-MIDNIGHT: Adaptive Emission — Highest EV Next Probe

Current `K=15 for all boxes` is blunt. Smarter emission uses confidence + margin to decide K per box.

### Three families to sweep (20-40 configs total, NOT a giant sweep)

#### Family A: Confidence + Margin Banding
After softmax with T=0.9:
- p1 ≥ 0.80 AND (p1 - p2) ≥ 0.25 → K=1
- p1 ≥ 0.60 AND (p1 - p2) ≥ 0.12 → K=5
- else → K=15
- Score cutoff raised to 0.015

#### Family B: A + Low-Det Cap
Same as A, plus:
- det_score < 0.10 → cap K=3
- det_score < 0.06 → force K=1
Targets garbage boxes that currently spray 15 labels.

#### Family C: Relative-Tail Emission
Always emit top-1. Emit rank r>1 only while:
- p_r ≥ 0.25 × p1
- AND final score ≥ 0.015
- Cap Kmax=15
Uses actual tail shape instead of fixed K.

### Realistic EV: +0.0005 to +0.0020

---

## Bullet Strategy: Isolated Probes, Not Stacked Combos

### Decision Tree

```
B2 (tonight): detcrop E4
    ↓ read live score
B3: best adaptive-emission probe
    - if detcrop scored > 0.9161: use detcrop weights
    - if detcrop flat: use 0.9161 newrecipe weights
    ↓ read live score
B4: combine ONLY confirmed live winners
    - both won → combine
    - one won → stay in winning family
    - neither won → classifier recipe scout (NOT detector tricks)
B5-B7: tune winning family OR adjacent classifier scouts
```

**Do NOT pre-assign bullets. React to live test feedback.**

---

## Demoted / Dead Levers

| Lever | Status | Why |
|-------|--------|-----|
| Detector multi-scale | **DEMOTED** — 30-min scout only | ONNX fixed at 1280×1280, flip TTA already exists, realistic EV 0 to +0.0015 |
| Crop padding | **DEAD** | Already killed at -0.0022. Inference-only padding is negative EV |
| 336px classifier | **Desperation tier** | scale320 TTA hurt. More pixels ≠ better for this data |
| Stack-all-passing | **WRONG** | 49-image fold + noisy transfer = compounded noise |

---

## If Detcrop AND Emission Both Flat

Before going to detector tricks or 336px, try:
1. **Adjacent detcrop family**: 75/25 GT/det-crop ratio instead of 50/50
2. **Adjacent classifier checkpoint**: sweep the 15 CE long-training checkpoints (free, already on server)
3. **Recipe variant**: detcrop + different aug strength or schedule

Stay on the classifier branch that has actually produced test gains.

---

## What We Will NOT Do
- Fake multi-scale (3+ hours for 0 to +0.0015)
- Soft-NMS, WBF (confirmed dead)
- Crop padding (confirmed dead)
- Combine three small fold wins into one bullet
- Pre-assign all 7 bullets to fixed plan

## Grounded Win Odds
| Scenario | Probability |
|----------|-------------|
| Detcrop lands live | 40-55% |
| Adaptive emission lands live | 30-45% |
| Both land + combo works | 15-25% |
| Beat 0.9255 | 10-15% |
| Realistic overnight ceiling | 0.919-0.923 |
