# NorgesGruppen Detection — Blitz

## Pre-Blitz Submissions (Mar 20-21)
| # | Date | Time (CET) | Size | Score | Time | Key Change | Learning |
|---|------|-----------|------|-------|------|------------|----------|
| 1 | Mar 20 | 01:12 | 280.1 MB | **0.7365** | 67.1s | First working submission | Baseline established — kNN classifier, early detector |
| 2 | Mar 20 | 02:25 | 280.5 MB | **FAILED** | — | ZIP validation error | run.py must be at ZIP root, no disallowed files |
| 3 | Mar 20 | 02:31 | 280.5 MB | **0.7459** | 69.1s | Fix from #2 | +0.0094 — packaging matters |
| 4 | Mar 20 | 10:45 | 271.6 MB | **0.7743** | 105.6s | Pipeline overhaul | +0.0284 — big jump |
| 5 | Mar 20 | 15:58 | 271.6 MB | **0.9034** | 79.0s | v4.1: YOLOv8x ONNX | +0.1291 — ONNX detector + scoring rework |
| 6 | Mar 21 | 01:10 | 304.3 MB | **0.9076** | 98.7s | v4.4: kNN + det-only score | +0.0042 |
| 7 | Mar 21 | 01:46 | 271.6 MB | **0.9104** | 97.8s | v5.0: removed kNN, additive fusion, top-K | +0.0028 |
| 8 | Mar 21 | 13:03 | 294.6 MB | **FAILED** | — | Size bug | predictions.json 18MB > 10MB limit |
| 9 | Mar 21 | 13:30 | 294.6 MB | **0.9140** | 128.3s | v5.4_cos20: MixUp cos/20 + K=15 + TTA | +0.0036 |
| 10 | Mar 21 | 18:59 | 271.5 MB | **0.9161** | 122.9s | Newrecipe E1 (strong aug + LS 0.05 + cos/60) | +0.0021 |
| 11 | Mar 21 | 20:56 | 294.6 MB | **0.9150** | 128.3s | Detcrop E4 (50/50 GT+det-crop) | -0.0011 — regression |

## Blitz (Mar 22, midnight → 03:00 CET)
| Blitz # | Score | Content | Key Insight |
|---------|-------|---------|-------------|
| B1 | **0.9154** | long_ce_e13 (15-epoch CE classifier) | -0.0007 vs best. long_ce family weaker than newrecipe. |
| B2 | **0.9168** | Model soup: newrecipe E1+E2 avg | **NEW BEST.** Soup works! +0.0007 over single checkpoint. |
| B3 | **0.9219** | **Finetuned detector** + soup_nr_e1e2 | **MASSIVE +0.0051.** Domain-specific detector = biggest lever since ONNX switch. |
| B4 | **0.9229** | Finetuned det + weighted cross soup (80/20) | **NEW BEST.** +0.0010 from cross-recipe soup. Confirmed: weighted blending helps. |
| B5 | **0.9212** | **All-data detector** + weighted cross soup | **REGRESSED -0.0017.** No holdout = overfitting. 248 images too few without early stopping. |
| B6 | — | TBD | **LAST BULLET** |

## Score Progression
```
0.7365 → 0.7459 → 0.7743 → 0.9034 → 0.9076 → 0.9104 → 0.9140 → 0.9161 → 0.9150 → 0.9154 → 0.9168 → 0.9219 → 0.9229 → 0.9212
  +94      +284     +1291      +42       +28       +36       +21      -11       -7        +14       +51       +10       -17
```

## Current Position
| Metric | Value |
|--------|-------|
| **Our best** | **0.9229** (Blitz 4) |
| **Leader** | **0.9255** |
| **Gap** | **0.0026** |
| **Bullets remaining** | **1** |

## What Moved the Board (Blitz Phase)
1. **Finetuned detector**: +0.0051 (OOF was +0.0097, 53% transfer — detector changes transfer better than classifier)
2. **Weighted cross soup**: +0.0010 (80/20 blend > pure soup)
3. **Model soup**: +0.0007 (OOF was +0.0035, 20% transfer — consistent with historical rate)
4. **long_ce_e13**: -0.0007 (contaminated OOF useless, classifier family exhausted)
5. **All-data detector**: -0.0017 (overfit without holdout, OOF gains didn't transfer)

## Remaining Strategy — 1 BULLET LEFT
**Pre-packaged option:** Finetuned detector + soup_cross_70_30 (heavier long_ce weight). At `/tmp/blitz_05_newdet_cross70_30.zip`. NOT E2E verified.

**Other soup ratios available on server:** soup_cross_60_40, soup_nr1_lce13 (50/50), soup_4way.

## Key Takeaways
1. **Two regime changes**: ONNX detector (+0.1291) and finetuned detector (+0.0051)
2. **Model soup is free money** — always average top checkpoints
3. **Detector is 70% of score** — small det improvements > large cls improvements
4. **Finetuned detector also improves cls** (+0.0106 cls_mAP via better crops)
5. **OOF transfer**: classifier ~20%, detector ~53% — detector changes more reliable
6. **Inference tricks are dead** — K/T/alpha/decay, emission strategies, NMS variants all exhausted
7. **All-data training without holdout = overfitting** — 248 images not enough to train blindly
8. **Confirmed dead levers**: all-data detector, long_ce family, detcrop, all inference tweaks
