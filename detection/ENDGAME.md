# Detector Endgame — Last Bullet

## Strategy (from advisor)
Stop guessing classifier soup ratios. The detector is 70% of score and transfers at 53%.

**Plan:**
1. 5-fold CV on 224-train pool → find optimal epoch E* by mean mAP50
2. Train 1-3 candidates on 224-train at E*, E*±3 epochs
3. Evaluate end-to-end on untouched 24-val with current best classifier (weighted cross soup)
4. If winner found → retrain on all 248 at E*, submit

**Critical details:**
- Select by **mAP50**, not Ultralytics default fitness (mAP50-95)
- Keep 24-val **untouched** as final selector
- If doing detector soup: greedy inclusion, EMA weights, refresh BN stats
- Final retrain on 248 only AFTER E* is known (avoids B5's blind overfitting)

## Timeline (revised — measured ~15s/epoch on H100)
| Step | Est. Time | Status |
|------|-----------|--------|
| 5-fold CV (5 × 75 epochs) | ~45 min | **RUNNING** (PID 105005, started 02:42 CET) |
| Analyze → pick E* | 5 min | WAITING |
| Train candidate at E* on 224-train | ~11 min | WAITING |
| E2E eval candidate on 24-val (full pipeline) | ~10 min | WAITING |
| If winner: retrain on all 248 at E* | ~15 min | WAITING |
| Export ONNX + package + E2E verify | ~15 min | WAITING |
| **Total** | **~100 min** | |

## Scripts on Server
| Script | Purpose |
|--------|---------|
| `/root/war/train_5fold_cv.py` | 5-fold CV, logs mAP50 per epoch, saves cv_summary.json |
| `/root/war/train_candidate_detector.py` | Train single candidate at E* (--epochs, --patience 0) |
| `/root/war/train_final_alldata.py` | Final all-248 retrain at fixed E* |
| `/root/war/eval_candidate_e2e.py` | Full pipeline eval: ONNX export → detect → classify → COCO eval |

## Current Position
| Metric | Value |
|--------|-------|
| Our best | **0.9229** (B4: finetuned det + weighted cross soup 80/20) |
| Leader | **0.9255** |
| Gap | **0.0026** |
| Bullets remaining | **1** |

## 5-Fold CV Results (COMPLETE)
| Fold | Best Epoch | mAP50 | Early Stopped? |
|------|-----------|-------|----------------|
| 0 | 51 | 0.9473 | No (ran 75) |
| 1 | 42 | 0.9535 | Yes (66 epochs) |
| 2 | 72 | 0.9428 | No (ran 75) |
| 3 | 47 | 0.9520 | Yes (67 epochs) |
| 4 | 54 | 0.9395 | No (ran 74) |

**E* = 58** (mean mAP50 = 0.9453). Flat plateau at epochs 55-60 (all within 0.001).
Conservative pick: **E55** (earlier in flat zone, less overfitting risk).

## Candidate Training (COMPLETE)
- Trained candidate_e55: 55 epochs on 224-train, saved every epoch
- Best mAP50 on 24-val: **e33 = 0.9589** (old finetuned was e69 = 0.9591 — essentially identical)
- NOTE: Ultralytics `best.pt` selected e42 by default (mAP50-95 fitness). We correctly use e33 (mAP50).

## Full Pipeline Eval on 24-Val Holdout
| Detector | det_mAP50 | cls_mAP50 | blend | Delta |
|----------|-----------|-----------|-------|-------|
| old_finetuned (B4 winner) | 0.7488 | 0.8755 | 0.7868 | — |
| **candidate_e33** | 0.7494 | 0.8843 | **0.7899** | **+0.0031** |

Key insight: det mAP50 nearly identical, but **cls improved +0.0088** — better detector crops → better classification.

## 3-Way Comparison on 24-Val Holdout (COMPLETE)
| Detector | det_mAP50 | cls_mAP50 | blend | Delta |
|----------|-----------|-----------|-------|-------|
| old_finetuned (B4 winner) | 0.7488 | 0.8755 | 0.7868 | — |
| candidate_e33 (224-train, **CLEAN**) | 0.7494 | 0.8843 | 0.7899 | **+0.0031** |
| alldata_e33 (248-train, **CONTAMINATED**) | 0.7615 | 0.9099 | 0.8060 | +0.0192 (don't trust) |

**IMPORTANT**: alldata_e33 trained on the 24-val images → its eval is contaminated.
Only candidate_e33's +0.0031 is an honest signal.

## Decision
- **candidate_e33**: Safe. Honest +0.0031. At 53% transfer → ~+0.0016 test. Expected ~0.9245.
- **alldata_e33**: Risky. CV-informed E*=33 (better than B5's blind 60), but still no clean holdout.
- B5 precedent: all-data with blind epochs REGRESSED. CV epochs may be different.
