# NorgesGruppen Detection — Blitz

## Pre-Blitz Submissions (Mar 20-21)
| # | Date | Time (CET) | Size | Score | Time | Key Change | Learning |
|---|------|-----------|------|-------|------|------------|----------|
| 1 | Mar 20 | 01:12 | 280.1 MB | **0.7365** | 67.1s | First working submission | Baseline established — kNN classifier, early detector |
| 2 | Mar 20 | 02:25 | 280.5 MB | **FAILED** | — | ZIP validation error | run.py must be at ZIP root, no disallowed files |
| 3 | Mar 20 | 02:31 | 280.5 MB | **0.7459** | 69.1s | Fix from #2 | +0.0094 — packaging matters |
| 4 | Mar 20 | 10:45 | 271.6 MB | **0.7743** | 105.6s | Pipeline overhaul | +0.0284 — big jump, likely detector + classifier upgrades |
| 5 | Mar 20 | 15:58 | 271.6 MB | **0.9034** | 79.0s | v4.1: YOLOv8x ONNX | +0.1291 — massive jump from ONNX detector + scoring rework |
| 6 | Mar 21 | 01:10 | 304.3 MB | **0.9076** | 98.7s | v4.4: kNN + det-only score | +0.0042 |
| 7 | Mar 21 | 01:46 | 271.6 MB | **0.9104** | 97.8s | v5.0: removed kNN, additive fusion, top-K | +0.0028 |
| 8 | Mar 21 | 13:03 | 294.6 MB | **FAILED** | — | Size bug | predictions.json 18MB > 10MB limit. Burned bullet. |
| 9 | Mar 21 | 13:30 | 294.6 MB | **0.9140** | 128.3s | v5.4_cos20: MixUp cos/20 + K=15 + TTA | +0.0036 |
| 10 | Mar 21 | 18:59 | 271.5 MB | **0.9161** | 122.9s | Newrecipe E1 (strong aug + LS 0.05 + cos/60) | +0.0021 — **current best** |
| 11 | Mar 21 | 20:56 | 294.6 MB | **0.9150** | 128.3s | Detcrop E4 (50/50 GT+det-crop) | -0.0011 — regression |

## Blitz (Mar 22, midnight → 03:00 CET)
| Blitz # | Score | Checkpoint | Status |
|---------|-------|------------|--------|
| B1 | — | long_ce_e13 | **READY** — packaged, E2E verified |
| B2 | — | TBD (react to B1) | Waiting |
| B3 | — | TBD | Waiting |
| B4 | — | TBD | Waiting |
| B5 | — | TBD | Waiting |
| B6 | — | TBD | Waiting |

## Decision Tree (after B1 score)
- **B1 > 0.9161** → B2: long_ce_e14 (adjacent, higher upside)
- **B1 ≈ 0.9161** (within 0.002) → B2: long_ce_e14 (one more probe)
- **B1 < 0.914** → STOP long_ce family, pivot to different direction

## Score Progression
```
0.7365 → 0.7459 → 0.7743 → 0.9034 → 0.9076 → 0.9104 → 0.9140 → 0.9161 → 0.9150
  +94      +284     +1291      +42       +28       +36       +21      -11
```

## Blitz Plan (00:00 → 03:00 CET)

### Strategy: Single fire → wait → react. No blind pairs.

**00:00** — Fire B1 (long_ce_e13). Wait ~2 min for score.

### If B1 > 0.9161 (long_ce family is live):
- **B2**: long_ce_e14 (adjacent, higher contaminated OOF)
- **B3**: long_ce_e12 (earlier epoch, maybe cleaner generalization)
- **B4-B6**: retune around best scorer (e.g. blend best 2 epochs, or try e11/e10)

### If B1 ≈ 0.9161 (within 0.002):
- **B2**: long_ce_e14 (one more probe before deciding)
- **B3-B6**: pivot if E14 also flat — try newrecipe with more epochs or different aug strength

### If B1 < 0.914 (family is dead):
- Do NOT submit E14
- **B2-B6**: completely different direction:
  - Train fold-1 version of long_ce recipe (honest OOF, not contaminated)
  - Or train newrecipe with 2-3 epochs instead of 1
  - Package on server, eval, then submit

### Packaging (pre-staged)
- B1 ZIP ready: `/tmp/submission_bullet3_e13.zip` (295MB, E2E verified)
- For subsequent bullets: same run.py + detector.onnx, only swap classifier.safetensors
- All 15 long_ce checkpoints on server at `/root/ng/output_long_ce/`

### Constraints
- Must be done by **03:00 CET** (user asleep after that)
- 6 bullets total, use wisely — react to live feedback
- Competition ends **Mar 22, 15:00 CET**

## Key Takeaways
1. **Detector upgrade was the biggest single jump** (+0.1291, sub #5)
2. **After detector plateau, only classifier recipe changes move the board**
3. **Post-process tricks are dead** — all tested, all failed or regressed
4. **OOF→test translation ~20%** — need +0.003 OOF minimum to trust
5. **Packaging errors cost bullets** — always E2E verify before submit
