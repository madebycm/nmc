# NorgesGruppen Detection — Full Submission History

| # | Date | Time (CET) | Size | Score | Time | Key Change | Learning |
|---|------|-----------|------|-------|------|------------|----------|
| 1 | Mar 20 | 01:12 | 280.1 MB | **0.7365** | 67.1s | First working submission | Baseline established — kNN classifier, early detector |
| 2 | Mar 20 | 02:25 | 280.5 MB | **FAILED** | — | ZIP validation error | run.py must be at ZIP root, no disallowed files |
| 3 | Mar 20 | 02:31 | 280.5 MB | **0.7459** | 69.1s | Fix from #2 | +0.0094 — packaging matters |
| 4 | Mar 20 | 10:45 | 271.6 MB | **0.7743** | 105.6s | Pipeline overhaul | +0.0284 — big jump, likely detector + classifier upgrades |
| 5 | Mar 20 | 15:58 | 271.6 MB | **0.9034** | 79.0s | v4.1: M=200, YOLOv8x ONNX | +0.1291 — massive jump from ONNX detector + scoring rework |
| 6 | Mar 21 | 01:10 | 304.3 MB | **0.9076** | 98.7s | v4.4: kNN + det-only score | +0.0042 |
| 7 | Mar 21 | 01:46 | 271.6 MB | **0.9104** | 97.8s | v5.0: removed kNN, additive fusion, top-K | +0.0028 |
| 8 | Mar 21 | 13:03 | 294.6 MB | **FAILED** | — | Size bug | predictions.json 18MB > 10MB limit. Burned bullet. |
| 9 | Mar 21 | 13:30 | 294.6 MB | **0.9140** | 128.3s | v5.4_cos20: MixUp cos/20 + K=15 + TTA | +0.0036 — classifier recipe works |
| 10 | Mar 21 | 18:59 | 271.5 MB | **0.9161** | 122.9s | Blitz 0 (pre-blitz): newrecipe E1 (strong aug + LS 0.05 + cos/60) | +0.0021 — **current best** |
| 11 | Mar 21 | 20:56 | 294.6 MB | **0.9150** | 128.3s | Blitz 0 (pre-blitz): detcrop E4 (50/50 GT+det-crop) | -0.0011 — regression, small OOF wins don't transfer |

## Summary
- **11 submissions** (2 failed, 9 scored)
- **Best: 0.9161** (submission #10, newrecipe E1)
- **Leader: 0.9255** (gap: 0.0094)
- **6 bullets remaining** (post-midnight)
- **Next: long_ce_e13** (Blitz 1, packaged and E2E verified, ready to fire at midnight)

## Score Progression (scored only)
```
0.7365 → 0.7459 → 0.7743 → 0.9034 → 0.9076 → 0.9104 → 0.9140 → 0.9161 → 0.9150
  +94      +284     +1291      +42       +28       +36       +21      -11
```

## Key Takeaways
1. **Detector upgrade was the biggest single jump** (+0.1291, submission #5)
2. **After detector plateau, only classifier recipe changes move the board** (#6→#10)
3. **Post-process tricks are dead** — all tested, all failed or regressed
4. **OOF→test translation ~20%** — need +0.003 OOF minimum to trust
5. **Packaging errors cost bullets** — always E2E verify before submit
