# Nightforce Astar — Autonomous Night Log

## Directive
Every 10 min: run solver, check scores, harvest GT, analyze results, sync H100.
Pattern: INCREASE → amplify with 0 risk. STALL/DECREASE → diagnose extensively.

---

## Session Start: 2026-03-21 ~21:20 UTC

- **Loop ID**: 173a6570 (10 min interval)
- **CandB live** as production v3
- **R17 pending** (z=0.402 healthy, closes ~20:49 UTC)
- **H100**: synced, idle, ready for nightforce training
- **Harvester**: running background, R9-R16 replay collection
- **Physics masking**: implemented, zero score impact (entropy-weighted nullifies)
- **Keyed calibration**: identified as #1 priority by Codex, NOT YET IMPLEMENTED

## Key Findings Pre-Session
- Model too peaked on Land(11) 67.7% and Forest(4) 29.3% of loss
- Right argmax, wrong tail probabilities
- Prior-shrinkage toward z-conditioned keyed priors = highest EV fix
- Eval/live parity gap must be fixed before trusting offline selection

---

## Round Log

| Time (UTC) | Round | Score | Weight | Weighted | z | Regime | Delta vs Prev | Action |
|------------|-------|-------|--------|----------|---|--------|---------------|--------|
| pending | R17 | ? | 2.292 | ? | 0.402 | healthy | first CandB OOS | waiting |

---

## Actions Taken

### 21:30 UTC — Loop tick
- R17 still active (not scored yet, expected close ~20:49 but delayed)
- Leaderboard: #60 at 187.1 (unchanged)
- H100 harvester running (R1-R8 replays flowing)
- Local harvesters killed (Mac stays lightweight)
- **Tested keyed prior-shrinkage**: HURTS at every λ (0.02 to 0.50, -0.09 to -2.93). Priors less accurate than NN.
- **Tested physics masking**: ±0.00 impact. Entropy-weighting nullifies static cells.
- **Conclusion**: No post-hoc correction improves the NN. Better training is the only path forward.
- **Next**: Wait for R17 score, then launch nightforce training on H100.

### 21:40 UTC — Nightforce v2 COMPLETE on H100 (50 seconds total!)

All 3 families trained. No R17 GT yet → forward validation pending.

**In-sample results (per-round breakdown):**

| Round | z | Regime | Moderate | Healthy | Robust |
|-------|-------|------|----------|---------|--------|
| R1 | 0.419 | healthy | 87.5 | 91.2 | 88.1 |
| R2 | 0.415 | healthy | 91.9 | 91.7 | 91.7 |
| R3 | 0.018 | catastrophic | 84.6 | 86.9 | 74.1 |
| R4 | 0.235 | moderate | **92.4** | 91.6 | 88.8 |
| R5 | 0.330 | moderate | **86.6** | 85.4 | 85.4 |
| R6 | 0.415 | healthy | 85.9 | **89.2** | 86.2 |
| R7 | 0.423 | healthy | 75.0 | **76.2** | 75.3 |
| R8 | 0.068 | catastrophic | 91.7 | **93.1** | 85.6 |
| R9 | 0.275 | moderate | **93.9** | 93.6 | 92.8 |
| R10 | 0.058 | catastrophic | 87.1 | **89.1** | 80.2 |
| R11 | 0.499 | healthy | 88.9 | **89.8** | 88.2 |
| R12 | 0.599 | healthy | 69.1 | **71.7** | 68.1 |
| R13 | 0.226 | moderate | **93.0** | 92.4 | 89.8 |
| R14 | 0.522 | healthy | 87.5 | **88.2** | 84.4 |
| R15 | 0.328 | moderate | 93.3 | 93.3 | **93.3** |
| R16 | 0.294 | moderate | **84.8** | 82.9 | 82.1 |

**Summary:**
- Moderate: avg 87.1 — wins on moderate rounds (R4,R5,R9,R13,R16)
- Healthy: avg 87.7 — wins on healthy + catastrophic (R6-R8,R10-R12,R14)
- Robust: avg 84.7 — never best, but stable
- **Healthy specialist is best overall** — surprisingly wins catastrophic too
- R17 forward validation BLOCKED — need R17 GT to proceed
- Regime routing confirmed valuable: best model differs by z

### 21:45 UTC — Loop tick
- R17 still active (not scored)
- H100 harvester running
- Nightforce v2 complete, awaiting R17 GT for promotion board

### 21:55 UTC — Loop tick
- R17 still active. Harvester on H100: 310 fetched, still running.
- Nightforce v2 done. All systems nominal. Waiting for R17 close.

