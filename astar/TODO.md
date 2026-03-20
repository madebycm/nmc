# Astar Island — TODO (2026-03-20 15:00 UTC → 2026-03-21 03:00 UTC)

> **Deadline**: March 22 15:00 CET (2026-03-22 14:00 UTC)
> **Position**: ~#55 (97.2) — need 85+ on R8+ to win
> **Leader**: #1 at 118.6

---

## IMMEDIATE (next 30 min)

- [x] Wait for R7 to complete and score
- [x] Harvest R7 GT (5 new files → 35 total)
- [x] Check R7 score — **DISASTER: 38.4 avg** (NN scored 13, Dirichlet alone: 65)
- [x] Auto-recalibrate Dirichlet with R7 data
- [x] Upload R7 GT to A100, start v2 retraining (35 files)
- [x] Reduce NN weight to 25% (Dirichlet-dominant safety)

**CRITICAL LESSON**: NN trained on R1-R6 does NOT generalize to R7 maps.
v2 scores 12-18 on R7 regardless of z. Pure Dirichlet would have scored 65.
The geometric blend at 70% NN weight dragged score from 65 → 38.

## HIGH PRIORITY (next 2 hours)

- [ ] **Retrain v3 on 35 GT files** — v3 still trained on R1-R5/R6, hurts ensemble on unseen rounds
  - Upload train_nn_v3.py to A100, run with 35 GT files
  - v3 LORO on R7 is ~10 (garbage). After retrain should be 60+
- [ ] **Retrain v4 with FIXED context** (Codex issue #2 still open)
  - Training ctx[0] = P(class 1) probability, inference = binary survival
  - Also need to verify v4 training channel order matches new encode_grid_v4
- [ ] **Harvest R8 GT** when R8 closes at 17:46 UTC
  - Recalibrate Dirichlet with R8 data (8 rounds total)
  - R8 z=0.051 provides second catastrophic data point (R3 z=0.018 was the only one)
- [ ] **Monitor R9 open** — auto-solve with improved pipeline
  - R9 weight = 1.551. Score 80 → 124.1 weighted → #1
  - Now with FIXED channel order, NN actually contributes!

## MEDIUM PRIORITY (next 6 hours)

- [ ] **Cannot resubmit past rounds** — API returns 400 on completed rounds
- [ ] **Optimize NN weights via LORO backtest**
  - Current z-adaptive: 0%/15%/35%/45% — calibrate with full pipeline backtest
  - Consider different weights for v2 vs v3 (v3 may need different treatment)
- [ ] **Train v2c on A100** — after R8 GT, 40 samples. More data = better generalization
- [ ] **FiLM conditioning** for v4 instead of broadcast context
- [ ] **Per-round fine-tuning** on observations (risky, observations are single samples)

## MONITORING (continuous)

- [x] `/loop 3m cd ~/www/nm/astar && python solver.py` — RUNNING (job 01721f45)
- [ ] Check leaderboard after each round scores
- [ ] Harvest GT immediately when rounds complete

## DONE THIS SESSION

- [x] **CRITICAL: Fixed channel order bug in nn_predict.py** — channels 11-12 were SWAPPED
  - Training: ...z(ch11), density(ch12). Inference had: density(ch11), z(ch12)
  - ALL previous NN predictions were garbage. v2 R7: 13→68.7, ensemble R7: 53→74.5
  - Also fixed v4 channel order (density before context)
- [x] Retrained v2b on 35 GT files (R1-R7) — LORO avg 74.1
- [x] Downloaded v2b model, swapped as primary `astar_nn.pt`
- [x] Resubmitted R8 with pure Dirichlet (z=0.051 catastrophic)
- [x] Implemented z-adaptive NN weighting (0%/15%/35%/45%)
- [x] Fixed empty-observations context bug in ensemble_predict
- [x] Implemented Global Context Vector (8-dim, grid-based)
- [x] Integrated v4 Conditional U-Net into ensemble
- [x] Added empirical anchoring (7% base, n_obs ≥ 2)
- [x] Added precision query allocation (remaining budget → high-density viewports)
- [x] Fixed seed misattribution, double-counting, precision query bias (Codex review)
- [x] Updated CLAUDE.md, TODO.md

## KEY NUMBERS

| Model | LORO avg | Best round | Worst round | Notes |
|-------|----------|------------|-------------|-------|
| v2b (35 files) | 74.1 | R1: 82.5 | R3: 63.7 | Channel-fixed, active |
| v3 (old) | ~60 est | ? | R7: 10.8 | Needs retrain |
| v4 | 71.9 | R2: 84.2 | R3: 35.7 | Context mismatch |
| Dirichlet | ~70 | R2: 82.9 | R3: 38.0 | Reliable backbone |
| **Ensemble (fixed)** | **~78-82 est** | — | — | **Channel fix = game changer** |

**Position**: #70 (97.2). Leader: #1 (118.6). Gap: 21.4 pts.
**To win**: score 80+ on R9 (weight 1.551) → 124.1 weighted → #1
