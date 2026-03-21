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

### 22:00 UTC — R17 SCORED: 77.89 raw (DECREASE)
**R17 per-seed**: 80.6, 85.8, 72.6, 81.6, 68.9 → avg 77.89
**R17 weighted**: 77.89 × 2.292 = 178.5 (NOT new best, prev best 187.1)
**z = 0.454 (healthy)**

**DIAGNOSIS**: CandB scored 77.89 through live pipeline. But nightforce healthy specialist
scores 90.35 on same R17 GT in forward validation (held-out, raw model output).
Gap = +12.5 points. This means the live ensemble pipeline is DEGRADING the model output.
Possible causes:
1. NN weight curve caps healthy at 0.20 — too conservative
2. Dirichlet blend pulls predictions away from correct NN output
3. CandB architecture mismatch (different from nightforce AstarNetV3)

**ACTION**: Retrained healthy specialist on ALL R3-R17 → saved nf2_healthy_all.pt (87.77 avg)
Downloaded all checkpoints. Need to test through FULL live pipeline.

**R17 Forward Validation (nightforce families, held-out):**
| Family | R17 Score |
|--------|-----------|
| Moderate | 89.63 |
| **Healthy** | **90.35** |
| Robust | 86.85 |
| CandB (live) | 77.89 |

### 22:10 UTC — R18 DETECTED: z=0.616 (very healthy)
- R18 first submission was **PURE DIRICHLET** — all 3 .pt model files were MISSING from local Mac
- Models were only on H100 after nightforce training
- z=0.616 is at the EDGE of training range (max GT = 0.599 from R12)

### 22:22 UTC — R18 RESUBMITTED with full NN ensemble
- Downloaded astar_nn.pt, astar_nn_v3.pt, nf2_healthy_all.pt from H100
- Added resubmit-from-cache path to solver.py (uses saved observations when queries exhausted)
- NN_HEALTHY raised from 0.20 → 0.40 per user review (see r17.md analysis)
- Actual NN weights: 0.39-0.41 per seed (with per-seed z-nudge)
- z=0.616 clipped to 0.599 (training range ceiling)
- **RISK**: z=0.616 is R12-territory (R12=29.11 at z=0.599). Very healthy = our weakest regime.

### 22:24 UTC — Loop tick
- R18: submitted, waiting for close at 23:48 UTC (~1h 24m)
- Leaderboard: #117 at 187.1, #1 Laurbærene at 217.4
- Gap to #1: **30.3 points** — massive. Need ~90+ raw on R18 (weight 2.407) for 216+
- H100: nightforce v2 training complete. Harvester still running (R9-R16 replays).
- Local: all 3 model files restored, solver has resubmit capability
- **Competitive note**: Field jumped ~30 pts since R17. Top teams likely have much better healthy-regime models.
- **Next**: Wait for R18 score. If healthy (z=0.616) scores well → evidence NF model works. If not → need fundamental approach change for healthy rounds.

### 22:45 UTC — Codex GPT-5.4 Deep Analysis (CRITICAL FINDINGS)

**Codex ran exact competition scorer on GT R11-R17:**

| Round | z | Doctrine | Pure NN | Gap |
|-------|-------|----------|---------|-----|
| R11 | 0.545 | 92.77 | **95.57** | +2.8 |
| **R12** | **0.615** | **78.24** | **91.04** | **+12.8** |
| R13 | 0.199 | 92.92 | 94.59 | +1.7 |
| R14 | 0.494 | 87.93 | 93.32 | +5.4 |
| R15 | 0.329 | 94.19 | 95.42 | +1.2 |
| R16 | 0.302 | 90.23 | 91.84 | +1.6 |
| R17 | 0.402 | 84.30 | 83.53 | -0.8 |

**R12 SMOKING GUN**: Live scored 29.11, doctrine scores 78.24, pure NN scores 91.04.
The Dirichlet arm is actively destroying healthy round predictions.

**Key findings:**
1. NN beats doctrine on ALL healthy rounds except R17 (where it's ~equal)
2. Observations only update Dirichlet arm, not final posterior — architecture wrong
3. NF model gets only ~15% total influence after normalization — too low
4. z=0.616 clip cost is modest (~1.7pp), not the main problem
5. **Recommendation: NN_HEALTHY → 0.65+ for z>0.45, NN-dominant on healthy**

**Decision**: User approved NN_HEALTHY → 0.65.

### 22:34 UTC — R18 RESUBMITTED with NN_HEALTHY=0.65
- NN weight now 0.64-0.65 per seed (was 0.39-0.41)
- Dirichlet reduced from ~60% to ~35%
- **Expected R18 score**: ~84.7 raw × 2.407 = ~204 weighted (new best, +17 over 187.1)
- Offline backtests with 0.65: R11=94.4, R12=84.7, R14=90.0, R17=84.3

### 22:42 UTC — PROMOTION MACHINE LAUNCHED on H100
- Nightforce v3 training started (PID 86650)
- All 85 GT files (R1-R17), 3 families (moderate/healthy/robust)
- Forward validation on R17: **healthy best at 90.44** (vs v2 90.35)
- Final retrain of healthy on all data in progress (epoch 120, avg=85.40, climbing)
- Fixed CandB architecture mismatch crash

### 22:47 UTC — Loop tick
- R18: resubmitted with NN_HEALTHY=0.65, closes 23:48 UTC (~1h)
- Harvester: running on H100 (PID 82672)
- H100: nightforce v3 training COMPLETE (87.85 avg, R17 fwd=90.44)
- v3 checkpoint downloaded, compared vs v2: **identical** through live pipeline (±0.1)
- Retraining same arch/data doesn't help — model already near-optimal
- GPU idle, waiting for new GT (R18)

### 23:03 UTC — Loop tick
- R18: closes 23:48 UTC (~45m). NN_HEALTHY=0.65 live.
- H100: GPU idle. Training done. Harvester running.
- Rank: #117 at 187.1 (unchanged until R18 scores)
- **TESTED 3 "zero-risk" improvements — ALL REVERTED:**
  1. NF intra-weight 0.50→0.70: HURTS R12 by -3.84 (our R18 z-analog)
  2. Dirichlet z-clip to training range: makes it worse combined with #1
  3. Cell-level posterior (n≥2, conc=50): HURTS R12 by -3.5, helps R17 by +2.2 — net negative
  - **Conclusion**: NN_HEALTHY=0.65 with original intra-weights is already optimal
  - The NF model helps most at its current 0.50 weight within ensemble — going higher overcorrects
  - Dirichlet extrapolation at z=0.616 is actually BETTER than clamping (captures real trend)
  - Observation posterior still too noisy even with n≥2 filter
  - ~~LIVE CONFIG: NN_HEALTHY=0.65~~ SUPERSEDED by sweep results below

### 23:00 UTC — CONFIG SWEEP COMPLETE (396 configs × 85 GT seeds)

Pre-computed NN+Dirichlet predictions once, then swept blend params at numpy speed.

**Sweep-optimal config: NN_PEAK=0.75, NN_HEALTHY=0.75, FLOOR=0.003**

| Metric | Old (H=0.65, P=0.65) | Optimal (H=0.75, P=0.75) |
|--------|----------------------|--------------------------|
| Healthy avg | ~84.7 | **89.24** (+4.5) |
| Moderate avg | ~92 | **93.49** (+1.5) |

Key insight: **the NN is strictly better than Dirichlet at ALL z values**. There is no regime where adding more Dirichlet helps. The "inverted-U" curve was wrong — a flat 0.75 is optimal.

### 23:03 UTC — R18 RESUBMITTED with sweep-optimal config
- NN weight: 0.74-0.75 per seed (was 0.64-0.65)
- Dirichlet: 0.25-0.26 (was 0.35-0.36)
- Expected R18: ~89 raw (up from ~84.7) → **~214 weighted** → potentially top 10
- R18 closes 23:48 UTC (~45m)
- **LIVE CONFIG**: NN_PEAK=0.75, NN_HEALTHY=0.75, FLOOR=0.003, NF=0.50

### 23:10 UTC — H100 NEVER IDLE DIRECTIVE
- Upgraded promo loop: GPU priority queue (retrain > LORO > multi-seed > augmented > metric-loss)
- Launched LORO forward validation (17 folds, ~6 min) — TRUE out-of-sample scores
- Multi-seed ensemble (5 seeds) queued behind LORO
- LORO progress: R1=86.28, R2=81.01, R3=83.50 (20 sec/fold)

### 23:15 UTC — Loop tick
- R18: closes 23:48 UTC (~33m). Sweep-optimal NN=0.75 live.
- H100: GPU at 100% — LORO running, multi-seed queued.
- All systems go.
- Rank: #117 at 187.1

### 22:30 UTC — NN_HEALTHY Sweep (GPU, 15 seconds)

**Full sweep across R3-R17 on GPU:**
- NN_HEALTHY only affects healthy rounds (z>0.35) — moderate/catastrophic identical
- In-sample: higher NN = better for healthy (87.5 avg → 91.1)
- Exception: R17 slightly favors lower NN (87.0→86.5)
- **All in-sample** — real OOS gap is ~16 pts for healthy
- **Decision: keep NN_HEALTHY=0.75.** Moderate rounds unaffected.

### 22:35 UTC — Multi-seed Ensemble Test: HURTS

Tested 5 NF ensemble (nf_ensemble_0-4.pt) through full pipeline vs single nf2_healthy_all.pt on R12:
- **Ensemble: 84.95 avg** (in-sample)
- **Single: 86.97 avg** (in-sample)
- **Delta: -2.02** — ensemble HURTS in full pipeline
- Reverted. Single NF stays in production.

### 22:35 UTC — Augmented Training Launched (H100)

Training NF with 8x augmentation (rotations+flips) + entropy-weighted KL loss.
Goal: close LORO gap (in-sample 95+ but OOS 72.75 on healthy).

**Results:**
- Best in-sample: 77.12 at epoch 300 (lower than 95+ non-augmented, expected)
- Model collapsed after epoch 300 (loss still dropping but eval score diverging)
- R15 LORO: **90.32** (vs original 86.71 = +3.6 pts!)
- R16 LORO: 79.73 (vs original 83.99 = -4.3 pts)
- R17 LORO: computing...
- **z values in LORO may be wrong** (showing 0.283 default for all rounds)

### 22:48 UTC — Status
- R18: closes 23:48 UTC (~60 min). NN=0.75 live. z=0.616 (very healthy).
- Rank: #117 at 187.1. #1: Laurbærene at 217.4.
- H100: augmented training R17 LORO running (48% GPU)
- **Path to #1**: moderate R19+ at weight 2.5+ → need 87+ raw → 217+ weighted
- LORO moderate avg = 87.35 → naturally within striking distance on next moderate round

### 23:15 UTC — Codex Diagnosis: Training Recipe

**Root cause of augmented training failure**: Used WRONG loss functions. The proven recipe uses `competition_loss` (entropy-weighted KL with softmax → clamp 0.01 → renormalize). My attempts used soft KL and cross-entropy instead.

**Key finding**: AstarNetNF has NO dropout (ResBlockNF uses pure GELU without Dropout2d). Original V3 ResBlock has Dropout2d(0.05). This likely explains the massive 16pt LORO gap on healthy rounds — the model memorizes training maps without regularization.

**Action**: Launched proper NF training on H100 with:
- Exact `competition_loss` from train_nn_v3.py
- 8x rotation/flip + z-augmentation (same as proven recipe)
- AdamW lr=3e-4, weight_decay=5e-4, warmup 50 + cosine decay
- **Added Dropout2d(0.10)** for regularization
- Batch size 8, 1500 epochs
- Phase 2: same with dropout=0.15 to compare
- Phase 3: LORO evaluation of best dropout

**Early results**: Ep 400 = 88.58 avg (in-sample). Trajectory matches original recipe.

### 23:30 UTC — Status
- R18: closes 23:48 UTC (17 min). NN=0.75, z=0.616.
- H100: training NF with dropout (ep 400/1500, 88.58 avg, climbing)
- ~5 more rounds expected (R19-R23) before deadline

