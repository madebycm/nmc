# Astar Island — Progress Report (2026-03-20 16:10 UTC)

## Competition Context

**NM i AI 2026** (Norwegian AI Championship). 69-hour competition, 1M NOK prize.
One of three tasks: **Astar Island** — predict the final state of a Norse civilization simulator.

**Deadline**: March 22 15:00 CET (2026-03-22 14:00 UTC) — ~46 hours remaining.

### The Game
- Each **round** presents a 40×40 grid map with settlements, ports, forests, mountains, ocean.
- A hidden simulation runs 50 years with hidden parameters (winter severity, food, raids, etc.).
- We observe the outcomes of simulation runs via a limited query budget (50 per round).
- We must predict the **probability distribution over 6 terrain classes** for every cell.
- Scoring: `score = 100 × exp(-3 × entropy_weighted_KL)` — higher is better, max 100.
- **Leaderboard uses ONLY your single best weighted round**: `max(round_score × 1.05^round_number)`.
- Later rounds have exponentially higher weight — R8=1.477, R9=1.551, R10=1.629.

### Leaderboard Metrics
- **`weighted_score`** (primary, determines rank): `max over all rounds (round_score × 1.05^round_number)`
- **`hot_streak_score`** (secondary): average of best 3 consecutive scored rounds

### Current Standing
| Metric | Us (J6X) | Leader (#1) |
|--------|----------|-------------|
| **Rank** | #70 of 256 | #1 |
| **weighted_score** | 97.16 | 118.63 |
| **hot_streak_score** | 57.54 | 84.83 |
| **rounds_participated** | 5 | 5 |
| **Best round** | R4: 79.9 × 1.216 | Likely R7: ~84.3 × 1.407 |

### Gap Analysis

**Gap to #1: 21.4 weighted points.**

The leader likely scored ~84.3 on R7 (118.63 / 1.407 = 84.3) or ~80.3 on R8 if they participated. Their hot streak of 84.8 means they averaged 84.8 across their best 3 consecutive rounds — extremely consistent.

**What we need to reach #1:**

| Round | Weight | Score Needed | Our Realistic Estimate | Feasible? |
|-------|--------|-------------|----------------------|-----------|
| R8 | 1.477 | 80.3 | ~76 (catastrophic z=0.051, pure Dirichlet) | Unlikely |
| **R9** | **1.551** | **76.5** | **78-82 (healthy round with fixed NN)** | **YES** |
| R10 | 1.629 | 72.8 | 78-82 | YES |

**Our R7 holdout backtest with fixed channels + v2b + observations: 74.5.** With v3 retrained (+3-5 pts estimated) and continued calibration improvements, **78-82 on a healthy R9 is achievable**.

**If R9 is catastrophic (z<0.08):** Pure Dirichlet scores ~76 → weighted 117.9. Close but not enough. Need R10.

### Our Hot Streak (Damage Assessment)

| Rounds | Scores | Avg |
|--------|--------|-----|
| R5-R7 (current best 3) | 75.4, 58.8, 38.4 | **57.5** |
| R5-R7 (if channels were fixed) | ~82, ~80, ~74.5 | **~78.8** |
| R4-R6 | 79.9, 75.4, 58.8 | 71.4 |

The channel bug cost us ~21 points on hot streak. Our actual pipeline capability was masked by garbage NN predictions.

---

## Architecture Overview

### Prediction Pipeline: 4-Model Ensemble

```
Observations (50 queries) → Context Vector (8-dim) → Four models → Weighted blend → Submit
```

1. **Dirichlet-Bayesian Model** — z-conditioned prior with 27 terrain-class keys
   - Backbone model. Reliable. LORO ~70 avg.
   - Updated via Bayesian updates from per-seed observations.

2. **NN v2b** (1.8M params, ResNet with 6 dilated blocks, 128 hidden)
   - Trained on 35 GT files (R1-R7) on A100 GPU.
   - LORO holdout avg: 74.1 (best: R1=82.5, worst: R3=63.7).
   - Takes 13-channel input: 8 terrain one-hot + distance + coastal + is_land + z + density.

3. **NN v3** (5.7M params, ResNet with 8 blocks + multi-scale, 192 hidden)
   - Currently trained on OLD data (R1-R5/R6 only). **Retraining on A100 NOW.**
   - Same 13-channel input as v2.

4. **NN v4** (16.8M params, Conditional U-Net)
   - 20-channel input: 12 spatial + 8-dim context vector broadcast.
   - Gated to moderate rounds only (z=0.10-0.38) where it outperforms v2.
   - Known issue: train/inference context mismatch (TODO: retrain).

### Blending Strategy

```python
# z-adaptive NN weight (Dirichlet is always the safe backbone)
z < 0.08  → 0% NN   (pure Dirichlet for catastrophic rounds)
z 0.08-0.15 → 15% NN
z 0.15-0.30 → 35% NN
z > 0.30  → 45% NN   (NN dominates on healthy rounds)

# Geometric mean blend: exp(w_nn * log(nn) + w_dir * log(dir))
# Then empirical anchoring: 7% blend with observation counts (n_obs >= 2 only)
# Floor at 0.01, renormalize
```

### Global Context Vector (8 dimensions)

Replaces scalar z parameter. Computed from grid comparison (initial vs observed cells) across all 5 seeds.

| Dim | Feature | Meaning |
|-----|---------|---------|
| 0 | Survival rate | Initially settled → still settled/port |
| 1 | Port survival | Initially port → still port |
| 2 | Ruin frequency | Ruin cells / land cells |
| 3 | Forest fraction | Forest cells / land cells |
| 4 | Collapse rate | 1 - survival rate |
| 5 | Expansion rate | Empty → settlement/port |
| 6 | Entropy proxy | Faction diversity + food level |
| 7 | z (backward compat) | = survival rate |

**Critical**: Context computed from base 45 tiling queries only — precision queries excluded to avoid biasing toward high-density areas.

---

## Scored Rounds

| Round | Score | Weight | Weighted | Strategy | z | Notes |
|-------|-------|--------|----------|----------|---|-------|
| R3 | 7.2 | 1.158 | 8.3 | proximity_v1 | 0.018 | Overwrite disaster (server poller) |
| R4 | **79.9** | 1.216 | **97.2** | dirichlet_v3 | 0.235 | **Current best weighted** |
| R5 | 75.4 | 1.276 | 96.2 | dirichlet_v4 | 0.330 | |
| R6 | 58.8 | 1.340 | 78.8 | dirichlet_v4 | 0.415 | |
| R7 | 38.4 | 1.407 | 54.0 | ensemble_ctx_v2 | 0.423 | NN disaster (broken channels) |
| R8 | pending | 1.477 | ? | pure_dirichlet | 0.051 | Catastrophic round, closes 17:46 UTC |

**Cannot resubmit past rounds** — API returns 400 on completed rounds. Only active round (R8) allows submission.

---

## Critical Bug Discovery: Channel Order Mismatch

### The Bug (found 2026-03-20 ~16:00 UTC)

The `nn_predict.py` inference code had **channels 11-12 SWAPPED** compared to the training scripts on the A100 server.

**Training script** (`train_nn.py` on A100):
```python
channels = [terrain(8), distance(1), coastal(1), is_land(1), z(1), density(1)]  # 13 total
#                                                               ^ch11   ^ch12
```

**Inference script** (`nn_predict.py`, BEFORE fix):
```python
spatial = [terrain(8), distance(1), coastal(1), is_land(1), density(1)]  # 12 ch
features = concatenate([spatial, z_channel])  # z appended LAST = ch12
#                                                density=ch11, z=ch12  ← WRONG!
```

The model was trained seeing z in channel 11 and density in channel 12, but at inference time it received density where it expected z and vice versa.

### Impact

**ALL NN predictions since the beginning of the competition were garbage at inference time.**

The LORO holdout scores (computed during training with correct channel order) showed v2 scoring 77.6 avg. But at inference time, the swapped channels meant the model produced near-random predictions.

| Model | Before Fix (R7 s0) | After Fix (R7 s0) | Improvement |
|-------|--------------------|--------------------|-------------|
| v2 | 13.2 | 68.7 | **+55.5** |
| v3 | 10.8 | 63.2 | **+52.4** |
| Full ensemble (R7) | 53.3 avg | 74.5 avg | **+21.2** |

The ensemble pre-fix scored only 53.3 because the NN was actively *hurting* the Dirichlet backbone with its garbage predictions. Post-fix, the NN genuinely helps.

### The Fix

Rewrote `encode_grid_v2v3()` to inline all channels in the exact training order instead of splitting into spatial + z append. Similarly fixed `encode_grid_v4()` to put density before context (matching v4 training order).

### Implications for Past Submissions

- **R5 (75.4)**: Was submitted with NN blend. The broken NN was dragging down Dirichlet. Pure Dirichlet would have scored ~79+.
- **R6 (58.8)**: Same issue. Should have been 70-80+ with proper NN or pure Dirichlet.
- **R7 (38.4)**: NN scored 13 with broken channels. At 70% NN weight, it dragged Dirichlet from ~65 → 38.
- **R8**: Submitted as pure Dirichlet (z=0.051 catastrophic), so unaffected by NN bug.
- **Future rounds (R9+)**: Will benefit from the fix. Estimated ensemble score: 78-82 on healthy rounds.

---

## Model Performance (LORO = Leave-One-Round-Out Cross-Validation)

### v2b (retrained with 35 GT files, R1-R7)

| Round | z | LORO Holdout | Notes |
|-------|---|-------------|-------|
| R1 | 0.419 | 82.5 | Healthy |
| R2 | 0.415 | 77.6 | Healthy |
| R3 | 0.018 | 63.7 | Catastrophic |
| R4 | 0.235 | 75.3 | Moderate |
| R5 | 0.330 | 78.6 | Moderate-healthy |
| R6 | 0.415 | 75.2 | Healthy |
| R7 | 0.423 | 66.0 | Healthy (new map layout) |
| **Avg** | | **74.1** | |

### Backtest with Fixed Channels (v2b, no observations)

| Round | z | NN Weight | Ensemble Score | Pure Dirichlet | Submitted |
|-------|---|-----------|----------------|----------------|-----------|
| R1 | 0.419 | 45% | 89.7* | 83.8 | — |
| R2 | 0.415 | 45% | 90.6* | 85.8 | — |
| R3 | 0.018 | 0% | 75.8 | 75.8 | 7.2 |
| R4 | 0.235 | 35% | 91.1* | 88.9 | 79.9 |
| R5 | 0.330 | 45% | 88.5* | 79.3 | 75.4 |
| R6 | 0.415 | 45% | 86.6* | 74.6 | 58.8 |
| R7 | 0.423 | 45% | 76.6* | 64.8 | 38.4 |

*\* Training fit — overestimates unseen round performance. Only R7 with observations (74.5) is a fair holdout test.*

---

## Infrastructure

### Local (Mac)
- Solver loop: `/loop 3m cd ~/www/nm/astar && python solver.py` (job 01721f45)
- State tracking: `state.json` prevents double-execution
- All submissions run from local only (server execution = R3 disaster)

### A100 VPS (XXx--xx-A100)
- NN training with CUDA
- v2b training complete (35 GT files, 1000 epochs, LORO 74.1)
- **v3b retraining in progress** — started ~16:10 UTC
- Workspace: `/astar/` with venv

### H100 Server (XXx--xx-H100) — for NorgesGruppen task, not used for Astar

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `solver.py` | Main loop: harvest GT → observe → predict → submit | Active, auto-running |
| `strategy.py` | Context vector, Dirichlet, ensemble blend, anchoring | Updated with z-adaptive weights |
| `nn_predict.py` | Multi-model NN inference (v2+v3+v4) with TTA | **FIXED channel order** |
| `calibrate.py` | z-conditioned Dirichlet calibration from GT | 7 rounds calibrated |
| `api.py` | HTTP client for all endpoints | Stable |
| `config.py` | Auth token, QUERIES_PER_SEED=10 | Stable |
| `state.json` | Persistent state, prevents double-execution | R1-R8 tracked |
| `astar_nn.pt` | v2b weights (35 GT files, channel-fixed) | **NEW — active** |
| `astar_nn_v3.pt` | v3 weights (OLD, needs retrain) | **Retraining on A100** |
| `astar_nn_v4.pt` | v4 Conditional U-Net weights | Active (z-gated) |
| `calibration.json` | Dirichlet priors + z-model (R1-R7) | Current |

---

## Bugs Found and Fixed (Chronological)

1. **R3 overwrite disaster** — Server poller overwrote good predictions with 1-observation predictions. Fixed: never run from server.
2. **Seed misattribution** — Flat observation list broke with precision queries. Fixed: pass `{seed_idx: [obs]}` dict.
3. **Double-counting observations** — Dirichlet + empirical anchoring applied same observations twice. Fixed: 7% weight, n_obs >= 2.
4. **Precision query context bias** — Extra queries on dense areas skewed global context. Fixed: base 45 only.
5. **v4 weight too high** — Reduced from 0.5 to 0.4 after LORO showed underperformance.
6. **R7 NN disaster** — NN at 70% weight dragged score from 65 → 38. Fixed: reduced to 25%.
7. **CHANNEL ORDER BUG** (critical, found today) — Channels 11-12 swapped between training and inference. ALL NN predictions were garbage. Fixed by rewriting encode functions to match training order exactly. **Single biggest improvement: +21 points on R7.**
8. **Empty observations context bug** — `compute_context_vector` called with empty observation lists produced garbage context. Fixed: check for actual observations before computing.

---

## What Needs to Happen Next (Priority Order)

### Immediate (next 2 hours)
1. **Harvest R8 GT** at 17:46 UTC → recalibrate Dirichlet (8 rounds)
2. **v3 retraining completes** on A100 → download → integrate
3. **R9 auto-solves** with fixed pipeline when it opens

### Short-term (next 6 hours)
4. **Retrain v4** with fixed context semantics (train/inference mismatch still exists)
5. **Optimize z-adaptive weights** via proper LORO backtest
6. **Consider v2c retraining** after R8 GT (40 samples)

### Strategy to Exceed #1 (118.63)

**Core thesis**: The channel fix transforms our NN from garbage (13 on R7) to genuinely useful (68.7). The ensemble with Dirichlet goes from 53 → 74.5 on holdout. With v3 retrained and more GT data, **78-82 on healthy rounds is achievable**.

**Path to #1 by scenario:**

| Scenario | R9 Score | R9 Weighted | Rank |
|----------|----------|-------------|------|
| Catastrophic (z<0.08) | ~76 (Dirichlet only) | 117.9 | #2-3 |
| Moderate (z 0.15-0.30) | ~80 (35% NN) | 124.1 | **#1** |
| Healthy (z>0.30) | ~82 (45% NN) | 127.2 | **#1** |
| Healthy + retrained v3 | ~85 | 131.8 | **#1 by large margin** |

**Improvement vectors (each adds 1-5 points):**

1. **Channel fix** (DONE): +21 points on R7. Base improvement for all future rounds.
2. **v3 retrain** (IN PROGRESS on A100): v3 currently scores 10-11 on R7 (garbage, old data). After retrain with 35 files, expected 60+. Ensemble improvement: +3-5 pts.
3. **v2c/v3c retrain after R8 GT**: 40 samples instead of 35. More catastrophic data (R3+R8). +1-2 pts generalization.
4. **v4 context fix**: Training uses P(class) probabilities, inference uses binary survival. Systematic shift ~0.06. Fixing this could add +2-3 pts on moderate rounds.
5. **Optimized z-adaptive weights**: Current weights (0/15/35/45%) are educated guesses. Proper LORO grid search could find optimal weights. +1-2 pts.
6. **More observations per round**: Using all 50 queries optimally (9 tiling + precision targeting). Better observation coverage = better empirical anchoring. +1 pt.
7. **Increased NN weight on healthy rounds**: With fixed channels + retrained v3, we might safely push to 55-60% NN on healthy rounds. +2-3 pts.

**Total potential**: 74.5 (current holdout) + 10-16 = **84-90 on healthy rounds**.

**Risk factors:**
- Catastrophic round → pure Dirichlet ceiling ~76 → weighted 117.9 → #2-3 (still massive improvement from #70)
- NN generalization to truly novel map layouts (R7 pattern) — mitigated by more training data
- Competition is tight at the top (top 5 all above 115)
- Only ~12 rounds remaining (~24 hours of rounds at ~2h each)

**Key decision points:**
- When R8 scores: validate our Dirichlet prediction on catastrophic round
- When v3 retraining completes: download, integrate, backtest before R9
- After each round: harvest GT immediately, recalibrate, retrain if time allows
- If we score 80+ on any round: that's likely #1. Don't risk resubmitting.
- If we score 75-80: consider resubmitting with tweaked weights during the round window

---

## Lessons Learned

1. **Always verify feature encoding matches between training and inference.** The channel swap bug cost us at least 20 points on every NN-blended submission. This single bug is the difference between #70 and potentially #1.
2. **Dirichlet is the reliable backbone.** On catastrophic rounds (z<0.08), it beats all NNs. Never trust NN alone.
3. **Observations are end-state, not mid-sim.** Each `/simulate` is a fresh 50-year run. Pool statistics across all queries.
4. **Resubmission is critical.** Unlimited resubmissions during active rounds. But cannot resubmit completed rounds (400 error).
5. **Later rounds matter exponentially more.** Scoring 80 on R10 (weight 1.629) = 130 weighted. Don't waste effort on old rounds.
6. **Test inference independently from training.** The LORO scores (77.6 avg) looked great, but the actual inference pipeline produced 13-point garbage. Always validate end-to-end.
7. **The leaderboard is max(single round × weight).** Consistency doesn't matter for rank — one great round wins. Hot streak is secondary.
