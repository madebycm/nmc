# Astar Island — Full Status Report
**Team J6X** | **2026-03-22 02:30 UTC** | Competition ends 14:00 UTC (~11.5h remaining)

---

## Leaderboard Position
- **Rank**: #131 / 380 teams
- **Score**: 187.1 (R15: 90.0 raw × 2.079 weight)
- **#1**: WinterIsComing_ at 219.0
- **Gap to #1**: 31.9 pts
- **Our ceiling on remaining rounds**: R22 moderate → 89 × 2.925 = **260.3** (beats everyone)

## Leaderboard Formula
```
leaderboard_score = max over all rounds (round_score × 1.05^round_number)
round_score = avg over 5 seeds (100 × exp(-3 × entropy_weighted_kl))
```
Only your BEST single weighted round counts. Earlier disasters (R3=7.2, R7=38.4) are irrelevant. Later rounds have exponentially higher multipliers — R22 weight is 2.53× R3 weight.

---

## Complete Round History

| Round | Raw Score | Weight | Weighted | z | Regime | Strategy | Notes |
|-------|-----------|--------|----------|------|--------|----------|-------|
| R3 | 7.17 | 1.158 | 8.3 | 0.018 | catastrophic | proximity_v1 | Overwritten by poller — disaster |
| R4 | 79.94 | 1.216 | 97.2 | 0.235 | moderate | dirichlet_v3 | Regime-aware priors |
| R5 | 75.40 | 1.276 | 96.2 | 0.330 | moderate | dirichlet_v4 | |
| R6 | 58.83 | 1.340 | 78.8 | 0.415 | healthy | dirichlet_v4 | Healthy rounds are our weakness |
| R7 | 38.39 | 1.407 | 54.0 | 0.423 | healthy | ensemble_nn_v1 | **Channel order bug** — NN garbage |
| R8 | 84.42 | 1.477 | 124.7 | 0.068 | catastrophic | ensemble_ctx_v2 | Pure Dirichlet (z<0.05) |
| R9 | 89.12 | 1.551 | 138.3 | 0.275 | moderate | champion_challenger | Best raw score at time |
| R10 | 86.25 | 1.629 | 140.5 | 0.058 | catastrophic | champion_challenger | |
| R11 | 79.94 | 1.710 | 136.7 | 0.499 | healthy | champion_challenger | |
| R12 | 29.11 | 1.796 | 52.3 | 0.638 | healthy | champion_challenger | **Very-healthy OOD disaster** |
| R13 | 92.28 | 1.886 | 174.0 | 0.226 | moderate | champion_challenger | Our best raw score |
| R14 | 80.0 | 1.980 | 158.4 | 0.464 | healthy | champion_challenger | |
| R15 | 90.0 | 2.079 | **187.1** | 0.329 | moderate | champion_challenger | **Current leaderboard score** |
| R16 | 85.1 | 2.183 | 185.8 | 0.312 | moderate | champion_challenger | Close to best |
| R17 | 77.89 | 2.292 | 178.5 | 0.454 | healthy | doctrine_v1 | |
| R18 | 74.33 | 2.407 | 178.9 | 0.616 | healthy | resubmit_nn | Very-healthy, expected weak |
| R19 | pending | 2.527 | ? | 0.041 | catastrophic | doctrine_v1 | Pure Dirichlet submitted |

**Pattern**: Moderate rounds (z=0.15-0.35) score 80-92. Healthy rounds (z>0.40) score 58-80. Catastrophic (z<0.08) score 84-86 via pure Dirichlet. Very-healthy (z>0.55) is our Achilles heel: R12=29 (z=0.64), R18=74 (z=0.63).

---

## Architecture: Full Prediction Pipeline

### Flow (per round, per seed)
1. **Observe**: 9 tiling viewports × 5 seeds = 45 base queries → full map coverage
2. **Precision queries**: Remaining budget on highest-entropy viewports (up to 50 total)
3. **Compute Global Context Vector**: 8-dim from grid comparison across ALL seeds
4. **Estimate z**: Settlement survival rate from context vector (backward compat)
5. **NN Ensemble**: NF (50%) + v2 (25%) + v3 (25%), each with 8× TTA (rotation+flip)
6. **Geometric blend**: `nn_weight × log(nn_pred) + dir_weight × log(dir_pred)` — logarithmic blend
7. **Physics mask**: Ocean (terrain 5)→always GT index 5, Mountain (terrain 10)→always GT index 0 (verified 100% in all GT)
8. **Floor & normalize**: Clamp min=0.003, renormalize (prevents KL→infinity)

### NN Weight Curve (z-adaptive)
```
z < 0.05:  0% NN (pure Dirichlet) — catastrophic regime
z 0.05-0.12: ramp up to 30%
z 0.12-0.25: ramp up to 60%
z 0.25-0.35: 75% NN (peak)
z 0.35-0.60: linear decay to 75% (sweep showed flat is optimal)
z > 0.60:  75% NN
```
Sweep over 396 configurations (floor, NN weight curve, concentration) landed on NN_PEAK=0.75, flat curve. This was counterintuitive — we expected healthy rolloff — but the data supports it.

### Per-Seed Nudge
Each seed's z is compared to the round-pooled z. If deviation > 0.03 and seed has full 9/9 coverage, a conservative 15% nudge is applied to NN weight (can decrease more than increase).

---

## Model Zoo

### Production Models (active)

| Model | File | Params | Architecture | Training | LORO |
|-------|------|--------|-------------|----------|------|
| **NF** (primary) | `nf2_healthy_all.pt` | 6.4M | 192-hidden, 8 ResBlocks, GELU, multi-scale | nightforce_v2 recipe, 600ep, lr=2e-3 | **86.13** overall |
| **v2** | `astar_nn.pt` | 1.8M | 128-hidden, 6 dilated ResBlocks (1,2,4,8,16,1), ReLU | train_nn_v3 recipe, 1500ep, lr=3e-4 | 73.1 late |
| **v3** | `astar_nn_v3.pt` | 5.7M | 192-hidden, 8 dilated ResBlocks (1,2,4,8,16,8,4,1), ReLU, multi-scale | train_nn_v3 recipe, 1500ep, lr=3e-4 | in-sample 92.7 |
| **Dirichlet** | `calibration.json` | - | 27-key z-conditioned, linear z-slope per key, concentration=30 | Fit on 90 GT + 1166 replays | LORO ~70 |

### NF (Nightforce) Architecture — Our Best Model
```
AstarNetNF:
  stem: Conv2d(13→192) + BN + GELU
  blocks: 8× ResBlockNF(192, no dilation, GELU)
  down: Conv2d(192→192, stride=2) + BN + GELU + ResBlockNF + Upsample(2×)
  merge: Conv2d(384→192, 1×1) + BN + GELU
  head: Conv2d(192→6, 1×1)
```
- Trained with `nightforce_v2.py` recipe: lr=2e-3, weight_decay=1e-4, CosineAnnealingLR, z_jitter=±0.06, batch=16, patience=80, 600 epochs
- Healthy specialist: 3× weight on z>0.40 samples
- Uses same `encode_grid_v2v3` as v2/v3 (13 channels)

### Feature Encoding (13 channels)
```
encode_grid_v2v3(grid, z):
  [0-7]:  One-hot terrain codes [0, 1, 2, 3, 4, 5, 10, 11]
  [8]:    Manhattan distance to settlements [1,2] / 40.0
  [9]:    Coastal adjacency to ocean [10] (4-connected)
  [10]:   Is land (grid != 10)
  [11]:   z (broadcast scalar)
  [12]:   Settlement density (5×5 window, np.pad) / 25.0
```

### NF LORO by Regime (new, retrained on 90 samples)

| Regime | Rounds | Avg LORO | Individual |
|--------|--------|----------|------------|
| Healthy (z>0.40) | R1,R2,R6,R7,R11,R12,R14,R17,R18 | **83.23** | 85.6, 91.5, 86.4, 72.0, 89.1, 65.4, 83.6, 90.9, 84.7 |
| Moderate (z=0.15-0.40) | R4,R5,R9,R13,R15,R16 | **89.08** | 90.3, 83.4, 93.2, 91.9, 93.8, 81.8 |
| Catastrophic (z<0.15) | R3,R8,R10 | **88.93** | 87.7, 90.5, 88.5 |
| **Overall** | R1-R18 | **86.13** | |

**Weak spots**: R7 (72.0, z=0.42 healthy), R12 (65.4, z=0.60 very-healthy), R16 (81.8, z=0.29 moderate). R12 is an outlier — very-healthy regime (z>0.55) where NN struggles.

### v2 Late LORO (R11-R18)
| Round | z | LORO |
|-------|------|------|
| R11 | 0.499 | 73.2 |
| R12 | 0.599 | 53.8 |
| R13 | 0.226 | 84.8 |
| R14 | 0.464 | 68.3 |
| R15 | 0.329 | 83.7 |
| R16 | 0.312 | 81.0 |
| R17 | 0.454 | 76.3 |
| R18 | 0.632 | 63.7 |
| **Late avg** | | **73.1** |

### Ensemble Weights
In `nn_predict.py`, models are blended before passing to strategy:
- **NF**: weight 0.50 (primary predictor)
- **v2**: weight 0.25
- **v3**: weight 0.25

Then the combined NN prediction is blended with Dirichlet via geometric mean (see weight curve above).

---

## What We Built (Chronological)

### Day 1 (March 19-20): Foundation
1. **API client** (`api.py`): All endpoints — `/start_round`, `/simulate`, `/submit`, `/analysis`
2. **Solver loop** (`solver.py`): Stateful, lock-protected, auto-harvests GT, auto-calibrates
3. **Dirichlet-Bayesian** (`strategy.py`): z-conditioned 27-key priors, linear z-slope per key
4. **Calibration system** (`calibrate.py`): Fits priors from GT files, z-model from replay variance
5. **NN v2** (`train_nn.py`): First neural net, 128-hidden dilated ResNet, trained on A100
6. **NN v3** (`train_nn_v3.py`): Larger 192-hidden multi-scale ResNet, z-augmentation
7. **Observation system**: Full 3×3 tiling + precision queries for dense areas
8. **State management** (`state.json`): Prevents double-execution (R3 disaster was dual-execution)

### Day 2 (March 20-21): Channel Fix + Context Vector
9. **CRITICAL BUG FIX**: Channels 11-12 (z, density) were SWAPPED between training and inference. All NN predictions were garbage. R7 went from 13→69 after fix.
10. **Global Context Vector**: 8-dim feature computed from observation grids (not settlement lists). Replaced scalar z for v4.
11. **NN v4** (`train_nn_v4.py`): Conditional U-Net, 16.8M params, takes 8-dim context vector
12. **Codex GPT-5.4 Review**: Found 5 issues — seed misattribution, context mismatch, double-counting, v4 weight too high, precision query bias
13. **Empirical anchoring**: Dirichlet conjugate posterior with observation counts (later disabled — net negative)
14. **Physics masking**: Force ocean/mountain to deterministic outcomes
15. **Per-seed z nudge**: Asymmetric NN weight adjustment per seed

### Day 3 (March 21-22): Nightforce + Retrain Sprint
16. **Nightforce model** (`nightforce_v2.py`): New training recipe — GELU, lr=2e-3, CosineAnnealing, z-jitter, healthy weighting. LORO jumped from 72→83 on healthy regime.
17. **encode_grid mismatch discovery**: `train_nf_proper.py` had completely wrong features (scipy distance-to-mountains, coastal to water[0], is_land excluded mountains). All "proper" NF training was broken. Only `nightforce_v2.py` had correct encoding.
18. **Full retrain on 90 samples**: All 3 production models (NF, v2, v3) retrained with correct encoding on R1-R18 ground truth.
19. **Moderate specialist experiment**: Trained NF variant with 3× weight on moderate z. LORO showed only +0.28 on late moderate rounds (gate was +1.0). Not promoted.
20. **External script review**: Found 4 critical encode_grid bugs in user-provided scripts (distance normalization, coastal includes mountains, is_land excludes mountains, density edge handling).
21. **396-config sweep**: Tested floor values (0.001-0.01), NN weight curves (flat, inverted-U, stepped), concentration (15-50). Best: floor=0.003, NN_PEAK=0.75, flat curve.
22. **v4 deprioritized**: LORO underperformed NF (83.4 vs 86.1). NF became primary model with 50% weight.

---

## Failed Experiments (Complete List)

### Architecture Failures
1. **NN v4 Conditional U-Net** (16.8M params, 8-dim context): R1 LORO=83.4, underperformed simpler NF (86.1). Context vector train/inference mismatch (training uses P(class 1), inference uses binary). Weight reduced from 0.5→0.4→0 (now excluded from ensemble).
2. **Multi-seed NF ensemble** (5 models, bagged): Tested on R12 through full pipeline → **-2.02 raw** vs single model. Diversity wasn't useful.
3. **Dropout regularization** (0.10, 0.15 on NF blocks): LORO worse across ALL regimes. Overfitting wasn't the problem.

### Training Failures
4. **train_nf_proper.py** (wrong encode_grid): Scipy distance-to-mountains instead of Manhattan distance-to-settlements. Coastal adjacency to water[0] instead of ocean[10]. is_land excluded mountains. ALL training with this encoding was wasted.
5. **Augmented training with entropy-weighted KL loss**: Model collapsed after epoch 300. Loss function was wrong for the augmented data regime.
6. **Augmented training with soft KL loss** (`-(target * log_softmax).mean()`): Peaked at epoch 1, then diverged to avg=33-41. Wrong loss.
7. **Two more augmented training attempts**: Wrong architecture (v2 instead of NF), wrong z-conditioning, wrong encoding.
8. **Moderate specialist**: LORO +0.28 on late moderate rounds (gate was +1.0). Healthy specialist beats it overall AND on catastrophic. Not promoted.

### Strategy Failures
9. **Empirical anchoring (Bayesian posterior update)**: Even with n_obs≥2 filter and concentration=50, hurts R12 by -3.5 while helping R17 by +2.2. Net negative — disabled.
10. **Concentration experiments**: Various sweep values for Dirichlet concentration. Marginal differences, not worth complexity.
11. **Live seed split-tests**: Considered but rejected — each round is one-shot, can't afford to waste submissions on A/B testing.

### Infrastructure Failures
12. **R3 dual-execution disaster** (score: 7.17): Solver running from BOTH local Mac AND H100 server simultaneously. Server overwrote good 9-obs predictions with 1-obs garbage. Led to CRITICAL SAFETY RULE: never run from two places.
13. **R7 channel-swap disaster** (score: 38.4): NN channels 11-12 swapped between training and inference. All NN predictions were random noise. R7 scored 38.4 instead of expected ~75.
14. **SSH timeout issues**: Background training tasks on H100 reported as "failed" but actually completed. Learned to check logs directly.

---

## Dirichlet-Bayesian System

### Calibration
- **27 spatial keys**: Terrain code × distance bucket (near/mid/far) × coast/inland
- **z-conditioned**: Each key has `intercept + slope × z` with concentration=30
- **Trained on**: 90 GT files (R1-R18, 5 seeds each) + 1166 replay samples
- **z distribution**: mean=0.345, std=0.173

### How z Is Estimated
From observations: compare initial grid to observed grid across all viewports. Settlement survival rate = z. This is `ctx[7]` in the context vector.

### Regime Classification
| Regime | z Range | Characteristics | Our Strength |
|--------|---------|-----------------|-------------|
| Catastrophic | z < 0.08 | Mass extinction, ruins everywhere | Strong (84-90) via Dirichlet |
| Low-moderate | 0.08-0.15 | Heavy damage, some survival | Good (80-85) |
| Moderate | 0.15-0.35 | Balanced growth/decay | **Best** (85-93) |
| Healthy | 0.35-0.55 | Most settlements survive, expansion | Moderate (77-85) |
| Very-healthy | z > 0.55 | Almost everything survives | **Weak** (29-75) |

### Why Very-Healthy Is Hard
When z>0.55, outcomes are very uniform (high survival, low entropy). The scoring formula heavily weights high-entropy cells. Small errors on the few interesting cells (which ones expanded? which rare ruins appeared?) dominate the score. Our NN hasn't seen enough very-healthy training data to capture these subtle patterns.

---

## Remaining Rounds & Projections

### R19 (Active, closing ~02:45 UTC)
- z = 0.041 (catastrophic)
- Pure Dirichlet submitted (NN weight = 0%)
- Expected: ~85 raw → 85 × 2.527 = **214.8 weighted**
- Would close gap to #1 significantly if it lands

### Remaining Schedule
| Round | Weight | For 219 need | For 230 need | For 260 need |
|-------|--------|-------------|-------------|-------------|
| R20 | 2.653 | 82.5 raw | 86.7 raw | 98.0 raw |
| R21 | 2.786 | 78.6 | 82.6 | 93.3 |
| R22 | 2.925 | 74.9 | 78.6 | 88.9 |
| R23 | 3.072 | 71.3 | 74.9 | 84.6 |

### Score Projections by Regime
| Regime | Expected Raw | At R20 (×2.65) | At R22 (×2.93) | At R23 (×3.07) |
|--------|-------------|----------------|----------------|----------------|
| Moderate | 89 | **235.9** | **260.8** | **273.2** |
| Healthy | 80 | 212.0 | 234.4 | 245.6 |
| Catastrophic | 85 | 225.5 | 249.1 | 261.0 |
| Very-healthy | 70 | 185.5 | 205.1 | 214.9 |

**Key insight**: By R22, even healthy regime (80 raw) beats current #1 (219). A moderate round at R22+ gives 260+ which is almost certainly unbeatable.

### Probability Analysis
- P(at least one moderate round in R20-R23) ≈ 60% (based on 6/18 historical moderate rounds)
- P(all very-healthy) ≈ 5% (worst case)
- P(we beat 219 at some point) ≈ 85% (catastrophic/moderate/healthy all sufficient by R22+)
- P(we beat 250) ≈ 40% (needs moderate at R22+ or excellent catastrophic)

---

## Production Stack (Current Files)

| File | Size | Purpose |
|------|------|---------|
| `solver.py` | - | Main loop: observe → predict → submit |
| `strategy.py` | - | Ensemble blend, Dirichlet, context vector, physics mask |
| `nn_predict.py` | - | Multi-model NN inference with TTA |
| `calibrate.py` | - | z-conditioned Dirichlet prior fitting |
| `api.py` | - | HTTP client for competition API |
| `config.py` | - | Token, query budget |
| `state.json` | - | Persistent state, prevents double-execution |
| `nf2_healthy_all.pt` | 25.7MB | NF healthy specialist (PRODUCTION) |
| `astar_nn.pt` | 7.2MB | v2 dilated ResNet (PRODUCTION) |
| `astar_nn_v3.pt` | 22.7MB | v3 multi-scale ResNet (PRODUCTION) |
| `calibration.json` | 12KB | 27-key z-conditioned Dirichlet priors |
| `ground_truth/` | 90 files | R1-R18 × 5 seeds GT data |

### H100 Server (86.38.238.86)
- GPU idle, standing by for R19 GT → retrain on 95 samples
- All retrain scripts uploaded: `retrain_nf_r18.py`, `retrain_v2_r18.py`, `retrain_v3_r18.py`
- Completed training logs available

---

## Structural Analysis: Why We're #131 Not #1

### Our Strengths
- Moderate rounds: avg 89.1 raw (LORO), consistently 85-93 in practice
- Catastrophic rounds: avg 88.9 raw (LORO), Dirichlet is very accurate here
- Principled ensemble: geometric blend, z-adaptive weighting, physics masking
- Solid infrastructure: stateful solver, auto-calibration, lock-protected execution

### Our Weaknesses
1. **Very-healthy regime (z>0.55)**: R12=29 (LORO 65.4), R18=74.3 (LORO 84.7). When everything survives, the model can't distinguish subtle cell-level outcomes.
2. **Healthy regime inconsistency**: R6=59, R7=38, R11=80, R14=80, R17=78. Variance is high.
3. **Top team delta**: Top teams appear to average ~91 raw across ALL regimes. Our healthy average is ~80 — an 11-point structural gap. The weight multiplier on later rounds may or may not close this.
4. **Early-round damage**: R3 (7.2), R7 (38.4) from bugs, not model quality. But they don't matter for leaderboard (only best round counts).

### What Would Close the Gap
- +10 on healthy regime → avg 90 → any R22+ round gives 263+ (wins)
- But our LORO says healthy NF averages 83.2 — getting to 90 would require a fundamentally different approach to healthy regime prediction
- The weight multiplier is doing the heavy lifting: even our current 80 healthy avg × R23 weight (3.07) = 245.6

---

## Critical Safety Rules
1. **NEVER run solver from two places simultaneously** — R3 dual-execution disaster
2. **state.json prevents double-execution** — always check before submitting
3. **50 queries per seed per round** — 9 tiling + precision (no waste)
4. **Submissions overwrite destructively** — no undo, verify before overwriting
5. **Floor at 0.003 ALWAYS** — prevents KL → infinity on any cell
6. **Unlimited resubmissions** — can resubmit with better model during round
7. **Context from base 45 queries only** — exclude precision queries to avoid bias
8. **LOCAL MAC ONLY for solver** — never run from H100 server

---

## What Matters Now
1. **R19 score**: If catastrophic lands ~85 → 214.8 weighted (close to #1)
2. **R20+ with upgraded models**: First live test of retrained NF/v2/v3 on 90 samples
3. **After R19 GT**: Retrain on 95 samples (marginal, ~0.5-1.0 LORO improvement expected)
4. **Ride the weight multiplier**: By R22, even our weakest regime (healthy=80) gives 234+ which beats current #1
5. **Hope for moderate**: One moderate round at R22+ → 260+ → almost certainly wins

## Risk Assessment
- **Best case** (60%): Moderate round at R22+ → 260+ weighted → top 10 or better
- **Good case** (25%): Healthy/catastrophic rounds only, but weight multiplier carries us past 220 by R22
- **Bad case** (10%): All very-healthy (z>0.55) → stuck at ~210-215
- **Worst case** (5%): Multiple very-healthy rounds, R12-style disasters → stay below 200
