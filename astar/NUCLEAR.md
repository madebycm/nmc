# NUCLEAR RESUBMIT — R23 Free Lottery Ticket

## Why This Is Free
- **Only BEST weighted round counts** for leaderboard
- Our best: **R21 = 249.4** (89.54 raw x 2.786) — LOCKED, can't change
- R23 needs **81.2+ raw** to beat R21 (at weight 3.072)
- Our healthy average: **74** (never hit 81 on healthy)
- **Conclusion: R23 cannot hurt us. Any experiment is free.**

## Time
- R23 closes: **14:00 UTC / 15:00 CET**
- Unlimited resubmissions until close
- Last submission wins

---

## COMPLETE TECHNICAL STACK — Everything We Know

### 1. The Problem

40x40 Norse island grid. Each cell starts as one of 8 terrain types: `{0: empty, 1: settlement, 2: port, 3: ruin, 4: forest, 5: ocean, 10: mountain, 11: grassland}`. A hidden stochastic simulation runs 50 years. We predict the **probability distribution over 6 GT categories** for each cell: `{0: land/mountain/grassland, 1: settlement, 2: port, 3: ruin, 4: forest, 5: ocean}`.

**Scoring**: `round_score = avg over 5 seeds (100 * exp(-3 * entropy_weighted_KL))`. Leaderboard = `max over all rounds (round_score * 1.05^round_number)`. Only BEST single weighted round matters.

### 2. Observation Mechanics

Each `/simulate` API call runs a **FULL independent 50-year simulation** with a different random seed. Returns a 15x15 viewport of the final Year 50 state.

- **Budget**: 50 queries per round (10 per seed)
- **Tiling**: 3x3 grid of 15x15 viewports at positions `[0, 13, 25]` x `[0, 13, 25]` = 9 base queries per seed (45 total)
- **Precision**: 1 extra query per seed on highest-entropy viewport
- **Key insight**: Observations ARE Monte Carlo samples from P(final_state). Different queries = different simulation runs = independent samples from the true distribution.
- Overlap zones: columns/rows 13-14 and 25-27 covered by 2 tiles

### 3. z-Value (Regime Indicator)

`z = settlement survival rate` = fraction of initially-settled cells that still have settlement/port at Year 50. Computed from grid comparison (NOT settlement list — API only reports alive ones).

| Regime | z range | Characteristics | Our avg score |
|--------|---------|-----------------|---------------|
| Catastrophic | < 0.08 | Mass die-off, most cells → ruin/forest | 68.0 (best 94.2) |
| Low-moderate | 0.08-0.15 | Heavy decline but some survive | 90.5 (one sample) |
| Moderate | 0.15-0.35 | Balanced — settlements persist, some decay | 85.3 (best 92.3) |
| Healthy | > 0.35 | Most settlements survive, expansion common | 62.6 (best 80.0) |

**R23 z = 0.463 (healthy)** — our weakest regime by far. Per-seed z: 0.474, 0.525, 0.414, 0.387, 0.585.

### 4. Dirichlet-Bayesian Prediction (calibrate.py → strategy.py)

**Concept**: For each cell, set a Dirichlet prior based on:
- Initial terrain code (8 types)
- Distance to nearest settlement (near ≤3, mid ≤6, far)
- Coastal adjacency (adjacent to mountain/ocean? — note: `_is_coastal` checks code 10)
- z-conditioning (linear model: `P(class | features) = intercept + slope * z`)

This gives 27 unique keys (e.g., `"11_near_coast"`, `"1_far_inland"`). Each key has a linear model trained from:
- **110 GT files** (R1-R22 x 5 seeds) — actual ground truth outcomes
- **7,767 replay samples** — Monte Carlo simulations we harvested, giving smooth z-coverage

**Process**:
1. Compute `alpha[cell] = concentration * (intercept + slope * z)` for each cell's calibration key
2. Add observation evidence counts: `posterior = alpha + obs_counts`
3. Normalize: `prediction = posterior / sum(posterior)`

**Concentration**: z_model uses per-key concentration (default 30). Higher concentration = model prior dominates over observations.

### 5. Neural Network Ensemble (nn_predict.py)

Three architectures, all convolutional:

| Model | Architecture | Params | Hidden | Input | Performance |
|-------|-------------|--------|--------|-------|-------------|
| v2 (AstarNet) | 128-hidden, 6 ResBlocks (dilations 1,2,4,8,16,1) | 1.8M | 128 | 13ch | LORO 77.6 |
| v3 (AstarNetV3) | 192-hidden, 8 ResBlocks + multi-scale (AvgPool2d branch) | 5.7M | 192 | 13ch | LORO 74.5 |
| v4 (ConditionalUNet) | U-Net with enc→bottleneck→dec skip connections | 16.8M | 160 | 20ch | LORO ~83 (partial) |
| NF (AstarNetNF) | 192-hidden, 8 ResBlocks, GELU, stride-2 downscale branch | 6.4M | 192 | 13ch | LORO **87.49** |

**Currently active**: Only NF model (`nf2_healthy_all.pt`, z-jitter=0.02, LORO 87.49). v2/v3/v4 disabled.

**Input channels (v2/v3/NF — 13 total)**:
- ch 0-7: One-hot terrain codes `[0, 1, 2, 3, 4, 5, 10, 11]`
- ch 8: Manhattan distance to nearest settlement/port (normalized by /40)
- ch 9: Coastal adjacency (adjacent to code 10)
- ch 10: Is-land mask (grid != 10)
- ch 11: z broadcast (scalar z filled across 40x40)
- ch 12: Settlement density (5x5 kernel, fraction of cells that are settlement/port)

**CRITICAL: ch 11 = z BEFORE ch 12 = density**. Previously swapped (channels bug — all NN predictions were garbage until R8 fix).

**v4 Input (20 channels)**: 12 spatial (no z broadcast) + 8-dim context vector broadcast

**TTA**: 8x test-time augmentation (4 rotations x 2 flips). Each augmentation → predict → reverse → average.

### 6. Global Context Vector (8-dim)

Replaces scalar z for v4. Computed from **base 45 tiling observations only** (precision queries excluded to avoid bias).

| Dim | Feature | How computed |
|-----|---------|-------------|
| 0 | Settlement survival rate | initially settled → still settled/port |
| 1 | Port survival rate | initially port → still port |
| 2 | Ruin frequency | ruin cells / land cells |
| 3 | Forest fraction | forest cells / land cells |
| 4 | Collapse rate | 1 - survival rate |
| 5 | Expansion rate | initially empty → settlement/port |
| 6 | Entropy proxy | faction_diversity * 0.5 + food_level * 0.5 |
| 7 | z (= survival rate) | backward compat with scalar z |

### 7. Ensemble Blend (strategy.py lines 455-510)

**z-adaptive NN weight curve** (sweep-optimized over 396 configs):
```
z < 0.05:            0% NN (pure Dirichlet) — catastrophic
z 0.05-0.12:  ramp 0→30% NN
z 0.12-0.25:  ramp 30→60% NN
z 0.25-0.35:       75% NN (peak)
z > 0.35:           75% NN (flat — no healthy penalty)
```

**Blend method**: Geometric mean (log-space):
```
log_blend = nn_weight * log(nn_pred) + (1 - nn_weight) * log(dir_pred)
blended = exp(log_blend)
```

**Per-seed nudge**: If a seed's z deviates from round z by >0.03, apply small weight adjustment (alpha=0.15, clipped to ±0.10/+0.05).

### 8. Physics Masking

Hard rules verified against ALL ground truth (0 exceptions):
- **Ocean cells (code 5) → always ocean (class 5)**. Force P(ocean) = 1.0.
- **Mountain cells (code 10) → always mountain (class 0)**. Force P(land) = 1.0.

Applied after ensemble, before final floor+normalize.

### 9. Probability Floor

`PROB_FLOOR = 0.003`. Every probability clamped to ≥ 0.003, then renormalized. Prevents KL divergence → infinity when true class gets near-zero probability. Sweep-optimized: 0.003 > 0.005 > 0.01.

### 10. Empirical Anchoring (currently DISABLED)

We built `empirical_anchor()` — Dirichlet conjugate posterior update where model prediction is prior (concentration=30) and observation counts are evidence. **But it was disabled** because testing showed it hurt R12 by -3.5 while only helping R17 by +2.2. The root cause: with n=1-2 observations per cell (from the 3x3 tiling), single one-hot samples are too noisy vs a good NN ensemble.

**KEY INSIGHT FOR NUCLEAR**: With 9-11 observations per seed (not 1-2), observation counting becomes much more reliable. The overlap zones (2 tiles) have even more. This is exactly what the nuclear approach exploits.

### 11. Data Assets

| Asset | Count | Description |
|-------|-------|-------------|
| Ground truth | 110 files (R1-R22 x 5 seeds) | True distributions P(class) for each cell |
| Replay samples | 7,767 | Monte Carlo simulations for calibration |
| Calibration keys | 27 | z-conditioned linear models per terrain key |
| R23 observations | 9-11 per seed | Independent Year 50 simulation outcomes |
| NF model | 1 (6.4M params) | LORO 87.49, z-jitter=0.02 trained |

---

## The Nuclear Idea: Direct Observation Counting

Each `/simulate` runs a FULL independent 50-year sim. We have **9-11 observations per seed** — these are Monte Carlo samples from the TRUE distribution P(final_state).

**Current approach (75% NN + 25% Dirichlet):** NN predicts from initial grid + z. Weak on healthy (avg 74, best 80).

**Nuclear approach:** For each cell, COUNT the actual outcomes across observations. This is the **maximum-likelihood estimate** of the true distribution. Smooth with a prior to avoid zeros.

### Why This Might Work

1. **NN weakness on healthy**: Our NN averages 62.6 on healthy (vs 85.3 on moderate). The NN was trained on limited healthy GT data and generalizes poorly.
2. **Observations ARE ground truth samples**: Each observation is a fully-converged 50-year simulation. They're not approximations — they're independent draws from P(final_state | initial_grid, hidden_params).
3. **9-11 samples is actually decent**: With Dirichlet smoothing (alpha=0.5), we get reasonable posteriors. For deterministic cells (ocean, mountain), even 1 sample confirms. For stochastic cells, 9 samples captures the mode reliably.
4. **Overlap zones get 2x coverage**: Cells in columns/rows 13-14 and 25-27 are covered by 2 tiles = 18-22 samples.

### Why This Might NOT Work

1. **9 samples can miss rare outcomes**: If a cell is forest 80% and settlement 20%, 9 samples might show 9/9 forest (probability ≈ 0.13). KL divergence for putting too little mass on the true 20% settlement outcome is punishing.
2. **No spatial coherence**: Raw counting treats each cell independently. NN sees spatial patterns. A rare cell surrounded by similar cells might need the NN's pattern recognition.
3. **High-z healthy regimes are MORE stochastic**: z=0.463 means lots of things are changing. More entropy = harder to pin down from few samples.

### Implementation Plan

For each cell (r, c) on the 40×40 map:
1. Find all observations whose viewport contains (r, c)
2. Extract terrain code → map to class via `CODE_TO_CLASS`
3. Count frequencies → empirical distribution over 6 classes
4. Smooth: `(count + alpha) / (total + 6*alpha)` where alpha=0.5
5. Blend with NN prediction using `empirical_anchor()` with tuned concentration

**Blend options** (what to ask experts):
- Pure empirical (alpha=0.5 smoothing, no NN) — maximum departure from current
- 50/50 blend: `0.5 * empirical + 0.5 * nn_pred`
- Bayesian posterior: use NN as prior (concentration=C), observations as evidence
- **What's the optimal C?** With n=9 observations: C=9 means 50/50. C=30 means NN dominates (75%). C=3 means observations dominate (75%).

---

## R23 Observation Coverage

| Seed | Observations | Per-seed z |
|------|-------------|-----------|
| 0 | 9 | 0.474 |
| 1 | 11 | 0.525 |
| 2 | 10 | 0.414 |
| 3 | 11 | 0.387 |
| 4 | 9 | 0.585 |

Viewport positions: `(0,0) (13,0) (25,0) (0,13) (13,13) (25,13) (0,25) (13,25) (25,25)` — each 15x15.
Seeds 1 and 3 have 11 obs (2 precision queries), seeds 2 has 10 (1 precision), seeds 0 and 4 have 9 (base only).

Grid terrain codes in observations: `{1, 2, 3, 4, 5, 10, 11}` — no code 0 observed.

---

## Complete Score History (All 23 Rounds)

| Round | Raw | Weight | Weighted | z | Regime | Strategy | Notes |
|-------|-----|--------|----------|------|--------|----------|-------|
| R1 | — | 1.050 | — | 0.419 | healthy | — | No submission |
| R2 | — | 1.103 | — | 0.415 | healthy | — | No submission |
| R3 | 7.17 | 1.158 | 8.3 | 0.018 | catastrophic | proximity_v1 | Overwrite disaster — server poller clobbered good predictions |
| R4 | 79.94 | 1.216 | 97.2 | 0.235 | moderate | dirichlet_v3 | First real score. Regime-aware priors. |
| R5 | 75.40 | 1.276 | 96.2 | 0.330 | moderate | dirichlet_v4 | |
| R6 | 58.83 | 1.340 | 78.8 | 0.415 | healthy | dirichlet_v4 | First healthy — weak |
| R7 | 38.39 | 1.407 | 54.0 | 0.423 | healthy | ensemble_nn_v1 | NN channel bug — garbage predictions |
| R8 | 84.42 | 1.477 | 124.7 | 0.068 | catastrophic | ensemble_ctx_v2 | First strong catastrophic |
| R9 | 89.12 | 1.551 | 138.3 | 0.275 | moderate | champion_challenger | Strong moderate |
| R10 | 86.25 | 1.629 | 140.5 | 0.058 | catastrophic | champion_challenger | |
| R11 | 79.94 | 1.710 | 136.7 | 0.499 | healthy | champion_challenger | Best healthy at the time |
| R12 | 29.11 | 1.796 | 52.3 | 0.638 | healthy | champion_challenger | Extreme healthy OOD disaster |
| R13 | 92.28 | 1.886 | 174.0 | 0.226 | moderate | champion_challenger | Peak moderate score |
| R14 | 79.99 | 1.980 | 158.4 | 0.464 | healthy | champion_challenger | Decent healthy |
| R15 | 90.00 | 2.079 | 187.1 | 0.329 | moderate | champion_challenger | |
| R16 | 85.10 | 2.183 | 185.8 | 0.312 | moderate | champion_challenger | |
| R17 | 77.89 | 2.292 | 178.5 | 0.454 | healthy | doctrine_v1 | z-jitter model era begins |
| R18 | 74.33 | 2.407 | 178.9 | 0.616 | healthy | resubmit_nn | High-z healthy |
| R19 | 94.19 | 2.527 | 238.0 | 0.041 | catastrophic | doctrine_v1 | SPIKE — best raw score ever |
| R20 | 90.54 | 2.653 | 240.2 | 0.130 | low-moderate | doctrine_v1+resubmit | Resubmit at 35% NN |
| R21 | 89.54 | 2.786 | **249.4** | 0.263 | moderate | doctrine_v1 | **CURRENT BEST WEIGHTED** |
| R22 | 83.93 | 2.925 | 245.5 | 0.215 | moderate | doctrine_v1 | Below moderate avg |
| R23 | pending | 3.072 | ? | 0.463 | healthy | doctrine_v1 | FINAL ROUND |

### Score Evolution by Regime

**Catastrophic** (z < 0.08): R3=7.2, R8=84.4, R10=86.2, **R19=94.2** → Our strength, improving trend
**Moderate** (z 0.15-0.35): R4=79.9, R5=75.4, R9=89.1, R13=92.3, R15=90.0, R16=85.1, R21=89.5, R22=84.0 → avg ~85
**Healthy** (z > 0.35): R6=58.8, R7=38.4, R11=79.9, R12=29.1, R14=80.0, R17=77.9, R18=74.3 → avg ~63, best 80

### Known Bugs (all fixed)
1. **Channel swap** (R5-R7): NN inference had z and density channels swapped vs training. All NN predictions were garbage. Fixed R8.
2. **Overwrite disaster** (R3): Server poller overwrote good 9-observation predictions with 1-observation junk.
3. **Seed misattribution**: Flat observation list broke with uneven per-seed counts from precision queries. Fixed: pass dict.
4. **Coastal check**: `_is_coastal` checks adjacency to code 10 (mountain), not code 5 (ocean). May be intentional (fjord-style maps) or bug.

---

## Current Position
- **#73 out of 392 teams** at 249.4
- **#1**: Lokka Language Models at 266.6
- **Gap**: 17.2 points (unbridgeable — would need 87+ raw on healthy, our best is 80)

---

## QUESTIONS FOR FRONTIER ML EXPERTS

Given our situation (final round, healthy regime z=0.463, ~90 min left, 9-11 Monte Carlo observations per cell from Year 50 simulations, existing NN ensemble + Dirichlet model), we need advice on:

### Q1: Optimal Bayesian Update — What Concentration?

We have an `empirical_anchor()` function that does Dirichlet conjugate updating:
```
posterior_alpha = model_pred * concentration + obs_counts
posterior = normalize(posterior_alpha)
```

With `n=9` observations and `concentration=C`:
- C=9: observations and model contribute equally (50/50)
- C=30: model dominates (77% weight)
- C=3: observations dominate (75% weight)

Our model's LORO score on healthy is ~74 (weak). The observations are exact Monte Carlo samples.

**What C value maximizes expected score given n=9 independent samples from the true distribution and a model that scores ~74/100?**

### Q2: Should We Trust Observations More on High-Entropy Cells?

Mountain and ocean cells are deterministic (always same). For those, n=1 is enough. But settlement/forest/ruin transitions are stochastic. Should we use a **cell-type-dependent concentration**: low C for stochastic cells (trust observations more), high C for deterministic cells?

### Q3: Log-Space vs Linear Blending

Our current NN+Dirichlet blend uses geometric mean (log-space). For the nuclear observation counting, should we:
- (a) Add observation counts directly to the Dirichlet posterior (conjugate update) then blend with NN in log-space?
- (b) Compute empirical distribution separately, blend with NN+Dirichlet in linear space?
- (c) Use observations as the likelihood in a proper Bayesian posterior with NN as prior?

### Q4: Is 75% NN Weight Too High for Healthy?

Our sweep said "no healthy penalty needed" (75% NN even at z>0.35). But our healthy scores are terrible (avg 63). The sweep was done on only 7 healthy GT rounds, some with buggy NN (R7). Should we trust the sweep or override to lower NN weight for healthy?

### Q5: Variance-Aware Smoothing

With 9 samples: P(seeing k=9 for category with true probability p) follows Binomial. If we see 9/9 of some category, the Beta posterior 95% CI for p is [0.72, 1.0]. Should we explicitly model uncertainty and use the posterior **mean** (which is (k+alpha)/(n+2*alpha)) or some other point estimate?

### Q6: The Real Question

Given that we score ~74 avg on healthy and need 81.2+ to beat our R21 record: **Is there ANY strategy that gives us >50% chance of hitting 81.2+ on a z=0.463 healthy round?** Or should we accept R21=249.4 as final? What would a team scoring 95+ on healthy be doing differently?
