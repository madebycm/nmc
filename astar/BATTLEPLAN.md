# ASTAR ISLAND BATTLEPLAN — REPLAY PIVOT

> **Updated**: 2026-03-20 19:30 UTC
> **Remaining**: ~43 hours (deadline March 22 15:00 CET)
> **Position**: ~#70 (97.2 best weighted) — R8 score pending (est 84.4 × 1.477 = 124.6)
> **Leader**: #1 at 140.3
> **R9 Status**: LIVE, safe baseline submitted (z=0.287), closes 20:47 UTC
> **Formula**: `leaderboard = max(round_score × 1.05^round_number)` — ONLY best single round

## THE PIVOT

**2026-03-20 18:00 UTC**: Discovered `POST /replay` endpoint returns **full 51-frame simulations with internal state** for any completed round. Each call returns a DIFFERENT random sim_seed. Zero query cost. Rate limited at ~0.7 req/s but unlimited total.

**Before:** Optimizing a predictor from 40 sparse GT files.
**Now:** Optimizing a **decision system** — round-latent inference, query allocation, seed-wise promotion — backed by an offline simulator with unlimited Monte Carlo samples.

### What Replay Data Contains
```
POST https://api.ainm.no/astar-island/replay
Body: {"round_id": "<uuid>", "seed_index": 0-4}

Response: {
    sim_seed: int,           // RANDOM each call
    frames: [                // 51 frames (year 0-50)
        {
            step: int,
            grid: int[40][40],         // terrain codes
            settlements: [
                {x, y, population, food, wealth, defense, has_port, alive, owner_id}
            ]
        }
    ]
}
```

### Key Properties
- **Unlimited samples**: different random outcome each call (Monte Carlo)
- **Zero cost**: no query budget impact
- **Completed rounds only**: R1-R8 available, NOT active rounds
- **Full internal state**: population, food, wealth, defense per settlement per year
- **Rate limit**: ~0.7 req/s sustained, generous backoff on 429
- **GT relationship**: API ground_truth = average over many Monte Carlo sims

### Why This Is NOT "4000 Training Examples"
Each round has ONE hidden parameter setting shared across 5 seeds. Replay samples
from the same round share the same regime. There are only **8 independent round-regimes**.
Random train/val splits will test memorization, not generalization.
**ALL evaluation MUST be leave-one-round-out (LORO).**

---

## PHASE 1: Validate Replay → GT Convergence (GO/NO-GO GATE)

**Time**: 15 min | **Compute**: Mac (HTTP only) | **Status**: IN PROGRESS

### What
Average N replay final-state grids → empirical probability distribution per cell.
Compare to `/analysis` ground truth tensor.

### Why This Gates Everything
If replay MC converges to official GT → replay IS the data generator → green light all phases.
If it diverges → replay still useful for feature learning but NOT as direct GT substitute.

### Execution
```python
# For R1 seed 0 (20 samples already harvested):
# 1. Load all 20 final grids (frame 50)
# 2. Count terrain type per cell → probability distribution
# 3. Compare to analysis GT via mean absolute error + KL divergence
# 4. Check convergence: does error decrease as N increases?
```

### Go/No-Go
- **GREEN**: MAE < 0.05 with 20 samples → full steam ahead
- **YELLOW**: MAE 0.05-0.10 → use for representation learning, not hard targets
- **RED**: MAE > 0.10 → replay uses different sim version, proceed cautiously

---

## PHASE 2: Replay-Backed Offline Harness (Decision System Tuning)

**Time**: 2h build + continuous use | **Compute**: A100 for NN inference | **Status**: PARTIALLY DONE

### What
Emulate full live rounds offline using replay data:
- Choose historical round
- Sample replay rollouts
- Crop synthetic 15×15 viewports (exactly like `/simulate`)
- Run current solver end-to-end
- Score against official GT

### Why
This is the ONLY way to tune the actual decision loop:
- z/latent thresholds
- NN/Dirichlet blend weights
- Query allocation policy (which viewports, how many per seed)
- Per-seed aggression gates
- Observation anchoring

### What Exists
- `replay_harness.py` — grid search over blend params (DONE, ran on A100)
- Found: nn=0.50, v2=0.35, v3=0.50, v4=OFF, anchor=0, floor=0.005

### What's Missing
- Synthetic viewport simulation (crop replay frames as fake /simulate responses)
- Query policy optimization (test different tiling strategies)
- End-to-end solver simulation (not just blend params)
- Per-seed promotion threshold tuning

### Execution
```bash
# On A100:
ssh root@135.181.8.209
cd /astar
# Upload replays + harness code
# Run full policy sweep
python replay_harness_v2.py --policy-sweep
```

---

## PHASE 3: Settlement Survival Model (Quick Win)

**Time**: 1h | **Compute**: A100 (seconds to train) | **Status**: NOT STARTED

### What
From replay step 0 settlement features, predict step 50 alive/dead.
Features: `[population, food, wealth, defense, has_port, x, y, local_terrain_context]`
Model: XGBoost or logistic regression.

### Why
- `/simulate` exposes EXACTLY these internal stats during live rounds
- Near-perfect z estimation without spending queries on full-map scanning
- Settlement survival IS the core mechanic that drives terrain transitions
- Enables smarter query allocation: focus queries on uncertain-survival settlements

### Impact on Live Rounds
Currently we spend 45 queries on tiled viewports to estimate z from grid changes.
With a settlement survival model, we can estimate z from settlement stats alone
(available in every /simulate response), freeing queries for high-value targets.

### Execution
```python
# 1. Extract training data from all replay frames:
#    X = step_0 settlement features (population, food, wealth, defense, has_port, terrain)
#    y = step_50 alive/dead
# 2. Train XGBoost with round-balanced sampling (NOT replay-balanced)
# 3. Validate LORO (hold out entire rounds)
# 4. Deploy as z_estimate = mean(predicted_survival) per round
```

---

## PHASE 4: Replay-Augmented Direct Predictor (Main A100 Job)

**Time**: 4h (data prep + training) | **Compute**: A100 GPU | **Status**: NOT STARTED

### What
Retrain v2/v3 architectures with replay data:
- Input: initial terrain channels + geom channels + round-latent/context
- Target: MC-averaged final probability distribution (from N replay samples)
- Auxiliary heads: survival rate, port survival, ruin fraction

### Why
- Zero architectural changes needed — same dataloaders, same models
- Replaces 40 GT files with much smoother MC-averaged targets
- Auxiliary heads force regime learning, not texture memorization
- Round-balanced sampling prevents overfitting to over-harvested rounds

### Critical Guardrails
- **Round-balanced sampling**: equal weight per round, NOT per replay sample
- **LORO validation only**: hold out entire rounds, never random splits
- **Promotion gate**: new model must pass full replay harness before going live
- **Fallback**: current v2b+v3b stay active until new model proven better

### Execution
```bash
# On A100:
# 1. Upload replay data: scp -r replays/ root@135.181.8.209:/astar/
# 2. Generate MC-averaged targets per round/seed config
# 3. Train v2c with replay targets + auxiliary heads
# 4. LORO validation against official GT
# 5. If LORO improves: download, test through nn_predict.py, deploy
nohup python3 -u train_nn_v2_replay.py > train_v2_replay.log 2>&1 &
```

---

## PHASE 5: Observation-Conditioned Model (Overnight Job)

**Time**: 6-8h | **Compute**: A100 GPU | **Status**: NOT STARTED

### What
Train a model on synthetic live episodes generated from replay data:
- Input per seed: initial map + observation tensor (from synthetic viewport queries) + visit mask
- Pooled input: round-latent summary from all 5 seeds' queries
- Target: final probability tensor

### Why
This is the true frontier — a model trained on the SAME information pattern we get
in competition: sparse stochastic observations, shared hidden round regime, full-map prediction.
Much closer to the actual game than initial-grid-only supervision.

### Architecture
```
[initial_grid_channels] + [observation_channels] + [visit_mask] + [round_latent_broadcast]
    → Encoder (ResNet backbone)
    → Decoder (per-cell 6-class softmax)
    + Auxiliary: z prediction, settlement count, port count
```

### Execution
```bash
# On A100 (start before sleep, check in morning):
# 1. Build synthetic episode generator from replay data
# 2. For each replay: crop random viewports → build observation tensor
# 3. Train obs-conditioned model
# 4. LORO + promotion gate in morning
nohup python3 -u train_obs_conditioned.py > train_obs.log 2>&1 &
```

### When to Start
- Only AFTER Phase 3 and Phase 4 are validated
- If Phase 4 model already beats 85+ LORO, this becomes lower priority
- Best scheduled as overnight job

---

## LIVE-ROUND OPERATING DOCTRINE (Post-Pivot)

### At Round Open (automated by solver.py)
1. **Submit conservative baseline for ALL 5 seeds immediately** (< 2 min)
2. Spend first 45 queries on validated coverage policy
3. Use remaining 5 queries for **maximum expected improvement**:
   - NOT 1 per seed — spend on the knife-edge seed
   - Target: high model disagreement, coastal settlement clusters, ruin zones
4. Per-seed promotion: keep conservative unless strong evidence → overwrite

### Query Allocation (Post Settlement Model)
- If settlement survival model deployed: estimate z from settlement stats alone
- Spend ALL 50 queries on high-entropy zones instead of z estimation
- Focus on: settlement borders, port adjacency, ruin clusters

### Seed-Wise Submission
- Safe baseline: ALL 5 seeds with conservative recipe
- Promotion: overwrite individual seeds with aggressive recipe IF:
  - z > 0.30 AND v2/v3 agreement > 0.85
  - OR settlement model confidence > 0.9
- Insurance: keep at least 2 seeds conservative, max 3 aggressive
- Catastrophic override: z < 0.08 → pure Dirichlet (overwrite NN blend)

---

## RECIPE SYSTEM (Harness v2 — Production-Matched)

```python
# Harness v2 full sweep: 14,400 recipes, 8 rounds, TTA
# Matches production stack exactly: replay/v2/v3 with arithmetic NN blend
OPTIMAL = {
    "nn_weight": 0.65,       # overall NN vs Dirichlet (geometric blend)
    "replay_weight": 0.0,    # replay model: NOT useful with 45 obs queries
    "v2_weight": 0.10,       # v2b: small regularization role
    "v3_weight": 0.70,       # v3b: dominant (overfitting = diversity!)
    "anchor_weight": 0.0,    # empirical anchoring: OFF
    "prob_floor": 0.003,     # lower floor = more confident
    "z_thresholds": [0.05, 0.12, 0.25],  # linear ramp within bands
}
# z-adaptive NN weight (linear ramp):
# z < 0.05: 0.0 (pure Dirichlet)
# z 0.05-0.12: ramp 0 → 0.26
# z 0.12-0.25: ramp 0.26 → 0.52
# z >= 0.25: 0.65
```

**Key findings from harness v2:**
- Replay model (LORO 78.3) adds NOTHING in obs-rich regime → weight=0
- v3b overfitting (93.1 fit, 72.3 LORO) is BENEFICIAL for ensemble diversity
- v3c (regularized, LORO 76.8) WORSE in ensemble than v3b
- Arithmetic NN blend (production) outperforms geometric (old harness)

Healthy-round backtests (TTA): R1=94.6, R2=94.9, R4=94.7, R5=94.2, R6=93.4, R7=92.7
Source: replay_harness.py v2 sweep over 14,400 recipes, 8 rounds.

---

## HARVEST STATUS

### Replay Data: 782 samples harvested (95-100 per round)

### Ground Truth (Official): 40 files (R1-R8 × 5 seeds)
| Round | z | Regime | GT | Replay | Harness Score (TTA) |
|-------|---|--------|----|--------|---------------------|
| R1 | 0.419 | healthy | 5 ✓ | 100 | 94.6 |
| R2 | 0.415 | healthy | 5 ✓ | 100 | 94.9 |
| R3 | 0.018 | catastrophic | 5 ✓ | 98 | 81.1 (Dirichlet) |
| R4 | 0.235 | moderate | 5 ✓ | 99 | 94.7 |
| R5 | 0.330 | moderate | 5 ✓ | 99 | 94.2 |
| R6 | 0.415 | healthy | 5 ✓ | 96 | 93.4 |
| R7 | 0.423 | healthy | 5 ✓ | 95 | 92.7 |
| R8 | 0.068 | catastrophic | 5 ✓ | 95 | 89.5 (Dirichlet) |

---

## SCORED ROUNDS

| Round | Score | Weight | Weighted | Strategy |
|-------|-------|--------|----------|----------|
| R3 | 7.2 | 1.158 | 8.3 | proximity_v1 (disaster) |
| R4 | **79.9** | 1.216 | **97.2** | dirichlet_v3 |
| R5 | 75.4 | 1.276 | 96.2 | dirichlet_v4 |
| R6 | 58.8 | 1.340 | 78.8 | dirichlet_v4 |
| R7 | 38.4 | 1.407 | 54.0 | broken NN channels |
| R8 | ~84.4 | 1.477 | ~124.6 | pure Dirichlet (z=0.068) |
| R9 | pending | 1.551 | ? | v2b+v3b ensemble nn=0.65 (z=0.272) |

**Projections for future rounds:**
- R10 (wt 1.629): healthy 93+ → **151+** | moderate 90+ → **146+**
- R11 (wt 1.710): healthy 93+ → **159+** | moderate 90+ → **154+**
- Only BEST single weighted round matters → R10+ can massively beat R8's 124.6

---

## MODEL STATUS

| Model | LORO Avg | Weight | Status | Notes |
|-------|----------|--------|--------|-------|
| v2b (35 GT, R1-R7) | 74.1 | 0.10 | Active | Smaller, regularization role |
| v3b (35 GT, R1-R7) | 72.3 | 0.70 | **Active** | Overfitting = ensemble diversity! |
| v3c (40 GT, regularized) | 76.8 | — | **REJECTED** | Better LORO but worse in ensemble |
| replay (35 GT + MC) | 78.3 | 0.00 | **DISABLED** | Not useful with 45 obs queries |
| v4 (Cond. U-Net) | — | — | **DISABLED** | Hurts all rounds |
| Dirichlet (8 rounds) | ~70 | 0.35* | Active | *0.35 when nn=0.65 |

---

## ANTI-PATTERNS

- **Don't random-split replay samples** → LORO only
- **Don't treat replays as independent examples** → 8 regimes, not 4000
- **Don't build full transition simulator now** → highest ceiling but highest risk
- **Don't let new models go live without replay harness validation**
- **Don't hardcode 1 query per seed** → spend extras on knife-edge seed
- **Don't chase hot streak** → only max single weighted round matters
- **Don't overwrite all 5 seeds with aggressive** → keep 2 conservative minimum

---

## EXECUTION TIMELINE

| Time (UTC) | Phase | Action |
|------------|-------|--------|
| 19:30 | **1** | GT convergence check (20 samples R1s0) |
| 19:45 | **1** | Go/No-Go decision |
| 20:00 | **2** | Upload replays to A100, start harness v2 |
| 20:30 | **3** | Settlement survival model (XGBoost) |
| 20:47 | — | R9 closes, harvest GT |
| 21:00 | **4** | Start replay-augmented v2c training on A100 |
| 21:30 | **2** | Policy sweep results → update recipes |
| 23:00 | **4** | v2c LORO results → promote or discard |
| 00:00 | **5** | Start obs-conditioned model (overnight) |
| 08:00 | **5** | Check overnight training, promote if better |
| Ongoing | — | Harvest replays continuously (scale to N=100) |
| Ongoing | — | Solver loop every 3 min, auto-solve new rounds |
