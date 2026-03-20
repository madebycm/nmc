# REPLAY DATA PIVOT — Discovery Report

> **Created**: 2026-03-20 18:56 UTC
> **Status**: PROPOSED — awaiting review before execution
> **Impact**: Potentially game-changing — unlimited ground truth generation

---

## Discovery: POST /replay Endpoint

The competition platform has a **replay API** that returns the **full 51-frame simulation** for any completed round. This is NOT documented prominently but is used by the frontend replay viewer at `https://app.ainm.no/submit/astar-island/replay`.

### API Details

```
POST https://api.ainm.no/astar-island/replay
Body: {"round_id": "<uuid>", "seed_index": 0-4}
Auth: Bearer token (standard)

Response: {
    "round_id": "<uuid>",
    "seed_index": 0,
    "sim_seed": 2139592963,      // RANDOM — different each call!
    "width": 40, "height": 40,
    "frames": [                   // 51 frames (step 0 through 50)
        {
            "step": 0,
            "grid": [[10, 10, ...], ...],   // 40x40 terrain codes
            "settlements": [
                {
                    "x": 3, "y": 30,
                    "population": 1.43,
                    "food": 0.712,
                    "wealth": 0.311,
                    "defense": 0.545,
                    "has_port": false,
                    "alive": true,
                    "owner_id": 0
                }, ...
            ]
        },
        ...  // frames 1-50
    ]
}
```

### Critical Properties

| Property | Finding | Implication |
|----------|---------|-------------|
| **sim_seed varies** | Each call returns a DIFFERENT random simulation | Unlimited Monte Carlo samples |
| **No query cost** | Budget unchanged after 20+ calls | Free data |
| **Completed rounds only** | Works on R1-R8 (all completed) | Cannot use on active rounds |
| **All seeds** | Works for seed_index 0-4 | 5 initial configs per round = 40 unique maps |
| **Rate limit** | ~0.7 req/s sustained (~6 before throttle, then ~1/2s) | ~2,500/hour, ~42/min |
| **Deterministic per call** | Same sim_seed → same outcome, but seed differs per call | Each call = independent sample |

### Rate Limit Details

- Burst: ~6 requests without delay
- Sustained: ~0.7 req/s (add 1.5s sleep between calls to be safe)
- Rate limit response: `{"detail": "Rate limit exceeded. Try again shortly."}`
- Recovery: ~2-3 seconds after hitting limit
- **Estimate**: ~2,500 replays/hour, ~60,000/day

---

## What This Data Contains (vs What We Had)

### Before: GT Files (40 files)
- Initial grid (40x40 terrain codes)
- Final probability distribution (40x40x6 class probs)
- One file per round per seed
- Probabilities = average over MANY simulations

### After: Replay Frames (unlimited)
- **51 time steps** per simulation (year 0 through year 50)
- **Exact grid state** at every step (terrain codes)
- **Settlement attributes**: population, food, wealth, defense, has_port, alive, owner_id
- **Different random outcome** each call — can sample distribution directly
- **1247/1600 cells are probabilistic** in GT — meaning there IS real variance to capture

### Settlement Data Per Frame
```
population: float  (e.g. 1.43)
food: float        (e.g. 0.712)
wealth: float      (e.g. 0.311)
defense: float     (e.g. 0.545)
has_port: bool
alive: bool
owner_id: int      (faction index)
```

This is the INTERNAL STATE of the simulator. We can learn:
- What population/food/wealth levels predict survival
- How settlements expand (land→settlement transitions)
- Port construction conditions
- Ruin creation dynamics
- Forest regeneration patterns

---

## Observed Dynamics (R7 healthy vs R8 catastrophic)

### R7 seed0 (z=0.423, healthy)
- Settlements: 44 → 220 (5x expansion!)
- Top transitions: land→settlement (133), forest→settlement (48), land→forest (24)
- Settlements thriving, building ports, expanding into forest

### R8 seed0 (z=0.068, catastrophic)
- Settlements: 60 → 17 (72% collapse)
- Top transitions: land→forest (42), settlement→land (36), forest→land (31)
- Mass die-off, forest reclaiming cleared land

### Step-by-step dynamics (R8 catastrophic)
| Step | Settlements | Alive | Ports | Avg Pop | Avg Food |
|------|------------|-------|-------|---------|----------|
| 0 | 60 | 60 | 4 | 0.99 | 0.556 |
| 10 | 31 | 31 | 2 | 1.32 | 0.704 |
| 25 | 20 | 20 | 2 | 1.22 | 0.643 |
| 40 | 29 | 29 | 1 | 1.16 | 0.547 |
| 50 | 24 | 24 | 1 | 1.19 | 0.632 |

---

## Data Harvesting Plan

### Phase 1: Fetch & Store Locally (10 min)

```python
# harvest_replays.py — run on Mac (lightweight HTTP only)
# Store in replays/ directory, one JSON per round/seed/sample

replays/
  round_1/
    seed_0/
      sample_000.json  # {sim_seed, frames: [{step, grid, settlements}]}
      sample_001.json
      ...
    seed_1/
    ...
  round_8/
    ...
```

**Fetching strategy** (respecting rate limits):
- 1.5s delay between calls
- 8 rounds × 5 seeds × N samples per config
- N=50 samples per config → 2,000 calls → ~50 min
- N=100 samples per config → 4,000 calls → ~100 min
- Start with N=20 (~13 min) for quick validation, then scale up

### Phase 2: Upload to A100 VPS

```bash
# Compress and upload
tar czf replays.tar.gz replays/
scp replays.tar.gz root@135.181.8.209:/astar/
ssh root@135.181.8.209 "cd /astar && tar xzf replays.tar.gz"
```

### Phase 3: Train on A100 GPU

Multiple training approaches become possible (see next section).

---

## Training Approaches Unlocked

### Approach A: Monte Carlo GT Estimation (Easiest, Highest Confidence)

Average N replay final frames → empirical probability distribution per cell.
This IS the ground truth, just estimated from samples.

```python
# For each round/seed, average 100 replay final grids
# Each grid cell → count terrain types → normalize → probability
# Compare to API ground_truth to verify convergence
```

**Why this matters**: With 100 samples we get much tighter GT estimates than the
competition's built-in GT (which may use fewer samples). Better GT = better calibration.

**Validation**: Compare our Monte Carlo GT to API GT. If they converge, we know
the exact sample count the competition uses internally.

### Approach B: Transition Model (Medium, High Impact)

Learn P(cell_state at step t+1 | neighborhood at step t, settlement attributes).

```python
# From 51 frames, extract 50 transition pairs per cell per replay
# Features: 3x3 neighborhood, settlement proximity, population, food, wealth
# Target: next-step terrain code
# Training data: 50 steps × 1600 cells × N_samples × 40 configs = MASSIVE
```

With transition probabilities, we can:
1. Run our own Monte Carlo forward simulation from any initial state
2. Estimate final distributions without the API
3. Understand exactly which features drive outcomes

### Approach C: Direct Final-State Prediction with Augmented Training (Easiest NN upgrade)

Current NN training: 40 GT files (initial_grid → final_probs).
New: Each replay gives a different final grid. Convert to soft targets:
- N=100 replays → average → soft probability targets (same as GT but more samples)
- OR use each replay as a separate training example with hard labels
- 100 replays × 40 configs = 4,000 training examples (vs 40 currently!)

### Approach D: Trajectory-Conditioned Prediction (Advanced)

Use intermediate frames as conditioning signal.
During live rounds, our /simulate observations ARE intermediate samples.
If we can match an observation pattern to a trajectory cluster, we know
which final-state distribution to predict.

### Approach E: Settlement Survival Model (Quick Win)

From settlement attributes at step 0, predict alive/dead at step 50.
Simple tabular model (XGBoost/logistic) on (population, food, wealth, defense, has_port, position).
Directly improves our z estimation and per-cell predictions.

---

## Priority Recommendation

| Priority | Approach | Effort | Impact | Time |
|----------|----------|--------|--------|------|
| **1** | **A: Monte Carlo GT** | Low | High | 1h fetch + 30min validate |
| **2** | **C: Augmented NN training** | Medium | Very High | 2h (after data fetched) |
| **3** | **E: Settlement survival** | Low | Medium | 1h |
| **4** | **B: Transition model** | High | Very High | 4-6h |
| **5** | **D: Trajectory conditioning** | Very High | Highest ceiling | 8h+ |

**Recommended immediate action**:
1. Start harvesting replays NOW (runs on Mac, just HTTP calls with 1.5s delays)
2. While fetching, implement Approach A validator on A100
3. Train augmented NN (Approach C) with replay data
4. Have replay-grounded model ready before R9 opens (~2h from now)

---

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| Rate limiting blocks harvest | 1.5s delay, retry with backoff, ~2500/hr is enough |
| Replay data doesn't match GT | Validate Monte Carlo avg vs API GT first — STOP if diverges |
| Training takes too long | Start with Approach A (just better calibration, no training) |
| R9 opens before model ready | Keep current optimized model as fallback (121.0 backtest) |
| API changes/blocks replay | Fetch ALL data upfront, store permanently |
| Overfitting to 8 map configs | Use replay variance (different sim seeds) as regularization |

---

## Current Backtest vs Potential

| Model | Best Weighted (backtest) | Notes |
|-------|------------------------|-------|
| Current (replay-harness optimized) | 121.0 | v2b+v3b, no v4, no anchor |
| + Monte Carlo calibration (A) | ~123-125 est | Better priors from exact distributions |
| + Augmented NN training (C) | ~127-130 est | 100x more training data |
| + Transition model (B) | ~130-135 est | Understanding the actual simulator |

To win: need 76.5+ on R9 (weight 1.551) → 118.7 weighted.
Current backtest already hits 121.0. With replay data, ceiling is much higher.

---

## Immediate Next Steps (if approved)

1. `python harvest_replays.py` — fetch 20 samples per config (13 min)
2. Validate Monte Carlo GT matches API GT
3. Upload to A100, start Approach C training
4. Keep solver loop running for R9 detection
5. Scale to 100 samples per config in background (~50 min)
