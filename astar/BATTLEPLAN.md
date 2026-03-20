# ASTAR ISLAND BATTLEPLAN

> **Goal**: Beat #1 score of 98.79 (Algebros). Currently 0 scored rounds.
> **Confidence**: HIGH — the path to 95+ is clear, 98+ requires calibration from ground truth.

## Why We Can Win

1. **Analysis endpoint is the cheat code** — after each round completes, we get the EXACT ground truth (H×W×6 probability tensors from hundreds of Monte Carlo runs). Top teams use this to calibrate.
2. **Round weights increase** — later rounds worth more. A round-10 score of 98 beats a round-1 score of 99.
3. **Unlimited resubmissions** — iterate within each round.
4. **69-hour competition** — enough time for ~20+ rounds of learning.

## The Math: What 98.8 Requires

```
score = 100 × exp(-3 × weighted_kl)
98.8 → weighted_kl ≈ 0.004
```

This is near-perfect. But most cells are STATIC:
- Ocean/Mountain: entropy=0, excluded from scoring entirely
- Forest: entropy≈0.2, mostly stays forest
- Plains far from settlements: entropy≈0.1

**Only ~100-200 cells out of 1600 are truly dynamic** (settlements, ports, ruins, nearby plains).
Getting those right = winning.

## Three-Phase Strategy

### Phase 1: Ground Truth Harvesting (IMMEDIATE)
Round 3 is active now. When it completes:
1. Download ground truth via `GET /analysis/{round_id}/{seed_index}` for all 5 seeds
2. Also get rounds 1 & 2 ground truth (we didn't submit, but analysis may still work — test this)
3. From ground truth, learn empirical transition probabilities:
   - P(final_class | initial_class, distance_to_settlement, is_coastal, neighbor_types)
   - These become our **calibrated Dirichlet priors**

### Phase 2: Dirichlet-Bayesian Prediction Engine (BUILD NOW)
Replace naive observation blending with principled Bayesian approach:

**For each cell:**
```python
# Step 1: Set informative prior from initial state + learned features
alpha = get_prior(initial_class, distance_to_settlement, is_coastal, ...)

# Step 2: Update with observation counts
posterior = alpha + observation_counts

# Step 3: Predict = normalized posterior
prediction = posterior / posterior.sum()

# Step 4: Floor at 0.01, renormalize
prediction = max(prediction, 0.01)
prediction = prediction / prediction.sum()
```

**Prior calibration (before ground truth data):**
| Initial Type | Prior α (6 classes: Empty, Settle, Port, Ruin, Forest, Mountain) |
|-------------|------------------------------------------------------------------|
| Ocean | [100, 0.01, 0.01, 0.01, 0.01, 0.01] |
| Mountain | [0.01, 0.01, 0.01, 0.01, 0.01, 100] |
| Forest | [0.5, 0.1, 0.05, 0.2, 20, 0.05] |
| Plains (far) | [20, 0.3, 0.1, 0.2, 1.0, 0.05] |
| Plains (near settlement) | [5, 2, 0.5, 1.5, 0.5, 0.05] |
| Settlement | [0.5, 5, 2, 3, 0.3, 0.05] |
| Port | [0.3, 1, 6, 2, 0.2, 0.05] |
| Ruin | [3, 1, 0.5, 3, 3, 0.05] |

**After ground truth calibration:** Replace hand-tuned priors with empirically learned ones.

### Phase 3: Smart Query Allocation
Current: naive 3×3 tiling (9 queries per seed, 45 total)

**Better:**
1. From initial state, identify dynamic zones (within 5 cells of any settlement)
2. Place viewports to cover ALL dynamic zones
3. Skip ocean/mountain-dominated areas entirely
4. Use remaining queries to RE-OBSERVE highest-entropy areas
5. Cross-seed pooling: observations from all seeds inform transition probability estimates

**Coverage math:**
- 40×40 = 1600 cells
- ~200-400 are "dynamic" (near settlements)
- These cluster around 8-12 settlement positions
- 5-6 well-placed 15×15 viewports can cover all dynamic zones
- Remaining 4-5 queries: repeat observations on settlement-dense areas

### Phase 4: Cross-Seed Transfer
All 5 seeds share SAME hidden parameters (expansion rate, raid aggression, winter severity).
After observing all seeds:
1. Pool transition statistics across seeds
2. Estimate shared parameters:
   - expansion_rate = new_settlements / initial_settlements
   - collapse_rate = ruins / initial_settlements
   - port_rate = ports / coastal_settlements
3. Apply to predictions for all seeds

### Phase 5: Iterative Calibration Loop
```
Round N completes → get ground truth → calibrate priors → better predictions for Round N+1
```
Each round gives us 5 × 40 × 40 = 8000 cell-level training examples.
After 3 rounds: 24,000 examples. This is MORE than enough to learn accurate transition probabilities.

## File Architecture

```
astar/
├── BATTLEPLAN.md          # This file
├── CLAUDE.md              # Project rules and safety
├── api.py                 # API client
├── config.py              # Auth token, settings
├── solver.py              # Main solver: poll → observe → predict → submit
├── strategy.py            # Prediction strategies (Dirichlet, feature-based)
├── calibrate.py           # Learn priors from ground truth data
├── state.json             # Persistent state: solved rounds, budgets, scores
└── ground_truth/          # Cached ground truth from analysis endpoint
    ├── round_1_seed_0.json
    ├── round_1_seed_1.json
    └── ...
```

## Safety Protocols

1. **state.json tracks everything** — rounds solved, queries used, predictions submitted
2. **Dry-run mode** — test locally before any API call
3. **Never overwrite better predictions** — compare observation count before resubmitting
4. **Single execution point** — ONLY the local Mac runs the solver (not the H100 server)
5. **Budget guard** — refuse to solve if insufficient queries for improvement

## Scoring Targets

| Round | Strategy | Expected Score |
|-------|----------|----------------|
| 3 (done) | 1 observation + naive priors | 20-50 (damaged by overwrite) |
| 4 | Dirichlet priors + full coverage | 50-70 |
| 5 | + ground truth calibration from round 3 | 70-85 |
| 6-7 | + cross-seed + smart viewports | 80-90 |
| 8+ | + feature model from 3+ rounds GT | 90-98 |

## Immediate TODO

1. [x] Write BATTLEPLAN.md
2. [ ] Add persistent state.json tracking
3. [ ] Implement Dirichlet posterior strategy
4. [ ] Add ground truth harvesting (calibrate.py)
5. [ ] Smart viewport placement
6. [ ] Set up /loop for local Mac polling
7. [ ] Wait for round 3 to complete → harvest ground truth
8. [ ] Calibrate priors from ground truth
9. [ ] Deploy improved strategy for round 4
