# Astar Island Solver

## Architecture
- `api.py` — API client for all endpoints
- `strategy.py` — Dirichlet-Bayesian prediction engine
- `solver.py` — Main solver: harvest GT → observe → predict → submit (stateful, safe)
- `calibrate.py` — Learn priors from ground truth data
- `state.py` / `state.json` — Persistent state: tracks solved rounds, prevents double-execution
- `config.py` — Auth token, settings
- `ground_truth/` — Cached ground truth from analysis endpoint

## Running
- **LOCAL MAC ONLY** — never run from H100 server. Mac stays on 24/7.
- **Monitoring**: Use `/loop 3m` to run `cd ~/www/nm/astar && python solver.py` every 3 minutes
- **Dry run**: `python solver.py --dry-run` (no API calls, logs what would happen)
- **Manual solve**: `python solver.py` (checks state.json, skips if already solved)
- **Calibrate**: `python calibrate.py` (after new ground truth is harvested)

## CRITICAL SAFETY RULES

### 1. NEVER run from two places simultaneously
Round 3 was ruined by running locally AND on server. The solver now tracks state
in state.json and refuses to re-solve a round that's already been solved.

### 2. NEVER overwrite without checking
state.json records every solved round with queries_used and seeds_submitted.
The solver checks this BEFORE doing anything.

### 3. API budget is FINITE and PRECIOUS
50 queries per round across 5 seeds. Every query counts.
The solver allocates queries evenly (10/seed) and prioritizes dynamic areas.

### 4. Submissions overwrite destructively
POST /submit replaces previous prediction. No undo.
Only submit if we have observations to back it up.

### 5. Floor at 0.01 ALWAYS
Never assign probability 0.0 to any class. KL divergence → infinity.

### 6. Deploy changes carefully
After code changes: test with `--dry-run` first. Then let /loop pick it up naturally.

## Strategy: Dirichlet-Bayesian

For each cell:
1. Compute prior alpha from (terrain_type, settlement_distance, is_coastal)
2. If calibration.json exists (learned from ground truth), use calibrated priors
3. Add observation counts to prior → posterior
4. Normalize posterior → prediction
5. Floor at 0.01, renormalize

## Ground Truth Pipeline
1. After each round completes, solver automatically harvests via GET /analysis
2. Saves to ground_truth/round_N_seed_M.json
3. Run `python calibrate.py` to recompute priors from all available GT
4. calibration.json is automatically loaded by strategy.py

## Round Timing
- Rounds happen ~every 3 hours (observed from rounds 1-3)
- Prediction window: ~165 min (2h45m)
- Round weights increase: 1.05 → 1.10 → 1.16 → ...

## Leaderboard Context
- #1 Algebros: 98.8 — requires weighted_kl ≈ 0.004 (near perfect)
- Only ~100-200 cells out of 1600 are truly dynamic (scored by entropy weighting)
- Key: get settlement/port/ruin transitions right. Static cells don't matter.
- Analysis endpoint is the cheat code: gives exact ground truth for past rounds

## Files
| File | Purpose |
|------|---------|
| `solver.py` | Main entry: `check_and_solve()` — harvest GT, check active round, solve |
| `strategy.py` | Dirichlet prediction: `predict_for_seed(grid, observations)` |
| `calibrate.py` | Learn priors from ground truth: `calibrate()` |
| `state.py` | Persistent state management |
| `state.json` | Solved rounds, queries used, strategies |
| `calibration.json` | Learned Dirichlet priors (auto-loaded by strategy.py) |
| `config.py` | Auth token (JWT expires ~2026-03-24) |
| `api.py` | Full API client |
| `BATTLEPLAN.md` | Complete strategy document |
| `ground_truth/` | Cached analysis endpoint data |
