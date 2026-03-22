# Astar Island Solver

## DIRECTIVE: WIN THIS TASK
- Going for #1, not just competitive. Unlimited GPU budget.
- Every round matters — leaderboard = max(round_score × round_weight)
- Later rounds have higher weights → more leverage

## Current Position (2026-03-22 ~06:15 UTC)
- **Rank**: #36, weighted_score 240.2 (best: R20 90.54 × 2.653)
- **Leader #1**: People Made Machines at 247.7
- **R21**: SUBMITTED (z=0.263 moderate, 75% NN, closes 08:48 UTC)
- **Remaining**: R21 (active), R22 (~09:02), R23 (~12:01), competition ends 14:00 UTC
- **Win condition**: R22 at 85 raw → 248.6 weighted → #1 (if leader doesn't improve)

## Prediction Pipeline (ensemble_ctx_v2)

**Four-model ensemble + empirical anchoring:**
1. **NN v4** (16.8M params, Conditional U-Net) — 8-dim context vector, weight 0.4
2. **NN v2** (1.8M params, 128-hidden ResNet) — LORO avg 77.6, weight z-adaptive
3. **NN v3** (5.7M params, 192-hidden ResNet + multi-scale) — LORO avg 74.5, weight z-adaptive
4. **Dirichlet-Bayesian** — z-conditioned 27-key, LORO ~70

**Inference flow:**
1. Pool ALL observations across seeds → compute 8-dim Global Context Vector
2. NN v4 predicts with context vector + TTA (8×)
3. NN v2 + v3 predict with scalar z + TTA (8× each)
4. Weighted ensemble (v4=0.4, v2/v3=z-adaptive, reduced when v4 present)
5. Geometric mean blend with Dirichlet-Bayesian (NN weight 0.3-0.7 by z deviation)
6. Empirical anchoring: blend with observation counts (7% base, n_obs ≥ 2 only)
7. Floor at 0.01, renormalize

**Performance (training fit):** v4: 93.5, v2: 93.3, v3: 91.8
**Performance (LORO holdout):** v4: R1=83.4, R2=84.2 (partial), v2: avg 77.6, v3: avg 74.5

## Global Context Vector (8-dim)

Replaces scalar z. Computed from grid comparison (initial vs observed cells)
across ALL seeds — NOT from settlement list (which only reports alive ones).

| Dim | Feature | Source |
|-----|---------|--------|
| 0 | Settlement survival rate | initially settled → still settled/port |
| 1 | Port survival rate | initially port → still port |
| 2 | Ruin frequency | ruin cells / land cells |
| 3 | Forest fraction | forest cells / land cells |
| 4 | Collapse rate | 1 - survival rate |
| 5 | Expansion rate | initially empty → settlement/port |
| 6 | Entropy proxy | faction diversity + food level |
| 7 | z (backward compat) | = survival rate |

**CRITICAL**: Context computed from base 45 tiling queries only — precision
queries excluded to avoid biasing toward high-density areas (Codex review).

## Architecture

| File | Purpose |
|------|---------|
| `solver.py` | Main loop: harvest GT → observe → precision queries → predict → submit |
| `strategy.py` | Context vector, empirical anchoring, ensemble blend |
| `nn_predict.py` | Multi-model NN inference (v2+v3+v4) with TTA |
| `calibrate.py` | z-conditioned Dirichlet calibration from GT |
| `train_nn.py` | NN v2 training (A100) |
| `train_nn_v3.py` | NN v3 training with z-augmentation |
| `train_nn_v4.py` | Conditional U-Net v4 training with Global Context Vector |
| `surrogate_sim.py` | Monte Carlo surrogate simulator (WIP) |
| `codex_advisor.py` | GPT-5.4 Codex consultation at key decisions |
| `api.py` | HTTP client for all endpoints |
| `config.py` | Auth token, QUERIES_PER_SEED=10 |
| `state.py` / `state.json` | Persistent state, prevents double-execution |

## Model Files

| File | Size | Description |
|------|------|-------------|
| `astar_nn.pt` | 7MB | NN v2 weights (30 GT files, A100) |
| `astar_nn_v3.pt` | 23MB | NN v3 weights (z-augmented, A100) |
| `astar_nn_v4.pt` | 64MB | Conditional U-Net v4 (8-dim context, A100) |
| `calibration.json` | 12KB | Dirichlet priors + z-model (27 keys) |

## Data

| Data | Count | Location |
|------|-------|----------|
| Ground truth | 100 files (R1-R20 × 5 seeds) | `ground_truth/` |
| Initial grids | 35 files (R1-R7 × 5 seeds) | `observations/round_N/` |
| Observations | R7 onwards | `observations/round_N/` |
| Calibration z values | 20 rounds, R1-R20 | `calibration.json` |

## Scored Rounds

| Round | Score | Weight | Weighted | z | Regime | Strategy |
|-------|-------|--------|----------|------|--------|----------|
| R3 | 7.2 | 1.158 | 8.3 | 0.018 | catastrophic | proximity_v1 |
| R4 | 79.9 | 1.216 | 97.2 | 0.235 | moderate | dirichlet_v3 |
| R5 | 75.4 | 1.276 | 96.2 | 0.330 | moderate | dirichlet_v4 |
| R6 | 58.8 | 1.340 | 78.8 | 0.415 | healthy | dirichlet_v4 |
| R7 | 38.4 | 1.407 | 54.0 | 0.423 | healthy | ensemble_nn_v1 |
| R8 | 84.4 | 1.477 | 124.7 | 0.068 | catastrophic | ensemble_ctx_v2 |
| R9 | 89.1 | 1.551 | 138.3 | 0.275 | moderate | champion_challenger |
| R10 | 86.2 | 1.629 | 140.5 | 0.058 | catastrophic | champion_challenger |
| R11 | 79.9 | 1.710 | 136.7 | 0.499 | healthy | champion_challenger |
| R12 | 29.1 | 1.796 | 52.3 | 0.638 | healthy | champion_challenger |
| R13 | 92.3 | 1.886 | 174.0 | 0.226 | moderate | champion_challenger |
| R14 | 80.0 | 1.980 | 158.4 | 0.464 | healthy | champion_challenger |
| R15 | 90.0 | 2.079 | 187.1 | 0.329 | moderate | champion_challenger |
| R16 | 85.1 | 2.183 | 185.8 | 0.312 | moderate | champion_challenger |
| R17 | 77.9 | 2.292 | 178.5 | 0.454 | healthy | doctrine_v1 |
| R18 | 74.3 | 2.407 | 178.9 | 0.616 | healthy | resubmit_nn |
| R19 | 94.2 | 2.527 | 238.0 | 0.041 | catastrophic | doctrine_v1 |
| R20 | 90.5 | 2.653 | **240.2** | 0.130 | low-moderate | doctrine_v1 (resubmit) |
| R21 | pending | 2.786 | ? | 0.263 | moderate | doctrine_v1 |

## Key Findings

### CRITICAL: Channel Order Bug (fixed 2026-03-20 ~16:00 UTC)
- **nn_predict.py had channels 11-12 SWAPPED vs training script**
- Training: terrain(8) + distance + coastal + is_land + **z** + **density** = 13ch
- Inference (BROKEN): terrain(8) + distance + coastal + is_land + **density** + **z** = 13ch
- **Impact**: ALL NN predictions were garbage at inference time. v2 scored 13 on R7 (should be 68+)
- **Fix**: Rewrote `encode_grid_v2v3()` to inline all channels in correct training order
- **Also fixed v4**: spatial(12 with density) + context(8), not spatial(11) + context(8) + density(1)
- **Result**: v2 on R7 went from 13.2 → 68.7. Full ensemble R7: 53.3 → 74.5 (+21 pts!)
- **All previous NN-blended submissions (R5-R7) were degraded by this bug**

### CRITICAL: Observation Mechanics (corrected 2026-03-20)
- Each `/simulate` call runs the FULL 50-year sim with DIFFERENT random seed
- All observations are **Year 50 final state** — NOT mid-sim
- Different queries = different simulation runs (parallel universes)
- Cannot stitch viewports — but CAN pool statistics across all 45 queries
- Settlement list only reports ALIVE settlements — use GRID comparison instead
- Previous "mid-sim survival unreliable" concern was WRONG

### Codex GPT-5.4 Review (2026-03-20 ~15:00 UTC)
**5 issues found and fixed:**
1. **Seed misattribution bug** — flat observation list broke with uneven counts from precision queries. Fixed: pass `{seed_idx: [obs]}` dict.
2. **Train/inference context mismatch** — training `ctx[0]` uses P(class 1) probability, inference uses binary observed outcome. Systematic shift ~0.06. Partially mitigated by noise injection. **TODO: retrain v4 with inference-compatible context.**
3. **Double-counting observations** — Dirichlet already uses obs counts, empirical anchoring was applying them again. Fixed: reduced to 7% weight, n_obs ≥ 2 only.
4. **v4 weight too high** — R1 LORO underperformed v2 (83.4 vs 87.4). Reduced from 0.5 to 0.4 until full LORO validates.
5. **Precision queries bias context** — extra queries on high-density areas skew global context. Fixed: compute context from base 45 tiling only.

### Nightforce v2 Model (promoted 2026-03-22 ~05:00 UTC)
- **AstarNetNF**: 192-hidden, 8 ResBlocks, 6.4M params, 13 input channels
- Trained on 95 GT files (R1-R19) on A100 with z-jitter=0.02
- LORO avg: **87.49** (best across all experiments)
- Active as `nf2_healthy_all.pt` (swapped from z-jitter=0.08 model, LORO 86.13)
- Backup: `nf2_healthy_all_backup_zj08.pt`
- A100 VPS (XXx--xx-A100) is DOWN — no further retraining possible

### z-Adaptive NN Weight (sweep-optimized)
- z < 0.05: 0% NN (pure Dirichlet) — catastrophic
- z 0.05-0.12: ramp 0→30% NN
- z 0.12-0.25: ramp 30→60% NN
- z 0.25-0.35: 75% NN (peak)
- z > 0.35: 75% NN (flat — sweep found no healthy penalty needed)

### Performance by Regime
- **Catastrophic** (z<0.08): avg 68.0, best 94.2 (R19), improving trend
- **Moderate** (z=0.15-0.35): avg 85.3, best 92.3 (R13), consistent
- **Healthy** (z>0.35): avg 62.6, best 80.0, inconsistent (our weakness)

## Running

- **LOCAL MAC ONLY** — never run solver from server
- **Loop**: `/loop 3m cd ~/www/nm/astar && python solver.py`
- **Manual**: `python solver.py` (safe, checks state.json)
- **Calibrate**: `python calibrate.py` (after new GT harvested)
- **Retrain on A100**: `ssh root@XXx--xx-A100`, workspace at `/astar`

## Training on A100 VPS

```bash
ssh root@XXx--xx-A100
source /astar/venv/bin/activate
cd /astar

# Upload new GT
scp ground_truth/round_*_seed_*.json root@XXx--xx-A100:/astar/ground_truth/
scp calibration.json root@XXx--xx-A100:/astar/

# Train
nohup python3 -u train_nn_v4.py > train_nn_v4.log 2>&1 &
tail -f train_nn_v4.log

# Download model
scp root@XXx--xx-A100:/astar/astar_nn_v4.pt .
```

## Auto-Pipeline (every round)

1. Solver detects completed round → harvests GT via `/analysis`
2. Auto-recalibrates Dirichlet priors (`calibrate()`)
3. Codex consulted on round results and calibration
4. New round detected → observe (9 viewports × 5 seeds = 45 queries)
5. Precision strikes (remaining budget on highest-density viewports)
6. Compute Global Context Vector from base 45 observations
7. Predict with 4-model ensemble + empirical anchoring
8. Submit all 5 seeds
9. **Manual step**: upload new GT to A100, retrain v4, download model, resubmit

## CRITICAL SAFETY RULES

1. **NEVER run from two places simultaneously** — R3 disaster
2. **state.json prevents double-execution** — always check before submitting
3. **50 queries per round** — 9 tiling + 1 precision per seed
4. **Submissions overwrite destructively** — no undo
5. **Floor at 0.01 ALWAYS** — prevents KL → infinity
6. **Unlimited resubmissions** — resubmit with better model anytime during round
7. **Context from base queries only** — exclude precision queries from context vector

## Leaderboard Formula

```
leaderboard_score = max over all rounds (round_score × round_weight)
round_weight = 1.05^round_number
round_score = avg over 5 seeds (100 × exp(-3 × entropy_weighted_kl))
```

Only your BEST single weighted round matters. R3 disaster is irrelevant.

## Next Steps (Final Hours)

1. **Monitor R21-R23** — auto-solver handles everything, just check scores
2. **R22 resubmit** — if regime is favorable, consider NN weight override
3. **Hope for catastrophic/moderate regime** — our model excels there
4. **GPU servers down** — A100 reprovisioned, DataCrunch SSH rejected, no retraining
5. **Production freeze** — no code changes, only weight swaps via config layer
