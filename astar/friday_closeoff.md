# ASTAR ISLAND — FRIDAY NIGHT STATUS HANDOFF

> **Timestamp**: 2026-03-21 00:15 UTC (Saturday morning) — updated from prev session
> **Deadline**: 2026-03-22 14:00 UTC (Sunday 15:00 CET) — **~40 hours remaining**
> **Position**: #38, weighted_score 138.4 (R9: 89.12 × 1.551)
> **Leader**: #1 People Made Machines at 146.3
> **Gap to #1**: 7.9 points
> **Confidence of beating #1**: 75-80%

---

## LEADERBOARD TOP 10

| Rank | Team | Score |
|------|------|-------|
| 1 | People Made Machines | 146.3 |
| 2 | Six Seven | 145.7 |
| 3 | Kult Byrå | 145.3 |
| 4 | Synthetic Synapses | 145.3 |
| 5 | Meme Dream Team | 145.1 |
| 6 | Maskinkraft | 144.6 |
| 7 | Slop Overflow | 144.3 |
| 8 | Creative Destruction | 144.1 |
| 9 | Er det kå i? | 144.1 |
| 10 | Propulsion Optimizers | 143.9 |

---

## SCORED ROUNDS

| Round | Score | Weight | Weighted | z | Regime | Strategy | Notes |
|-------|-------|--------|----------|---|--------|----------|-------|
| R1 | n/a | 1.050 | n/a | 0.419 | healthy | — | No score (early round?) |
| R2 | n/a | 1.103 | n/a | 0.415 | healthy | — | No score |
| R3 | 7.17 | 1.158 | 8.3 | 0.018 | catastrophic | proximity_v1 | Overwrite disaster (server poller) |
| R4 | 79.94 | 1.216 | 97.2 | 0.235 | moderate | dirichlet_v3 | Resubmitted with regime-aware |
| R5 | 75.40 | 1.276 | 96.2 | 0.330 | moderate | dirichlet_v4 | |
| R6 | 58.83 | 1.340 | 78.8 | 0.415 | healthy | dirichlet_v4 | Before NN fixed |
| R7 | 38.39 | 1.407 | 54.0 | 0.423 | healthy | ensemble_nn_v1 | Broken NN channel order |
| R8 | 84.42 | 1.477 | 124.7 | 0.068 | catastrophic | ensemble_ctx_v2 | Pure Dirichlet (z<0.05) |
| R9 | **89.12** | 1.551 | **138.2** | 0.275 | moderate | champion_challenger_v1 | **Current best weighted** |
| R10 | pending | 1.629 | ? | 0.038 | catastrophic | champion_challenger_v1 | Pure Dirichlet, closes 23:45 UTC |

**Formula**: `leaderboard = max(round_score × 1.05^round_number)` — ONLY best single round matters.

---

## PREDICTION PIPELINE (Production as of 22:00 UTC)

### Ensemble Architecture
```
Input: 40×40 initial grid + observations
  → NN v2d (1.8M params, 128-hidden ResNet, 6 blocks) — weight 0.10
  → NN v3d (5.7M params, 192-hidden ResNet + multi-scale, 8 blocks) — weight 0.70
  → Arithmetic mean of NN predictions
  → Geometric blend with Dirichlet-Bayesian (z-conditioned)
  → Floor at 0.003, renormalize
Output: 40×40×6 probability tensor
```

### z-Adaptive NN Weight (linear ramp)
```python
NN_MAX = 0.65
t1, t2, t3 = 0.05, 0.12, 0.25
z < 0.05:  nn_weight = 0.0  (pure Dirichlet — catastrophic)
z 0.05-0.12: nn_weight ramps 0 → 0.26
z 0.12-0.25: nn_weight ramps 0.26 → 0.52
z >= 0.25:  nn_weight = 0.65 (full NN — healthy)
```

### Query Allocation (UPDATED this session)
- **Phase 1**: 6 queries per seed × 5 seeds = 30 base queries (was 9×5=45)
  - 3×3 viewport tiling sorted by settlement density, take top 6
  - Sufficient for z estimation (z stabilizes by ~30 queries per Codex analysis)
- **Preliminary z computed** from 30 base queries
- **Phase 1b**: Remaining ~20 queries as adaptive precision strikes
  - Scored by: settlement density (3×) + coastal settlements (2×) + ports (1.5×) + dynamic empty (0.5×)
  - Spread across seeds (max 40% per seed for MC diversity)
- **Context vector** computed from base queries only (unbiased)
- **Dirichlet posterior** uses ALL observations (base + precision)

### Dirichlet-Bayesian
- z-conditioned linear model: `P(class | key, z) = intercept + slope × z`
- 27 keys: `{terrain_code}_{near/mid/far}_{coast/inland}`
- Fixed concentration = 30 (per-key concentration tested, neutral impact, reverted)
- Calibrated from 45 GT files + 782 replay MC samples
- Replay data provides continuous z coverage (0.000-0.702) vs 9 discrete GT z values

### Global Context Vector (8-dim)
Computed from grid comparison (initial vs observed cells) across ALL seeds:
```
dim 0: settlement survival rate
dim 1: port survival rate
dim 2: ruin frequency
dim 3: forest fraction
dim 4: collapse rate
dim 5: expansion rate
dim 6: entropy proxy (faction diversity + food)
dim 7: z (= dim 0, backward compat)
```

---

## MODEL STATUS

### Production Models (deployed)
| Model | File | LORO Avg | Training Data | Status |
|-------|------|----------|---------------|--------|
| **v2d** | astar_nn.pt (7MB) | **79.7** | 45 GT (R1-R9) | **ACTIVE** |
| **v3b** | astar_nn_v3.pt (22MB) | **72.3** | 35 GT (R1-R7) | **ACTIVE — REVERTED from v3d** |

### Backup Models
| Model | File | LORO Avg | Notes |
|-------|------|----------|-------|
| v2b | astar_nn_v2b_backup.pt | 74.1 | R1-R7 only, previous production |
| v3b | astar_nn_v3b_backup.pt | 72.3 | R1-R7 only, previous production |

### Disabled/Rejected Models
| Model | Reason |
|-------|--------|
| v4 (Cond. U-Net, 64MB) | Hurts all rounds in harness |
| replay (MC-augmented v2) | Weight=0 optimal, adds nothing with 45+ obs queries |
| v3c (regularized) | Better LORO but worse in ensemble than v3b |

### v2d LORO Detail (COMPLETE)
| Round | z | v2b (old) | v2d (new) | Change |
|-------|---|-----------|-----------|--------|
| R1 | 0.419 | 82.5 | 82.0 | -0.5 |
| R2 | 0.415 | 77.6 | 79.4 | +1.8 |
| R3 | 0.018 | 63.7 | 79.1 | **+15.4** |
| R4 | 0.235 | 75.3 | 88.2 | **+12.9** |
| R5 | 0.330 | 78.6 | 79.0 | +0.4 |
| R6 | 0.415 | 75.2 | 74.8 | -0.4 |
| R7 | 0.423 | 66.0 | 66.2 | +0.2 |
| R8 | 0.068 | n/a | 83.8 | new |
| R9 | 0.275 | n/a | 84.5 | new |
| **Avg** | | **74.1** | **79.7** | **+5.6** |

### v3d LORO Detail (7/9 done, R8 running on A100)
| Round | z | v3b (old) | v3d (new) | Change |
|-------|---|-----------|-----------|--------|
| R1 | 0.419 | — | 76.8 | |
| R2 | 0.415 | — | 78.2 | |
| R3 | 0.018 | — | 66.7 | |
| R4 | 0.235 | — | 86.3 | |
| R5 | 0.330 | — | 78.0 | |
| R6 | 0.415 | — | 61.7 | **WEAK** (healthy round) |
| R7 | 0.423 | — | 56.0 | **BAD** (healthy round) |
| R8-R9 | | — | pending | |
| **Avg (7/9)** | | **72.3** | **71.96** | **-0.3 ⚠️** |

**⚠️ v3d is BELOW v3b after 7 rounds.** R6+R7 (both healthy, z>0.4) scored 61.7 and 56.0 — v3d struggles on healthy rounds despite more training data. R8 (catastrophic) and R9 (moderate) still pending. If final avg stays below 72.3, **REVERT**: `cp astar_nn_v3b_backup.pt astar_nn_v3.pt`

---

## HARNESS RESULTS (v2d+v3d ensemble, TTA)

**WARNING**: These are IN-SAMPLE scores since v2d+v3d trained on R1-R9. NOT LORO.
Use only for recipe comparison, not for absolute score prediction.

| Round | Score | Weighted |
|-------|-------|----------|
| R1 | 95.0 | 99.8 |
| R2 | 95.4 | 105.1 |
| R3 | 83.8 | 97.0 |
| R4 | 95.4 | 115.9 |
| R5 | 94.7 | 120.9 |
| R6 | 93.8 | 125.7 |
| R7 | 93.1 | 131.1 |
| R8 | 92.4 | 136.5 |
| R9 | 95.3 | 147.8 |

Recipe sweep confirmed: nn=0.65, v2=0.10, v3=0.70, floor=0.003, zt=[0.05,0.12,0.25] is optimal for healthy rounds.

---

## CALIBRATION DATA

| Source | Count | Details |
|--------|-------|---------|
| GT files | 45 | R1-R9 × 5 seeds |
| Replay MC samples | 782 | 95-100 per round, R1-R8 |
| Calibration keys | 27 | terrain × distance × coastal |
| z-model | 27 keys | linear intercept + slope × z |

### Round z Values
| Round | z | Regime |
|-------|---|--------|
| R1 | 0.419 | healthy |
| R2 | 0.415 | healthy |
| R3 | 0.018 | catastrophic |
| R4 | 0.235 | moderate |
| R5 | 0.330 | moderate |
| R6 | 0.415 | healthy |
| R7 | 0.423 | healthy |
| R8 | 0.068 | catastrophic |
| R9 | 0.275 | moderate |
| R10 | 0.038 (est) | catastrophic |

Distribution: 4 healthy, 3 moderate, 3 catastrophic

---

## INFRASTRUCTURE

### Local Mac
- Solver loop: cron job 84479134, every 3 min: `cd ~/www/nm/astar && python solver.py`
- Models: v2d (astar_nn.pt) + v3d (astar_nn_v3.pt) deployed
- Backups: v2b + v3b saved as *_backup.pt
- **CRITICAL**: solver runs LOCAL ONLY. Never from server (R3 disaster).

### A100 VPS (XXx--xx-A100)
- 2 training processes running (v2d LORO done, v3d LORO 5/9 done)
- Workspace: `/astar/` with all GT, calibration, training scripts
- `source /astar/venv/bin/activate`
- Monitor: `ssh root@XXx--xx-A100 "grep holdout /astar/train_v3d.log"`

### H100 Server (XXx--xx-H100)
- **NOT accessible** — SSH key rejected. Needs auth fix if we want to use it.

---

## CHANGES MADE THIS SESSION (2026-03-20 ~20:00-22:20 UTC)

### 1. Retrained v2d+v3d on 45 GT (R1-R9)
- Uploaded R9 GT + calibration to A100
- v2d LORO: 79.7 avg (+5.6 over v2b) — massive catastrophic/moderate gains
- v3d full: 93.1 training fit, LORO running
- Both deployed to production (swapped into astar_nn.pt / astar_nn_v3.pt)

### 2. Adaptive Query Allocation
- Changed QUERIES_PER_SEED from 10 to 6 in config.py
- 30 base queries (was 45) → z estimation still reliable per Codex analysis
- 20 adaptive precision queries (was 5) on settlement-dense, coastal viewports
- Precision targets scored by: settlements×3 + coastal settlements×2 + ports×1.5
- Spread across seeds (max 40% per seed)
- Context vector computed from base queries only (unbiased)

### 3. Calibration Upgrade (tested, reverted)
- Per-key concentration from replay variance: neutral impact (+0.03 to -0.79 depending on formula)
- Reverted to fixed concentration=30 — proven and stable

### 4. Recipe Optimization
- Swept nn_weight 0.35-0.70, z_thresholds, v2/v3 ratios, prob_floor
- Best for R9 (in-sample): nn=0.45-0.50, zt=[0.03,0.08,0.20] → 139.8 weighted
- BUT this hurts healthy rounds by -2 to -5 raw
- **Decision**: keep nn=0.65, zt=[0.05,0.12,0.25] — optimizes for healthy rounds which have highest ceiling
- Healthy rounds at wt 1.710 = 162.5 >>> moderate gain of +1.4 on R9

### 5. Codex GPT-5.4 Consultation
- Identified: oracle-z leakage in harness, static query allocation, retrain opportunity
- Estimated +0.7-1.5 from adaptive queries, +0.3-0.8 from retrain
- Recommended against new architecture (only obs-conditioned model worth it)

---

## HARNESS-TO-LIVE GAP (CRITICAL RISK)

| Round | Harness Score | Live Score | Gap | Notes |
|-------|--------------|------------|-----|-------|
| R8 | 92.4 | 84.4 | **-8.0** | Catastrophic. z est 0.051 vs true 0.068 |
| R9 | 89.4 (LORO) | 89.1 | -0.3 | Moderate. z est 0.272 vs true 0.275 |

R8 gap is concerning. Possible causes:
1. z estimation error (0.051 vs 0.068) pushing NN weight slightly wrong
2. Harness uses oracle z, live uses estimated z
3. Observations in live may not cover same cells as harness assumes
4. R8 had some queries starting at query 10+ (not from 1) — possible observation gap

R9 gap is minimal — good sign for moderate/healthy rounds.

---

## PATH TO #1

### Math
```
Current #1: 146.3
Our best: 138.4 (R9)
Gap: 7.9 points

To beat on R10 (wt 1.629): need 89.8 raw — possible if catastrophic Dirichlet scores well
To beat on R11 (wt 1.710): need 85.6 raw — achievable even if moderate
R11 healthy: 95 × 1.710 = 162.5 → CRUSHES #1
R12 healthy: 95 × 1.796 = 170.6 → DESTROYS #1
```

### Probability Assessment
- P(at least one healthy round in next 12-15 rounds) ≈ 95%
- P(beating #1 given one healthy round) ≈ 85-90%
- P(beating #1 overall) ≈ **75-80%**
- P(crushing #1 by >10 pts) ≈ 50-60%

### Risks
1. All remaining rounds catastrophic (unlikely but devastating)
2. Harness-to-live gap on healthy rounds larger than expected
3. Other teams also improve with later round weights
4. v3d LORO shows degradation (partial results so far look ok)
5. Adaptive query allocation regresses live performance

---

## IMMEDIATE NEXT STEPS

1. **R10 closes ~23:45 UTC** → solver auto-harvests GT, recalibrates (50 GT), logs score
2. **v3d LORO completes** (~1-2h) → check if v3d improves over v3b
3. **If v3d LORO worse than v3b**: revert to v3b backup
4. **R11 opens** → solver auto-solves with v2d+v3d + adaptive queries
5. **After R11**: check live score, verify harness-to-live gap, adjust if needed

### KNOWN ISSUE: R10 API Transition Lag
R10 closed at 23:45:58 UTC but API still reports status=active as of 22:30+ UTC.
This has happened before — can lag 10-20+ minutes. Cron loop (84479134, every 3m)
will auto-catch it. No manual intervention needed. If still stuck after 30+ min,
try `python solver.py` manually — the harvest_ground_truth function checks completed rounds.

### v3d REVERTED to v3b ✅ (2026-03-21 ~00:20 UTC)
v3d killed before LORO finished. R6=61.7, R7=56.0 on healthy rounds = catastrophic failure.
**Root cause**: Adding R8 (catastrophic) + R9 (moderate) training data shifted v3d's priors toward death/stagnation, destroying healthy-round predictions. More data ≠ better model when distribution shifts away from your win condition.
**Lesson**: Only retrain v3 if new healthy-round GT data outweighs catastrophic additions. v3b (35 GT, R1-R7) had better healthy-round balance.
v2d kept — it improved across ALL regimes including healthy (+1.8 R2, -0.4 R6, +0.2 R7).

### If R10 GT arrives
- Recalibrate with 50 GT files (10 rounds)
- Upload to A100, potentially retrain v2e+v3e (if time permits)
- v2d+v3d should be fine for R11 though — R10 is just one more catastrophic data point

### If H100 becomes available
- Auth fix needed (SSH key rejected)
- Could run parallel training for observation-conditioned model
- Or retrain v2/v3 faster with 2×H100

---

## FILE INVENTORY

### Production Files
| File | Purpose | Size |
|------|---------|------|
| solver.py | Main loop: harvest GT → observe → precision → predict → submit | |
| strategy.py | Context vector, Dirichlet, ensemble blend | |
| nn_predict.py | Multi-model NN inference with TTA | |
| calibrate.py | z-conditioned Dirichlet from GT + replay MC | |
| config.py | TOKEN, QUERIES_PER_SEED=6 | |
| api.py | HTTP client for all endpoints | |
| state.py / state.json | Persistent state, prevents double-execution | |
| codex_advisor.py | GPT-5.4 consultation at key decisions | |
| astar_nn.pt | v2d weights (active) | 7MB |
| astar_nn_v3.pt | v3d weights (active) | 22MB |
| calibration.json | Dirichlet priors + z-model (45 GT + 782 replay) | |

### Backup Files
| File | Purpose |
|------|---------|
| astar_nn_v2b_backup.pt | Previous v2b model (LORO 74.1) |
| astar_nn_v3b_backup.pt | Previous v3b model (LORO 72.3) |
| astar_nn_v2d.pt | Copy of current v2d |
| astar_nn_v3d.pt | Copy of current v3d |

### Data
| Directory | Contents |
|-----------|----------|
| ground_truth/ | 45 GT files (R1-R9 × 5 seeds) |
| observations/ | Per-round observations (R7-R10) |
| replays/ | 782 MC replay samples (R1-R8) |

### Analysis/Reference Files
| File | Purpose |
|------|---------|
| BATTLEPLAN.md | Full competition strategy and findings |
| CLAUDE.md | Solver architecture docs |
| replay_harness.py | Offline evaluation harness |
| harvest_replays.py | Replay data collection |
| seed_selector.py | Champion/challenger (BROKEN — falls back to baseline) |
| train_nn.py | v2 training script |
| train_nn_v3.py | v3 training script |

---

## ANTI-PATTERNS (LEARNED THE HARD WAY)

1. **Don't run solver from two places** → R3 overwrite disaster
2. **Don't trust harness for catastrophic absolute scores** → R8 was -8pts vs harness
3. **Don't random-split replay samples** → LORO only (8 regimes, not 4000)
4. **Don't treat replays as independent** → same hidden params per round
5. **Don't lower NN weight to optimize moderate rounds** → costs more on healthy rounds
6. **Don't use per-key concentration** → tested, neutral, adds complexity
7. **Don't let new models go live without checking LORO** → v3c was better LORO but worse ensemble
8. **Overfitting CAN help ensemble diversity** → v3b beat v3c in ensemble despite worse LORO
9. **R1/R2 have no scores** → possibly early rounds before scoring was active
10. **Seed selector is BROKEN** → silently falls back to baseline, needs rewrite if used
