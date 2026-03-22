# ASTAR OVERNIGHT LOG — 2026-03-21

## Setup (23:00 UTC / 00:00 CET)
- **Cron loop**: job 84479134, every 3 min — `cd ~/www/nm/astar && python solver.py`
- **H100 training**: v2e + v3e running at XXx--xx-H100:/astar/
- **Production models**: v2d (LORO 79.7) + v3b (LORO 72.3)
- **v3d reverted** to v3b — healthy rounds R6=61.7, R7=56.0 unacceptable
- **R10**: still active, waiting for API transition (catastrophic, z≈0.038)

## H100 Training Jobs
| Job | Script | Output | Status |
|-----|--------|--------|--------|
| v2e | train_nn.py (45 GT) | train_v2e.log | RUNNING |
| v3e | train_v3_nozaug.py (45 GT, no z-aug) | train_v3e.log | QUEUED |

## Decision Rules for Morning
- v2e LORO > 79.7 → deploy as astar_nn.pt
- v3e LORO > 72.3 AND no healthy fold < 60 → deploy as astar_nn_v3.pt
- Otherwise → keep v2d + v3b

## Event Log
| Time (UTC) | Event |
|------------|-------|
| 23:00 | Session started, v3d reverted to v3b |
| 23:02 | H100 overnight training launched (v2e started) |
| 23:30 | R10 transitioned — "No active round". GT not yet harvestable (status not "completed" yet) |
| 00:54 | **R10 GT harvested!** 50 GT files now. R10 score: **86.25 avg** (85.4-87.5). Weighted: **140.5** — NEW BEST! |
| 00:54 | Recalibrated with 50 GT + 782 replay. R10 z=0.058 (catastrophic). Uploaded new calibration to H100. |
| 00:54 | Between rounds. Waiting for R11. |
| 00:55 | **R10 API confirmed**: 86.25 avg, weighted 140.5. Seeds: 85.4, 86.3, 85.5, 87.5, 86.5. New best! |
| 01:00 | **v2e LORO complete**: 79.6 avg (v2d=79.7). No improvement. Keep v2d. v3e training started. |
| 01:00 | **#1 updated to 152.1**. Gap now 11.6 pts. Need R11 ≥ 89 × 1.710 = 152.2 to beat. |
| 01:31 | **R11 OPEN — HEALTHY z=0.521!** All 5 seeds submitted. NN weight=0.65. Closes 03:11 UTC. |
| 01:31 | Leaderboard: #1 Maskinkraft 152.1, #2 Meme Dream 152.0, #3 Synthetic Synapses 151.5 |
| 02:10 | **v3e LORO complete: 79.6 avg** (+7.3 over v3b). No z-aug confirmed as fix. Healthy: R6=71.7, R7=66.7 (vs v3d 61.7, 56.0). |
| 02:10 | **DECISION**: v3e deployed for R12+ but NOT resubmitting R11 (overwrite guard). v3b backup at v3b_backup2.pt. |
| 02:21 | v3e downloaded from H100, deployed as astar_nn_v3.pt. R12 will use v2d+v3e automatically. |
| 04:21 | **R11 GT harvested**: avg 79.94, weighted 136.7. BELOW R10 (140.5) and #1 (152.1). |
| 04:21 | Seeds: 82.5, 83.5, 73.9, 79.6, 80.4. Seed 2 weak (73.9). z=0.499 (healthy). |
| 04:21 | Harness-to-live gap on healthy round confirmed: expected ~88-90, got 79.9. Gap ~9 pts. |
| 04:21 | Recalibrated with 55 GT (11 rounds). Uploaded. Between rounds, waiting for R12. |
| 04:31 | **R12 OPEN — HEALTHY z=0.638!** First round with v3e. All 5 seeds submitted. Closes 06:06 UTC. |
| 04:31 | **Leaderboard exploded**: #1 Six Seven 158.1, #2 People Made Machines 157.8. Everyone scored R11 well. |
| 04:31 | Need R12 ~89 × 1.796 = 159.8 to take #1. R12 weight is huge. |
| 07:00 | **R12 DISASTER: 29.1 avg** (26.9, 34.0, 26.3, 28.6, 29.8). Weighted 52.3. v3e first live = catastrophic failure. |
| 07:00 | z=0.638 (healthy) but scored 29. Something fundamentally broken with v3e live inference. |
| 07:00 | **IMMEDIATE ACTION: investigate v3e vs v3b. May need to revert v3e.** |
| 07:05 | **ROOT CAUSE**: v3e trained z range [0.018-0.499]. R12 z=0.638 is OUT OF DISTRIBUTION. |
| 07:05 | v3e score on R12 S0: 15.8 vs v3b: 35.5. z-augmentation was needed for extrapolation! |
| 07:05 | **REVERTED to v3b.** Lesson: z-aug fixes OOD extrapolation. Need z-aug but only upward. |
| 07:05 | Neither v3b nor v3e are good at z=0.638 (both low). Fundamental coverage gap. |
| 07:21 | **R13 solved**: moderate z=0.215, NN weight=0.45. v3b (reverted). Closes 09:02 UTC. Weight 1.886. |
| 07:21 | If R13 ~84 × 1.886 = 158.5 → beats #1 (158.1). Moderate rounds are our strength (R9=89.1). |
| 07:21 | Leaderboard unchanged: #1 Six Seven 158.1 |
| 10:11 | **R13 GT: 92.28 avg × 1.886 = 174.1 WEIGHTED!!!** |
| 10:11 | Seeds: 93.2, 92.2, 91.3, 92.3, 92.4. ALL above 91! |
| 10:11 | **CRUSHES #1 (158.1) BY 16 POINTS. NEW #1 INCOMING.** |
| 10:11 | Moderate z=0.226, NN weight=0.45, v2d+v3b ensemble. The sweet spot. |
| 10:11 | Recalibrated with 65 GT (13 rounds). Between rounds, waiting for R14. |
| 10:21 | **R14 auto-solved**: healthy z=0.464, NN weight=0.65, v2d+v3b. Closes 11:59 UTC. Weight 1.980. |
| 10:22 | **Leaderboard exploded on R13**: #1 Matriks 177.1, #2 Laurbærene 176.7. Our 174.1 NOT enough for #1. |
| 10:22 | Everyone scored well on R13 moderate round. Need R14 to score higher than competition. |
| 10:22 | R14 at weight 1.980: need ~90 raw to get 178.2 → beat #1. z=0.464 is healthy but R11 (z=0.499) only got 79.9. |
| 10:30 | **v3f training launched on H100** (GPU 0). 65 GT, NO z-aug. v3e LORO was 79.6 but failed at z=0.638 (OOD). Now R12 data (z=0.599) is in training set — high-z gap should close. |
| 10:30 | Uploaded 20 new GT files (R10-R13) + calibration to H100. 65 GT total. |
| 10:30 | Leaderboard: we're **#13 J6X at 174.0**. Top is EXTREMELY tight — #7 to #13 span 0.8 points. |
| 10:30 | v3f training progress: epoch 600/1500, avg fit=91.2. LORO will start after ~ep 1500, 13 folds × ~8 min. ETA ~12:30 UTC. |
| 11:06 | **v3f LORO COMPLETE: 79.9 avg** (+7.6 over v3b 72.3, +0.3 over v3e 79.6). All 13 folds done. |
| 11:06 | v3f LORO: R1=83.7, R2=83.6, R3=83.5, R4=89.1, R5=81.5, R6=71.3, R7=70.0, R8=81.5, R9=86.8, R10=86.0, R11=75.0, R12=59.0, R13=87.9 |
| 11:06 | R12 holdout 59.0 (vs v3e live 29.1) — OOD gap closing. R13=87.9 best moderate score. |
| 11:07 | **v3f DEPLOYED** as astar_nn_v3.pt. v3b backed up as v3b_backup3.pt. |
| 11:07 | **R14 RESUBMITTED** with v3f + inverted-U curve. NN weight=0.58 (down from 0.65 due to z=0.464 healthy dropoff). |
| 11:10 | **v2f training launched** on H100. 65 GT, v2 architecture (1.8M params). |
| 11:49 | **v2f LORO complete: 79.2 avg** (vs v2d 79.7 on 9 rounds). R1-R9 avg=81.4 (+1.7 over v2d). R12=48.3 (hard outlier). |
| 11:49 | v2f deployed as astar_nn.pt. **R14 resubmitted AGAIN** with v2f+v3f+inverted-U. |
| 11:17 | Fixed nn_predict.py v2 weight: 0.10→0.15 (matching harness optimal). R14 resubmitted final time. |
| 12:02 | **R14 GT harvested**: avg 80.0, weighted 158.4. Seeds: 79.2, 80.6, 81.5, 79.4, 79.2. Consistent healthy round. |
| 12:02 | Recalibrated with 70 GT (14 rounds). Between rounds, waiting for R15. |
| 12:02 | Leaderboard unchanged: #13 J6X 174.0. Gap to #1 Matriks 177.1 = 3.1 pts. |
| 12:02 | **R15 weight=2.079**: need raw ~85 to get 176.7 (tie #2), ~86 to get 178.7 (beat #1). |
| 12:10 | **R15 OPEN** — closes 14:52 UTC. HOLDING submission per user directive. |

---

## FULL STRATEGIC ANALYSIS — Sources: Claude Architect Agent, Codex GPT-5.4, Harness Experiments

### Position & Math
- **#13 at 174.0**, #1 Matriks at 177.1. Gap: 3.1 pts.
- On R13 (w=1.886), #1 scored 93.9 raw vs our 92.3. Gap: **1.6 raw pts**.
- Competition: Sat 13:00 CET → Sun 15:00 CET (~26h remain). ~13-16 more rounds.

| Target | R15 (w=2.079) | R20 (w=2.653) | R25 (w=3.386) |
|--------|--------------|--------------|--------------|
| Beat #1 (177.1) | 85.2 raw | 66.8 raw | 52.3 raw |
| Decisive #1 (185) | 89.0 raw | 69.7 raw | 54.6 raw |

### Critical Findings from Three Independent Reviews

**CODEX GPT-5.4 (key quotes):**
> "The pipeline is not ignoring per-cell observation evidence. `dirichlet_predict()` already does per-cell Bayesian updating. The real limit is sample size (n≈2) plus prior concentration (30.0) — observations only move the posterior by ~6%."

> "The bigger leak is query allocation. A 40x40 map only needs 9 distinct 15x15 tiles for full coverage, but saved R11-R14 data only covers 72-77% of cells because the same tiles are repeated."

> "If I replaced predictions with oracle truth only on observed cells, gains ranged from +5.8 (R13) to +15.4 (R14). The information IS valuable — but n≈2 isn't enough to extract it."

> "Fixing seed_selector.py needs a full rewrite, not a one-line fix. Multiple stale assumptions about data format and model APIs."

**ARCHITECT AGENT (controversial finding):**
Found nn_weight=0.93 optimal in full-model harness (+4 avg vs current). **HOWEVER** this is full-model (in-sample) — live LORO is ~80 not ~93. The finding is directionally useful: our NN_PEAK=0.65 may be too conservative, but 0.93 would be catastrophic in live.

**HARNESS EXPERIMENTS (my data):**

Inverted-U sweep with v2f+v3f (full model, indicative only):
| Peak | Floor | R13 (mod) | R11 (healthy) | R12 (v.healthy) | R14 (healthy) | Best Weighted |
|------|-------|-----------|---------------|-----------------|---------------|---------------|
| 0.65 | 0.30 | 93.3 | 92.7 | 84.6 | 80.6 | 175.9 |
| 0.70 | 0.30 | 93.5 | 93.0 | 85.2 | 80.7 | 176.3 |
| 0.75 | 0.40 | 93.7 | 93.8 | 88.1 | 80.9 | 176.7 |
| 0.80 | 0.40 | 93.9 | 94.0 | 88.6 | 81.0 | 177.0 |

Higher peak + higher floor monotonically better in harness. But harness overestimates NN quality.

Per-cell coverage analysis (R14 seed 0):
- 1144/1600 cells observed (72%), 456 unobserved (28%)
- Average 2.2 obs/cell on observed cells
- 50% of cells have exactly 2 observations

### PRIORITIZED ACTION PLAN (ranked by expected value × feasibility)

**#1. FULL 9-TILE COVERAGE BEFORE REPEATS (+0.5-1.5 all rounds)** ⏱ 1h
Currently 6 base tiles + 4 precision repeats → 72% coverage, 28% blind spots.
Fix: Use all 9 tiling positions [0,13,25]×[0,13,25] as base → 100% coverage.
With 10 queries/seed: 9 unique + 1 best repeat. Precision queries come from the remaining 10 cross-seed budget.
**Both Claude and Codex agree this is #1 priority.**
Risk: Very low. Strictly more information.

**#2. NN WEIGHT CURVE TUNING (+0.5-1.5 moderate, +1-2 healthy)** ⏱ 1h
Raise NN_PEAK: 0.65 → 0.70 (conservative step from harness evidence)
Lower healthy dropoff: start decay at t4=0.35 (earlier), NN_HEALTHY=0.20 (lower)
Codex: "keep the idea of decay, but start it earlier and push it lower"
Test by resubmitting R15 with different curves and checking harness.
Risk: Low if conservative. Don't go above 0.75 peak without LORO validation.

**#3. REWRITE SEED_SELECTOR FOR PER-SEED OVERWRITE (+0.5-1.5 on bad seeds)** ⏱ 2h
Codex confirmed it's fundamentally broken — not just KeyError but wrong model APIs and data format.
Needs full rewrite against viewport schema and current nn_predict.py signatures.
Enables: submit safe baseline first, then try aggressive recipe per-seed, overwrite only if better.
Risk: Medium — need safeguard: only overwrite if aggressive score > safe + margin.

**#4. OBSERVATION-LIKELIHOOD GATING (+0.5-1.0)** ⏱ 2h
Novel idea from Codex: instead of fixed z-based blend, use observations as MODEL SELECTION signal.
For each seed, compute log-likelihood of observed outcomes under NN vs Dirichlet:
```
ll_nn = sum(log(nn_pred[y,x,observed_class])) for all observed cells
ll_dir = sum(log(dir_pred[y,x,observed_class])) for all observed cells
→ seed where NN explains data better → increase NN weight for that seed
→ seed where Dirichlet explains better → decrease NN weight
```
This uses observations optimally: not to update per-cell (too noisy at n≈2) but to ROUTE between models.
Risk: Low — only adjusts existing blend weights, safe baseline already submitted.

**#5. RETRAIN V3G + V2G WITH 70 GT (+0.3)** ⏱ background, 45min
GPU free on H100. Launch immediately. 5 more samples (R14).
Risk: None — keep backups.

**#6. DIRICHLET PRIOR CONCENTRATION TUNING (+0.2-0.5)** ⏱ 30min
Codex noted prior concentration fixed at 30.0 in calibrate.py:230.
With n≈2 observations, this means obs shift posterior by only 6%.
If we LOWER concentration to 15-20, observations have 2x more influence on the ~72% of covered cells.
Test in harness — may help healthy rounds where Dirichlet dominates.
Risk: Low if tested.

### What We Ruled OUT
- **Per-cell Bayesian posterior updating**: All three sources agree n≈2 is too few samples. Codex: "replace observed cells with empirical frequencies" test was **disastrous** (-79 to -85 points). Not worth pursuing.
- **Cross-seed cell-level pooling**: Different seeds have different initial grids (~57% cell overlap). Invalid to pool cell observations across seeds.
- **nn_weight=0.93**: Architect Agent's recommendation is based on in-sample full model. Would be catastrophic in live holdout. Maximum safe peak is ~0.75.
- **More complex blending layers**: Codex: "strong diminishing returns... mostly add double-counting and overfitting risk."

### 10-HOUR AUTONOMOUS PLAN

| Hour | Task | Expected Gain |
|------|------|---------------|
| 0-1 | **Fix query coverage to 9-tile full map** | +0.5-1.5 all rounds |
| 1-2 | **Tune NN weight curve** (peak→0.70, floor→0.20, t4→0.35) | +0.5-2.0 |
| 0-1 (bg) | **Retrain v3g/v2g** with 70 GT on H100 | +0.3 |
| 2-4 | **Rewrite seed_selector.py** for per-seed overwrite | +0.5-1.5 |
| 4-5 | **Observation-likelihood gating** per seed | +0.5-1.0 |
| 5-6 | **Dirichlet concentration tuning** | +0.2-0.5 |
| 6-10 | **Monitor, auto-solve, resubmit, retrain** | continuous |

**Cumulative expected gain: +2.5-7.5 raw points** (would push us from 92→95+ on moderate rounds, 80→82+ on healthy)

### For R15 Specifically
R15 closes 14:52 UTC. Plan:
1. Implement query coverage fix first
2. Then observe R15 with full 9-tile coverage
3. Check z. If moderate: current setup + coverage fix should score 90+.
4. If healthy: implement lower NN floor before submitting.
5. Submit safe baseline → then resubmit with improvements before close.
