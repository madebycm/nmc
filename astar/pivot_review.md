# Pivot Review — Saturday March 21, 2026

## What we changed (the doctrine pivot)

Three surgical production fixes deployed between R15 and R16:

1. **45-query fixed base scan**: 9 unique tiles per seed, deterministic order, no density sorting
2. **Z-clip + z-TTA**: NN input clamped to [0.018, 0.599], z-bagging on healthy rounds
3. **Tuned NN curve**: earlier decay (t4=0.35), lower floor (NN_HEALTHY=0.20), faster convergence (t5=0.60)
4. **JS-divergence precision targeting**: 5 remaining queries by model disagreement
5. **Per-seed asymmetric blend nudge**: small correction based on seed-local z
6. **Regime-adaptive v2:v3**: v2 share boosted on healthy rounds (more stable OOD)

Plus P0 safety fixes:
- Removed broken Phase 3 (seed_selector.py)
- Added filesystem lock (prevents concurrent runs)
- Only mark solved on 5/5 seeds (partial submits retry)
- Active round checked FIRST, GT harvest AFTER
- Codex removed from critical path
- poll_loop.py quarantined

## Results

| Round | z | Regime | Raw | Weight | Weighted | Pipeline |
|-------|---|--------|-----|--------|----------|----------|
| R13 | 0.226 | moderate | **92.28** | 1.886 | 174.0 | pre-pivot (v2d+v3b) |
| R14 | 0.522 | healthy | 80.0 | 1.980 | 158.4 | pre-pivot (v2f+v3f+invU) |
| R15 | 0.328 | moderate | **90.0** | 2.079 | **187.1** | MID-pivot (resubmitted with 9-tile context) |
| R16 | 0.312 | moderate | 85.1 | 2.183 | 185.8 | FULL doctrine pipeline |

## Analysis

### R15 (90.0 raw) — resubmitted with partial pivot
- Originally submitted with old 6-tile base, then resubmitted with 9-tile context
- z=0.328 moderate — hit NN peak (0.65)
- Per-seed scores: 92.8, 92.1, 83.4, 91.8, 89.9
- Seed 2 dragged average down (83.4 vs 91+ on others)
- **Best weighted score: 187.1** — new leaderboard best

### R16 (85.1 raw) — full doctrine, but rate-limited
- z=0.312 moderate — should be kill zone
- Full 9-tile pipeline, but 429 errors dropped 4/5 seeds to 8/9 base coverage (41/45)
- Per-seed nudge active (seeds 2,3 nudged down)
- Per-seed scores: 85.5, 83.1, 86.6, 85.3, 85.0
- **Much tighter spread than R15** (83-87 vs 83-93) but lower ceiling
- **5 points below R15 despite same regime** — why?

### R16 vs R15 gap (85.1 vs 90.0): possible causes
1. **Rate-limit coverage gap**: 41/45 base queries vs 45/45 — 4 missing tiles = biased context on 4 seeds
2. **Per-seed nudge may have over-corrected**: seed 3 went 0.650→0.623 NN weight (z_seed=0.222 vs z_round=0.312)
3. **Different hidden params**: z similar (0.312 vs 0.328) but port survival dramatically different (R16 has ports, R15 didn't)
4. **R15 was resubmitted** — used 6-tile observations for predictions but 9-tile context. R16 used 9-tile for everything.
5. **Natural variance**: R15 seed 0/1/3 were exceptionally strong (91-93), R16 was uniformly ~85

### What the pivot DID fix
- No disasters — R16 at 85.1 is a clean healthy-ish moderate score
- Per-seed nudge working (seeds with lower local z got reduced NN weight)
- Z-clip + z-TTA ready for next healthy round (not tested yet)
- Filesystem lock preventing concurrent runs
- No broken Phase 3 overwrites

### What still needs investigation
- **Rate-limit is the #1 operational issue** — 0.5s sleep still too aggressive for 45 queries
- R16 scored 5pts below R15 in the same regime — is this pipeline regression or just variance?
- Per-seed nudge strength (alpha=0.30) may need tuning

## Leaderboard (Astar task)

| Rank | Team | Score |
|------|------|-------|
| #1 | Meme Dream Team | 196.6 |
| #2 | SiddisAI | 196.5 |
| #3 | People Made Machines | 196.1 |
| #60 | J6X (us) | 187.1 |

**Gap to #1: 9.5 pts**

Overall competition: **#39** (all 3 tasks combined)

## To match #1 (196.6 weighted):
- R17 (w=2.292): need **85.8 raw**
- R18 (w=2.407): need **81.7 raw**
- R19 (w=2.527): need **77.8 raw**
- R20 (w=2.653): need **74.1 raw**

## Ablation Study (R11–R16, current doctrine vs no-nudge vs pure Dirichlet)

| Round | z | Regime | Doctrine | No-nudge | Dirichlet | Nudge fx | NN lift |
|-------|---|--------|----------|----------|-----------|----------|---------|
| R11 | 0.545 | healthy | 91.2 | 91.2 | 85.5 | +0.05 | +5.67 |
| R12 | 0.615 | healthy | 75.8 | 75.9 | 65.0 | -0.08 | +10.90 |
| R13 | 0.199 | moderate | 93.8 | 93.8 | 90.2 | -0.01 | +3.56 |
| R14 | 0.494 | healthy | 80.8 | 80.8 | 76.8 | +0.01 | +3.98 |
| R15 | 0.329 | moderate | 90.0 | 90.0 | 87.5 | -0.01 | +2.53 |
| R16 | 0.302 | moderate | 85.4 | 85.4 | 84.8 | +0.00 | +0.65 |

### Key findings
1. **Nudge is essentially zero** — never more than ±0.08. Now gated (alpha reduced 0.30→0.15, requires 9/9 coverage + |delta|>0.03)
2. **NN adds real value everywhere** — +0.65 to +10.90 over pure Dirichlet
3. **NN lift varies enormously** — R16 only +0.65, R12 was +10.90. Same model, different rounds.
4. **R12 z-clip working** — 75.8 with clip vs historical 29.1 live. Clip alone worth +47 pts on OOD.
5. **R13 offline = 93.8 vs live 92.3** — offline has full 9-tile context, live had 6-tile biased context. +1.5 from coverage fix.
6. **R11 offline = 91.2 vs live 79.9** — massive gap. Live used broken v2d+v3b, now v2f+v3f with z-clip.

### Verdict
- R16 drop is **the round itself** (NN lift only +0.65), not pipeline regression
- Coverage fix is worth +1-2 pts on average
- Z-clip is worth +47 pts on OOD rounds (R12-class prevention)
- Nudge is noise — gated now

## Fixes applied (post-R17)
1. **Interleaved base scan** — tile-by-tile across all seeds (not seed-by-seed)
2. **Mandatory retry** — 429s retry same tile up to 5x with increasing backoff
3. **0.8s between queries** — conservative, 45/45 matters more than speed
4. **Nudge gated** — alpha=0.15, requires 9/9 coverage + |delta|>0.03
5. **GT harvest moved after solve** — nothing non-essential before submission

## H100 Training Plan
- **Candidate A**: Conservative v3 refresh — all GT through R16, fine-tune from current v3f
- **Candidate B**: Healthy specialist v3 — overweight healthy rounds, upward z-jitter
