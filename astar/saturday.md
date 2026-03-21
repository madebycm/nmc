# Saturday Directive — March 21, 2026

> Win thesis: survive cleanly to the next moderate/catastrophic round at late weights.
> R13 already proved the system can produce a championship score (92.28 raw).
> The enemy is live failure, not model capacity.

## Position

- **#13 J6X**: 174.0 (best: R13 × 1.886 = 174.1)
- **#1 Matriks**: 177.1 — gap: 3.1 pts
- **Remaining**: ~21 hours (closes Sun 15:00 CET)
- **R15**: submitted (z=0.329 moderate, weight=2.079), score pending

## Regime Recipes

Three distinct recipes. Not one universal curve.

### Moderate (z = 0.15–0.40) — KILL SHOT

This is where we win. R13 recipe proven at 92.28 raw.

- NN peak weight: **0.65** (proven, don't chase higher)
- v2:v3 ratio: 0.15:0.70 (harness optimal)
- 5 extra queries: highest **model disagreement** tiles (KL between NN and Dirichlet)
- Aggressive but proven — this recipe doesn't change

### Catastrophic (z < 0.08) — DON'T TOUCH

Already strong: R8=84.4, R10=86.3.

- Pure Dirichlet (0% NN)
- Full 9-tile coverage still helps
- Don't destabilize chasing +0.5

### Healthy (z > 0.40) — CONTAINMENT

This is the weak regime. R11=79.9, R14=80.0, R12=29.1.
Objective: score 80–84 safely. NOT 90 by force.

- **Clip z input**: `z_nn = clip(z_est, 0.018, 0.599)` — training support range
- Earlier healthy decay: t4 = **0.35** (was 0.40)
- Lower NN floor: NN_HEALTHY = **0.20** (was 0.30)
- Higher v2 share inside NN block on healthy: v2=0.25, v3=0.60 (v2 is more stable OOD)
- **Observation-based veto**: if challenger log-likelihood on observed cells is >2x worse than baseline, reject it
- Fallback: if z > 0.55, always have Dirichlet-dominant backup ready

## Implementation Plan — Priority Order

### 1. Full 9-tile coverage (IMMEDIATE)

**The single highest-value fix.**

Current: QUERIES_PER_SEED=6, density-sorted → 28% of cells unobserved, selection bias in context vector.

Fix:
- `config.py`: QUERIES_PER_SEED = **9** (use all 9 unique tiles)
- `solver.py compute_smart_viewports()`: return tiles in fixed 3x3 order (no density sort for base pass)
- Precision budget: 50 - 45 = **5 queries** (was 20)
- Those 5 extras: pick by **model disagreement** (KL between NN pred and Dirichlet pred), not settlement density
- Context vector: computed from all 45 base queries (unbiased, full map)

Expected: +0.5–1.5 pts from unbiased context + better cell coverage

### 2. Z-clip for NN input (IMMEDIATE)

Prevents R12-class OOD disaster.

```python
# In nn_predict.py predict():
Z_TRAIN_MIN, Z_TRAIN_MAX = 0.018, 0.599
z_nn = np.clip(z, Z_TRAIN_MIN, Z_TRAIN_MAX)
```

For healthy rounds (z > 0.45), also do z-TTA:
- Average predictions at z_clip, z_clip-0.03, z_clip+0.03
- Reduces sensitivity to z estimation errors

Expected: prevents disasters, +2–5 pts on OOD rounds

### 3. Tune NN weight curve (IMMEDIATE)

Based on **live evidence only** (not in-sample harness):

```python
NN_PEAK = 0.65        # KEEP — proven at R13
NN_HEALTHY = 0.20     # was 0.30 — live evidence says less NN on healthy
t4 = 0.35             # was 0.40 — start decay earlier
t5 = 0.60             # was 0.70 — reach floor faster
```

Also: on healthy rounds (z > 0.40), shift v2:v3 ratio toward v2:
```python
if z > 0.40:
    v2_weight = 0.25   # was 0.15 — v2 more stable OOD
    v3_weight = 0.60   # was 0.70
```

### 4. Per-seed blend adjustment (SMALL, SAFE)

NOT a full seed_selector rewrite. Minimal inline logic in solver.py:

```python
# After computing global z_est from 45 base queries:
for seed_idx in range(seeds_count):
    z_seed = estimate_z_from_seed(base_observations[seed_idx], initial_grids[seed_idx])
    # Nudge NN weight based on seed deviation from global
    alpha = 0.15  # small correction
    nn_weight_seed = clip(nn_weight_global + alpha * (z_seed - z_est), 0.0, NN_PEAK)
```

This helps weak seeds (like R11 seed 2) without a dangerous rewrite.

### 5. Observation-based veto (SAFETY NET)

After safe baseline submit, before any resubmit:

```python
# For each seed, compute log-likelihood of baseline vs challenger on observed cells
ll_baseline = sum(log(pred_baseline[r,c,observed_class]) for observed cells)
ll_challenger = sum(log(pred_challenger[r,c,observed_class]) for observed cells)

# Only overwrite if challenger is not dramatically worse
if ll_challenger < ll_baseline - threshold:
    log.warning(f"Challenger rejected: ll={ll_challenger:.1f} vs baseline={ll_baseline:.1f}")
    continue  # keep safe baseline
```

This would have caught the v3e R12 disaster.

### 6. H100 background work

Two tasks only:

**A) Generate LORO prediction tensors for blend tuning**
- For each holdout round, predict with v2f, v3f, Dirichlet separately
- Search blend parameters on OOF predictions (not in-sample)
- Optimize regime-specific scores, especially healthy floor

**B) Train one healthy-specialist v3 variant**
- Overweight healthy/very healthy rounds in loss
- z-augmentation biased upward
- Accept ONLY if healthy holdout folds improve without hurting moderate
- Reject immediately if any fold collapses

## What we are NOT doing

- No full seed_selector rewrite (bug surface too large under pressure)
- No Dirichlet concentration tuning (fix acquisition first)
- No broad replay-soft-label retrain as mainline (background only)
- No in-sample harness tuning (misleading, overfits)
- No pushing NN peak above 0.65–0.70 without OOF evidence
- No architecture churn in production

## Decision Table — Round Time

```
IF z < 0.08:
  → CATASTROPHIC recipe: 0% NN, pure Dirichlet
  → Full 9-tile, no precision queries needed
  → Submit once, done

IF 0.08 <= z < 0.40:
  → MODERATE recipe: NN_PEAK=0.65, v2=0.15, v3=0.70
  → Full 9-tile base, 5 disagreement-based precision queries
  → Submit safe baseline, then resubmit with per-seed adjustment
  → THIS IS THE KILL SHOT REGIME

IF z >= 0.40:
  → HEALTHY recipe: NN decayed, v2 share boosted, z clipped
  → Full 9-tile base, 5 queries on highest-entropy tiles
  → Submit safe baseline
  → Challenger only if observation veto passes
  → Objective: 80–84 safely, DO NOT force 90
```

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| Next moderate round raw | 92+ | 92.28 (R13) |
| Next catastrophic round raw | 85+ | 86.25 (R10) |
| Healthy round floor | 80+ | 79.9 (R11) |
| Zero live disasters | 0 | R3, R7, R12 = 3 |
| Gap to #1 | < 0 | -3.1 |

## The Math

R16 at weight 2.183: need raw 81.2 to match current 177.1
R17 at weight 2.292: need raw 77.3 to match current 177.1
R18 at weight 2.407: need raw 73.6 to match current 177.1

**One clean moderate round at R17+ weight = 92 × 2.29 = 210.7 → clear #1**

Even a good catastrophic: 86 × 2.29 = 197.0 → clear #1

The later weights do the work. We just need to not blow up before then.
