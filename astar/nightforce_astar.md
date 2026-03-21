# Nightforce — H100 Overnight Promotion Machine

**Window**: ~18h remaining (now → March 22 15:00 CET)
**Goal**: Turn H100 into a promotion machine, not a checkpoint factory.

---

## Status (2026-03-21 20:00 UTC)

- **CandB swapped to production** — `astar_nn_v3.pt` is now CandB (v3f backed up)
- **R16 resubmit FAILED** — round completed, API rejects submissions
- **R17 resubmitted 5/5 with CandB** — z=0.402 healthy, NN weight ~0.55, pending score
- **Scorer bug found** — eval_candidates.py used plain mean KL, not entropy-weighted KL. Fixed.

### Corrected replay (exact competition metric, in-sample):

| Model | R14 | R15 | R16 |
|-------|-----|-----|-----|
| v3f | 80.8 | 90.0 | 85.4 |
| CandA | 85.0 | 93.6 | 91.8 |
| **CandB (now live)** | **86.8** | **94.3** | **93.1** |

**WARNING**: These are in-sample. Cannot distinguish improvement from memorization.

---

## Doctrine: Promotion Machine, Not Checkpoint Factory

### Core principle
We need the **best model for future late-weight rounds**, not the best average model.
Plain 16-fold LORO average is the wrong metric — it overweights early/catastrophic rounds.

### The right validation: forward/shadow testing

**Primary**: Rolling forward validation on late rounds
- Train on rounds `< r`, evaluate on round `r`, for R13-R17
- Matches deployment: no future data leakage, emphasizes recent distribution

**Secondary**: Fixed shadow set (regime-balanced)
- R12 = healthy/OOD stress
- R14 = healthy
- R15 or R16 = moderate
- R10 = catastrophic floor

### What NOT to use as promotion metric
- Plain in-sample all-round eval
- Plain mean fold score across all 16 rounds
- Training fit numbers (CandA 90.1, CandB 92.2 = meaningless)

---

## Three Families (not 5 random variants)

### Family 1: Moderate Specialist (HIGHEST EV)

**Thesis**: Moderate is our kill zone (R13=92.3, R15=90.0). Late weights + moderate 85-90 wins the task.

Training:
- Overweight moderate rounds (z=0.15-0.38) 3x
- Keep doctrine-compatible z behavior
- Do NOT optimize for catastrophic
- Do NOT over-chase healthy extrapolation

### Family 2: Healthy Specialist (Containment)

**Thesis**: Healthy rounds leak points (R11=79.9, R14=80.0). Goal is not "force 90" but "never blow up, turn 80 into 83-85."

Training:
- Overweight healthy rounds (z>0.40) 3x
- Strong z-jitter / clip-aware
- Explicitly test against R12-class OOD stress
- CandB is conceptually this family

### Family 3: Robust Fallback (Hedge)

**Thesis**: Many failures are z-sensitivity / OOD. A stable lower-ceiling model has value at regime boundaries.

Training:
- Stronger z-jitter (±0.15)
- Reduced z-dependence
- Prioritize worst-case stability over flashy train fit

### Family 4: One v3-XL Shot (Controlled Experiment)

256 hidden, 10 ResBlocks = ~12M params. Run exactly one.
- Make it the **moderate specialist** variant (highest EV family + more capacity)
- If it wins, great. If not, lost only one experiment.

---

## Overnight Queue (sequential on GPU, eval on CPU/Mac)

| Step | What | Time | Notes |
|------|------|------|-------|
| 0 | Fix evaluator (entropy-weighted scorer) | 5 min | Before anything else |
| 1 | Train moderate specialist (v3, 400 epochs) | ~5 min | Overweight z=0.15-0.38 |
| 2 | Train healthy specialist (v3, 400 epochs) | ~5 min | Overweight z>0.40 |
| 3 | Train robust fallback (v3, 400 epochs) | ~5 min | Heavy z-jitter |
| 4 | Train v3-XL moderate (800 epochs) | ~15 min | Capacity experiment |
| 5 | **WAIT for R17 GT** | variable | Most valuable single sample |
| 6 | Forward validation board (R13-R17) | ~30 min | Train on `<r`, eval on `r` |
| 7 | Promotion decision | — | Per-regime metrics |
| 8 | Retrain winner on all R1-R17 | ~5 min | Only after family selected |
| 9 | Deploy: regime-routed model | — | See deployment structure |

**Total GPU time**: ~70 min training + ~30 min forward eval = ~100 min

---

## Promotion Metrics (per checkpoint)

Log ALL of these — do NOT rank by one scalar:

| Metric | What |
|--------|------|
| `moderate_avg` | Avg score on moderate shadow rounds |
| `healthy_avg` | Avg score on healthy shadow rounds |
| `stress_healthy` | R12-type OOD stress score |
| `catastrophic_floor` | Worst catastrophic round score |
| `late_weight_proxy` | Best held-out `raw × 1.05^round` |
| `worst_round` | Absolute floor across all held-out rounds |

### Promotion Rules

**Moderate specialist ships if**:
- Gains ≥ +0.8 raw on moderate shadow/forward rounds
- Does not lose > 1 raw on healthy
- Does not crater catastrophic floor

**Healthy specialist ships if**:
- Gains ≥ +1.5 raw on healthy shadow rounds
- Clearly improves R12 stress case
- Does not lose too much on moderate

**Robust fallback ships if**:
- Never the flashy winner
- Materially improves worst-case behavior

---

## Deployment Structure: Regime Routing

Not one global v3 winner. Route by z:

| z Range | Model | Rationale |
|---------|-------|-----------|
| z < 0.08 | Dirichlet only | Catastrophic — NN unreliable |
| 0.08 ≤ z < ~0.38 | Moderate specialist | Kill zone — max NN leverage |
| z ≥ ~0.38 | Healthy specialist | Containment — stability over ceiling |
| Boundary / weird | Robust fallback | Hedge against regime misclassification |

---

## When R17 GT Lands

R17 is not "just one more sample." It is:
- Healthy-ish (z=0.402)
- Post-pivot, under current doctrine
- Exactly the distribution we're fighting for next 6 rounds

Immediately:
1. Add R17 to healthy shadow/forward set
2. Re-run promotion board with R17 included
3. Only THEN retrain winning family on all R1-R17
4. Deploy as regime-routed specialist pair

---

## Evaluator Safety Checklist

Before trusting ANY number:
- [ ] Score = exact entropy-weighted KL: `100 * exp(-3 * weighted_kl)`
- [ ] Clear `nn_predict._models` cache between candidates
- [ ] Historical obs: use correct base subset (not just `obs[:9]` blindly)
- [ ] Forward tests: roll calibration forward too, not final all-round snapshot
- [ ] No file mutation during eval (don't swap `astar_nn_v3.pt` in place)

---

## Reviewer Corrections (2026-03-21 ~21:00 UTC)

### 1. Re-evaluate baselines with corrected scorer ✅
Entropy-weighted scorer changes everything. Models that looked "weak" at static cells
may actually be great at contested cells. **Re-run all historical models through exact scorer.**

### 2. Smooth regime blending (CRITICAL FIX)
Hard cutoffs at z=0.38 are a landmine. If z estimation says 0.375 but true z is 0.385,
we route to wrong specialist → catastrophic boundary failure.

**Fix**: Sigmoid/linear ramp between z=0.35 and z=0.41:
- z ≤ 0.35 → 100% Moderate specialist
- z = 0.38 → 50/50 blend
- z ≥ 0.41 → 100% Healthy specialist

Apply same smooth blending at catastrophic/moderate boundary (z=0.06-0.10).

### 3. v3-XL is a trap — demote to stretch
12M params on ~17 rounds = instant memorization. If training fit spikes to 98, it's
overfitting, not improving. **Trust smaller v3 specialists. v3-XL is stretch goal only.**
If run: heavy z-jitter mandatory, kill at first sign of overshoot.

### 4. R17 shadow discipline (CRITICAL)
Do NOT fold R17 into training immediately. Keep it strictly held-out for forward
validation of R18 deployment. Only fold into training data for **final deployment checkpoint**
after architecture and routing are locked.

**Sequence**:
1. Train families on R3-R16 only
2. Forward-validate all candidates on R17 (held-out)
3. Pick winner based on R17 forward score
4. THEN retrain winner on R3-R17 for deployment

---

## Codex GPT-5.4 Review (2026-03-21 ~21:15 UTC)

### Finding 1: Keyed post-blend calibration (HIGHEST EV)
- Model gets RIGHT argmax but distribution too peaked on Land(11) and Forest(4) cells
- 67.7% of loss is Land, 29.3% is Forest — these are THE battleground
- We ALREADY have z-conditioned priors in calibration.json for these keys
- **Fix**: `p' = (1-λ) * p_model + λ * q_key(z)` — prior-shrinkage, NOT global temperature
- Better than temperature because it sends mass to correct secondary classes
- Physics masking = ZERO impact (confirmed: entropy-weighting nullifies static cells)

### Finding 2: Eval/live parity gap
- Harness uses ALL observations for context; live uses base-only (9 tiles)
- Harness references replay model; live only loads v2+v3
- **Must force parity before trusting offline selection**

### Finding 3: Script/plan misalignment
- Plan says 3 families + forward validation
- nightforce_astar_train.py runs plain LORO + XL (outdated)
- **Must rewrite script to match plan**

### Finding 4: Replay as calibrator, not separate model
- Use replay to fit per-key calibration on `11` and `4` cells
- MC-average replay → soft targets → fine-tune CandB backbone
- Use replay variance for per-key concentration (not fixed 30.0)
- Replay model as standalone = low EV; replay as calibration data = high EV

### Finding 5: What 196+ teams likely do
- Better calibrated probabilities on dynamic land cells
- Smooth specialist blending by regime
- Stronger observation utilization (local posterior, not just scalar z)
- Clean offline/live parity

### Finding 6: Temperature verdict
- Global temperature = wrong fix
- Keyed de-peaking per cell type = right fix
- Per-key `T[key, regime]` or prior-shrinkage on `11_*` and `4_*`

---

## Reality Check

Do not comfort ourselves with "R20 only needs 74.1 raw" math.
That is true for everyone. The winning target is **low-80s to high-80s** on moderate/healthy while avoiding disasters. Every team has the same exponential tailwind.
