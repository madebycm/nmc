# Saturday Push Plan — March 22, 2026

## Ground Truth
| Fact | Value |
|------|-------|
| Our best test | 0.9140 (v5.4_cos20) |
| Leader | 0.9255 (14 subs) |
| Gap | 0.0115 |
| Bullets remaining | 2 now (before midnight) + 6 after midnight = **8 total** |
| Competition end | March 22, 15:00 CET |
| OOF→test translation | ~20% (0.0182 OOF → 0.0036 test) |

## What We Know (hard data, not hope)

1. **Soft-NMS is dead.** Tested exhaustively — hurts Top-K by -0.002 to -0.006. Dense shelves have side-by-side products, not overlapping. Hard NMS at IoU=0.5 is correct.

2. **Detection is at ceiling.** det_mAP is ~0.777 OOF across ALL NMS variants (0.7761-0.7781). No code-only detection change moves the needle.

3. **Classification is the only live lever.** Our cls_mAP moved from 0.658 (v5.0) to 0.696 (v5.4). Leader's implied cls_mAP is ~0.845-0.868 depending on their det. Gap is entirely classification.

4. **Top-K is our best discovery.** K=10→K=15 gave +0.003-0.005 OOF consistently. But it's blunt — same K for every box regardless of confidence.

5. **MixUp > CE on OOF.** MixUp cos/20 E3 blend=0.7598 vs CE-only=0.7545. MixUp is the stronger model.

## Strategy: Probe → Combine → Tune

### ~~Lever A: FP16 Detector + Dual-Classifier Ensemble~~ — BLOCKED
**Finding:** Detector is ALREADY FP16 (136.5MB). Cannot shrink further.
136.5 + 172.3 + 172.3 = 481MB > 420MB limit. 61MB over budget.
**Only possible with smaller detector (YOLOv8m ~51MB) but that sacrifices detection.**
**Verdict: NOT VIABLE without accepting worse detection.**

### Lever B: Selective Top-K Emission
**Why:** Current K=15 for every box is blunt. The eval structure has an asymmetry:
- Detection: globally constrained (maxDets), more predictions = dilution
- Classification: per-category, more recall candidates = better

**Smart emission rule:**
- High-confidence boxes (rank-0 prob > 0.8): emit K=1-3 (classifier is sure)
- Ambiguous boxes (rank-0 prob < 0.5 or high entropy): emit K=10-15 (classifier uncertain)
- Score threshold: stop emitting when decayed score < 0.02
- This reduces total predictions (helping detection ranking) while preserving classification recall where it matters

**Implementation:** Pure code change in run.py scoring loop. OOF-validate on H100.

### Lever C: Ensemble Weight Tuning
**Why:** Equal CE+MX averaging is wrong — MixUp is stronger AND CE is its parent (high correlation). The blend weight matters.

**Sweep:** MX weight in [0.5, 0.6, 0.7, 0.8, 0.9] with OOF eval.

## Bullet Allocation

### Pre-midnight (2 bullets)
Offline prep running NOW on H100 (3 parallel agents):
- [ ] FP16 ONNX export + parity check
- [ ] CE+MixUp ensemble OOF sweep (MX-heavy ratios)
- [ ] Selective top-K emission OOF sweep

**Bullet 1: Selective top-K probe** (on v5.4 base)
- Adaptive K based on best OOF strategy (entropy/margin/cumulative)
- Isolates: does smarter emission help on test?

**Bullet 2: Based on B1 result**
- If selective-K positive: tune thresholds
- If selective-K flat: try other code-level changes (score formula, different T per confidence band)

### Post-midnight (6 bullets)
**Bullet 3: Combine positives from B1+B2**
- Only stack probes that showed positive signal on test

**Bullet 4: Tune the winner**
- If ensemble won: sweep MX:CE ratio
- If selective-K won: tune thresholds
- If both won: tune combined parameters

**Bullet 5: Secondary lever or parameter refinement**
- Based on B3+B4 feedback

**Bullet 6-8: Iterate / safety**
- Final refinements based on accumulated test feedback
- One bullet reserved for "best known config" safety shot

### If ensemble doesn't work (FP16 breaks or OOF negative)
- Shift all bullets to selective-K + parameter tuning
- Consider: score formula alternatives, different temperature per confidence band

## What We Are NOT Doing
- ❌ Soft-NMS (proven negative)
- ❌ Crop padding (side quest, not frontier)
- ❌ Multi-scale detection (high risk, timeout/output concerns, detection already at ceiling)
- ❌ Equal-weight ensemble (MixUp is stronger, equal dilutes it)
- ❌ Blind parameter sweeps without structural change
- ❌ Any submission without full end-to-end verification

## Honest Probability Assessment
- Top-15 climb from #25: **high** (ensemble or selective-K likely helps)
- Cross 0.920: **plausible** with one positive probe
- Beat 0.9255: **10-20%** (need two positive probes + successful integration)
- The path: ensemble adds +0.003-0.005, selective-K adds +0.002-0.003, tuning adds +0.001-0.002 = total +0.006-0.010. Gets us to ~0.920-0.924. Closing the last 0.001-0.005 requires luck or the leader staying flat.

## Priority Order for Tonight
1. FP16 ONNX export + parity check
2. OOF ensemble eval (MX:CE ratio sweep)
3. Selective top-K implementation + OOF eval
4. End-to-end verification of best config
