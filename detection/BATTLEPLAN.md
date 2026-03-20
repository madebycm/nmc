# BATTLEPLAN: 0.9034 → 0.9200+

**Time:** Friday 17:34 → Saturday midnight+
**Submissions:** 1 tonight (before midnight UTC), 3 tomorrow
**Leader:** 0.9200 | **Us:** 0.9034 | **Gap:** 0.0166
**Score:** `0.7 × det_mAP@0.5 + 0.3 × cls_mAP@0.5`

---

## I. What We Know (Hard Facts)

### Val Baseline (M=300, no kNN)
| Metric | Value |
|--------|-------|
| det_mAP@0.5 | 0.8030 |
| cls_mAP@0.5 | 0.9060 |
| blend | 0.8339 |

⚠️ Val is contaminated (classifier trained on all 248 images including val 50).
⚠️ Val blend 0.8332 → test score 0.9034. Test is a different distribution.

### Submissions
| Version | Config | Val blend | Test score | Status |
|---------|--------|-----------|------------|--------|
| v4.1 | M=200, no kNN | 0.8332 | **0.9034** | Scored |
| v4.2 | M=300, no kNN | 0.8339 | Pending | Submitted |
| v4.3 | Tonight's shot | TBD | TBD | **BUILD NOW** |

### What We Tested & Killed Today
| Change | Val Δ | Verdict |
|--------|-------|---------|
| T=1.5 temperature scaling | −0.0028 | **DEAD** globally |
| 5% crop expansion | −0.0022 | **DEAD** globally |
| Global kNN blend α=0.7 | −0.0013 | **DEAD** |
| Global kNN blend α=0.9 | −0.0004 | **DEAD** |
| kNN override (pure) t=0.5 | −0.0033 | **DEAD** |
| Routed kNN t=0.5 a=0.5 T=0.05 (sharp) | −0.0008 | Hurts |
| **Routed kNN t=0.5 a=0.5 T=0.10 (smooth)** | **+0.0000** | **Val-neutral → ALIVE** |

### Key Insight from Sweep
- 4496/13700 crops (33%) are uncertain (softmax conf < 0.5)
- Sharp kNN (T=0.05) damages common classes → net negative
- Smooth kNN (T=0.10) is harmless → coexists with supervised head
- **`amax` aggregator has many-shot bias** — classes with 20 refs beat classes with 2 refs by random chance, reintroducing the bias we're trying to fix

---

## II. The Failure Mode (Why We're Losing 0.0166)

84 of 356 categories have ≤5 training examples. The supervised head:
1. Never saw enough examples of these classes to learn them
2. Confidently misclassifies them into frequent classes (overfit logits are sharp)
3. These misclassifications cost us ~0.02-0.05 on cls_mAP on unseen test data

The kNN reference bank has embeddings for ALL 356 classes. But the current `amax` aggregator gives many-shot classes (20 refs) an unfair advantage over few-shot classes (2 refs). This is why kNN hasn't helped yet — it reproduces the same bias.

**Fix the aggregator → fix the tail → close the gap.**

---

## III. Tonight's Submission (v4.3) — Ship Before Midnight

### Build 1: Count-Corrected kNN Aggregator

Replace `amax` per-class with count-normalized scoring:

```python
# Option A: count-normalized logsumexp
score_knn(c) = logsumexp(sims_c / tau) - beta * log(n_c)
# tau = 0.07, beta = 0.03-0.07

# Option B: prototype + top-r support (preferred)
proto_c = normalize(mean(ref_embeds[class == c]))   # [356, 768]
score_proto = query @ proto.T                        # [B, 356]
score_topr = mean(top-min(r, n_c) sims within class c)  # [B, 356]
score_knn = 0.6 * score_proto + 0.4 * score_topr - beta * log(n_c)
```

**Why prototype is better:** One vector per class = inherently count-invariant. Class with 20 refs and class with 2 refs each get one prototype. The `top-r` term adds evidence strength, and `-beta*log(n_c)` penalizes the remaining many-shot advantage in top-r.

### Build 2: Two-Signal Routing Gate

```python
conf = p_softmax.max(dim=1)        # max confidence
margin = top1_prob - top2_prob      # decision margin

route = (conf < 0.65) | (margin < 0.12) | (head_pred != knn_pred & conf < 0.75)

# Confident branch: pure softmax
# Routed branch: 0.45 * p_softmax + 0.55 * p_knn
```

### Build 3: Routed-Only Score Calibration

```python
# Don't change labels — only shape scores for COCO ranking
if routed:
    fused_score = 0.7 * det_s + 0.3 * (cls_s ** 0.7)  # dampen brittle FPs
else:
    fused_score = 0.7 * det_s + 0.3 * cls_s  # unchanged
```

### Tonight's Timeline

| Time | Action |
|------|--------|
| 17:34-18:00 | Precompute class prototypes + per-class ref counts on A100 |
| 18:00-19:00 | Implement count-corrected aggregator in sweep harness |
| 19:00-20:00 | Sweep tau, beta, r, routing params (reuse existing harness) |
| 20:00-20:30 | Implement two-signal gate, sweep thresholds |
| 20:30-21:00 | Add routed-only score calibration, final sweep |
| 21:00-21:30 | Pick best config, integrate into run.py, validate full pipeline |
| 21:30-22:00 | Package submission_4.3.zip, sanity checks |
| 22:00-23:30 | Buffer for debugging / iteration |
| 23:30-00:00 | **SUBMIT v4.3** |

### Ship Criteria
Submit if ANY of:
- Val blend ≥ 0.8340 (beats baseline by +0.001)
- Val cls ≥ 0.9070 with det unchanged (classification clearly improved)
- Count-corrected kNN shows consistent improvement across multiple configs

Do NOT submit if:
- Best config is val-neutral or negative AND no config beats 0.8339
- In that case, hold and use tomorrow's 3 bullets with more iteration

---

## IV. Tomorrow's 3 Bullets (Saturday)

### Submission 5.0: Best kNN from tonight's sweep
If tonight's submission lands well, iterate on the same axis. If it didn't work, try the alternative aggregator (logsumexp if prototype failed, or vice versa).

### Submission 5.1: Gate refinement
Use test score feedback from 5.0 to calibrate:
- If test improved but less than expected: widen the gate (route more crops)
- If test regressed: narrow the gate (route fewer crops)
- If test improved a lot: try even more aggressive kNN weight

### Submission 5.2: Score calibration + final ensemble
- Routed-only score shaping
- Final tuned parameters from all prior feedback
- This is the LAST SHOT. Must be the culmination of everything.

---

## V. Expected Outcomes (Honest Assessment)

### Test Score Decomposition (estimated)
We don't know the exact decomposition. Two plausible scenarios:

| Scenario | det_test | cls_test | blend | cls needed for 0.92 |
|----------|----------|----------|-------|---------------------|
| A: strong det | 0.96 | 0.77 | 0.903 | 0.827 (+0.057) |
| B: balanced | 0.94 | 0.82 | 0.904 | 0.873 (+0.053) |

### Realistic Gain Budget

| Change | Val Δ | Test Δ (est.) | Confidence |
|--------|-------|---------------|------------|
| M=300 (already in v4.2) | +0.0006 | +0.001-0.003 | HIGH |
| Count-corrected kNN (new) | +0.000-0.002 | +0.005-0.015 | MEDIUM |
| Two-signal routing | +0.000-0.001 | +0.002-0.005 | MEDIUM |
| Routed score calibration | +0.000-0.001 | +0.001-0.003 | LOW |

### Where We Land

| Scenario | Total test gain | Final score | vs Leader |
|----------|----------------|-------------|-----------|
| **Conservative** | +0.005 | 0.908 | −0.012 |
| **Expected** | +0.012 | 0.915 | −0.005 |
| **Optimistic** | +0.018 | 0.921 | **+0.001 🏆** |

The expected case puts us within striking distance. The optimistic case beats the leader. The conservative case still moves us up the leaderboard significantly.

**The fundamental bet:** Count-corrected kNN should help MORE on unseen test data than on contaminated val. Every point of evidence supports this — val is artificially easy for the supervised head, test is where rare classes actually matter.

---

## VI. Assets & Constraints

### Files to Ship
| File | Size | Status |
|------|------|--------|
| detector.onnx | 130 MB | Unchanged |
| classifier.safetensors | 164 MB | Unchanged |
| ref_embeddings_finetuned.npy | 10 MB | Exists on server |
| ref_labels.json | 31 KB | Exists on server |
| class_prototypes.npy | ~535 KB (356×768×FP16) | **BUILD TONIGHT** |
| ref_counts.npy | ~1.4 KB (356×int16) | **BUILD TONIGHT** |
| run.py | ~12 KB (est.) | **BUILD TONIGHT** |
| **Total** | **~305 MB** | Within 420 MB |

### Compute
- A100 80GB at 135.181.8.209 — alive, idle, ready
- Sweep harness (`sweep_knn.py`) — exists, needs new aggregator configs

### Sandbox Constraints (reminder)
- L4 GPU, 300s timeout, 8GB RAM, no network
- No: os, sys, subprocess, pickle, yaml
- Current runtime: ~95s (M=300). Budget: 205s headroom.
- kNN adds ~5-10s (one matmul per batch). Total ~105s. Safe.

---

## VII. What's Dead (Do Not Revisit)

| Idea | Why Dead |
|------|----------|
| T=1.5 global temperature | −0.0028 on val. Doesn't change argmax. |
| 5% crop expansion | −0.0022 on val. Distribution shift from tight-crop training. |
| Multi-scale ONNX (1024/1280/1536) | Fixed 1280×1280 ONNX. Cannot resize. |
| SAHI tiled detection | Previously −0.045 det_mAP. Floods maxDets=100. |
| Checkpoint averaging (avg5) | −0.005 cls_mAP. Dilutes converged model. |
| WBF (flip TTA merge) | −0.005 blend. Averaging misaligned boxes hurts. |
| Classifier flip TTA | −0.002 cls_mAP. Flipped grocery text confuses model. |
| Global kNN blend (any α) | All negative on val. Many-shot bias in aggregator. |

---

## VIII. The Decisive Insight

The gap lives in the **84 rare classes**. The supervised head is blind to them. The kNN has the embeddings to see them. But `amax` aggregation gives common classes an unfair advantage — 20 lottery tickets vs 2.

**Count-corrected prototypes are the surgical fix.** One prototype per class = no count bias. Top-r support adds evidence. `-beta*log(n_c)` penalizes residual advantage.

This is not a hack. This is the theoretically correct approach to few-shot classification with an imbalanced reference bank.

**Build it. Sweep it. Ship it.**
