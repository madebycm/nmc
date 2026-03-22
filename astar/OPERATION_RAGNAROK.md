# OPERATION RAGNAROK
### The 69-Hour Norse Island Campaign — Team J6X

> *"The island burned through 23 winters and we fought every one."*

---

## The Mission
Predict the fate of a 40x40 Norse island — 1,600 cells, 6 possible destinies, 50-year stochastic simulations. Score = `100 × exp(-3 × KL)`. Only your single best round counts. Prize: 1,000,000 NOK.

## The Final Weapon: Operation Ragnarok

| Component | Value |
|-----------|-------|
| Codename | **RAGNAROK** — cross-seed hierarchical empirical Bayes with confusion-aware soft evidence |
| Observations pooled | **10,125** across 5 seeds × 25 calibration keys |
| NN role | 25% smoothing prior (down from 75%) |
| Key-level concentration | A=5 (aggressive — 1000+ samples per key) |
| Soft evidence | Codex GPT-5.4 confusion matrix for n=1 cells |
| Cell-level C | 3-8 adaptive by class stability + observation count |
| Physics mask | Ocean=ocean, Mountain=mountain (0 exceptions in 110 GT files) |
| Probability floor | 0.003 (sweep-optimized, prevents KL=infinity) |
| Expert consultations | 3 (Expert 1: hierarchical, Expert 2: aggressive, Codex GPT-5.4: soft evidence) |

---

## The 69-Hour War — Round by Round

| Hour | Round | Raw | Weighted | z | Regime | Strategy | Moment |
|------|-------|-----|----------|------|--------|----------|--------|
| 0 | R1-R2 | — | — | 0.42 | healthy | — | Didn't know the task existed |
| 6 | R3 | 7.2 | 8.3 | 0.02 | catastrophic | proximity_v1 | SERVER POLLER OVERWRITES OUR PREDICTIONS |
| 10 | R4 | 79.9 | 97.2 | 0.24 | moderate | dirichlet_v3 | First real score. Hope. |
| 12 | R5 | 75.4 | 96.2 | 0.33 | moderate | dirichlet_v4 | Regime-aware priors online |
| 15 | R6 | 58.8 | 78.8 | 0.42 | healthy | dirichlet_v4 | First healthy round. Pain begins. |
| 18 | R7 | 38.4 | 54.0 | 0.42 | healthy | ensemble_nn_v1 | CHANNEL SWAP BUG. NN outputs garbage. |
| 21 | R8 | 84.4 | 124.7 | 0.07 | catastrophic | ensemble_ctx_v2 | Bug fixed. Strong catastrophic. |
| 24 | R9 | 89.1 | 138.3 | 0.28 | moderate | champion_challenger | Into the groove. |
| 27 | R10 | 86.2 | 140.5 | 0.06 | catastrophic | champion_challenger | Consistent. |
| 30 | R11 | 79.9 | 136.7 | 0.50 | healthy | champion_challenger | Best healthy yet. |
| 33 | R12 | 29.1 | 52.3 | 0.64 | healthy | champion_challenger | EXTREME OOD DISASTER. Score collapses. |
| 36 | R13 | **92.3** | 174.0 | 0.23 | moderate | champion_challenger | PEAK MODERATE. Our finest hour. |
| 39 | R14 | 80.0 | 158.4 | 0.46 | healthy | champion_challenger | Decent healthy. Hope again. |
| 42 | R15 | 90.0 | 187.1 | 0.33 | moderate | champion_challenger | Strong. Climbing. |
| 45 | R16 | 85.1 | 185.8 | 0.31 | moderate | champion_challenger | Solid. |
| 48 | R17 | 77.9 | 178.5 | 0.45 | healthy | doctrine_v1 | Nightforce model era begins |
| 51 | R18 | 74.3 | 178.9 | 0.62 | healthy | resubmit_nn | High-z. We suffer. |
| 54 | R19 | **94.2** | 238.0 | 0.04 | catastrophic | doctrine_v1 | SPIKE. Best raw score EVER. |
| 57 | R20 | 90.5 | 240.2 | 0.13 | low-moderate | doctrine_v1 | Resubmit gambit pays off. |
| 60 | R21 | 89.5 | **249.4** | 0.26 | moderate | doctrine_v1 | OUR BEST. LOCKED FOREVER. |
| 63 | R22 | 83.9 | 245.5 | 0.22 | moderate | doctrine_v1 | Disappointing. Falling. |
| 66 | R23 | ? | ? | 0.46 | healthy | **RAGNAROK** | Free lottery ticket. All in. |

## Algorithm Evolution

| Era | Rounds | Algo | Key Innovation | Avg Score |
|-----|--------|------|----------------|-----------|
| Stone Age | R3-R6 | Dirichlet priors | Hand-tuned priors by terrain type | 55.3 |
| Iron Age | R7 | NN ensemble v1 | Neural net... with swapped channels | 38.4 |
| Renaissance | R8-R16 | Champion/Challenger | 4-model ensemble + empirical anchoring | 83.4 |
| Nightforce | R17-R22 | Doctrine v1 | 6.4M param NF model, z-jitter, LORO 87.5 | 84.8 |
| Ragnarok | R23 | Soft Bayesian | Cross-seed pooling + confusion matrix | ? |

## The Desperation Curve

```
Confidence
100% |
 90% |          "we're going to win this"
 80% |     *         *
 70% |   *   *     *   *  *
 60% |  *     *   *     **  *
 50% | *       * *          *  *
 40% |*         *            *
 30% |                        *  "GPU servers died"
 20% |                         *  "healthy rounds destroy us"
 10% |                          *  "17.2 pts behind #1"
  0% |___*_________________________*___*
     R3  R7  R9  R12  R15  R19 R21 R23
         BUG  groove OOD  SPIKE LOCK RAGNAROK
```

## The Confusion Matrix That Saved Us

Codex GPT-5.4 discovered: a single observation is NOT what it seems.

| You observe | Actually that class again | Most likely true state |
|-------------|-------------------------|----------------------|
| Land (0) | 78% | Land (78%) |
| Settlement (1) | **27%** | Land (52%) |
| Port (2) | **38%** | Port (38%) / Land (27%) |
| Ruin (3) | **2.4%** | Land (48%) / Settlement (38%) |
| Forest (4) | 65% | Forest (65%) |
| Ocean (5) | 100% | Ocean (100%) |

**If you see a ruin, it's only a ruin 2.4% of the time.** Our previous approach treated every observation as gospel truth. Ragnarok spreads the probability mass across likely alternatives.

## Key Stats

| Metric | Value |
|--------|-------|
| Total rounds played | 21 of 23 |
| Best raw score | 94.19 (R19, catastrophic) |
| Worst raw score | 7.17 (R3, overwrite disaster) |
| Best weighted | 249.4 (R21) |
| Final rank | #73 of 392 |
| #1 score | 266.6 (Lokka Language Models) |
| Gap to #1 | 17.2 points |
| NN architectures built | 4 (v2, v3, v4 U-Net, Nightforce) |
| Total params trained | 30.7M across all models |
| Ground truth files | 110 (22 rounds x 5 seeds) |
| Replay MC samples | 7,767 |
| Expert AIs consulted | 3 |
| GPU servers lost | 2 (A100 reprovisioned, H100 SSH rejected) |
| Resubmits on R23 | 4 (doctrine → moderate → ultra → RAGNAROK) |
| Bugs that nearly killed us | 3 (channel swap, server overwrite, seed misattribution) |
| Hours without sleep | ~69 |
| Coffee consumed | Immeasurable |

## The Four Resubmits of R23

| # | Codename | NN% | Cell C | Key A | Innovation | Status |
|---|----------|-----|--------|-------|------------|--------|
| 1 | Doctrine | 75% | N/A | N/A | Standard pipeline | Overwritten |
| 2 | Moderate | 43% | 3-6 | 20 | Key-level update, lower NN | Overwritten |
| 3 | Ultra | 20% | 1-3 | 8 | Observation-dominant | Overwritten |
| 4 | **RAGNAROK** | 25% | 3-8 adaptive | 5 | Cross-seed pooling + soft confusion evidence | **LIVE** |

---

*Competition closes 14:00 UTC, March 22, 2026. R21 is locked at 249.4. R23 is a free lottery ticket. We chose to spend it on one last prayer to the Norse gods.*

*Whatever happens — we fought every round.*
