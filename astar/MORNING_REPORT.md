# Astar Island — Morning Report

> Last updated: 2026-03-20 21:04 UTC

## Round Status

| Round | Status | Score | Queries | Submissions | Strategy |
|-------|--------|-------|---------|-------------|----------|
| 1 | completed | pending | 0/50 | 0/5 seeds | — |
| 2 | completed | pending | 0/50 | 0/5 seeds | — |
| 3 | completed | 7.2 (rank #89) | 50/50 | 5/5 seeds | proximity_v1_overwritten_by_poller |
| 4 | completed | 79.9 (rank #32) | 45/50 | 5/5 seeds | dirichlet_v3_regime |
| 5 | completed | 75.4 (rank #46) | 45/50 | 5/5 seeds | dirichlet_v4_mixed |
| 6 | completed | 58.8 (rank #86) | 45/50 | 5/5 seeds | dirichlet_v4_mixed |
| 7 | completed | 38.4 (rank #147) | 45/50 | 5/5 seeds | ensemble_nn_v1 |
| 8 | completed | 84.4 (rank #49) | 50/50 | 5/5 seeds | ensemble_ctx_v2 |
| 9 | completed | 89.1 (rank #39) | 50/50 | 5/5 seeds | champion_challenger_v1 |
| 10 | active | pending | 50/50 | 5/5 seeds | champion_challenger_v1 |

## Leaderboard Top 10
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

## Our Position
Not yet ranked

## Ground Truth & Calibration
- GT files: 45
- Calibration keys: 27

## Actions Taken Overnight
- (waiting for events...)

- [2026-03-20 02:00 UTC] Autonomous mode started. Loop running every 3min. Goodnight!


- [2026-03-20 02:57 UTC] Round 3 scored: avg 7.2 (damaged by overwrite). GT harvested (15 files). Recalibrated (27 keys). Ready for round 4.

- [2026-03-20 03:03 UTC] Solved round #4 with calibrated Dirichlet strategy

- [2026-03-20 03:04 UTC] ROUND 4 SOLVED! Calibrated Dirichlet, 45/50 queries, 5/5 seeds. Dynamics: 100% survival, 52 factions, low conflict. Awaiting score.

- [2026-03-20 04:30 UTC] STRATEGY UPGRADE: Regime-aware Dirichlet (v3). Backtested: +5.6 to +8.7 pts on healthy rounds, +24 pts on catastrophic rounds. Separate "healthy" vs "catastrophic" calibrations, selected by observed dynamics.

- [2026-03-20 04:32 UTC] ROUND 4 RESUBMITTED with v3 regime-aware strategy (healthy priors, survival=100%). Expected ~84-87 vs ~78 for v1.


- [2026-03-20 05:51 UTC] ROUND 4 SCORED: avg 79.9, rank #32! GT harvested (20 files). Recalibrated with 4 rounds of data.

- [2026-03-20 06:54 UTC] STRATEGY FIX: Regime detection from mid-sim observations is UNRELIABLE (round 4 showed 100% mid-sim survival but only 4-18% end-state survival). Switched to v4_mixed: uses blended calibration from all GT, no regime selection. Backtested avg 74.0 across all rounds. This is safest default.

- [2026-03-20 07:04 UTC] Solved round #5 with v4_mixed strategy.

- [2026-03-20 09:54 UTC] ROUND 5 SCORED: avg 75.4, rank #46. GT harvested (25 files). Recalibrated with 5 rounds.


- [2026-03-20 09:10 UTC] Solved round #6 with calibrated Dirichlet strategy

- [2026-03-20 12:28 UTC] Solved round #7 with calibrated Dirichlet strategy


- [2026-03-20 15:04 UTC] Solved round #8 with calibrated Dirichlet strategy



- [2026-03-20 18:12 UTC] Solved round #9 with calibrated Dirichlet strategy


- [2026-03-20 21:04 UTC] Solved round #10 with calibrated Dirichlet strategy

## Issues / Alerts
- Round 3 predictions degraded by server overwrite (fixed — server poller killed)
