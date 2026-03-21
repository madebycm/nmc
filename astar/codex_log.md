
## [2026-03-20 12:25 UTC] Calibration update (30 files)

Prioritize `2`, but keep it very low-DOF.

`1)` No. `30` GT files is still small relative to a `27`-key model, and your own LORO result already says higher key counts overfit. I would not add feature keys yet.

`2)` Yes, but only a tiny nonlinearity: `quadratic` or a `1-knot piecewise/hinge` z term. The round z values are clearly not linear, especially the round-3 collapse and rebound in 4/5/6. That is the one change here with a real expected gain.

`3)` Only if you already see calibration error in residuals vs confidence. Without that evidence, confidence scaling feels lower-yield than fixing the z-shape.

So: keep the feature set fixed, try a 2-3 parameter nonlinear z conditioner, and re-run LORO. If that does not beat the current linear version, I would stop there rather than widen the model.

## [2026-03-20 12:26 UTC] Round 6 scored 58.8

1. `Probably yes, but not conclusively.` A 58.8 on round 6 is consistent with `z` being overestimated from mid-sim, especially since you already saw survival-based mid-sim `z` fail badly in round 4. But it could also mean the `slope*z` term is too aggressive. Treat this as evidence of both `z` noise and overreaction to `z`.

2. `Yes.` Replace raw `survival_rate` with a small feature set from the mid-sim state:
   - class share vector / entropy
   - leader margin and concentration
   - momentum over the last few checkpoints
   - per-class hazard/elimination rate
   - disagreement between independent simulators / seeds  
   Fit `z` from these jointly, not from survival alone.

3. `Yes, lower it.` If `z` is estimated from mid-sim, cut confidence from `30` to roughly `10-15`, or make it dynamic:
   - high confidence only when multiple signals agree
   - otherwise shrink toward the global prior

4. Quick wins:
   - gate the `z` prior: only use it when estimated `z` is extreme and stable
   - clip the slope effect
   - ensemble `z-conditioned prior` with `global prior`
   - calibrate on residuals by round to detect when mid-sim `z` is untrustworthy

## [2026-03-20 12:27 UTC] Pre-solve round 7, z=0.1611109346960873

Use `z ≈ 0.09` rather than `0.161`.

Reasoning: the only signal here I’d trust strongly is `avg_food = 0.67`, which is deep in your catastrophe-indicating zone, while `survival_rate = 1.0` is explicitly not reliable from prior rounds. The rest also looks weak-to-stagnant, not healthy: near-zero wealth, tiny population growth, low port activity. So I would downshift from the regime-hedging mean, but not all the way to the floor since settlements/factions are still intact. `0.08–0.10` feels like the right band.

## [2026-03-20 14:54 UTC] Calibration update (35 files)

Do `2`, but keep it tiny: add a **1-breakpoint piecewise z term** (or quadratic with strong shrinkage). That’s the only change here with a clear expected gain.

Why:
- The new round z’s are not smooth around one line: rounds `1,2,6,7` cluster near `0.41`, round `4` is mid, and round `3` is a near-zero outlier. That pattern is exactly where linear conditioning underfits.
- Going from 27 keys to more keys is still the wrong direction. Your earlier 310/475 result already says feature expansion is not the bottleneck.
- Confidence scaling is only worth touching if LORO shows systematic miscalibration by confidence bucket. Otherwise expected gain is small.

Recommendation:
1. **Do not add keys.**
2. **Try piecewise/quadratic z conditioning, heavily regularized.**
3. **Leave confidence scaling alone unless validation plots show calibration error.**

## [2026-03-20 14:54 UTC] Round 7 scored 38.4

1. Yes, if your deployed `z_hat` was pushed above `z_mean` by mid-sim survival. A 38.4 outcome is consistent with an over-optimistic `z` estimate, not proof, but enough to treat survival-based `z` as unreliable unless late-round calibration supports it.

2. Replace `survival_rate` as the main proxy. Try a small `z` estimator using:
- improvement slope over time
- top-k score gap / score concentration
- duplicate-state or collision rate
- terminal/dead-end mix
- expansion efficiency (useful children per expansion)

3. Reduce prior confidence when `z` is estimated. Keep `30` only for oracle/near-oracle `z`. For mid-sim `z_hat`, shrink hard toward the global prior:
- start with confidence `10-15`
- or `conf = 30 * reliability(z_hat)`, where reliability is learned from LORO

4. Quick wins for next round:
- default to `z_mean` unless estimator confidence clears a threshold
- cap `z_hat` to a conservative band around mean
- ensemble multiple mid-sim signals instead of survival alone
- log `z_hat`, uncertainty, and score by time slice so you can fit a calibration curve immediately after the round

## [2026-03-20 15:04 UTC] Pre-solve round 8, z=0.05723905563354492

Use `z = 0.05` (slight downward adjustment from `0.0572`).

Reasoning: mid-sim `survival_rate=1.0` is too optimistic to trust, and the rest of the state still looks weak rather than healthy: low ports, low population, low wealth, fragmented factions. I would not push all the way to catastrophic `0.018` unless you have a truly extreme end-state signal, but I also wouldn’t leave z unchanged because that bakes in known survival bias. `0.05` keeps the hedge while modestly correcting toward the fragile regime.

## [2026-03-20 17:52 UTC] Calibration update (40 files)

`2` is the only change with clear expected gain.

`1)` No. `40` GT files does not materially change the overfitting story. You already learned that `310/475` keys hurt, which says feature expansion is not the bottleneck.

`2)` Yes, but keep it tiny: add a `1-breakpoint piecewise/hinge` z term, or a weak quadratic with strong shrinkage. Your round z’s still look regime-like, not linear: `1/2/6/7` cluster near `0.41`, `4/5` are mid, `3/8` are low. That is exactly where a single line underfits.

`3)` Not yet. Leave confidence scaling alone unless LORO shows systematic miscalibration by confidence bucket, or unless deployment uses noisy `z_hat` rather than oracle z. Expected gain there is smaller than fixing the z-shape first.

## [2026-03-20 17:52 UTC] Round 8 scored 84.4

1. Probably yes. An 84.4 on round 8 is hard to reconcile with a very low `z` unless either `z_hat` was too low or the linear `intercept + slope*z` effect is misspecified for low-`z` rounds. Treat this as evidence to distrust low-`z` mid-sim estimates, not to lean harder on them.

2. Yes. Replace raw `survival_rate` with a small `z_hat` model using:
- elimination hazard slope over time
- leader/rank volatility
- survivor concentration or entropy
- matchup asymmetry / spread
- snapshot-to-snapshot stability

3. Yes. Lower prior confidence from `30` to roughly `10-15` when `z` is estimated mid-sim. Better: make confidence dynamic, `conf = f(z_hat uncertainty, snapshot time)`, and shrink toward the global prior when uncertainty is high.

4. Quick wins for next round:
- Use a mixture prior: `w * Dir(z_hat) + (1-w) * Dir(global)`
- Learn `w` from backtests, not by hand
- Clip extreme `z_hat` influence
- Replace linear `z` effect with bins or a spline
- Prefer later-sim snapshots only; early survival looks too noisy to be trusted

## [2026-03-20 18:12 UTC] Pre-solve round 9, z=0.28658536076545715

Use `z = 0.27`.

`0.2866` is a solid prior, but I’d shade slightly downward because mid-sim `survival_rate=1.0` has historically overstated end-state health. The rest of the snapshot doesn’t scream catastrophe: ports exist, faction/settlement counts are still broad, and there’s no clear extreme-regime signal unless `avg_food` is on the same raw scale as your `<10` rule. So I would not swing hard either way. `0.27` keeps the hedge across regimes while adding a modest correction for known mid-sim optimism.

## [2026-03-20 20:54 UTC] Calibration update (45 files)

1. Don’t add more keys.  
45 files is still tiny relative to 27 keys, and your 310/475-key result already says extra capacity is being spent on noise, not signal.

2. Don’t add quadratic/piecewise z yet.  
That’s effectively more capacity again. Unless LORO residuals vs. z show a consistent curved pattern across folds, expected gain is weak.

3. The only change with a clear expected upside is confidence scaling.  
Your round z’s are uneven and some are near-zero (`3`, `8`), which screams “variable reliability,” not “needs more features.” I’d try a stronger learned shrinkage so z-conditioned predictions get pulled toward the global mean when confidence is low, and trusted more only when z signal is strong/stable.

So: keep the 27-key linear model, but improve calibration/shrinkage. That’s the safest likely gain.

## [2026-03-20 20:55 UTC] Round 9 scored 89.1

1. Not by itself. `z_9 = 0.275` is very close to `z_mean = 0.289`, so Round 9’s `89.1` does not strongly point to a bad z estimate. Action: compare Round 9 score under `oracle-z`, `z_hat`, and `z_mean`; if `z_hat` is not clearly better than `z_mean`, your z estimator is not adding value.

2. Yes: stop using `survival_rate` alone. Action: predict `z` from a small pre-end-state feature set:
   - elimination hazard over the last few windows
   - change in alive count, not just level
   - stack/score concentration (`Gini`, entropy, top-1/top-3 share)
   - leader margin and its trend
   - rank volatility / churn
Train a regularized regressor and evaluate `R²(z_hat, z_true)` LORO.

3. Yes. If `z` is inferred mid-sim, reduce prior confidence from `30` unless predictor confidence is high. Action: use adaptive confidence, e.g. `alpha_eff = 30 * w`, where `w` comes from z-prediction quality or interval width.

4. Quick wins:
   - shrink toward global prior: `prior = w*Dir(z_hat) + (1-w)*Dir(mean_z)`
   - clip `z_hat` to training range
   - gate z-conditioning off when mid-sim signals are unstable
   - log per-round uplift vs `z_mean` immediately after each round

## [2026-03-20 21:04 UTC] Pre-solve round 10, z=0.037854891270399094

Use `z = 0.04` (effectively keep `0.037854891270399094`, just round).

Reasoning: don’t push higher on `survival=1.0` because mid-sim survival has already proven noisy, and the economy/activity signals are still weak (`port_rate`, `wealth`, food not strong). But there also isn’t a true catastrophe flag here, so forcing z down toward `0.018` would be overreacting. Low-but-not-minimum is the best hedge across regimes.

## [2026-03-20 23:54 UTC] Calibration update (50 files)

1. Don’t add more keys. `50` GT files is still only `10` round-level `z` values, and your `310/475` result already says capacity is not the bottleneck.

2. Don’t add quadratic/piecewise yet. With only `10` true z support points, extra z-shape is still easy to overfit. I’d only try it if LORO residuals vs `z` show a consistent curve across folds.

3. Change confidence scaling only for live `z_hat`, not for oracle-z backtests. Fixed `30` was already neutral in prior testing, but deployment uses noisy `z_hat`, so the clear upside is adaptive shrinkage:
`alpha_eff = 30 * w`, where `w` reflects z-estimator reliability, or just drop estimated-z concentration to `10-15` and blend toward global mean.

So the only change with clear expected gain is smarter confidence shrinkage for estimated `z`.

## [2026-03-20 23:55 UTC] Round 10 scored 86.2

1. Not by itself. `86.2` only suggests the prior helped overall. It implies your `z` estimate was off only if your live `z_hat` for round 10 was materially above the true low `z=0.0575`. Check that first; if `|z_hat - 0.058|` was large, treat the round as luck/robustness, not validation.

2. Yes: stop using raw `survival_rate` alone. Add mid-sim features that capture collapse dynamics:
- first-death timing
- death-rate slope over recent windows
- concentration/entropy of survivors
- leader change frequency / margin stability
- pairwise matchup imbalance or board/control asymmetry
Use a small calibrated regressor to predict `z` from these, not a single proxy.

3. Yes: lower or make confidence adaptive. Keep `30` only when the `z` estimator is high-confidence. Quick rule:
- `alpha = c * [w * p(.|z_hat) + (1-w) * p(.|z_mean)]`
- start with `c=20`
- let `w` come from estimator reliability; if only survival-based, keep `w <= 0.3`

4. Quick wins next round:
- fall back to `z_mean` when signals disagree
- predict a `z` interval, not point `z`
- clip extreme `z_hat` early in sim
- backtest adaptive confidence immediately on prior rounds

## [2026-03-21 00:31 UTC] Pre-solve round 11, z=0.5208333134651184

Use `z = 0.419` (clip to the top of the GT-supported healthy range), not `0.5208`.

Reasoning: `0.5208` is an extrapolation beyond the range the z-conditioned model was fit on, so it adds model risk without clear evidence. Mid-sim `survival=1.0` is not trustworthy enough to justify that boost, and you don’t have a clean catastrophic signal either. Clipping at `0.419` preserves the “healthy” interpretation while staying in-distribution and avoiding overconfidence.

## [2026-03-21 03:21 UTC] Calibration update (55 files)

1. Do not add more keys yet. `55` GT files is still small for expanding a `27`-key linear model, and your earlier `310/475` result already says capacity is not the bottleneck.

2. Try non-linear `z` conditioning, but only as a simple piecewise model, not quadratic. Your round `z` values are clearly multi-regime: very low (`~0.02–0.07`), mid (`~0.23–0.33`), and high (`~0.41–0.50`). That pattern is exactly where a linear mapping usually misses. A 3-bin or hinge function has a plausible gain with limited overfit risk.

3. Don’t change confidence scaling as a separate experiment yet. The clearer likely gain is to make scaling `z`-aware via the piecewise conditioning above. Global rescaling alone doesn’t have strong evidence from what you shared.

## [2026-03-21 03:21 UTC] Round 11 scored 79.9

1. `79.9` suggests round 11 was likely a genuinely high-`z` round, not a mean-`z` round. Action: compare your pre-round predicted score at `z_hat` vs at `z_mean`. If `79.9` is much closer to the high-`z` prediction, the estimate was directionally right; if not, `z_hat` was off. Don’t overread a 3-point gap from oracle `82.9`.

2. Yes. Replace raw `survival_rate` with a small feature set:
- survival by subgroup/class, not just overall
- time-derivative of survival (`d surv / dt`) and acceleration
- entropy/concentration of survivors across classes
- early elimination order / hazard rank
- variance across parallel sims; unstable rounds should imply low confidence

3. Yes: lower fixed confidence `30`, or make it adaptive. Action: use `alpha = alpha_base * R²_z` or similar, where `R²_z` is out-of-fold z-prediction quality. If z is uncertain, shrink toward the global prior automatically.

4. Quick wins:
- Stop using a point `z`; use a 3-point mixture (`z_low`, `z_mid`, `z_high`) and average priors.
- Calibrate on features from the same mid-sim timestamp only.
- Add a “trust gate”: if z-estimator uncertainty is high, fall back toward global mean.

## [2026-03-21 03:32 UTC] Pre-solve round 12, z=0.6380090713500977

Use `z = 0.419` (or `0.40` if you want a tiny extra hedge), not `0.638`.

Reasoning: `0.638` is outside the GT-supported range, so it forces risky extrapolation in a linear `intercept + slope*z` model. The current dynamics do not show your only reliable catastrophe signals, so I would not push `z` downward. But mid-sim survival is clearly too optimistic historically, so I also would not reward the 100% survival signal with an above-range `z`. Best move: cap at the healthiest in-range value.

## [2026-03-21 06:22 UTC] Pre-solve round 13, z=0.21544715762138367

Use `z = 0.14`.

`0.215` feels too optimistic: mid-sim `survival=1.0` has already proven unreliable, while `avg_food=0.66`, `avg_wealth=0.006`, low population, and weak port activity all point to a fragile state. If your `food < 10` rule is on the same scale, this is exactly the kind of extreme signal you said is actually trustworthy. I would not go full-catastrophic, but I would shift clearly below the mean hedge. A sensible band is `0.12-0.15`, centered at `0.14`.

## [2026-03-21 09:11 UTC] Calibration update (65 files)

1. **Do not add more feature keys yet.**  
65 GT files is still small relative to a 27-key z-conditioned linear model, and your earlier 310/475-key result already says variance is beating bias. More keys is the least likely clear win.

2. **Try confidence scaling first.**  
That’s the highest-probability gain: low-dimensional, directly targets calibration error, and is much less overfit-prone than adding features. I’d test a simple shrinked scaling or clipped/saturating scaling.

3. **Only then try mild non-linear z conditioning.**  
Not full quadratic. Use a **piecewise-linear / 2-3 bin** transform on `z` only, with strong shrinkage toward linear. That can capture the obvious low-z vs high-z regime changes without exploding capacity.

So: **3 first, then a very constrained version of 2, and skip 1 for now.**

## [2026-03-21 09:11 UTC] Round 13 scored 92.3

1. Yes: `92.3` is evidence your round-13 `z` was probably misestimated, especially if the model pushed away from the global prior because `z13=0.226` is below `z_mean=0.306`. Given oracle-`z` beats mean by `+13`, bad `z` can hurt a lot. Treat this as “`z` helps only when estimated well,” not proof the linear prior is wrong.

2. Replace `survival_rate` with early-trajectory signals:
- death timing / hazard curve, not just alive %
- class-specific survival and elimination order
- board-control / resource / position advantage proxies
- velocity features: change over last 1-2 sim checkpoints
- disagreement/variance across sims as an uncertainty feature

3. Yes: lower effective confidence when `z` is inferred from mid-sim only. Don’t keep confidence fixed at `30`.
- Use `confidence = 30 * reliability(zhat)`
- If `zhat` comes mostly from survival, cut to ~`10-15`
- Or shrink slope toward 0 when `z` uncertainty is high

4. Quick wins for next round:
- Blend priors: `P = w*P(.|zhat) + (1-w)*P(.|z_mean)`
- Make `w` depend on `z`-estimator confidence
- Add hazard/velocity features to the `z` predictor immediately
- Clip extreme `zhat` toward mean unless multiple signals agree

## [2026-03-21 09:22 UTC] Pre-solve round 14, z=0.46396395564079285

Use `z = 0.419` (or `0.41` if you want a tiny safety margin), not `0.46396`.

Reasoning: your current estimate is above the GT-supported range, so feeding `0.46396` into a linear `intercept + slope*z` model is extrapolation, which is usually less reliable than the mid-sim “healthy” signal itself. The catastrophic indicators you trust are not present here: survival is high, food is low-ish but nowhere near the disaster threshold, and port/population look stable. So I’d treat this as healthy-but-not-superhuman and clip to the top of the validated range.

## [2026-03-21 12:46 UTC] Pre-solve round 15, z=0.32863849401474

Use `z = 0.26`.

Reasoning: I would not submit `0.3286` unchanged. `survival_rate=1.0` is explicitly not trustworthy mid-sim, while `avg_food=0.699` is far into your own “extreme catastrophe” warning zone if that metric is on the GT scale. Tiny `avg_wealth` and low `port_rate` also don’t support a clearly healthy regime. I also wouldn’t crash all the way to catastrophic `z`, because settlement count and faction count don’t scream total collapse. So a moderate downward adjustment to about `0.26` is the best hedge.

## [2026-03-21 15:01 UTC] Calibration update (75 files)

1. No. `75` files is still small for expanding past `27` keys, and your earlier `310/475` result already says capacity is not the bottleneck. I’d expect more keys to add variance, not gain.

2. Maybe, but only in a very constrained way. Test one extra `z^2` term or a single hinge/piecewise split if LORO residuals vs `z` show curvature. The round values suggest possible nonlinearity at very low `z`, but I would not add flexible piecewise bins.

3. Yes. Confidence scaling is the clearest next lever with expected gain. It changes shrinkage without materially increasing model capacity, so it’s the lowest-overfit path. I’d tune it first with LORO, then only keep quadratic `z` if it adds incremental lift on top.

## [2026-03-21 15:02 UTC] Round 15 scored 90.0

1. No, not by itself. Round 15’s realized `z=0.328` is almost exactly the overall mean `0.322`, so a `90.0` score does not point to a bad z estimate. It suggests the round was won mostly by the non-z features, or by interactions your linear `intercept + slope*z` term does not capture.

2. Yes. Replace `survival_rate` as the main z proxy with mid-sim signals that are closer to end-state structure:
- class-count vector / alive-by-class composition
- elimination order and recent hazard rate
- concentration/entropy of survivors
- leader margin / pairwise matchup imbalance
- trend features: last-N-step changes, not just snapshot levels

3. Yes: lower or gate the prior confidence. With weak z observability, `30` is likely too rigid. Quick rule:
- use `10-15` when z is inferred only from noisy mid-sim signals
- keep `30` only when z posterior is tight
- blend `z-conditioned prior` with `global prior` using uncertainty-based weights

4. Quick wins for next round:
- stop point-estimating z; average predictions over a posterior on z
- add nonlinear z terms or bin z into regimes
- train a direct mid-sim -> end-state model, then feed that into the prior
- backtest confidence shrinkage first; it’s probably the fastest gain

## [2026-03-21 15:06 UTC] Pre-solve round 16, z=0.3115384578704834

Use `z ≈ 0.24` rather than `0.3115`.

Reasoning: I would discount the `100%` mid-sim survival heavily, because you already know it failed badly in round 4. The stronger signal is stress: `avg_food=0.735` and near-zero wealth suggest a fragile regime, even if not full catastrophe yet. I would not slam it to catastrophic `z`, because settlements/factions are still broad and there’s no clear collapse signal beyond scarcity. So a moderate downward adjustment from the hedge value is the safer play.
