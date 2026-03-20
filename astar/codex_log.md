
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
