#!/usr/bin/env python3
"""Test empirical anchoring at different weights with exact competition scorer.

Sweeps anchor_weight to find optimal value on R13-R16.
"""
import json, numpy as np, logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from strategy import (
    compute_context_vector, estimate_z_from_context,
    ensemble_predict, load_calibration, compute_empirical_observations,
    empirical_anchor, floor_and_normalize, NUM_CLASSES,
)
import nn_predict

GT_DIR = Path("ground_truth")
OBS_DIR = Path("observations")
CAL = json.loads(Path("calibration.json").read_text())
ROUND_Z = {int(k): v for k, v in CAL.get("round_z", {}).items()}


def exact_score(pred, gt):
    """Exact competition metric: entropy-weighted KL."""
    eps = 1e-10
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    entropy = -np.sum(gt * np.log(np.clip(gt, eps, 1)), axis=-1)
    kl = np.sum(gt * np.log(np.clip(gt, eps, 1) / np.clip(pred, eps, 1)), axis=-1)
    total_ent = entropy.sum()
    if total_ent < eps:
        return 100.0
    w = entropy / total_ent
    return float(100 * np.exp(-3 * (w * kl).sum()))


def eval_round_with_anchor(round_num, concentration):
    """Evaluate one round with specific anchor weight."""
    obs_dir = OBS_DIR / f"round_{round_num}"
    if not obs_dir.exists():
        return None

    initial_grids = []
    all_observations = {}
    base_observations = {}

    for seed_idx in range(5):
        grid_file = obs_dir / f"initial_seed_{seed_idx}.json"
        if not grid_file.exists():
            return None
        initial_grids.append(json.loads(grid_file.read_text()))
        obs_file = obs_dir / f"observations_seed_{seed_idx}.json"
        if obs_file.exists():
            obs = json.loads(obs_file.read_text())
            all_observations[seed_idx] = obs
            base_observations[seed_idx] = obs[:9]
        else:
            all_observations[seed_idx] = []
            base_observations[seed_idx] = []

    context = compute_context_vector(base_observations, initial_grids)
    z = estimate_z_from_context(context)
    calibration = load_calibration()

    scores = []
    for seed_idx in range(5):
        gt_file = GT_DIR / f"round_{round_num}_seed_{seed_idx}.json"
        if not gt_file.exists():
            continue
        gt = json.loads(gt_file.read_text())["ground_truth"]

        # Get base prediction (NN + Dirichlet blend, NO anchoring)
        from strategy import dirichlet_predict
        dir_pred = dirichlet_predict(initial_grids[seed_idx], all_observations[seed_idx], calibration, z=z)

        nn_pred = nn_predict.predict(initial_grids[seed_idx], z=z, context=context)
        if nn_pred is not None:
            # Apply NN weight curve
            NN_PEAK = 0.65; NN_HEALTHY = 0.20
            t1, t2, t3, t4, t5 = 0.05, 0.12, 0.25, 0.35, 0.60
            if z < t1: nn_weight = 0.0
            elif z < t2: nn_weight = NN_PEAK * 0.4 * ((z-t1)/(t2-t1))
            elif z < t3: nn_weight = NN_PEAK * (0.4 + 0.4*((z-t2)/(t3-t2)))
            elif z < t4: nn_weight = NN_PEAK
            else: nn_weight = NN_PEAK * (1.0 - min((z-t4)/(t5-t4), 1.0)) + NN_HEALTHY * min((z-t4)/(t5-t4), 1.0)

            if nn_weight > 0:
                eps = 1e-8
                log_blend = nn_weight * np.log(nn_pred + eps) + (1-nn_weight) * np.log(dir_pred + eps)
                blended = np.exp(log_blend)
                blended = floor_and_normalize(blended)
            else:
                blended = dir_pred
        else:
            blended = dir_pred

        # Apply Dirichlet conjugate posterior update
        if concentration < 999999:
            obs_counts, n_observed = compute_empirical_observations(all_observations[seed_idx])
            if n_observed.max() >= 1:
                blended = empirical_anchor(blended, obs_counts, n_observed, concentration=concentration)

        scores.append(exact_score(blended, gt))

    return np.mean(scores) if scores else None


# Sweep concentration values on recent rounds
# concentration = inf means pure model (no observation update)
# concentration = 0 means pure observations
test_rounds = [13, 14, 15, 16]
concentrations = [999999, 200, 100, 50, 30, 20, 15, 10, 5]

print(f"{'conc':>8}", end="")
for rn in test_rounds:
    z = ROUND_Z.get(rn, 0)
    regime = 'H' if z > 0.40 else 'M' if z > 0.15 else 'C'
    print(f" | R{rn}({regime})", end="")
print(" |   AVG")
print("-" * 70)

best_conc = 999999
best_avg = 0
for conc in concentrations:
    nn_predict._models.clear()
    scores = {}
    print(f"{conc:>8}", end="")
    for rn in test_rounds:
        s = eval_round_with_anchor(rn, conc)
        scores[rn] = s
        print(f" | {s:>7.2f}" if s else " |    N/A", end="")
    avg = np.mean([v for v in scores.values() if v])
    print(f" | {avg:>6.2f}")
    if avg > best_avg:
        best_avg = avg
        best_conc = conc

print(f"\nBest: concentration={best_conc} → avg={best_avg:.2f}")
