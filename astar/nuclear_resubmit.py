"""Nuclear resubmit for R23 — hierarchical empirical Bayes + lower healthy NN weight.

TWO EXPERT REVIEWS RECONCILED:
  Expert 1: Key-level grouped update, conservative cell C=8-12, NN 55-60%
  Expert 2: Aggressive C=3-4, NN 30-40%, trust observations heavily

  Key insight (Expert 1): n_eff per cell is 1-4, NOT 9. Each of the 9 tiles
  shows different cells. Only overlap zones (cols/rows 13-14, 25-27) get 2-4x.

SYNTHESIS (lean aggressive — free lottery ticket):
1. Key-level grouped Dirichlet update (A_key=20) — Expert 1's hierarchical idea
2. NN weight ~45% at z~0.46, ~35% at z~0.58 — between both experts
3. Cell-level Bayesian posterior: C=6(n=1), 4(n=2), 3(n≥4+agreement)
4. Linear-space posterior (both agree): P = (C*q0 + counts) / (C + n)
5. Physics masks + floor preserved

FREE LOTTERY TICKET: R21=249.4 locked. R23 needs 81.2+ raw to beat it.
"""

import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[nuclear] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent
ROUND_ID = "93c39605-628f-4706-abd9-08582f8b61d7"
NUM_CLASSES = 6
PROB_FLOOR = 0.003
A_KEY = 8  # ULTRA AGGRESSIVE key-level: trust R23-specific observations heavily

CODE_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def load_seed_data(seed_idx):
    """Load initial grid and observations for a seed."""
    ig = json.loads((BASE / f"observations/round_23/initial_seed_{seed_idx}.json").read_text())
    obs = json.loads((BASE / f"observations/round_23/observations_seed_{seed_idx}.json").read_text())
    return np.array(ig), obs


def settlement_distance_map(grid):
    h, w = grid.shape
    positions = list(zip(*np.where(np.isin(grid, [1, 2]))))
    if not positions:
        return np.full((h, w), 999.0)
    dist = np.full((h, w), 999.0)
    for sy, sx in positions:
        yy = np.abs(np.arange(h) - sy)[:, None]
        xx = np.abs(np.arange(w) - sx)[None, :]
        dist = np.minimum(dist, yy + xx)
    return dist


def is_coastal(grid, y, x):
    h, w = grid.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 10:
            return True
    return False


def calibration_key(code, dist, coastal):
    bucket = "near" if dist <= 3 else "mid" if dist <= 6 else "far"
    return f"{code}_{bucket}_{'coast' if coastal else 'inland'}"


def floor_and_normalize(pred):
    pred = np.maximum(pred, PROB_FLOOR)
    return pred / pred.sum(axis=-1, keepdims=True)


def apply_physics_mask(pred, initial_grid):
    grid = np.asarray(initial_grid, dtype=np.int32)
    result = pred.copy()
    ocean = grid == 5
    if ocean.any():
        result[ocean] = 0.0
        result[ocean, 5] = 1.0
    mountain = grid == 10
    if mountain.any():
        result[mountain] = 0.0
        result[mountain, 0] = 1.0
    return floor_and_normalize(result)


def compute_key_counts(initial_grid, observations):
    """Count observed class frequencies grouped by calibration key."""
    grid = np.array(initial_grid)
    h, w = grid.shape
    dist_map = settlement_distance_map(grid)

    key_counts = {}  # key -> np.array of shape (NUM_CLASSES,)

    for obs in observations:
        vp = obs.get("viewport", {})
        obs_grid = np.array(obs.get("grid", []))
        vx, vy = vp.get("x", 0), vp.get("y", 0)

        for dy in range(obs_grid.shape[0]):
            for dx in range(obs_grid.shape[1]):
                gy, gx = vy + dy, vx + dx
                if gy >= h or gx >= w:
                    continue
                key = calibration_key(
                    int(grid[gy, gx]), dist_map[gy, gx], is_coastal(grid, gy, gx)
                )
                cls = CODE_TO_CLASS.get(int(obs_grid[dy, dx]), 0)
                if key not in key_counts:
                    key_counts[key] = np.zeros(NUM_CLASSES, dtype=np.float32)
                key_counts[key][cls] += 1

    return key_counts


def compute_cell_counts(observations, h=40, w=40):
    """Count per-cell observations and class frequencies."""
    counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    n_eff = np.zeros((h, w), dtype=np.float32)

    for obs in observations:
        vp = obs.get("viewport", {})
        obs_grid = np.array(obs.get("grid", []))
        vx, vy = vp.get("x", 0), vp.get("y", 0)

        for dy in range(obs_grid.shape[0]):
            for dx in range(obs_grid.shape[1]):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < h and 0 <= gx < w:
                    cls = CODE_TO_CLASS.get(int(obs_grid[dy, dx]), 0)
                    counts[gy, gx, cls] += 1
                    n_eff[gy, gx] += 1

    return counts, n_eff


def updated_dirichlet_predict(initial_grid, observations, calibration, z, key_counts):
    """Dirichlet prediction with round-specific key-level posterior update."""
    grid = np.array(initial_grid)
    h, w = grid.shape
    dist_map = settlement_distance_map(grid)

    pred = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            key = calibration_key(int(grid[y, x]), dist_map[y, x], is_coastal(grid, y, x))

            # Get base prior from calibration z-model
            z_model = calibration.get("z_model", {})
            if key in z_model:
                model = z_model[key]
                intercept = np.array(model["intercept"], dtype=np.float32)
                slope = np.array(model["slope"], dtype=np.float32)
                conc = model.get("concentration", 30.0)
                prior_mean = np.clip(intercept + slope * z, 0.001, None)
                prior_mean = prior_mean / prior_mean.sum()
                prior_alpha = prior_mean * conc
            else:
                prior_alpha = np.ones(NUM_CLASSES, dtype=np.float32)

            # Step 1: Key-level grouped posterior update
            if key in key_counts:
                kc = key_counts[key]
                n_key = kc.sum()
                # Posterior predictive: (A_key * prior_mean + counts) / (A_key + n)
                prior_mean_norm = prior_alpha / prior_alpha.sum()
                updated = (A_KEY * prior_mean_norm + kc) / (A_KEY + n_key)
            else:
                updated = prior_alpha / prior_alpha.sum()

            pred[y, x] = updated

    return floor_and_normalize(pred)


def nn_weight_for_seed_z(z_seed):
    """ULTRA AGGRESSIVE: NN is just a smoothing prior, not the oracle."""
    # Flat 20% — NN exists only to put mass on unobserved classes
    return 0.20


def predict_seed(seed_idx, calibration, all_obs_by_seed, all_initial_grids):
    """Full prediction for one seed."""
    initial_grid, observations = load_seed_data(seed_idx)

    # Compute per-seed z from context
    from strategy import compute_context_vector, estimate_z_from_context
    seed_ctx = compute_context_vector({0: observations}, [initial_grid])
    z_seed = estimate_z_from_context(seed_ctx)

    # Also compute round-level z for NN
    round_ctx = compute_context_vector(all_obs_by_seed, all_initial_grids)
    z_round = estimate_z_from_context(round_ctx)

    log.info(f"Seed {seed_idx}: z_seed={z_seed:.3f}, z_round={z_round:.3f}")

    # Step 1: Key-level grouped counts from this seed's base 9 observations
    base_obs = observations[:9]  # first 9 are base tiling
    key_counts = compute_key_counts(initial_grid, base_obs)
    log.info(f"  Key-level: {len(key_counts)} keys with counts")

    # Step 1: Updated Dirichlet with round-specific key-level posterior
    dir_pred = updated_dirichlet_predict(initial_grid, observations, calibration, z_seed, key_counts)

    # Step 2: NN prediction
    try:
        import nn_predict
        nn_pred = nn_predict.predict(initial_grid, z=z_seed, context=seed_ctx)
    except Exception as e:
        log.warning(f"  NN failed: {e}")
        nn_pred = None

    # Step 2: Blend with lower NN weight
    nn_w = nn_weight_for_seed_z(z_seed)
    if nn_pred is not None:
        dir_w = 1.0 - nn_w
        eps = 1e-8
        log_blend = nn_w * np.log(nn_pred + eps) + dir_w * np.log(dir_pred + eps)
        q0 = np.exp(log_blend)
        q0 = floor_and_normalize(q0)
        log.info(f"  Blended: nn_weight={nn_w:.3f} (was 0.75)")
    else:
        q0 = dir_pred
        log.info(f"  Dirichlet-only (NN unavailable)")

    # Step 3: Cell-level Bayesian nudge
    cell_counts, n_eff = compute_cell_counts(observations)

    pred = q0.copy()
    h, w = 40, 40
    nudged = 0
    for y in range(h):
        for x in range(w):
            n = n_eff[y, x]
            if n < 1:
                continue
            # ULTRA AGGRESSIVE: observations are ground truth samples
            # PROB_FLOOR=0.003 is our only safety net
            counts = cell_counts[y, x]
            max_frac = counts.max() / n if n > 0 else 0
            if n == 1:
                C_cell = 3.0   # n=1: observation gets 25% weight
            elif n == 2:
                C_cell = 2.0   # n=2: observations get 50% weight
            elif n >= 4 and max_frac >= 0.75:
                C_cell = 1.0   # n≥4 + agreement: observations dominate (80%)
            else:
                C_cell = 2.0   # default: trust observations

            # Proper Bayesian posterior: (C * q0 + counts) / (C + n)
            pred[y, x] = (C_cell * q0[y, x] + counts) / (C_cell + n)
            nudged += 1

    log.info(f"  Cell nudge: {nudged} cells updated")

    # Step 4: Physics mask + floor
    pred = apply_physics_mask(pred, initial_grid)

    return pred.tolist()


def main():
    from api import AstarAPI
    from config import TOKEN

    api = AstarAPI(TOKEN)
    calibration = json.loads((BASE / "calibration.json").read_text())

    # Load all observations and initial grids for round-level context
    all_obs_by_seed = {}
    all_initial_grids = []
    for s in range(5):
        ig, obs = load_seed_data(s)
        all_obs_by_seed[s] = obs
        all_initial_grids.append(ig.tolist())

    # Predict and submit each seed
    for seed_idx in range(5):
        log.info(f"=== Seed {seed_idx} ===")
        prediction = predict_seed(seed_idx, calibration, all_obs_by_seed, all_initial_grids)

        resp = api.submit(ROUND_ID, seed_idx, prediction)
        log.info(f"  SUBMITTED seed {seed_idx}: {resp}")

    log.info("=== ALL 5 SEEDS SUBMITTED — NUCLEAR RESUBMIT COMPLETE ===")


if __name__ == "__main__":
    main()
