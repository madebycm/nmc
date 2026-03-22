"""FINAL NUCLEAR — Codex GPT-5.4 guided. Cross-seed pooled key-level update.

Key findings from Codex analysis:
- Settlement repeat probability: 27%. Ruin: 2.4%. Cell-level counting DANGEROUS for these.
- Cross-seed key-level JS divergence < 0.04 — safe to pool all 5 seeds.
- 10,125 pooled key observations — extremely robust empirical distributions.
- Use z_round (not z_seed) for calibration to reduce noise.

Strategy:
1. Pool ALL observations across ALL 5 seeds for key-level update (10K+ samples)
2. Very aggressive A_key=5 (we have thousands of samples — can trust them)
3. Cell-level C ADAPTIVE by observed class stochasticity:
   - If obs shows class 0/4/5 (land/forest/ocean, high repeat): C=3
   - If obs shows class 1/2/3 (settlement/port/ruin, low repeat): C=8
4. NN at 25% — smoothing prior only
5. Use z_round for all calibration lookups
"""

import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[FINAL] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent
ROUND_ID = "93c39605-628f-4706-abd9-08582f8b61d7"
NUM_CLASSES = 6
PROB_FLOOR = 0.003
A_KEY = 5  # aggressive — we have 1000+ samples per key from cross-seed pooling
NN_WEIGHT = 0.25  # NN is just a smoothing prior

CODE_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}

# Codex leave-one-out confusion matrix: P(true class | observed class)
# From repeated-cell analysis across all seeds.
# Row = observed class, values = P(true class is 0,1,2,3,4,5)
CONFUSION = {
    0: np.array([0.778, 0.164, 0.005, 0.014, 0.039, 0.000]),  # 78% repeat
    1: np.array([0.517, 0.270, 0.011, 0.031, 0.170, 0.000]),  # 27% repeat
    2: np.array([0.270, 0.216, 0.378, 0.054, 0.081, 0.000]),  # 38% repeat
    3: np.array([0.476, 0.381, 0.024, 0.024, 0.095, 0.000]),  # 2.4% repeat
    4: np.array([0.135, 0.198, 0.004, 0.011, 0.653, 0.000]),  # 65% repeat
    5: np.array([0.000, 0.000, 0.000, 0.000, 0.000, 1.000]),  # 100% repeat
}
STABLE_CLASSES = {0, 4, 5}
VOLATILE_CLASSES = {1, 2, 3}


def load_seed_data(seed_idx):
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


def compute_cross_seed_key_counts():
    """Pool key-level observations across ALL 5 seeds. ~10K samples."""
    key_counts = {}
    for s in range(5):
        ig, obs = load_seed_data(s)
        grid = np.array(ig)
        h, w = grid.shape
        dist_map = settlement_distance_map(grid)
        for ob in obs[:9]:  # base 9 only
            vp = ob.get("viewport", {})
            og = np.array(ob.get("grid", []))
            vx, vy = vp.get("x", 0), vp.get("y", 0)
            for dy in range(og.shape[0]):
                for dx in range(og.shape[1]):
                    gy, gx = vy + dy, vx + dx
                    if gy >= h or gx >= w:
                        continue
                    key = calibration_key(int(grid[gy, gx]), dist_map[gy, gx], is_coastal(grid, gy, gx))
                    cls = CODE_TO_CLASS.get(int(og[dy, dx]), 0)
                    if key not in key_counts:
                        key_counts[key] = np.zeros(NUM_CLASSES, dtype=np.float32)
                    key_counts[key][cls] += 1
    return key_counts


def compute_cell_counts(observations, h=40, w=40):
    counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    n_eff = np.zeros((h, w), dtype=np.float32)
    # Track which class was observed (for stability-aware C)
    obs_class = np.full((h, w), -1, dtype=np.int32)  # last observed class
    for obs in observations:
        vp = obs.get("viewport", {})
        og = np.array(obs.get("grid", []))
        vx, vy = vp.get("x", 0), vp.get("y", 0)
        for dy in range(og.shape[0]):
            for dx in range(og.shape[1]):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < h and 0 <= gx < w:
                    cls = CODE_TO_CLASS.get(int(og[dy, dx]), 0)
                    counts[gy, gx, cls] += 1
                    n_eff[gy, gx] += 1
                    obs_class[gy, gx] = cls
    return counts, n_eff, obs_class


def predict_seed(seed_idx, calibration, z_round, pooled_key_counts):
    initial_grid, observations = load_seed_data(seed_idx)
    grid = np.array(initial_grid)
    h, w = grid.shape
    dist_map = settlement_distance_map(grid)

    # Step 1: Build Dirichlet prediction using CROSS-SEED pooled key-level update
    dir_pred = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            key = calibration_key(int(grid[y, x]), dist_map[y, x], is_coastal(grid, y, x))

            # Base prior from z-model using z_round (not z_seed — less noise)
            z_model = calibration.get("z_model", {})
            if key in z_model:
                model = z_model[key]
                intercept = np.array(model["intercept"], dtype=np.float32)
                slope = np.array(model["slope"], dtype=np.float32)
                prior_mean = np.clip(intercept + slope * z_round, 0.001, None)
                prior_mean = prior_mean / prior_mean.sum()
            else:
                prior_mean = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES

            # Cross-seed pooled key-level posterior
            if key in pooled_key_counts:
                kc = pooled_key_counts[key]
                n_key = kc.sum()
                updated = (A_KEY * prior_mean + kc) / (A_KEY + n_key)
            else:
                updated = prior_mean

            dir_pred[y, x] = updated

    dir_pred = floor_and_normalize(dir_pred)

    # Step 2: NN prediction
    try:
        from strategy import compute_context_vector, estimate_z_from_context
        all_obs = {seed_idx: observations}
        ctx = compute_context_vector(all_obs, [initial_grid.tolist()])
        import nn_predict
        nn_pred = nn_predict.predict(initial_grid, z=z_round, context=ctx)
    except Exception as e:
        log.warning(f"  NN failed: {e}")
        nn_pred = None

    # Step 2: Geometric blend NN + updated Dirichlet → q0
    if nn_pred is not None:
        eps = 1e-8
        log_blend = NN_WEIGHT * np.log(nn_pred + eps) + (1 - NN_WEIGHT) * np.log(dir_pred + eps)
        q0 = np.exp(log_blend)
        q0 = floor_and_normalize(q0)
        log.info(f"  NN blended at {NN_WEIGHT:.0%}")
    else:
        q0 = dir_pred

    # Step 3: Cell-level Bayesian nudge — SOFT EVIDENCE via confusion matrix
    # For n=1: instead of hard one-hot [0,0,1,0,0,0], use confusion row
    #   P(true|obs=c) from Codex leave-one-out analysis. This spreads mass
    #   to likely alternatives, preventing KL blow-up on volatile classes.
    # For n>=2: use hard counts (multiple samples self-correct)
    cell_counts, n_eff, obs_class = compute_cell_counts(observations)

    pred = q0.copy()
    nudged = soft_nudged = 0
    for y in range(h):
        for x in range(w):
            n = n_eff[y, x]
            if n < 1:
                continue
            counts = cell_counts[y, x]
            dominant_class = int(counts.argmax())

            if n == 1:
                # SOFT EVIDENCE: replace one-hot with confusion-matrix row
                obs_cls = int(obs_class[y, x])
                soft_counts = CONFUSION.get(obs_cls, counts)
                # C_cell controls how much model vs soft evidence
                if obs_cls in STABLE_CLASSES:
                    C_cell = 4.0  # stable: trust soft evidence moderately
                else:
                    C_cell = 6.0  # volatile: trust model more
                pred[y, x] = (C_cell * q0[y, x] + soft_counts) / (C_cell + 1.0)
                soft_nudged += 1
            else:
                # n>=2: hard counts, stability-aware C
                max_frac = counts.max() / n
                if max_frac >= 0.75 and dominant_class in STABLE_CLASSES:
                    C_cell = 3.0
                elif dominant_class in VOLATILE_CLASSES:
                    C_cell = 8.0
                else:
                    C_cell = 5.0
                pred[y, x] = (C_cell * q0[y, x] + counts) / (C_cell + n)
            nudged += 1

    log.info(f"  Cell nudge: {nudged} cells ({soft_nudged} soft-evidence n=1)")

    # Step 4: Physics mask + floor
    pred = apply_physics_mask(pred, initial_grid)

    return pred.tolist()


def main():
    from api import AstarAPI
    from config import TOKEN
    from strategy import compute_context_vector, estimate_z_from_context

    api = AstarAPI(TOKEN)
    calibration = json.loads((BASE / "calibration.json").read_text())

    # Compute round-level z from ALL observations
    all_obs = {}
    all_grids = []
    for s in range(5):
        ig, obs = load_seed_data(s)
        all_obs[s] = obs
        all_grids.append(ig.tolist())
    round_ctx = compute_context_vector(all_obs, all_grids)
    z_round = estimate_z_from_context(round_ctx)
    log.info(f"z_round = {z_round:.4f} (pooled across all seeds)")

    # Cross-seed pooled key counts (10K+ samples)
    pooled_key_counts = compute_cross_seed_key_counts()
    total = sum(v.sum() for v in pooled_key_counts.values())
    log.info(f"Cross-seed pooled: {len(pooled_key_counts)} keys, {total:.0f} total observations")

    for seed_idx in range(5):
        log.info(f"=== Seed {seed_idx} ===")
        prediction = predict_seed(seed_idx, calibration, z_round, pooled_key_counts)
        resp = api.submit(ROUND_ID, seed_idx, prediction)
        log.info(f"  SUBMITTED seed {seed_idx}: {resp}")

    log.info("=== FINAL NUCLEAR COMPLETE ===")


if __name__ == "__main__":
    main()
