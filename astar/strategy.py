"""Prediction strategies for Astar Island — ensemble with Global Context Vector."""

import numpy as np
from pathlib import Path
import json
import logging

log = logging.getLogger(__name__)

CODE_TO_CLASS = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0,
}

PROB_FLOOR = 0.003  # Harness v2 full sweep: 0.003 > 0.005
NUM_CLASSES = 6
CONTEXT_DIM = 8

CALIBRATION_FILE = Path(__file__).parent / "calibration.json"


def floor_and_normalize(pred: np.ndarray) -> np.ndarray:
    """CRITICAL: enforce min probability floor, renormalize. Prevents KL=infinity."""
    pred = np.maximum(pred, PROB_FLOOR)
    return pred / pred.sum(axis=-1, keepdims=True)


# ── Global Context Vector ────────────────────────────────────────────


def compute_context_vector(
    observations_by_seed: dict[int, list[dict]],
    initial_grids: list,
) -> np.ndarray:
    """Compute 8-dim Global Context Vector from ALL observations across seeds.

    Takes dict {seed_idx: [obs, ...]} to avoid seed misattribution bug.
    Uses GRID comparison (initial vs observed cells) since API settlement
    list only contains alive settlements.

    All 5 seeds share same hidden params. Pool all observations.

    Dimensions:
    0: settlement survival rate (initially settled → still settled/port)
    1: port survival rate (initially port → still port)
    2: ruin frequency (ruin cells / land cells)
    3: forest fraction (forest cells / land cells)
    4: collapse rate (initially settled → empty/ruin/forest)
    5: expansion rate (initially empty → settlement/port)
    6: entropy proxy (faction diversity + food level)
    7: z (= survival rate, backward compat)
    """
    ctx = np.zeros(CONTEXT_DIM, dtype=np.float32)

    init_arrays = {}
    if initial_grids:
        for i, g in enumerate(initial_grids):
            init_arrays[i] = np.array(g) if not isinstance(g, np.ndarray) else g

    # Grid-based statistics
    initially_settled_seen = 0
    initially_settled_survived = 0
    initially_port_seen = 0
    initially_port_survived = 0
    initially_empty_seen = 0
    initially_empty_expanded = 0
    ruin_cells = 0
    forest_cells = 0
    land_cells = 0

    # Settlement list stats (secondary)
    total_food = 0.0
    alive_count = 0
    n_factions = set()

    for seed_idx, obs_list in observations_by_seed.items():
        init_grid = init_arrays.get(seed_idx)
        if init_grid is None:
            continue

        for obs in obs_list:
            for s in obs.get("settlements", []):
                if s.get("alive", False):
                    alive_count += 1
                    total_food += s.get("food", 0)
                if s.get("owner_id") is not None:
                    n_factions.add(s["owner_id"])

            vp = obs.get("viewport", {})
            obs_grid = obs.get("grid")
            if obs_grid is None:
                continue

            arr = np.array(obs_grid)
            vx, vy = vp.get("x", 0), vp.get("y", 0)
            for dy in range(arr.shape[0]):
                for dx in range(arr.shape[1]):
                    gy, gx = vy + dy, vx + dx
                    if gy >= init_grid.shape[0] or gx >= init_grid.shape[1]:
                        continue

                    init_cell = init_grid[gy, gx]
                    obs_cell = arr[dy, dx]

                    if obs_cell != 10:
                        land_cells += 1
                    if obs_cell == 3:
                        ruin_cells += 1
                    if obs_cell == 4:
                        forest_cells += 1

                    if init_cell == 1:
                        initially_settled_seen += 1
                        if obs_cell in (1, 2):
                            initially_settled_survived += 1

                    if init_cell == 2:
                        initially_port_seen += 1
                        initially_settled_seen += 1
                        if obs_cell == 2:
                            initially_port_survived += 1
                        if obs_cell in (1, 2):
                            initially_settled_survived += 1

                    if init_cell in (0, 11):
                        initially_empty_seen += 1
                        if obs_cell in (1, 2):
                            initially_empty_expanded += 1

    if initially_settled_seen > 0:
        ctx[0] = initially_settled_survived / initially_settled_seen
    if initially_port_seen > 0:
        ctx[1] = initially_port_survived / initially_port_seen
    if land_cells > 0:
        ctx[2] = ruin_cells / land_cells
        ctx[3] = forest_cells / land_cells
    if initially_settled_seen > 0:
        ctx[4] = 1.0 - (initially_settled_survived / initially_settled_seen)
    if initially_empty_seen > 0:
        ctx[5] = initially_empty_expanded / initially_empty_seen
    faction_diversity = min(len(n_factions) / 10.0, 1.0)
    food_avg = total_food / max(alive_count, 1)
    ctx[6] = faction_diversity * 0.5 + min(food_avg / 200.0, 0.5)
    ctx[7] = ctx[0]

    log.info(f"Context vector: survive={ctx[0]:.3f} port={ctx[1]:.3f} ruin={ctx[2]:.3f} "
             f"forest={ctx[3]:.3f} collapse={ctx[4]:.3f} expand={ctx[5]:.3f} "
             f"entropy={ctx[6]:.3f} z={ctx[7]:.3f}")
    return ctx


def estimate_z_from_context(ctx: np.ndarray) -> float:
    """Extract scalar z from context vector for v2/v3 backward compat."""
    return float(ctx[7]) if ctx is not None else 0.283


# ── Empirical Anchoring ──────────────────────────────────────────────


def compute_empirical_observations(
    observations: list[dict], map_h: int = 40, map_w: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Count observed cell types from simulation queries.

    Returns (counts, n_observed) where:
    - counts: (H, W, NUM_CLASSES) — observation counts per cell per class
    - n_observed: (H, W) — number of times each cell was observed
    """
    counts = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)
    n_observed = np.zeros((map_h, map_w), dtype=np.float32)

    for obs in observations:
        vp = obs.get("viewport", {})
        obs_grid = obs.get("grid")
        if obs_grid is None:
            continue
        arr = np.array(obs_grid)
        vx, vy = vp.get("x", 0), vp.get("y", 0)
        for dy in range(arr.shape[0]):
            for dx in range(arr.shape[1]):
                gy, gx = vy + dy, vx + dx
                if 0 <= gy < map_h and 0 <= gx < map_w:
                    cls = CODE_TO_CLASS.get(arr[dy, dx], 0)
                    counts[gy, gx, cls] += 1
                    n_observed[gy, gx] += 1

    return counts, n_observed


def empirical_anchor(
    nn_pred: np.ndarray,
    obs_counts: np.ndarray,
    n_observed: np.ndarray,
    anchor_weight: float = 0.07,
) -> np.ndarray:
    """Blend NN prediction with empirical observations.

    Only anchors cells with n_obs >= 2 to avoid double-counting with Dirichlet
    (which already incorporates single observations as Bayesian updates).
    Conservative weight: observations are single stochastic samples.
    """
    result = nn_pred.copy()
    anchor_mask = n_observed >= 2  # Only anchor with 2+ observations

    if not anchor_mask.any():
        return result

    empirical = obs_counts.copy()
    empirical[anchor_mask] /= n_observed[anchor_mask, None]

    # Weight scales gently: 2 obs = base, 3+ = 1.5x base
    weight_map = np.zeros_like(n_observed)
    weight_map[anchor_mask] = anchor_weight * np.minimum(n_observed[anchor_mask] - 1, 2)

    for c in range(NUM_CLASSES):
        result[:, :, c] = (1 - weight_map) * nn_pred[:, :, c] + weight_map * empirical[:, :, c]

    return floor_and_normalize(result)


# ── Dirichlet Bayesian (legacy) ──────────────────────────────────────


def _settlement_distance_map(grid: np.ndarray) -> np.ndarray:
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


def _is_coastal(grid: np.ndarray, y: int, x: int) -> bool:
    h, w = grid.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 10:
            return True
    return False


def _calibration_key(code: int, dist: float, coastal: bool) -> str:
    dist_bucket = "near" if dist <= 3 else "mid" if dist <= 6 else "far"
    return f"{code}_{dist_bucket}_{'coast' if coastal else 'inland'}"


def _get_z_conditioned_prior(key, calibration, z, confidence=30):
    z_model = calibration.get("z_model", {})
    if z is not None and key in z_model:
        model = z_model[key]
        intercept = np.array(model["intercept"], dtype=np.float32)
        slope = np.array(model["slope"], dtype=np.float32)
        mean_prob = np.clip(intercept + slope * z, 0.001, None)
        mean_prob = mean_prob / mean_prob.sum()
        # Use per-key concentration from replay variance (calibrated)
        conc = model.get("concentration", confidence)
        return np.maximum(mean_prob * conc, 0.01)
    if key in calibration.get("priors", {}):
        return np.array(calibration["priors"][key], dtype=np.float32)
    return None


def get_dirichlet_prior(code, dist_to_settlement, coastal, calibration=None, z=None):
    if calibration:
        key = _calibration_key(code, dist_to_settlement, coastal)
        prior = _get_z_conditioned_prior(key, calibration, z)
        if prior is not None:
            return prior
    if code == 10:
        return np.array([200, 0.01, 0.01, 0.01, 0.01, 0.01])
    if code == 5:
        return np.array([0.01, 0.01, 0.01, 0.01, 0.01, 200])
    if code == 4:
        if dist_to_settlement <= 3:
            return np.array([0.5, 0.3, 0.1, 0.2, 15, 0.05])
        return np.array([0.3, 0.05, 0.02, 0.1, 25, 0.05])
    if code == 11:
        if dist_to_settlement <= 2:
            return np.array([3, 3, 1.0 if coastal else 0.3, 2, 0.5, 0.05])
        if dist_to_settlement <= 5:
            return np.array([8, 1.5, 0.5 if coastal else 0.2, 1, 1, 0.05])
        return np.array([25, 0.3, 0.1, 0.2, 2, 0.05])
    if code == 1:
        if coastal:
            return np.array([0.5, 3, 4, 3, 0.2, 0.05])
        return np.array([0.5, 5, 0.3, 4, 0.3, 0.05])
    if code == 2:
        return np.array([0.3, 1, 7, 3, 0.2, 0.05])
    if code == 3:
        if dist_to_settlement <= 3:
            return np.array([2, 1.5, 0.5 if coastal else 0.2, 2, 2, 0.05])
        return np.array([3, 0.3, 0.1, 2, 4, 0.05])
    return np.array([10, 0.5, 0.2, 0.5, 2, 0.05])


def load_calibration() -> dict | None:
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text())
    return None


def extract_round_dynamics(observations: list[dict]) -> dict:
    """Extract dynamics signals from settlement stats in observations."""
    total_settlements = 0
    total_alive = 0
    total_ports = 0
    total_pop = 0.0
    total_food = 0.0
    total_wealth = 0.0
    n_factions = set()

    for obs in observations:
        for s in obs.get("settlements", []):
            total_settlements += 1
            if s.get("alive", False):
                total_alive += 1
            if s.get("has_port", False):
                total_ports += 1
            total_pop += s.get("population", 0)
            total_food += s.get("food", 0)
            total_wealth += s.get("wealth", 0)
            if s.get("owner_id") is not None:
                n_factions.add(s["owner_id"])

    if total_settlements == 0:
        return {}

    return {
        "survival_rate": total_alive / total_settlements,
        "port_rate": total_ports / max(total_alive, 1),
        "avg_population": total_pop / max(total_alive, 1),
        "avg_food": total_food / max(total_alive, 1),
        "avg_wealth": total_wealth / max(total_alive, 1),
        "n_factions": len(n_factions),
        "total_settlements": total_settlements,
    }


def dirichlet_predict(initial_grid, observations, calibration=None, z=None):
    """z-conditioned Dirichlet-Bayesian prediction."""
    grid = np.array(initial_grid)
    h, w = grid.shape
    dist_map = _settlement_distance_map(grid)

    alpha = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            coastal = _is_coastal(grid, y, x)
            alpha[y, x] = get_dirichlet_prior(
                grid[y, x], dist_map[y, x], coastal, calibration, z=z,
            )

    counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for obs in observations:
        vp = obs.get("viewport", {})
        obs_grid = obs.get("grid")
        if obs_grid is None:
            continue
        arr = np.array(obs_grid)
        for dy in range(arr.shape[0]):
            for dx in range(arr.shape[1]):
                gy, gx = vp.get("y", 0) + dy, vp.get("x", 0) + dx
                if 0 <= gy < h and 0 <= gx < w:
                    cls = CODE_TO_CLASS.get(arr[dy, dx], 0)
                    counts[gy, gx, cls] += 1

    posterior = alpha + counts
    pred = posterior / posterior.sum(axis=-1, keepdims=True)
    return floor_and_normalize(pred)


# ── Main ensemble ────────────────────────────────────────────────────


def ensemble_predict(
    initial_grid: list[list[int]],
    observations: list[dict],
    calibration: dict | None = None,
    context: np.ndarray | None = None,
    observations_by_seed: dict[int, list[dict]] | None = None,
    initial_grids: list | None = None,
    nn_weight_nudge: float = 0.0,
) -> np.ndarray:
    """Full ensemble: NN (v2+v3+v4) + Dirichlet + Empirical Anchoring.

    1. Compute context vector from ALL observations (not just this seed)
    2. NN predicts with context + TTA
    3. Blend with Dirichlet-Bayesian
    4. Empirical anchoring with per-seed observations (n_obs >= 2 only)
    """
    if context is None and observations_by_seed:
        # Only compute if there are actual observations (not empty lists)
        has_obs = any(len(obs) > 0 for obs in observations_by_seed.values())
        if has_obs:
            context = compute_context_vector(observations_by_seed, initial_grids or [])

    z = estimate_z_from_context(context) if context is not None else 0.283

    # Dirichlet prediction
    dir_pred = dirichlet_predict(initial_grid, observations, calibration, z=z)

    # NN prediction (v2+v3+v4 ensemble)
    try:
        import nn_predict
        nn_pred = nn_predict.predict(initial_grid, z=z, context=context)
    except Exception as e:
        log.warning(f"NN prediction failed: {e}")
        nn_pred = None

    if nn_pred is None:
        blended = dir_pred
        log.info("Using Dirichlet-only (no NN available)")
    else:
        # Inverted-U NN weight curve (saturday.md directive):
        # NN peaks at moderate z, decays earlier and deeper for healthy z
        # Live evidence: R13 moderate=92.3 (good), R11/R14 healthy≈80 (weak), R12=29 (OOD)
        # Don't push peak above 0.65 without OOF evidence
        NN_PEAK = 0.65      # max NN weight at moderate z (proven at R13)
        NN_HEALTHY = 0.20   # lower floor for healthy (was 0.30 — less NN on healthy)
        t1, t2, t3 = 0.05, 0.12, 0.25  # ramp-up thresholds
        t4 = 0.35           # earlier healthy dropoff (was 0.40)
        t5 = 0.60           # reach floor faster (was 0.70)
        if z < t1:
            nn_weight = 0.0
            log.info(f"Catastrophic z={z:.3f} — pure Dirichlet (0% NN)")
        elif z < t2:
            frac = (z - t1) / (t2 - t1)
            nn_weight = NN_PEAK * 0.4 * frac
        elif z < t3:
            frac = (z - t2) / (t3 - t2)
            nn_weight = NN_PEAK * (0.4 + 0.4 * frac)
        elif z < t4:
            nn_weight = NN_PEAK
        else:
            # Healthy dropoff: linear decrease from NN_PEAK to NN_HEALTHY
            frac = min((z - t4) / (t5 - t4), 1.0)
            nn_weight = NN_PEAK * (1.0 - frac) + NN_HEALTHY * frac

        # Per-seed asymmetric nudge: can decrease NN weight more than increase
        # Avoiding a bad healthy seed > over-promoting a good one
        if nn_weight_nudge != 0.0 and nn_weight > 0:
            nn_weight_before = nn_weight
            nudge = 0.30 * nn_weight_nudge  # scale factor
            nn_weight = np.clip(nn_weight + nudge, nn_weight - 0.15, nn_weight + 0.10)
            nn_weight = max(0.0, min(nn_weight, NN_PEAK))
            if abs(nn_weight - nn_weight_before) > 0.01:
                log.info(f"Per-seed nudge: nn_weight {nn_weight_before:.3f} -> {nn_weight:.3f}")

        if nn_weight > 0:
            dir_weight = 1.0 - nn_weight
            eps = 1e-8
            log_blend = nn_weight * np.log(nn_pred + eps) + dir_weight * np.log(dir_pred + eps)
            blended = np.exp(log_blend)
            blended = floor_and_normalize(blended)
            log.info(f"Ensemble: NN weight={nn_weight:.2f}, Dirichlet weight={dir_weight:.2f}, z={z:.3f}")
        else:
            blended = dir_pred

    # Empirical anchoring: replay harness shows 0.0 is optimal, skip
    # obs_counts, n_observed = compute_empirical_observations(observations)
    # blended = empirical_anchor(blended, obs_counts, n_observed, anchor_weight=0.0)

    return blended


def predict_for_seed(
    initial_grid: list[list[int]],
    observations: list[dict],
    dynamics: dict | None = None,
    context: np.ndarray | None = None,
    observations_by_seed: dict[int, list[dict]] | None = None,
    initial_grids: list | None = None,
    seed_idx: int | None = None,
) -> np.ndarray:
    """Main entry point: best available prediction for one seed.

    If seed_idx provided + observations_by_seed available, computes per-seed
    z and applies asymmetric blend nudge (can decrease more than increase).
    """
    calibration = load_calibration()

    # Compute per-seed z nudge
    z_nudge = 0.0
    if seed_idx is not None and observations_by_seed and initial_grids:
        try:
            z_round = estimate_z_from_context(context) if context is not None else 0.283
            # Seed-local z from this seed's observations only
            seed_obs = observations_by_seed.get(seed_idx, observations)
            seed_ctx = compute_context_vector({0: seed_obs}, [initial_grids[seed_idx] if seed_idx < len(initial_grids) else initial_grid])
            z_seed = estimate_z_from_context(seed_ctx)
            z_nudge = z_seed - z_round
            log.info(f"Seed {seed_idx}: z_seed={z_seed:.3f}, z_round={z_round:.3f}, nudge={z_nudge:+.3f}")
        except Exception as e:
            log.warning(f"Per-seed z failed: {e}")

    return ensemble_predict(
        initial_grid, observations, calibration,
        context=context,
        observations_by_seed=observations_by_seed,
        initial_grids=initial_grids,
        nn_weight_nudge=z_nudge,
    )
