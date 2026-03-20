"""
Seed-wise champion/challenger system.

Instead of using the same blend recipe for all 5 seeds in a round,
compute per-seed trust signals and pick the best recipe for each seed
independently. The API accepts one seed at a time, and only the last
submission per seed counts.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy

from nn_predict import (
    _load_model,
    _predict_single_v2v3,
    _predict_single_v4,
    _predict_with_tta_v2v3,
    _predict_with_tta_v4,
    encode_grid_v2v3,
)
from strategy import (
    dirichlet_predict,
    floor_and_normalize,
    load_calibration,
    compute_empirical_observations,
    empirical_anchor,
)

# ---------------------------------------------------------------------------
# Default recipes
# ---------------------------------------------------------------------------
DEFAULT_RECIPES: dict[str, dict] = {
    "catastrophic": {
        "nn_weight": 0.30,
        "v2_rel": 0.5,
        "v3_rel": 0.5,
        "v4_enabled": False,
        "anchor_weight": 0.70,
        "prob_floor": 0.02,
    },
    "conservative": {
        "nn_weight": 0.50,
        "v2_rel": 0.5,
        "v3_rel": 0.5,
        "v4_enabled": False,
        "anchor_weight": 0.50,
        "prob_floor": 0.015,
    },
    "moderate": {
        "nn_weight": 0.65,
        "v2_rel": 0.45,
        "v3_rel": 0.55,
        "v4_enabled": True,
        "anchor_weight": 0.35,
        "prob_floor": 0.01,
    },
    "aggressive": {
        "nn_weight": 0.80,
        "v2_rel": 0.40,
        "v3_rel": 0.40,
        "v4_enabled": True,
        "anchor_weight": 0.20,
        "prob_floor": 0.005,
    },
}


# ---------------------------------------------------------------------------
# Trust signal computation
# ---------------------------------------------------------------------------
def compute_seed_trust(
    seed_idx: int,
    initial_grid: list[list[int]],
    observations: list[dict],
    context: np.ndarray,
    z: float,
) -> dict:
    """Compute trust signals for one seed.

    Returns dict with:
    - z: estimated z from context (global)
    - local_z: z estimated from THIS seed's observations only
    - v2v3_agreement: 1.0 - mean_absolute_difference (higher = more agreement)
    - v4_outlier: how much v4 disagrees with v2/v3 consensus
    - observation_coverage: fraction of dynamic cells observed
    - settlement_density: fraction of initially-settled cells
    - dynamic_entropy: mean entropy of NN predictions on dynamic cells
    """
    grid = np.array(initial_grid)
    H, W = grid.shape

    # Settlement density
    settled = (grid == 1).sum()
    settlement_density = float(settled) / (H * W)

    # Identify dynamic cells: cells that are NOT always the same as initial
    # A cell is "static" if it's water (0) far from land, or interior land.
    # For simplicity: any cell within distance 2 of a land/water boundary is dynamic.
    from scipy.ndimage import binary_dilation

    land_mask = grid == 1
    water_mask = grid == 0
    boundary_land = land_mask & binary_dilation(water_mask, iterations=2)
    boundary_water = water_mask & binary_dilation(land_mask, iterations=2)
    dynamic_mask = boundary_land | boundary_water
    n_dynamic = dynamic_mask.sum()

    # Observation coverage: fraction of dynamic cells that have observations
    observed_cells = set()
    for obs in observations:
        r, c = obs["row"], obs["col"]
        if dynamic_mask[r, c]:
            observed_cells.add((r, c))
    observation_coverage = len(observed_cells) / max(n_dynamic, 1)

    # Local z estimation from observations
    local_z = _estimate_local_z(grid, observations)

    # V2/V3 predictions (no TTA for speed)
    model_v2v3 = _load_model("v2v3")
    model_v4 = _load_model("v4")

    pred_v2v3 = _predict_single_v2v3(model_v2v3, grid, context, z)
    pred_v4 = _predict_single_v4(model_v4, grid, context, z)

    # Split v2v3 into separate v2 and v3 by running with slightly perturbed z
    # to measure sensitivity. Instead, we compare v2v3 vs v4 as agreement proxy.
    # For true v2/v3 split we'd need separate models. Use v2v3 vs v4 difference.

    # V2V3 agreement: measure internal consistency via observation fit
    # Compute how well v2v3 predictions match observations
    v2v3_obs_fit = _observation_fit(pred_v2v3, observations, grid)
    v4_obs_fit = _observation_fit(pred_v4, observations, grid)

    # Agreement between v2v3 and v4
    if n_dynamic > 0:
        diff = np.abs(pred_v2v3[dynamic_mask] - pred_v4[dynamic_mask])
        mad = float(diff.mean())
        v2v3_agreement = 1.0 - min(mad, 1.0)

        # V4 outlier score: how much v4 deviates from v2v3
        v4_outlier = float(mad)
    else:
        v2v3_agreement = 1.0
        v4_outlier = 0.0

    # Dynamic entropy: mean entropy of predictions on dynamic cells
    if n_dynamic > 0:
        # pred shape is (H, W, C) with C classes
        if pred_v2v3.ndim == 3:
            dyn_probs = pred_v2v3[dynamic_mask]  # (n_dynamic, C)
            ent = np.array([
                scipy_entropy(p + 1e-10) for p in dyn_probs
            ])
            dynamic_entropy = float(ent.mean())
        elif pred_v2v3.ndim == 2:
            # Binary predictions (H, W) — treat as prob of class 1
            dyn_p = pred_v2v3[dynamic_mask]
            ent = -(dyn_p * np.log(dyn_p + 1e-10) +
                    (1 - dyn_p) * np.log(1 - dyn_p + 1e-10))
            dynamic_entropy = float(ent.mean())
        else:
            dynamic_entropy = 0.0
    else:
        dynamic_entropy = 0.0

    return {
        "seed_idx": seed_idx,
        "z": float(z),
        "local_z": float(local_z),
        "v2v3_agreement": float(v2v3_agreement),
        "v4_outlier": float(v4_outlier),
        "observation_coverage": float(observation_coverage),
        "settlement_density": float(settlement_density),
        "dynamic_entropy": float(dynamic_entropy),
        "v2v3_obs_fit": float(v2v3_obs_fit),
        "v4_obs_fit": float(v4_obs_fit),
    }


def _estimate_local_z(grid: np.ndarray, observations: list[dict]) -> float:
    """Estimate z from a single seed's observations.

    z represents change rate. Count fraction of observed cells that
    changed from their initial state.
    """
    if not observations:
        return 0.15  # prior

    changed = 0
    total = 0
    for obs in observations:
        r, c = obs["row"], obs["col"]
        initial_val = grid[r, c]
        observed_val = obs["value"]
        total += 1
        if observed_val != initial_val:
            changed += 1

    if total == 0:
        return 0.15
    return changed / total


def _observation_fit(
    pred: np.ndarray,
    observations: list[dict],
    grid: np.ndarray,
) -> float:
    """Measure how well predictions fit observations. Returns mean confidence
    assigned to the correct observed class (higher = better fit)."""
    if not observations:
        return 0.5

    fits = []
    for obs in observations:
        r, c = obs["row"], obs["col"]
        observed_val = obs["value"]

        if pred.ndim == 3:
            # (H, W, C) — get probability of observed class
            if observed_val < pred.shape[2]:
                fits.append(float(pred[r, c, observed_val]))
            else:
                fits.append(0.0)
        elif pred.ndim == 2:
            # (H, W) — probability of class 1
            p = float(pred[r, c])
            if observed_val == 1:
                fits.append(p)
            else:
                fits.append(1.0 - p)
        else:
            fits.append(0.5)

    return float(np.mean(fits)) if fits else 0.5


# ---------------------------------------------------------------------------
# Recipe selection
# ---------------------------------------------------------------------------
def select_recipe(trust: dict, recipes: dict | None = None) -> str:
    """Select best recipe name for this seed based on trust signals.

    Rules:
    - If z < 0.08: always 'catastrophic'
    - If v2v3_agreement < 0.7: use 'conservative' (models disagree)
    - If z > 0.30 and v2v3_agreement > 0.85: use 'aggressive'
    - Otherwise: use 'moderate'
    """
    z = trust["z"]
    local_z = trust.get("local_z", z)
    agreement = trust["v2v3_agreement"]

    # Use the more conservative z estimate between global and local
    effective_z = min(z, local_z) if local_z > 0 else z

    if effective_z < 0.08:
        return "catastrophic"

    if agreement < 0.7:
        return "conservative"

    if effective_z > 0.30 and agreement > 0.85:
        return "aggressive"

    return "moderate"


# ---------------------------------------------------------------------------
# Prediction with recipe
# ---------------------------------------------------------------------------
def predict_with_recipe(
    recipe_name: str,
    recipes: dict | None,
    initial_grid: list[list[int]],
    observations: list[dict],
    context: np.ndarray,
    z: float,
    calibration: dict,
) -> np.ndarray:
    """Apply a specific recipe to produce a prediction.

    Each recipe is a dict with keys:
    - nn_weight: float — weight for NN ensemble vs empirical anchor
    - v2_rel: float — relative v2 weight within NN blend
    - v3_rel: float — relative v3 weight within NN blend
    - v4_enabled: bool — whether to include v4 in the blend
    - anchor_weight: float — weight for empirical anchor
    - prob_floor: float — minimum probability floor

    Uses geometric blending (log-space averaging) for the NN models,
    then linear interpolation with the empirical anchor.
    """
    if recipes is None:
        recipes = DEFAULT_RECIPES
    recipe = recipes[recipe_name]

    grid = np.array(initial_grid)

    # --- NN predictions with TTA ---
    model_v2v3 = _load_model("v2v3")
    pred_v2v3 = _predict_with_tta_v2v3(model_v2v3, grid, context, z)

    if recipe["v4_enabled"]:
        model_v4 = _load_model("v4")
        pred_v4 = _predict_with_tta_v4(model_v4, grid, context, z)

    # --- Geometric blend of NN models ---
    eps = 1e-10
    log_v2v3 = np.log(pred_v2v3 + eps)

    if recipe["v4_enabled"]:
        log_v4 = np.log(pred_v4 + eps)
        # Distribute NN weight: v2_rel and v3_rel for v2v3, remainder for v4
        v2v3_share = recipe["v2_rel"] + recipe["v3_rel"]
        v4_share = 1.0 - v2v3_share
        total = v2v3_share + v4_share

        log_blend = (v2v3_share * log_v2v3 + v4_share * log_v4) / total
    else:
        log_blend = log_v2v3

    nn_pred = np.exp(log_blend)

    # Normalize NN predictions
    nn_pred = _normalize_probs(nn_pred)

    # --- Empirical anchor ---
    emp_obs = compute_empirical_observations(observations, grid.shape)
    anchor = empirical_anchor(grid, emp_obs, calibration, z)

    # --- Blend NN with anchor ---
    nn_w = recipe["nn_weight"]
    anc_w = recipe["anchor_weight"]
    # Normalize weights
    total_w = nn_w + anc_w
    nn_w /= total_w
    anc_w /= total_w

    blended = nn_w * nn_pred + anc_w * anchor

    # --- Apply probability floor ---
    blended = floor_and_normalize(blended, recipe["prob_floor"])

    return blended


def _normalize_probs(pred: np.ndarray) -> np.ndarray:
    """Normalize predictions so probabilities sum to 1 along last axis."""
    if pred.ndim == 3:
        sums = pred.sum(axis=-1, keepdims=True)
        sums = np.maximum(sums, 1e-10)
        return pred / sums
    elif pred.ndim == 2:
        return np.clip(pred, 0, 1)
    return pred


# ---------------------------------------------------------------------------
# Convenience: full per-seed pipeline
# ---------------------------------------------------------------------------
def run_seed_selection(
    seed_idx: int,
    initial_grid: list[list[int]],
    observations: list[dict],
    context: np.ndarray,
    z: float,
    calibration: dict,
    recipes: dict | None = None,
) -> tuple[np.ndarray, str, dict]:
    """Full pipeline: compute trust, select recipe, predict.

    Returns:
        (prediction, recipe_name, trust_signals)
    """
    if recipes is None:
        recipes = DEFAULT_RECIPES

    trust = compute_seed_trust(seed_idx, initial_grid, observations, context, z)
    recipe_name = select_recipe(trust, recipes)

    pred = predict_with_recipe(
        recipe_name, recipes,
        initial_grid, observations, context, z, calibration,
    )

    return pred, recipe_name, trust
