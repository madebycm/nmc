"""Prediction strategies for Astar Island — Dirichlet-Bayesian approach."""

import numpy as np
from pathlib import Path
import json

CODE_TO_CLASS = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0,
}

PROB_FLOOR = 0.01
NUM_CLASSES = 6

# Calibration file (learned from ground truth via calibrate.py)
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"


def floor_and_normalize(pred: np.ndarray) -> np.ndarray:
    """CRITICAL: enforce min probability floor, renormalize. Prevents KL=infinity."""
    pred = np.maximum(pred, PROB_FLOOR)
    return pred / pred.sum(axis=-1, keepdims=True)


def _settlement_distance_map(grid: np.ndarray) -> np.ndarray:
    """Compute Manhattan distance to nearest settlement/port for each cell."""
    h, w = grid.shape
    positions = list(zip(*np.where(np.isin(grid, [1, 2]))))
    if not positions:
        return np.full((h, w), 999.0)
    dist = np.full((h, w), 999.0)
    for sy, sx in positions:
        for y in range(h):
            for x in range(w):
                d = abs(y - sy) + abs(x - sx)
                if d < dist[y, x]:
                    dist[y, x] = d
    return dist


def _is_coastal(grid: np.ndarray, y: int, x: int) -> bool:
    """Check if cell is adjacent to ocean."""
    h, w = grid.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 10:
            return True
    return False


def get_dirichlet_prior(
    code: int, dist_to_settlement: float, coastal: bool,
    calibration: dict | None = None,
) -> np.ndarray:
    """Get Dirichlet prior alpha for a cell based on initial state features.

    Returns alpha vector of shape (6,) for classes:
    [Empty, Settlement, Port, Ruin, Forest, Mountain]
    """
    # If we have learned calibration from ground truth, use it
    if calibration and "priors" in calibration:
        key = _calibration_key(code, dist_to_settlement, coastal)
        if key in calibration["priors"]:
            return np.array(calibration["priors"][key], dtype=np.float32)

    # Hand-tuned priors (pre-calibration)
    if code == 10:  # Ocean — ALWAYS stays Empty
        return np.array([200, 0.01, 0.01, 0.01, 0.01, 0.01])
    if code == 5:   # Mountain — ALWAYS stays Mountain
        return np.array([0.01, 0.01, 0.01, 0.01, 0.01, 200])
    if code == 4:   # Forest — mostly stable
        if dist_to_settlement <= 3:
            return np.array([0.5, 0.3, 0.1, 0.2, 15, 0.05])
        return np.array([0.3, 0.05, 0.02, 0.1, 25, 0.05])
    if code == 11:  # Plains
        if dist_to_settlement <= 2:
            return np.array([3, 3, 1.0 if coastal else 0.3, 2, 0.5, 0.05])
        if dist_to_settlement <= 5:
            return np.array([8, 1.5, 0.5 if coastal else 0.2, 1, 1, 0.05])
        return np.array([25, 0.3, 0.1, 0.2, 2, 0.05])
    if code == 1:   # Settlement
        if coastal:
            return np.array([0.5, 3, 4, 3, 0.2, 0.05])
        return np.array([0.5, 5, 0.3, 4, 0.3, 0.05])
    if code == 2:   # Port
        return np.array([0.3, 1, 7, 3, 0.2, 0.05])
    if code == 3:   # Ruin
        if dist_to_settlement <= 3:
            return np.array([2, 1.5, 0.5 if coastal else 0.2, 2, 2, 0.05])
        return np.array([3, 0.3, 0.1, 2, 4, 0.05])
    # Empty (code 0) or unknown
    return np.array([10, 0.5, 0.2, 0.5, 2, 0.05])


def _calibration_key(code: int, dist: float, coastal: bool) -> str:
    """Create lookup key for calibration table."""
    dist_bucket = "near" if dist <= 3 else "mid" if dist <= 6 else "far"
    return f"{code}_{dist_bucket}_{'coast' if coastal else 'inland'}"


def load_calibration() -> dict | None:
    """Load learned calibration from ground truth analysis."""
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text())
    return None


def dirichlet_predict(
    initial_grid: list[list[int]],
    observations: list[dict],
    calibration: dict | None = None,
) -> np.ndarray:
    """Dirichlet-Bayesian prediction: prior from initial state, update with observations.

    This is the core prediction engine. For each cell:
    1. Compute Dirichlet prior alpha from (terrain_type, distance, coastal)
    2. Count observation occurrences per class
    3. Posterior = prior + counts
    4. Prediction = normalized posterior
    5. Floor at 0.01, renormalize
    """
    grid = np.array(initial_grid)
    h, w = grid.shape
    dist_map = _settlement_distance_map(grid)

    # Build prior alpha for every cell
    alpha = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            coastal = _is_coastal(grid, y, x)
            alpha[y, x] = get_dirichlet_prior(
                grid[y, x], dist_map[y, x], coastal, calibration
            )

    # Count observations per cell per class
    counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    for obs in observations:
        vp = obs["viewport"]
        obs_grid = np.array(obs["grid"])
        for dy in range(obs_grid.shape[0]):
            for dx in range(obs_grid.shape[1]):
                gy, gx = vp["y"] + dy, vp["x"] + dx
                if 0 <= gy < h and 0 <= gx < w:
                    cls = CODE_TO_CLASS.get(obs_grid[dy, dx], 0)
                    counts[gy, gx, cls] += 1

    # Posterior = prior + counts
    posterior = alpha + counts
    pred = posterior / posterior.sum(axis=-1, keepdims=True)

    return floor_and_normalize(pred)


def predict_for_seed(
    initial_grid: list[list[int]],
    observations: list[dict],
) -> np.ndarray:
    """Main entry point: best available prediction for one seed."""
    calibration = load_calibration()
    return dirichlet_predict(initial_grid, observations, calibration)
