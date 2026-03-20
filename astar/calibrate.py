"""Learn Dirichlet priors from ground truth data (analysis endpoint).

Run after completed rounds to improve predictions for future rounds.
Reads ground truth from ground_truth/ directory, computes optimal priors
per (terrain_type, distance_bucket, coastal) combination.
"""

import json
import numpy as np
from pathlib import Path

GT_DIR = Path(__file__).parent / "ground_truth"
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
NUM_CLASSES = 6


def _settlement_distance_map(grid: np.ndarray) -> np.ndarray:
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
    h, w = grid.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 10:
            return True
    return False


def _dist_bucket(d: float) -> str:
    if d <= 3:
        return "near"
    if d <= 6:
        return "mid"
    return "far"


def calibrate():
    """Learn priors from all available ground truth files."""
    gt_files = sorted(GT_DIR.glob("round_*_seed_*.json"))
    if not gt_files:
        print("No ground truth files found. Run solver to harvest after rounds complete.")
        return

    print(f"Found {len(gt_files)} ground truth files")

    # Accumulate: for each (code, dist_bucket, coastal) → list of ground truth distributions
    accum = {}

    for gt_file in gt_files:
        data = json.loads(gt_file.read_text())
        gt = np.array(data["ground_truth"])  # H×W×6
        initial_grid = np.array(data.get("initial_grid", []))

        if initial_grid.size == 0:
            print(f"  {gt_file.name}: no initial_grid, skipping")
            continue

        h, w = initial_grid.shape
        dist_map = _settlement_distance_map(initial_grid)

        for y in range(h):
            for x in range(w):
                code = int(initial_grid[y, x])
                dist = dist_map[y, x]
                coastal = _is_coastal(initial_grid, y, x)
                key = f"{code}_{_dist_bucket(dist)}_{'coast' if coastal else 'inland'}"

                if key not in accum:
                    accum[key] = []
                accum[key].append(gt[y, x].tolist())

    # Compute mean ground truth distribution per key → use as Dirichlet prior
    priors = {}
    for key, distributions in accum.items():
        arr = np.array(distributions)
        mean_dist = arr.mean(axis=0)
        # Scale to reasonable Dirichlet alpha (higher = more confident)
        # Use count-based scaling: more samples = more confident
        n_samples = len(distributions)
        confidence = min(50, max(5, n_samples / 10))
        alpha = mean_dist * confidence
        alpha = np.maximum(alpha, 0.01)  # Floor
        priors[key] = alpha.tolist()
        print(f"  {key}: n={n_samples}, alpha={[f'{a:.2f}' for a in alpha]}")

    calibration = {
        "priors": priors,
        "n_gt_files": len(gt_files),
        "n_keys": len(priors),
    }

    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2))
    print(f"\nCalibration saved to {CALIBRATION_FILE} ({len(priors)} keys)")


if __name__ == "__main__":
    calibrate()
