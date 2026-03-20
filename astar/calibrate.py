"""Learn Dirichlet priors from ground truth + replay Monte Carlo data.

Run after completed rounds to improve predictions for future rounds.
Produces z-conditioned calibration: for each key, stores linear model
P(class | features, z) = intercept + slope * z, where z = settlement alive rate.

Replay enhancement: 782 MC samples give 782 unique z values (vs 8 from GT alone),
producing much smoother regression fits especially at extreme z values.
"""

import json
import numpy as np
from pathlib import Path

GT_DIR = Path(__file__).parent / "ground_truth"
REPLAY_DIR = Path(__file__).parent / "replays"
CALIBRATION_FILE = Path(__file__).parent / "calibration.json"
NUM_CLASSES = 6
GRID_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


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


def _dist_bucket(d: float) -> str:
    if d <= 3:
        return "near"
    if d <= 6:
        return "mid"
    return "far"


def _compute_round_z(round_num: int) -> float | None:
    """Compute z = settlement alive rate for a round from GT data."""
    alive_prob_sum = 0.0
    settlement_count = 0
    for seed in range(5):
        gt_file = GT_DIR / f"round_{round_num}_seed_{seed}.json"
        if not gt_file.exists():
            continue
        data = json.loads(gt_file.read_text())
        gt = np.array(data["ground_truth"])
        grid = np.array(data.get("initial_grid", []))
        if grid.size == 0:
            continue
        mask = np.isin(grid, [1, 2])
        settlement_count += int(mask.sum())
        # alive = P(settlement) + P(port) for initially-settled cells
        alive_prob_sum += float((gt[..., 1] + gt[..., 2])[mask].sum())

    if settlement_count == 0:
        return None
    return alive_prob_sum / settlement_count


def _load_replay_data() -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Load replay frame-50 outcomes as one-hot arrays with per-sample z.

    Returns list of (z_sample, initial_grid, onehot_outcome) tuples.
    Each onehot_outcome is (H, W, 6) with 1.0 in the observed class.
    """
    results = []
    if not REPLAY_DIR.exists():
        return results

    for round_dir in sorted(REPLAY_DIR.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round_"):
            continue
        for seed_dir in sorted(round_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for sample_path in sorted(seed_dir.glob("sample_*.json")):
                try:
                    data = json.loads(sample_path.read_text())
                    frames = data["frames"]
                    grid0 = np.array(frames[0]["grid"], dtype=np.int32)
                    grid50 = np.array(frames[50]["grid"], dtype=np.int32)

                    h, w = grid0.shape
                    # Per-sample z from frame 50
                    settle_mask = np.isin(grid0, [1, 2])
                    n_settle = int(settle_mask.sum())
                    if n_settle > 0:
                        alive = np.isin(grid50[settle_mask], [1, 2]).sum()
                        z_sample = float(alive) / n_settle
                    else:
                        z_sample = 0.283

                    # Convert grid50 to one-hot (H, W, 6)
                    onehot = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
                    for code, cls in GRID_TO_CLASS.items():
                        mask = (grid50 == code)
                        onehot[mask, cls] = 1.0

                    results.append((z_sample, grid0, onehot))
                except Exception:
                    continue
    return results


def _build_priors(gt_files: list[Path], confidence_cap: float = 50) -> dict:
    """Build Dirichlet prior table from a set of GT files (global mean)."""
    accum = {}
    for gt_file in gt_files:
        data = json.loads(gt_file.read_text())
        gt = np.array(data["ground_truth"])
        initial_grid = np.array(data.get("initial_grid", []))
        if initial_grid.size == 0:
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

    priors = {}
    for key, distributions in accum.items():
        arr = np.array(distributions)
        mean_dist = arr.mean(axis=0)
        n_samples = len(distributions)
        confidence = min(confidence_cap, max(5, n_samples / 10))
        alpha = np.maximum(mean_dist * confidence, 0.01)
        priors[key] = alpha.tolist()

    return priors


def _build_z_conditioned(gt_files: list[Path], round_z: dict[int, float],
                         replay_data: list | None = None) -> dict:
    """Build z-conditioned linear model: P(class | key, z) = intercept + slope * z.

    Uses GT data (smooth probabilities, weighted as 20 effective samples each)
    plus replay MC samples (one-hot outcomes, 1 effective sample each).
    782 replay samples → 782 unique z values (vs 8 from GT alone).
    """
    # Collect per-key: lists of (z, probability_vector, weight)
    key_data: dict[str, list[tuple[float, np.ndarray, float]]] = {}

    # GT data: smooth probability vectors, high weight
    GT_WEIGHT = 20.0  # each GT cell ≈ 20 MC samples
    for gt_file in gt_files:
        data = json.loads(gt_file.read_text())
        rn = int(gt_file.stem.split("_")[1])
        z = round_z.get(rn)
        if z is None:
            continue

        gt = np.array(data["ground_truth"])
        initial_grid = np.array(data.get("initial_grid", []))
        if initial_grid.size == 0:
            continue

        h, w = initial_grid.shape
        dist_map = _settlement_distance_map(initial_grid)

        for y in range(h):
            for x in range(w):
                code = int(initial_grid[y, x])
                dist = dist_map[y, x]
                coastal = _is_coastal(initial_grid, y, x)
                key = f"{code}_{_dist_bucket(dist)}_{'coast' if coastal else 'inland'}"
                key_data.setdefault(key, []).append((z, gt[y, x], GT_WEIGHT))

    # Replay data: one-hot outcomes, low weight but unique z values
    n_replay = 0
    if replay_data:
        for z_sample, grid0, onehot in replay_data:
            h, w = grid0.shape
            dist_map = _settlement_distance_map(grid0)
            for y in range(h):
                for x in range(w):
                    code = int(grid0[y, x])
                    dist = dist_map[y, x]
                    coastal = _is_coastal(grid0, y, x)
                    key = f"{code}_{_dist_bucket(dist)}_{'coast' if coastal else 'inland'}"
                    key_data.setdefault(key, []).append((z_sample, onehot[y, x], 1.0))
            n_replay += 1

    print(f"  Z-model: {len(gt_files)} GT files (w={GT_WEIGHT}) + {n_replay} replay samples (w=1)")

    # Weighted linear regression per key + per-key concentration from replay variance
    z_model = {}
    for key, points in key_data.items():
        zs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        ws = np.array([p[2] for p in points])

        w_sum = ws.sum()
        z_wmean = (ws * zs).sum() / w_sum
        y_wmean = (ws[:, None] * ys).sum(axis=0) / w_sum

        z_var = (ws * (zs - z_wmean) ** 2).sum()

        n_unique_z = len(np.unique(np.round(zs, 3)))

        if z_var < 1e-12 or n_unique_z < 3:
            slope = np.zeros(NUM_CLASSES)
            intercept = y_wmean
        else:
            slope = (ws[:, None] * (zs - z_wmean)[:, None] * (ys - y_wmean)).sum(axis=0) / z_var
            intercept = y_wmean - slope * z_wmean

        # Fixed concentration — per-key variance from replay tested but neutral impact
        concentration = 30.0

        z_model[key] = {
            "intercept": intercept.tolist(),
            "slope": slope.tolist(),
            "concentration": concentration,
        }

    return z_model


def calibrate():
    """Learn z-conditioned priors from GT + replay Monte Carlo data."""
    gt_files = sorted(GT_DIR.glob("round_*_seed_*.json"))
    if not gt_files:
        print("No ground truth files found. Run solver to harvest after rounds complete.")
        return

    print(f"Found {len(gt_files)} ground truth files")

    # Compute z for each round
    round_nums = sorted({int(f.stem.split("_")[1]) for f in gt_files})
    round_z = {}
    for rn in round_nums:
        z = _compute_round_z(rn)
        if z is not None:
            round_z[rn] = z
            regime = "catastrophic" if z < 0.05 else "moderate" if z < 0.35 else "healthy"
            print(f"  Round {rn}: z={z:.3f} ({regime})")

    # Load replay data (782 MC samples with per-sample z)
    print(f"\nLoading replay data...")
    replay_data = _load_replay_data()
    if replay_data:
        replay_zs = [r[0] for r in replay_data]
        print(f"  {len(replay_data)} replay samples, z range: {min(replay_zs):.3f} — {max(replay_zs):.3f}")
    else:
        print("  No replay data found (using GT only)")

    # Build global mean priors (GT only — fallback)
    priors_all = _build_priors(gt_files)

    # Build z-conditioned model (GT + replay)
    z_model = _build_z_conditioned(gt_files, round_z, replay_data=replay_data)

    print(f"\n  Global: {len(priors_all)} keys from {len(gt_files)} files")
    print(f"  Z-conditioned: {len(z_model)} keys ({len(replay_data)} replay + {len(gt_files)} GT)")
    z_vals = list(round_z.values())
    print(f"  GT z range: {min(z_vals):.3f} — {max(z_vals):.3f} ({len(z_vals)} unique)")
    if replay_data:
        print(f"  Replay z range: {min(replay_zs):.3f} — {max(replay_zs):.3f} ({len(replay_data)} unique)")

    # Show high-impact keys
    for key in ["11_near_inland", "11_near_coast", "1_near_inland", "1_near_coast"]:
        if key in z_model:
            m = z_model[key]
            print(f"  {key}: intercept={[f'{v:.3f}' for v in m['intercept']]}")
            print(f"  {' '*len(key)}  slope  ={[f'{v:.3f}' for v in m['slope']]}")

    calibration = {
        "priors": priors_all,
        "z_model": z_model,
        "round_z": {str(k): v for k, v in round_z.items()},
        "z_mean": float(np.mean(z_vals)),
        "z_std": float(np.std(z_vals)),
        "n_gt_files": len(gt_files),
        "n_replay_samples": len(replay_data),
        "n_keys": len(priors_all),
    }

    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2))
    print(f"\nCalibration saved to {CALIBRATION_FILE}")


if __name__ == "__main__":
    calibrate()
