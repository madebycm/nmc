#!/usr/bin/env python3
"""Evaluate candidate models vs current v3f on all GT rounds.

Runs the FULL production prediction path (strategy.py + nn_predict.py)
with each model swapped in as v3, comparing scores against ground truth.
"""
import json, sys, logging, numpy as np, shutil
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# Import production code
from strategy import (
    compute_context_vector, estimate_z_from_context,
    predict_for_seed, dirichlet_predict, load_calibration,
    floor_and_normalize, NUM_CLASSES, CODE_TO_CLASS,
)
import nn_predict

GT_DIR = Path("ground_truth")
OBS_DIR = Path("observations")
V3_PATH = Path("astar_nn_v3.pt")
CAND_A = Path("candidate_a_v3.pt")
CAND_B = Path("candidate_b_v3.pt")
BACKUP = Path("astar_nn_v3.pt.bak")

CAL = json.loads(Path("calibration.json").read_text())
ROUND_Z = {int(k): v for k, v in CAL.get("round_z", {}).items()}


def score_prediction(pred: np.ndarray, gt_dist: np.ndarray) -> float:
    """Exact competition score: 100 * exp(-3 * entropy_weighted_kl).

    GT is (40,40,6) probability distribution. Uses entropy-weighted KL,
    not plain mean KL — high-entropy (uncertain) cells matter more.
    """
    eps = 1e-10
    gt = np.maximum(gt_dist, eps)
    pred_safe = np.maximum(pred, eps)
    # Per-cell entropy and KL
    entropy = -np.sum(gt * np.log(gt), axis=-1)  # (H, W)
    kl = np.sum(gt * np.log(gt / pred_safe), axis=-1)  # (H, W)
    # Entropy-weighted average
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0
    weights = entropy / total_entropy
    weighted_kl = (weights * kl).sum()
    return float(100.0 * np.exp(-3.0 * weighted_kl))


def eval_round(round_num: int) -> dict:
    """Evaluate one round using saved observations."""
    obs_dir = OBS_DIR / f"round_{round_num}"
    if not obs_dir.exists():
        return {}

    z = ROUND_Z.get(round_num, 0.283)
    seeds_count = 5
    initial_grids = []
    all_observations = {}
    base_observations = {}

    for seed_idx in range(seeds_count):
        grid_file = obs_dir / f"initial_seed_{seed_idx}.json"
        if not grid_file.exists():
            return {}
        grid = json.loads(grid_file.read_text())
        initial_grids.append(grid)
        obs_file = obs_dir / f"observations_seed_{seed_idx}.json"
        if obs_file.exists():
            obs = json.loads(obs_file.read_text())
            all_observations[seed_idx] = obs
            base_observations[seed_idx] = obs[:9]
        else:
            all_observations[seed_idx] = []
            base_observations[seed_idx] = []

    # Context from base observations (same as production)
    context = compute_context_vector(base_observations, initial_grids)

    seed_scores = []
    for seed_idx in range(seeds_count):
        gt_file = GT_DIR / f"round_{round_num}_seed_{seed_idx}.json"
        if not gt_file.exists():
            continue
        gt = json.loads(gt_file.read_text())
        gt_dist = np.array(gt["ground_truth"])

        pred = predict_for_seed(
            initial_grids[seed_idx],
            all_observations.get(seed_idx, []),
            context=context,
            observations_by_seed=base_observations,
            initial_grids=initial_grids,
            seed_idx=seed_idx,
        )
        score = score_prediction(pred, gt_dist)
        seed_scores.append(score)

    if not seed_scores:
        return {}
    return {"avg": np.mean(seed_scores), "min": min(seed_scores), "seeds": seed_scores}


def swap_model(path: Path):
    """Swap v3 model file for evaluation."""
    # Clear cached model
    nn_predict._models.pop("v3", None)
    shutil.copy2(path, V3_PATH)


def main():
    # Get all rounds with GT
    rounds = sorted(set(
        int(f.stem.split("_")[1]) for f in GT_DIR.glob("round_*_seed_*.json")
    ))
    print(f"Evaluating {len(rounds)} rounds: {rounds}")

    # Backup current model
    shutil.copy2(V3_PATH, BACKUP)

    results = {}
    models = {
        "v3f (current)": BACKUP,
        "Candidate A": CAND_A,
        "Candidate B": CAND_B,
    }

    for name, path in models.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"\n=== {name} ===")
        swap_model(path)
        results[name] = {}
        for rn in rounds:
            r = eval_round(rn)
            if r:
                regime = "healthy" if ROUND_Z.get(rn, 0) > 0.40 else "moderate" if ROUND_Z.get(rn, 0) > 0.15 else "catastrophic"
                results[name][rn] = r
                print(f"  R{rn:2d} (z={ROUND_Z.get(rn,0):.3f} {regime:12s}): avg={r['avg']:.1f}  min={r['min']:.1f}")

    # Restore original
    shutil.copy2(BACKUP, V3_PATH)
    nn_predict._models.pop("v3", None)
    BACKUP.unlink()

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Round':>6} {'z':>6} {'Regime':>12} ", end="")
    for name in models:
        print(f"| {name:>15} ", end="")
    print()
    print("-" * 70)

    for rn in rounds:
        z = ROUND_Z.get(rn, 0)
        regime = "healthy" if z > 0.40 else "moderate" if z > 0.15 else "catastrophic"
        print(f"  R{rn:2d}  {z:6.3f} {regime:>12} ", end="")
        for name in models:
            if rn in results.get(name, {}):
                print(f"| {results[name][rn]['avg']:>14.1f} ", end="")
            else:
                print(f"| {'N/A':>14} ", end="")
        print()

    # Weighted scores (what matters for leaderboard)
    print("\n--- Weighted (score × 1.05^round) ---")
    for name in models:
        if name not in results:
            continue
        best_weighted = 0
        best_round = 0
        for rn, r in results[name].items():
            w = 1.05 ** rn
            weighted = r["avg"] * w
            if weighted > best_weighted:
                best_weighted = weighted
                best_round = rn
        print(f"  {name}: best weighted = {best_weighted:.1f} (R{best_round})")


if __name__ == "__main__":
    main()
