"""Replay harness — grid-search over ensemble parameters against ground truth.

Tests the EXACT end-to-end inference path (strategy.py -> nn_predict.py)
across all 7 rounds of ground truth. Optimizes for "healthy-round upside
with catastrophic floor": maximize best weighted round while keeping every
round above a catastrophic threshold.

Key optimization: NN model outputs are cached per (round, seed) since they
are independent of recipe parameters. Only the blending math varies.

Usage:
    python replay_harness.py              # full grid search
    python replay_harness.py --fast       # coarse grid, no TTA
    python replay_harness.py --validate   # re-run top 3 with TTA
"""

import itertools
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── project imports ──────────────────────────────────────────────────
from strategy import (
    compute_context_vector,
    compute_empirical_observations,
    dirichlet_predict,
    empirical_anchor,
    estimate_z_from_context,
    floor_and_normalize,
    load_calibration,
)
import nn_predict

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
GT_DIR = ROOT / "ground_truth"
OBS_DIR = ROOT / "observations"

CODE_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6
ROUND_WEIGHTS = {r: 1.05 ** r for r in range(1, 12)}


# ── Recipe (parameter set) ──────────────────────────────────────────

@dataclass
class Recipe:
    nn_weight: float = 0.50         # overall NN vs Dirichlet blend
    replay_weight: float = 0.15     # relative weight within NN ensemble
    v2_weight: float = 0.25         # relative weight within NN ensemble
    v3_weight: float = 0.60         # relative weight within NN ensemble
    anchor_weight: float = 0.0      # empirical anchoring strength
    prob_floor: float = 0.005       # minimum probability
    z_thresholds: list = field(default_factory=lambda: [0.08, 0.15, 0.30])

    def label(self) -> str:
        return (
            f"nn={self.nn_weight:.2f} rep={self.replay_weight:.2f} v2={self.v2_weight:.2f} "
            f"v3={self.v3_weight:.2f} anch={self.anchor_weight:.3f} "
            f"floor={self.prob_floor:.3f} zt={self.z_thresholds}"
        )


# ── Scoring ──────────────────────────────────────────────────────────

def kl_divergence(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-cell KL divergence: sum_c gt[c] * log(gt[c] / pred[c])."""
    safe_gt = np.maximum(gt, eps)
    safe_pred = np.maximum(pred, eps)
    return np.sum(safe_gt * np.log(safe_gt / safe_pred), axis=-1)


def score_prediction(pred: np.ndarray, gt: np.ndarray) -> float:
    """Competition scoring: 100 * exp(-3 * sum(entropy_weights * kl_per_cell))."""
    eps = 1e-12
    gt_safe = np.maximum(gt, eps)
    gt_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)  # (H, W)
    total_entropy = gt_entropy.sum()
    if total_entropy < eps:
        return 100.0
    entropy_weights = gt_entropy / total_entropy  # (H, W)
    kl = kl_divergence(pred, gt)  # (H, W)
    weighted_kl = (entropy_weights * kl).sum()
    return 100.0 * math.exp(-3.0 * weighted_kl)


# ── Data loading ─────────────────────────────────────────────────────

def load_ground_truth(round_num: int) -> list[dict]:
    """Load all 5 seeds of ground truth for a round."""
    results = []
    for seed in range(5):
        path = GT_DIR / f"round_{round_num}_seed_{seed}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        gt = np.array(data["ground_truth"], dtype=np.float64)  # (H, W, 6)
        grid = data["initial_grid"]
        results.append({
            "seed": seed,
            "gt": gt,
            "initial_grid": grid,
            "height": data["height"],
            "width": data["width"],
        })
    return results


def load_observations(round_num: int) -> tuple[dict, list]:
    """Load observations and initial grids for a round."""
    obs_dir = OBS_DIR / f"round_{round_num}"
    observations_by_seed: dict[int, list] = {}
    initial_grids: list = []

    for seed in range(5):
        init_path = obs_dir / f"initial_seed_{seed}.json"
        if init_path.exists():
            initial_grids.append(json.loads(init_path.read_text()))
        else:
            initial_grids.append(None)

        obs_path = obs_dir / f"observations_seed_{seed}.json"
        if obs_path.exists():
            observations_by_seed[seed] = json.loads(obs_path.read_text())
        else:
            observations_by_seed[seed] = []

    return observations_by_seed, initial_grids


# ── Precomputed round data (cached once) ─────────────────────────────

@dataclass
class SeedData:
    """All precomputed data for a single (round, seed) pair."""
    seed: int
    gt: np.ndarray              # (H, W, 6)
    initial_grid: list          # list[list[int]]
    observations: list          # list[dict]
    z: float
    context: Optional[np.ndarray]
    dir_pred: np.ndarray        # (H, W, 6) Dirichlet prediction
    nn_replay: Optional[np.ndarray]  # (H, W, 6) raw replay model output
    nn_v2: Optional[np.ndarray]      # (H, W, 6) raw NN v2 output
    nn_v3: Optional[np.ndarray]      # (H, W, 6) raw NN v3 output
    obs_counts: np.ndarray      # (H, W, 6) observation counts
    n_observed: np.ndarray      # (H, W)
    gt_entropy: np.ndarray      # (H, W) precomputed for scoring
    gt_entropy_weights: np.ndarray  # (H, W) normalized


@dataclass
class RoundData:
    round_num: int
    z: float
    context: Optional[np.ndarray]
    seeds: list  # list[SeedData]


def precompute_round(round_num: int, tta: bool = False) -> Optional[RoundData]:
    """Precompute all expensive data for a round (NN inference, Dirichlet, etc)."""
    gt_data = load_ground_truth(round_num)
    if not gt_data:
        return None

    obs_by_seed, initial_grids_obs = load_observations(round_num)

    # Context vector
    has_obs = any(len(v) > 0 for v in obs_by_seed.values())
    context = None
    if has_obs and initial_grids_obs:
        grids_for_ctx = [g for g in initial_grids_obs if g is not None]
        if grids_for_ctx:
            context = compute_context_vector(obs_by_seed, grids_for_ctx)

    calibration = load_calibration()
    round_z = calibration.get("round_z", {}).get(str(round_num)) if calibration else None

    if context is not None:
        z = estimate_z_from_context(context)
    elif round_z is not None:
        z = round_z
    else:
        z = 0.283

    # Synthetic context if no observations
    if context is None:
        context = np.zeros(8, dtype=np.float32)
        context[0] = z
        context[4] = 1 - z
        context[7] = z

    # Load NN models once (replay + v2 + v3, v4 disabled)
    model_replay = nn_predict._load_model("replay")
    model_v2 = nn_predict._load_model("v2")
    model_v3 = nn_predict._load_model("v3")

    seeds = []
    for item in gt_data:
        seed = item["seed"]
        gt = item["gt"]
        initial_grid = item["initial_grid"]
        observations = obs_by_seed.get(seed, [])

        # Dirichlet prediction
        dir_pred = dirichlet_predict(initial_grid, observations, calibration, z=z)

        # NN predictions (the expensive part — cached per seed)
        grid_np = np.array(initial_grid, dtype=np.int32)

        nn_replay = None
        if model_replay is not None:
            if tta:
                nn_replay = nn_predict._predict_with_tta_v2v3(model_replay, grid_np, z)
            else:
                nn_replay = nn_predict._predict_single_v2v3(model_replay, grid_np, z)

        nn_v2 = None
        if model_v2 is not None:
            if tta:
                nn_v2 = nn_predict._predict_with_tta_v2v3(model_v2, grid_np, z)
            else:
                nn_v2 = nn_predict._predict_single_v2v3(model_v2, grid_np, z)

        nn_v3 = None
        if model_v3 is not None:
            if tta:
                nn_v3 = nn_predict._predict_with_tta_v2v3(model_v3, grid_np, z)
            else:
                nn_v3 = nn_predict._predict_single_v2v3(model_v3, grid_np, z)

        # Empirical observation data
        obs_counts, n_observed = compute_empirical_observations(observations)

        # Precompute GT scoring data
        eps = 1e-12
        gt_safe = np.maximum(gt, eps)
        gt_entropy = -np.sum(gt_safe * np.log(gt_safe), axis=-1)
        total_entropy = gt_entropy.sum()
        gt_entropy_weights = gt_entropy / max(total_entropy, eps)

        seeds.append(SeedData(
            seed=seed, gt=gt, initial_grid=initial_grid,
            observations=observations, z=z, context=context,
            dir_pred=dir_pred, nn_replay=nn_replay, nn_v2=nn_v2, nn_v3=nn_v3,
            obs_counts=obs_counts, n_observed=n_observed,
            gt_entropy=gt_entropy, gt_entropy_weights=gt_entropy_weights,
        ))

    return RoundData(round_num=round_num, z=z, context=context, seeds=seeds)


# ── Fast recipe evaluation (no NN inference, just blending) ──────────

def _compute_nn_weight(recipe: Recipe, z: float) -> float:
    """Map z to NN weight using recipe's z_thresholds."""
    t1, t2, t3 = recipe.z_thresholds
    if z < t1:
        return 0.0
    elif z < t2:
        frac = (z - t1) / max(t2 - t1, 1e-6)
        return recipe.nn_weight * 0.4 * frac
    elif z < t3:
        frac = (z - t2) / max(t3 - t2, 1e-6)
        return recipe.nn_weight * (0.4 + 0.4 * frac)
    else:
        return recipe.nn_weight


def _blend_nn_models(recipe: Recipe, sd: SeedData) -> Optional[np.ndarray]:
    """Blend NN model outputs using recipe weights. ARITHMETIC mean (matches production).

    Production nn_predict.predict() uses: avg = sum(w * p for w, p in zip(weights, preds))
    """
    preds = []
    weights = []

    if sd.nn_replay is not None:
        preds.append(sd.nn_replay)
        weights.append(recipe.replay_weight)

    if sd.nn_v2 is not None:
        preds.append(sd.nn_v2)
        weights.append(recipe.v2_weight)

    if sd.nn_v3 is not None:
        preds.append(sd.nn_v3)
        weights.append(recipe.v3_weight)

    if not preds:
        return None

    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    # Arithmetic mean — matches production nn_predict.predict()
    result = sum(wi * p for wi, p in zip(w, preds))
    result = np.clip(result, 0.005, None)
    result = result / result.sum(axis=-1, keepdims=True)
    return result


def score_seed_fast(recipe: Recipe, sd: SeedData) -> float:
    """Score a single seed with a recipe. All NN data is precomputed."""
    nn_pred = _blend_nn_models(recipe, sd)
    nn_w = _compute_nn_weight(recipe, sd.z)

    if nn_pred is not None and nn_w > 0:
        eps = 1e-8
        dir_w = 1.0 - nn_w
        log_blend = nn_w * np.log(nn_pred + eps) + dir_w * np.log(sd.dir_pred + eps)
        blended = np.exp(log_blend)
    else:
        blended = sd.dir_pred.copy()

    # Empirical anchoring
    blended = empirical_anchor(blended, sd.obs_counts, sd.n_observed, anchor_weight=recipe.anchor_weight)

    # Floor and normalize
    blended = np.maximum(blended, recipe.prob_floor)
    blended = blended / blended.sum(axis=-1, keepdims=True)

    # Score using precomputed entropy weights
    kl = kl_divergence(blended, sd.gt)
    weighted_kl = (sd.gt_entropy_weights * kl).sum()
    return 100.0 * math.exp(-3.0 * weighted_kl)


# ── Grid search infrastructure ──────────────────────────────────────

@dataclass
class RoundResult:
    round_num: int
    avg_score: float
    seed_scores: list[float]
    weighted_score: float


@dataclass
class RecipeResult:
    recipe: Recipe
    rounds: list[RoundResult]
    best_weighted: float
    worst_round_score: float
    mean_score: float

    def summary_line(self) -> str:
        rnd_str = " | ".join(
            f"R{r.round_num}:{r.avg_score:5.1f}(w{r.weighted_score:5.1f})"
            for r in sorted(self.rounds, key=lambda x: x.round_num)
        )
        return (
            f"best_w={self.best_weighted:6.1f}  worst={self.worst_round_score:5.1f}  "
            f"mean={self.mean_score:5.1f}  || {rnd_str}\n"
            f"  {self.recipe.label()}"
        )


def evaluate_recipe_fast(recipe: Recipe, round_cache: list[RoundData]) -> RecipeResult:
    """Evaluate a recipe using precomputed round data (no NN inference)."""
    round_results = []
    for rd in round_cache:
        seed_scores = [score_seed_fast(recipe, sd) for sd in rd.seeds]
        avg = sum(seed_scores) / len(seed_scores) if seed_scores else 0.0
        rw = ROUND_WEIGHTS.get(rd.round_num, 1.0)
        round_results.append(RoundResult(
            round_num=rd.round_num, avg_score=avg,
            seed_scores=seed_scores, weighted_score=avg * rw,
        ))
    scores = [r.avg_score for r in round_results if r.avg_score > 0]
    return RecipeResult(
        recipe=recipe,
        rounds=round_results,
        best_weighted=max((r.weighted_score for r in round_results), default=0),
        worst_round_score=min(scores) if scores else 0,
        mean_score=sum(scores) / len(scores) if scores else 0,
    )


def build_grid(fast: bool = True) -> list[Recipe]:
    """Build parameter grid. Searches replay/v2/v3 weights (matching production stack)."""
    if fast:
        nn_weights = [0.30, 0.40, 0.50, 0.60]
        # Relative weights within NN (replay/v2/v3) — will be normalized
        replay_weights = [0.0, 0.10, 0.15, 0.25]
        v2_weights = [0.15, 0.25, 0.35]
        v3_weights = [0.40, 0.50, 0.60, 0.70]
        anchor_weights = [0.0]  # harness already showed 0 is optimal
        prob_floors = [0.005]
        z_threshold_sets = [
            [0.08, 0.15, 0.30],
            [0.05, 0.12, 0.25],
            [0.10, 0.20, 0.35],
        ]
    else:
        nn_weights = [0.25, 0.35, 0.45, 0.55, 0.65]
        replay_weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
        v2_weights = [0.10, 0.20, 0.30, 0.40]
        v3_weights = [0.30, 0.40, 0.50, 0.60, 0.70]
        anchor_weights = [0.0, 0.03]
        prob_floors = [0.003, 0.005, 0.01]
        z_threshold_sets = [
            [0.05, 0.12, 0.25],
            [0.08, 0.15, 0.30],
            [0.10, 0.20, 0.35],
            [0.06, 0.18, 0.32],
        ]

    recipes = []
    for nn_w, rep_w, v2_w, v3_w, anch, pf, zt in itertools.product(
        nn_weights, replay_weights, v2_weights, v3_weights,
        anchor_weights, prob_floors, z_threshold_sets,
    ):
        recipes.append(Recipe(
            nn_weight=nn_w, replay_weight=rep_w, v2_weight=v2_w,
            v3_weight=v3_w, anchor_weight=anch, prob_floor=pf,
            z_thresholds=list(zt),
        ))
    return recipes


def rank_recipes(results: list[RecipeResult], catastrophic_floor: float = 70.0) -> list[RecipeResult]:
    """Rank by "healthy-round upside with catastrophic floor".

    Primary: maximize best_weighted (leaderboard score)
    Constraint: worst round score must be >= catastrophic_floor
    Fallback: if nothing meets floor, relax and sort by composite.
    """
    above_floor = [r for r in results if r.worst_round_score >= catastrophic_floor]
    if above_floor:
        return sorted(above_floor, key=lambda r: r.best_weighted, reverse=True)

    # Relax progressively
    for relaxed in [60.0, 50.0, 40.0, 30.0, 0.0]:
        above = [r for r in results if r.worst_round_score >= relaxed]
        if above:
            print(f"  (relaxed catastrophic floor to {relaxed})")
            return sorted(above, key=lambda r: r.best_weighted, reverse=True)

    return sorted(
        results,
        key=lambda r: r.best_weighted + 0.3 * r.worst_round_score,
        reverse=True,
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s: %(message)s",
    )

    validate_mode = "--validate" in sys.argv
    fast_mode = "--fast" in sys.argv or not validate_mode

    # Discover rounds
    available_rounds = sorted({
        int(p.stem.split("_")[1])
        for p in GT_DIR.glob("round_*_seed_*.json")
    })
    print(f"Ground truth available for rounds: {available_rounds}")
    print(f"Round weights: {', '.join(f'R{r}={ROUND_WEIGHTS[r]:.3f}' for r in available_rounds)}")

    # Phase 1: precompute all NN outputs (the slow part, done ONCE)
    tta_precompute = validate_mode
    print(f"\nPhase 1: Precomputing NN outputs (tta={tta_precompute})...")
    t0 = time.time()
    round_cache: list[RoundData] = []
    for rn in available_rounds:
        t1 = time.time()
        rd = precompute_round(rn, tta=tta_precompute)
        if rd is not None:
            round_cache.append(rd)
            n_seeds = len(rd.seeds)
            print(f"  R{rn}: z={rd.z:.3f}, {n_seeds} seeds, "
                  f"replay={'Y' if rd.seeds[0].nn_replay is not None else 'N'} "
                  f"v2={'Y' if rd.seeds[0].nn_v2 is not None else 'N'} "
                  f"v3={'Y' if rd.seeds[0].nn_v3 is not None else 'N'} "
                  f"({time.time()-t1:.1f}s)")
    print(f"  Total precompute: {time.time()-t0:.1f}s\n")

    # Phase 2: sweep recipes (fast — just numpy blending)
    recipes = build_grid(fast=fast_mode)
    print(f"Phase 2: Sweeping {len(recipes)} recipes...")
    t0 = time.time()
    results: list[RecipeResult] = []
    total = len(recipes)
    for i, recipe in enumerate(recipes):
        if i % max(total // 10, 1) == 0:
            pct = 100 * i / total
            print(f"  [{i+1}/{total}] ({pct:.0f}%)", flush=True)
        results.append(evaluate_recipe_fast(recipe, round_cache))
    print(f"  Sweep done in {time.time()-t0:.1f}s\n")

    # Rank
    ranked = rank_recipes(results)

    print(f"{'='*80}")
    print("TOP 3 RECIPES")
    print(f"{'='*80}\n")
    for i, rr in enumerate(ranked[:3]):
        print(f"#{i+1}")
        print(rr.summary_line())
        print()

    # Baseline (current production: nn=0.50, replay=0.15, v2=0.25, v3=0.60)
    print(f"{'='*80}")
    print("BASELINE (current production params)")
    print(f"{'='*80}")
    baseline = Recipe()  # defaults match production
    baseline_result = evaluate_recipe_fast(baseline, round_cache)
    print(baseline_result.summary_line())
    print()

    # Score distribution
    all_best = [r.best_weighted for r in results]
    print(f"Score distribution (best_weighted across {len(results)} recipes):")
    print(f"  min={min(all_best):.1f}  median={sorted(all_best)[len(all_best)//2]:.1f}  "
          f"max={max(all_best):.1f}")

    # Validate top 3 with TTA if not already
    if not validate_mode and not tta_precompute:
        print(f"\nPhase 3: Re-validating top 3 with TTA=True...")
        for i, rr in enumerate(ranked[:3]):
            print(f"\n  Precomputing TTA for recipe #{i+1}...")
            tta_cache = []
            for rn in available_rounds:
                rd = precompute_round(rn, tta=True)
                if rd is not None:
                    tta_cache.append(rd)
            validated = evaluate_recipe_fast(rr.recipe, tta_cache)
            print(f"  #{i+1} (TTA):")
            print(f"  {validated.summary_line()}")
            # Only precompute TTA once — all 3 recipes share the same NN cache
            break  # Compute TTA cache once, then reuse
        # Reuse TTA cache for remaining top recipes
        if len(ranked) >= 2:
            for i in range(1, min(3, len(ranked))):
                validated = evaluate_recipe_fast(ranked[i].recipe, tta_cache)
                print(f"  #{i+1} (TTA):")
                print(f"  {validated.summary_line()}")

    # Save top 10
    top10_path = ROOT / "replay_top10.json"
    top10_data = []
    for rr in ranked[:10]:
        top10_data.append({
            "recipe": {
                "nn_weight": rr.recipe.nn_weight,
                "replay_weight": rr.recipe.replay_weight,
                "v2_weight": rr.recipe.v2_weight,
                "v3_weight": rr.recipe.v3_weight,
                "anchor_weight": rr.recipe.anchor_weight,
                "prob_floor": rr.recipe.prob_floor,
                "z_thresholds": rr.recipe.z_thresholds,
            },
            "best_weighted": round(rr.best_weighted, 3),
            "worst_round_score": round(rr.worst_round_score, 3),
            "mean_score": round(rr.mean_score, 3),
            "rounds": {
                str(r.round_num): {
                    "avg_score": round(r.avg_score, 2),
                    "weighted_score": round(r.weighted_score, 2),
                    "seed_scores": [round(s, 2) for s in r.seed_scores],
                }
                for r in rr.rounds
            },
        })
    top10_path.write_text(json.dumps(top10_data, indent=2))
    print(f"\nTop 10 saved to {top10_path}")


if __name__ == "__main__":
    main()
