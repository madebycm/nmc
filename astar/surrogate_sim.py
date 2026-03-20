"""Surrogate Norse Civilization Simulator.

Reverse-engineers the hidden parameters of the Astar Island sim, then runs
Monte Carlo simulations to produce probability distributions.

Simulation phases per year (50 years total):
1. Growth: food production, population growth, settlement expansion
2. Conflict: raids between settlements
3. Trade: resource exchange between ports
4. Winter: food consumption, potential collapse
5. Environment: ruins → forest, settlement reclamation

Hidden parameters are fitted to ground truth data using differential evolution.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import time

GT_DIR = Path("ground_truth")
CAL_FILE = Path("calibration.json")
NUM_CLASSES = 6
NUM_YEARS = 50
NUM_MONTE_CARLO = 200  # runs per parameter set


@dataclass
class Settlement:
    y: int
    x: int
    population: float = 10.0
    food: float = 20.0
    wealth: float = 0.0
    defense: float = 1.0
    tech: float = 0.0
    has_port: bool = False
    alive: bool = True
    faction: int = 0


@dataclass
class SimParams:
    """Hidden parameters that control simulation behavior."""

    # Growth
    food_per_forest: float = 5.0  # food produced per adjacent forest tile
    food_per_plains: float = 3.0  # food produced per adjacent plains tile
    food_per_port_trade: float = 2.0  # extra food from port operations
    growth_rate: float = 0.08  # population growth per food surplus
    expansion_pop_threshold: float = 25.0  # min population to found new settlement
    expansion_range: int = 5  # max distance for new settlement
    expansion_prob: float = 0.1  # probability of expansion per eligible settlement
    port_develop_prob: float = 0.08  # probability coastal settlement builds port

    # Conflict
    raid_food_threshold: float = 8.0  # below this food level, settlement raids
    raid_range: int = 6  # base raid range
    raid_strength_mult: float = 0.2  # fraction of population used in raid
    raid_loot_frac: float = 0.25  # fraction of defender's food looted
    longship_range_bonus: int = 4  # extra range for settlements with ports

    # Trade
    trade_range: int = 8  # max trade distance between ports
    trade_food_bonus: float = 4.0  # food per trade connection
    trade_wealth_bonus: float = 2.0  # wealth per trade connection

    # Winter
    winter_severity_mean: float = 0.5  # mean food cost per population
    winter_severity_std: float = 0.2  # variance in severity
    collapse_food_threshold: float = -15.0  # food below this → collapse

    # Environment
    ruin_forest_rate: float = 0.03  # probability ruin → forest per year
    ruin_reclaim_range: int = 3  # range for settlement to reclaim ruin
    ruin_reclaim_prob: float = 0.08  # probability of reclaim per eligible ruin

    def to_vector(self) -> np.ndarray:
        """Convert to parameter vector for optimization."""
        return np.array([
            self.food_per_forest, self.food_per_plains, self.food_per_port_trade,
            self.growth_rate, self.expansion_pop_threshold, self.expansion_range,
            self.expansion_prob, self.port_develop_prob,
            self.raid_food_threshold, self.raid_range, self.raid_strength_mult,
            self.raid_loot_frac, self.longship_range_bonus,
            self.trade_range, self.trade_food_bonus, self.trade_wealth_bonus,
            self.winter_severity_mean, self.winter_severity_std,
            self.collapse_food_threshold,
            self.ruin_forest_rate, self.ruin_reclaim_range, self.ruin_reclaim_prob,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "SimParams":
        return cls(
            food_per_forest=v[0], food_per_plains=v[1], food_per_port_trade=v[2],
            growth_rate=v[3], expansion_pop_threshold=v[4],
            expansion_range=int(max(1, round(v[5]))),
            expansion_prob=np.clip(v[6], 0, 1),
            port_develop_prob=np.clip(v[7], 0, 1),
            raid_food_threshold=v[8], raid_range=int(max(1, round(v[9]))),
            raid_strength_mult=np.clip(v[10], 0, 1),
            raid_loot_frac=np.clip(v[11], 0, 1),
            longship_range_bonus=int(max(0, round(v[12]))),
            trade_range=int(max(1, round(v[13]))),
            trade_food_bonus=v[14], trade_wealth_bonus=v[15],
            winter_severity_mean=max(0.1, v[16]),
            winter_severity_std=max(0.01, v[17]),
            collapse_food_threshold=v[18],
            ruin_forest_rate=np.clip(v[19], 0, 1),
            ruin_reclaim_range=int(max(1, round(v[20]))),
            ruin_reclaim_prob=np.clip(v[21], 0, 1),
        )

    @staticmethod
    def bounds():
        """Parameter bounds for optimization."""
        return [
            (0.5, 5.0),   # food_per_forest
            (0.2, 3.0),   # food_per_plains
            (0.0, 2.0),   # food_per_port_trade
            (0.01, 0.5),  # growth_rate
            (10, 100),     # expansion_pop_threshold
            (2, 10),       # expansion_range
            (0.01, 0.5),  # expansion_prob
            (0.01, 0.3),  # port_develop_prob
            (1.0, 20.0),  # raid_food_threshold
            (3, 15),       # raid_range
            (0.05, 0.8),  # raid_strength_mult
            (0.05, 0.8),  # raid_loot_frac
            (0, 10),       # longship_range_bonus
            (3, 20),       # trade_range
            (0.5, 10.0),  # trade_food_bonus
            (0.5, 10.0),  # trade_wealth_bonus
            (0.3, 3.0),   # winter_severity_mean
            (0.05, 1.0),  # winter_severity_std
            (-20, -1),     # collapse_food_threshold
            (0.01, 0.2),  # ruin_forest_rate
            (1, 6),        # ruin_reclaim_range
            (0.01, 0.3),  # ruin_reclaim_prob
        ]


CODE_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def _adjacent_terrain(grid, y, x) -> dict:
    """Count terrain types adjacent to (y, x)."""
    h, w = grid.shape
    counts = {}
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            t = int(grid[ny, nx])
            counts[t] = counts.get(t, 0) + 1
    return counts


def _manhattan(y1, x1, y2, x2):
    return abs(y1 - y2) + abs(x1 - x2)


def _find_empty_near(grid, y, x, max_dist, rng):
    """Find a random empty/plains cell near (y, x)."""
    h, w = grid.shape
    candidates = []
    for dy in range(-max_dist, max_dist + 1):
        for dx in range(-max_dist, max_dist + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if grid[ny, nx] in (0, 11) and abs(dy) + abs(dx) <= max_dist:
                    candidates.append((ny, nx))
    if candidates:
        return candidates[rng.integers(len(candidates))]
    return None


def run_simulation(initial_grid: np.ndarray, params: SimParams, rng=None) -> np.ndarray:
    """Run one Monte Carlo simulation → final grid state.

    Returns 40×40 array of class codes (0-5).
    """
    if rng is None:
        rng = np.random.default_rng()

    grid = initial_grid.copy()
    h, w = grid.shape

    # Initialize settlements from grid
    settlements = []
    faction_id = 0
    for y in range(h):
        for x in range(w):
            if grid[y, x] in (1, 2):
                s = Settlement(
                    y=y, x=x,
                    population=8 + rng.uniform(0, 8),
                    food=15 + rng.uniform(0, 15),
                    has_port=(grid[y, x] == 2),
                    faction=faction_id,
                )
                settlements.append(s)
                faction_id += 1

    for year in range(NUM_YEARS):
        alive_settlements = [s for s in settlements if s.alive]
        if not alive_settlements:
            break

        # ── Phase 1: Growth ──
        for s in alive_settlements:
            adj = _adjacent_terrain(grid, s.y, s.x)
            food_gain = (
                adj.get(4, 0) * params.food_per_forest
                + adj.get(11, 0) * params.food_per_plains
                + (params.food_per_port_trade if s.has_port else 0)
            )
            s.food += food_gain

            # Population growth
            if s.food > s.population:
                s.population += params.growth_rate * (s.food - s.population)
                s.population = min(s.population, 200)

            # Expansion
            if s.population > params.expansion_pop_threshold:
                if rng.random() < params.expansion_prob:
                    pos = _find_empty_near(grid, s.y, s.x, params.expansion_range, rng)
                    if pos is not None:
                        ny, nx = pos
                        # Check if coastal for port
                        is_coastal = any(
                            0 <= ny + dy < h and 0 <= nx + dx < w
                            and grid[ny + dy, nx + dx] == 10
                            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        )
                        new_s = Settlement(
                            y=ny, x=nx,
                            population=s.population * 0.3,
                            food=s.food * 0.3,
                            has_port=is_coastal and rng.random() < params.port_develop_prob,
                            faction=s.faction,
                        )
                        s.population *= 0.7
                        s.food *= 0.7
                        settlements.append(new_s)
                        grid[ny, nx] = 2 if new_s.has_port else 1

            # Port development
            if not s.has_port and rng.random() < params.port_develop_prob:
                is_coastal = any(
                    0 <= s.y + dy < h and 0 <= s.x + dx < w
                    and grid[s.y + dy, s.x + dx] == 10
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
                if is_coastal:
                    s.has_port = True
                    grid[s.y, s.x] = 2

        # ── Phase 2: Conflict ──
        alive_settlements = [s for s in settlements if s.alive]
        for attacker in alive_settlements:
            if attacker.food > params.raid_food_threshold:
                continue  # not desperate enough to raid

            raid_range = params.raid_range
            if attacker.has_port:
                raid_range += params.longship_range_bonus

            targets = [
                t for t in alive_settlements
                if t is not attacker
                and t.faction != attacker.faction
                and _manhattan(attacker.y, attacker.x, t.y, t.x) <= raid_range
            ]
            if not targets:
                continue

            target = targets[rng.integers(len(targets))]
            attack_str = attacker.population * params.raid_strength_mult
            defend_str = target.population * target.defense * 0.5

            if attack_str > defend_str * rng.uniform(0.5, 1.5):
                # Successful raid
                loot = target.food * params.raid_loot_frac
                attacker.food += loot
                target.food -= loot
                target.population *= 0.85  # casualties
                target.defense *= 0.9

        # ── Phase 3: Trade ──
        ports = [s for s in settlements if s.alive and s.has_port]
        for i, p1 in enumerate(ports):
            for p2 in ports[i + 1 :]:
                if _manhattan(p1.y, p1.x, p2.y, p2.x) <= params.trade_range:
                    p1.food += params.trade_food_bonus
                    p2.food += params.trade_food_bonus
                    p1.wealth += params.trade_wealth_bonus
                    p2.wealth += params.trade_wealth_bonus

        # ── Phase 4: Winter ──
        severity = max(
            0.1,
            rng.normal(params.winter_severity_mean, params.winter_severity_std),
        )
        for s in settlements:
            if not s.alive:
                continue
            s.food -= s.population * severity

            if s.food < params.collapse_food_threshold:
                s.alive = False
                grid[s.y, s.x] = 3  # ruin

        # ── Phase 5: Environment ──
        for y in range(h):
            for x in range(w):
                if grid[y, x] == 3:  # ruin
                    # Forest reclaims
                    if rng.random() < params.ruin_forest_rate:
                        grid[y, x] = 4
                        continue

                    # Nearby settlement reclaims
                    for s in settlements:
                        if (
                            s.alive
                            and _manhattan(s.y, s.x, y, x) <= params.ruin_reclaim_range
                            and s.population > 15
                        ):
                            if rng.random() < params.ruin_reclaim_prob:
                                is_coastal = any(
                                    0 <= y + dy < h and 0 <= x + dx < w
                                    and grid[y + dy, x + dx] == 10
                                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                )
                                new_s = Settlement(
                                    y=y, x=x,
                                    population=s.population * 0.2,
                                    food=s.food * 0.2,
                                    has_port=is_coastal,
                                    faction=s.faction,
                                )
                                s.population *= 0.8
                                s.food *= 0.8
                                settlements.append(new_s)
                                grid[y, x] = 2 if is_coastal else 1
                                break

    # Convert final grid to class codes
    result = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            result[y, x] = CODE_TO_CLASS.get(int(grid[y, x]), 0)
    return result


def monte_carlo_predict(
    initial_grid: np.ndarray, params: SimParams, n_runs: int = NUM_MONTE_CARLO
) -> np.ndarray:
    """Run Monte Carlo → probability distribution (H, W, 6)."""
    h, w = initial_grid.shape
    counts = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)

    for i in range(n_runs):
        rng = np.random.default_rng(seed=i * 1000 + 42)
        final = run_simulation(initial_grid, params, rng)
        for y in range(h):
            for x in range(w):
                counts[y, x, final[y, x]] += 1

    prob = counts / n_runs
    # Floor at 0.01
    prob = np.maximum(prob, 0.01)
    prob = prob / prob.sum(axis=-1, keepdims=True)
    return prob


# ── Scoring ───────────────────────────────────────────────────────────


def compute_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Competition scoring metric."""
    eps = 1e-10
    entropy = -np.sum(gt * np.log(np.clip(gt, eps, 1)), axis=-1)
    kl = np.sum(gt * np.log(np.clip(gt, eps, 1) / np.clip(pred, eps, 1)), axis=-1)
    w = entropy / (entropy.sum() + eps)
    wkl = (w * kl).sum()
    return max(0, min(100, 100 * np.exp(-3 * wkl)))


# ── Parameter Fitting ─────────────────────────────────────────────────


def fit_params(gt_files: list, n_mc_fit: int = 50, n_generations: int = 50):
    """Fit simulation parameters to ground truth using differential evolution.

    Uses scipy.optimize.differential_evolution for global optimization.
    """
    from scipy.optimize import differential_evolution

    # Load GT data
    gt_data = []
    for gt_file in gt_files:
        data = json.loads(gt_file.read_text())
        grid = np.array(data["initial_grid"], dtype=np.int32)
        gt = np.array(data["ground_truth"], dtype=np.float32)
        gt_data.append((grid, gt))

    print(f"Fitting params on {len(gt_data)} GT files, {n_mc_fit} MC runs each")

    eval_count = [0]
    best_score = [0]

    def objective(param_vector):
        params = SimParams.from_vector(param_vector)

        total_score = 0
        for grid, gt in gt_data:
            pred = monte_carlo_predict(grid, params, n_runs=n_mc_fit)
            total_score += compute_score(pred, gt)

        avg_score = total_score / len(gt_data)
        eval_count[0] += 1

        if avg_score > best_score[0]:
            best_score[0] = avg_score
            print(f"  Eval {eval_count[0]}: avg_score={avg_score:.1f} (new best)")
        elif eval_count[0] % 20 == 0:
            print(f"  Eval {eval_count[0]}: avg_score={avg_score:.1f}")

        return -avg_score  # minimize negative score

    bounds = SimParams.bounds()

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=n_generations,
        popsize=15,
        tol=0.001,
        seed=42,
        workers=1,  # MC is already parallelizable
        disp=True,
    )

    best_params = SimParams.from_vector(result.x)
    print(f"\nBest score: {-result.fun:.1f}")
    print(f"Evaluations: {result.nfev}")

    return best_params, -result.fun


def main():
    print("Surrogate Simulator — Parameter Fitting")
    print("=" * 60)

    gt_files = sorted(GT_DIR.glob("round_*_seed_*.json"))
    if not gt_files:
        print("No GT files found!")
        return

    print(f"Found {len(gt_files)} GT files")

    # Quick test: run one simulation with default params
    print("\nQuick test with default params...")
    data = json.loads(gt_files[0].read_text())
    grid = np.array(data["initial_grid"], dtype=np.int32)
    gt = np.array(data["ground_truth"], dtype=np.float32)

    t0 = time.time()
    pred = monte_carlo_predict(grid, SimParams(), n_runs=50)
    t1 = time.time()
    score = compute_score(pred, gt)
    print(f"  Default params: score={score:.1f}, time={t1-t0:.1f}s for 50 MC runs")

    # Fit parameters on a subset (5 files, one per round)
    print("\n" + "=" * 60)
    print("Fitting parameters...")
    print("=" * 60)

    # Use one seed per round to speed up fitting
    fit_files = []
    seen_rounds = set()
    for f in gt_files:
        rn = int(f.stem.split("_")[1])
        if rn not in seen_rounds:
            seen_rounds.add(rn)
            fit_files.append(f)

    print(f"Using {len(fit_files)} files (1 per round) for fitting")

    best_params, best_score = fit_params(
        fit_files, n_mc_fit=50, n_generations=30
    )

    # Save params
    params_file = Path("sim_params.json")
    params_file.write_text(json.dumps({
        "params": best_params.to_vector().tolist(),
        "score": best_score,
    }))
    print(f"\nParams saved to {params_file}")

    # Evaluate on ALL GT files
    print("\n" + "=" * 60)
    print("Full evaluation with fitted params...")
    print("=" * 60)

    scores = []
    for gt_file in gt_files:
        data = json.loads(gt_file.read_text())
        grid = np.array(data["initial_grid"], dtype=np.int32)
        gt = np.array(data["ground_truth"], dtype=np.float32)

        pred = monte_carlo_predict(grid, best_params, n_runs=100)
        s = compute_score(pred, gt)
        scores.append(s)
        print(f"  {gt_file.stem}: {s:.1f}")

    print(f"\nOverall avg: {np.mean(scores):.1f}, min: {np.min(scores):.1f}")


if __name__ == "__main__":
    main()
