"""Astar Island solver — safe, stateful, Dirichlet-Bayesian predictions."""

import time
import json
import logging
import numpy as np
from pathlib import Path

from api import AstarAPI
from config import TOKEN, QUERIES_PER_SEED
from strategy import predict_for_seed, floor_and_normalize
import state as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

GT_DIR = Path(__file__).parent / "ground_truth"
GT_DIR.mkdir(exist_ok=True)


def compute_smart_viewports(
    initial_grid: list[list[int]], map_w: int, map_h: int,
) -> list[tuple[int, int, int, int]]:
    """Compute viewports prioritizing dynamic areas (near settlements).

    Returns list of (x, y, w, h) viewport specs.
    """
    grid = np.array(initial_grid)
    # Find settlement positions
    sy, sx = np.where(np.isin(grid, [1, 2, 3]))

    if len(sy) == 0:
        # No settlements — fall back to full tiling
        return _full_tiling(map_w, map_h)

    # Cluster settlement positions into viewport-sized groups
    # For now: full tiling but sorted by settlement density
    tiles = _full_tiling(map_w, map_h)

    # Score each tile by number of dynamic cells it covers
    scored = []
    for x, y, w, h in tiles:
        dynamic_count = 0
        for sy2, sx2 in zip(sy, sx):
            if x <= sx2 < x + w and y <= sy2 < y + h:
                dynamic_count += 1
        scored.append((dynamic_count, x, y, w, h))

    # Sort: most dynamic cells first (observe important areas first)
    scored.sort(reverse=True)
    return [(x, y, w, h) for _, x, y, w, h in scored]


def _full_tiling(map_w: int, map_h: int, vp: int = 15) -> list[tuple[int, int, int, int]]:
    """Generate 3×3 viewport tiling for 40×40 map."""
    tiles = []
    positions = [0, 13, 25]  # Covers 0-14, 13-27, 25-39 with overlap
    for y in positions:
        for x in positions:
            w = min(vp, map_w - x)
            h = min(vp, map_h - y)
            tiles.append((x, y, w, h))
    return tiles


def solve_round(api: AstarAPI, round_info: dict, dry_run: bool = False):
    """Solve one active round. Safe, stateful, no double-execution."""
    round_id = round_info["id"]
    round_num = round_info["round_number"]
    persistence = st.load()

    # SAFETY: Check if already solved with sufficient quality
    existing = st.get_round_info(persistence, round_id)
    if existing:
        log.info(
            f"Round #{round_num} already solved: "
            f"queries={existing.get('queries_used')}, "
            f"seeds={existing.get('seeds_submitted')}, "
            f"strategy={existing.get('strategy')}"
        )
        return

    log.info(f"=== Solving round #{round_num} (id={round_id}) ===")

    # Get round details
    detail = api.get_round_detail(round_id)
    map_w, map_h = detail["map_width"], detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    # Check budget
    budget = api.get_budget()
    queries_left = budget["queries_max"] - budget["queries_used"]
    log.info(f"Map: {map_w}x{map_h}, seeds: {seeds_count}, budget: {queries_left}/{budget['queries_max']}")

    if queries_left <= 0:
        log.warning("No queries left!")
        return

    queries_per_seed = min(QUERIES_PER_SEED, queries_left // seeds_count)
    all_observations = {}

    # Phase 1: Observe all seeds
    log.info(f"Phase 1: Observing ({queries_per_seed} queries/seed)...")
    for seed_idx in range(seeds_count):
        viewports = compute_smart_viewports(
            initial_states[seed_idx]["grid"], map_w, map_h
        )
        observations = observe_seed(
            api, round_id, seed_idx, viewports, queries_per_seed, dry_run
        )
        all_observations[seed_idx] = observations
        log.info(f"  Seed {seed_idx}: {len(observations)} observations collected")

    # Phase 2: Predict and submit all seeds
    log.info("Phase 2: Predicting and submitting...")
    seeds_submitted = 0
    for seed_idx in range(seeds_count):
        pred = predict_for_seed(
            initial_states[seed_idx]["grid"],
            all_observations.get(seed_idx, []),
        )
        if submit_prediction(api, round_id, seed_idx, pred, dry_run):
            seeds_submitted += 1

    # Record state
    total_queries = sum(len(obs) for obs in all_observations.values())
    st.mark_round_solved(persistence, round_id, {
        "round_number": round_num,
        "queries_used": budget["queries_used"] + total_queries,
        "seeds_submitted": seeds_submitted,
        "strategy": "dirichlet_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    log.info(f"=== Round #{round_num} done: {seeds_submitted} seeds, {total_queries} queries ===")


def observe_seed(
    api: AstarAPI,
    round_id: str,
    seed_index: int,
    viewports: list[tuple[int, int, int, int]],
    max_queries: int,
    dry_run: bool,
) -> list[dict]:
    """Run simulation queries for one seed."""
    observations = []
    for i in range(min(max_queries, len(viewports))):
        vx, vy, vw, vh = viewports[i]
        if dry_run:
            log.info(f"  [DRY RUN] Seed {seed_index} query {i+1}: viewport ({vx},{vy},{vw},{vh})")
            continue
        try:
            time.sleep(0.25)  # Rate limit: 5 req/s
            result = api.simulate(round_id, seed_index, vx, vy, vw, vh)
            observations.append(result)
            log.info(
                f"  Seed {seed_index} query {i+1}/{max_queries}: "
                f"viewport ({vx},{vy}) — {result['queries_used']}/{result['queries_max']}"
            )
        except Exception as e:
            log.error(f"  Seed {seed_index} query {i+1} failed: {e}")
            if "429" in str(e):
                time.sleep(2)
            else:
                break
    return observations


def submit_prediction(
    api: AstarAPI, round_id: str, seed_index: int, pred: np.ndarray,
    dry_run: bool,
) -> bool:
    """Submit prediction with retry and safety."""
    pred = floor_and_normalize(pred)

    if dry_run:
        log.info(f"  [DRY RUN] Would submit seed {seed_index}: shape={pred.shape}, sum_check={pred[0,0].sum():.3f}")
        return True

    for attempt in range(3):
        try:
            time.sleep(0.6)  # Rate limit: 2 req/s
            result = api.submit(round_id, seed_index, pred.tolist())
            log.info(f"  Submitted seed {seed_index}: {result.get('status')}")
            return True
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                log.warning(f"  Rate limited, retrying in 3s...")
                time.sleep(3)
            else:
                log.error(f"  Submit seed {seed_index} failed: {e}")
                return False
    return False


def harvest_ground_truth(api: AstarAPI):
    """Download ground truth from all completed rounds for calibration."""
    try:
        my_rounds = api.get_my_rounds()
    except Exception as e:
        log.error(f"Failed to get my rounds: {e}")
        return

    for r in my_rounds:
        if r["status"] != "completed":
            continue
        round_id = r["id"]
        round_num = r["round_number"]
        seeds = r.get("seeds_count", 5)

        for seed_idx in range(seeds):
            gt_file = GT_DIR / f"round_{round_num}_seed_{seed_idx}.json"
            if gt_file.exists():
                continue
            try:
                analysis = api.get_analysis(round_id, seed_idx)
                gt_file.write_text(json.dumps(analysis))
                log.info(f"  Saved ground truth: round {round_num} seed {seed_idx} (score={analysis.get('score')})")
            except Exception as e:
                log.info(f"  No analysis for round {round_num} seed {seed_idx}: {e}")


def check_and_solve(dry_run: bool = False):
    """Main entry: check for active round, solve if new, harvest ground truth.

    Minimal output when nothing changes — saves context tokens in /loop.
    """
    api = AstarAPI(TOKEN)

    # Always try to harvest ground truth from completed rounds
    harvest_ground_truth(api)

    # Check for active round
    active = api.get_active_round()
    if active:
        persistence = st.load()
        if st.is_round_solved(persistence, active["id"]):
            # One-liner: already solved, just print status
            print(f"[astar] Round #{active['round_number']} solved. Waiting. closes={active.get('closes_at','?')}")
            return
        log.info(f"NEW ROUND: #{active['round_number']} (closes: {active.get('closes_at', '?')})")
        solve_round(api, active, dry_run)
        # Show leaderboard after solving
        try:
            lb = api.get_leaderboard()
            if lb:
                log.info("--- Leaderboard top 5 ---")
                for entry in lb[:5]:
                    log.info(f"  #{entry['rank']} {entry['team_name']}: {entry['weighted_score']:.1f}")
        except Exception:
            pass
    else:
        print("[astar] No active round.")


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("=== DRY RUN MODE ===")
    check_and_solve(dry_run=dry)
