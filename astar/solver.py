"""Astar Island solver — safe, stateful, Dirichlet-Bayesian predictions."""

import time
import json
import logging
import numpy as np
from pathlib import Path

from api import AstarAPI
from config import TOKEN, QUERIES_PER_SEED
from strategy import (
    predict_for_seed, floor_and_normalize, extract_round_dynamics,
    load_calibration, compute_context_vector, estimate_z_from_context,
)
import state as st
import codex_advisor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

GT_DIR = Path(__file__).parent / "ground_truth"
GT_DIR.mkdir(exist_ok=True)
OBS_DIR = Path(__file__).parent / "observations"
OBS_DIR.mkdir(exist_ok=True)


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


def _pick_precision_targets(
    initial_states: list, seeds_count: int, n_queries: int,
    z: float = None, base_observations: dict = None,
) -> list[tuple[int, int, int, int, int]]:
    """Pick (seed, x, y, w, h) for adaptive precision queries.

    Scoring: settlement density + coastal settlements + port cells.
    With z available, boost viewports containing high-uncertainty cells.
    Spreads queries across seeds for MC diversity.
    """
    scored = []
    for seed_idx in range(seeds_count):
        grid = np.array(initial_states[seed_idx]["grid"])
        h_map, w_map = grid.shape
        tiles = _full_tiling(w_map, h_map)

        # Build coastal mask for this grid
        ocean = (grid == 10)
        coastal = np.zeros_like(ocean)
        if h_map > 1:
            coastal[1:] |= ocean[:-1]
            coastal[:-1] |= ocean[1:]
        if w_map > 1:
            coastal[:, 1:] |= ocean[:, :-1]
            coastal[:, :-1] |= ocean[:, 1:]

        for x, y, w, h in tiles:
            region = grid[y:y+h, x:x+w]
            coast_region = coastal[y:y+h, x:x+w]

            # Settlement cells (highest entropy)
            settle_mask = np.isin(region, [1, 2])
            n_settle = settle_mask.sum()
            # Coastal settlements (port transitions = very high entropy)
            n_coastal_settle = (settle_mask & coast_region).sum()
            # Empty land near settlements (expansion zones)
            n_dynamic_empty = np.isin(region, [0, 11, 3]).sum()
            # Ports
            n_ports = (region == 2).sum()

            score = (n_settle * 3.0
                     + n_coastal_settle * 2.0
                     + n_ports * 1.5
                     + n_dynamic_empty * 0.5)
            scored.append((score, seed_idx, x, y, w, h))

    scored.sort(reverse=True)

    # Spread across seeds: don't give >40% of queries to one seed
    max_per_seed = max(n_queries * 2 // 5, 1)
    seed_counts = {i: 0 for i in range(seeds_count)}
    result = []
    for _, s, x, y, w, h in scored:
        if len(result) >= n_queries:
            break
        if seed_counts[s] < max_per_seed:
            result.append((s, x, y, w, h))
            seed_counts[s] += 1

    return result


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


def _save_round_data(round_num: int, detail: dict, all_observations: dict):
    """Persist full round data: initial states, observations, settlements."""
    round_dir = OBS_DIR / f"round_{round_num}"
    round_dir.mkdir(exist_ok=True)

    # Save round metadata (initial grids for all seeds, map dims, etc.)
    meta = {
        "round_number": round_num,
        "map_width": detail["map_width"],
        "map_height": detail["map_height"],
        "seeds_count": detail["seeds_count"],
    }
    # Save initial grids separately (large)
    for seed_idx, state in enumerate(detail["initial_states"]):
        init_file = round_dir / f"initial_seed_{seed_idx}.json"
        if not init_file.exists():
            init_file.write_text(json.dumps(state["grid"]))

    # Save all observations per seed
    for seed_idx, observations in all_observations.items():
        obs_file = round_dir / f"observations_seed_{seed_idx}.json"
        obs_file.write_text(json.dumps(observations))

    meta_file = round_dir / "meta.json"
    meta_file.write_text(json.dumps(meta))
    log.info(f"  Saved round data: {round_dir} ({sum(len(o) for o in all_observations.values())} obs)")


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

    # Snapshot base observations (before precision queries) for unbiased context
    base_observations = {k: list(v) for k, v in all_observations.items()}

    # Compute preliminary z from base observations for adaptive targeting
    initial_grids = [initial_states[i]["grid"] for i in range(seeds_count)]
    context = compute_context_vector(base_observations, initial_grids)
    z_est = estimate_z_from_context(context)
    log.info(f"Preliminary z from {sum(len(v) for v in base_observations.values())} base queries: {z_est:.3f}")

    # Phase 1b: Adaptive precision — use remaining budget on high-value viewports
    budget_after = api.get_budget()
    precision_left = budget_after["queries_max"] - budget_after["queries_used"]
    if precision_left > 0 and not dry_run:
        log.info(f"Phase 1b: {precision_left} adaptive queries (settlement-dense, coastal, spread across seeds)...")
        precision_targets = _pick_precision_targets(
            initial_states, seeds_count, precision_left,
            z=z_est, base_observations=base_observations,
        )
        for seed_idx, vx, vy, vw, vh in precision_targets:
            try:
                time.sleep(0.25)
                result = api.simulate(round_id, seed_idx, vx, vy, vw, vh)
                all_observations[seed_idx].append(result)
                log.info(f"  Precision: seed {seed_idx} viewport ({vx},{vy}) — {result['queries_used']}/{result['queries_max']}")
            except Exception as e:
                log.warning(f"  Precision query failed: {e}")
                break

    # Save all data at full fidelity for future training
    _save_round_data(round_num, detail, all_observations)

    # Analyze dynamics from all observations (base + precision)
    all_obs_flat = [o for obs_list in all_observations.values() for o in obs_list]
    dynamics = extract_round_dynamics(all_obs_flat)
    if dynamics:
        log.info(f"Round dynamics: survival={dynamics.get('survival_rate',0):.0%} "
                 f"ports={dynamics.get('port_rate',0):.0%} "
                 f"avg_pop={dynamics.get('avg_population',0):.1f} "
                 f"avg_food={dynamics.get('avg_food',0):.1f} "
                 f"factions={dynamics.get('n_factions',0)} "
                 f"total_settlements={dynamics.get('total_settlements',0)}")

    # Recompute context from BASE observations only (unbiased — excludes precision)
    # Precision queries are biased toward settlement areas, so context uses base only
    context = compute_context_vector(base_observations, initial_grids)
    z_est = estimate_z_from_context(context)
    log.info(f"Final z estimate: {z_est:.3f} (from {sum(len(v) for v in base_observations.values())} base queries)")

    # Consult Codex before submitting
    codex_advisor.on_pre_solve(round_num, dynamics, z_est)

    # Phase 2: SAFE BASELINE — submit conservative for ALL 5 seeds immediately
    log.info("Phase 2: Safe baseline — conservative recipe for all seeds...")
    seeds_submitted = 0
    for seed_idx in range(seeds_count):
        pred = predict_for_seed(
            initial_states[seed_idx]["grid"],
            all_observations.get(seed_idx, []),
            context=context,
            observations_by_seed=base_observations,
            initial_grids=initial_grids,
        )
        if submit_prediction(api, round_id, seed_idx, pred, dry_run):
            seeds_submitted += 1

    log.info(f"  Safe baseline: {seeds_submitted}/5 seeds submitted")

    # Phase 3: CHAMPION/CHALLENGER — selective aggressive overwrites
    try:
        from seed_selector import compute_seed_trust, select_recipe, predict_with_recipe, DEFAULT_RECIPES
        log.info("Phase 3: Champion/challenger — per-seed trust evaluation...")
        calibration = load_calibration()
        aggressive_count = 0
        for seed_idx in range(seeds_count):
            trust = compute_seed_trust(
                seed_idx,
                initial_states[seed_idx]["grid"],
                all_observations.get(seed_idx, []),
                context, z_est,
            )
            recipe_name = select_recipe(trust, DEFAULT_RECIPES)
            log.info(f"  Seed {seed_idx}: recipe={recipe_name}, "
                     f"z={trust['z']:.3f}, agreement={trust['v2v3_agreement']:.3f}")

            # Insurance: max 3 aggressive overwrites
            if recipe_name == "aggressive" and aggressive_count >= 3:
                recipe_name = "conservative"
                log.info(f"  Seed {seed_idx}: downgraded to conservative (insurance cap)")

            # Only overwrite if recipe differs from baseline conservative
            if recipe_name != "conservative":
                pred = predict_with_recipe(
                    recipe_name, DEFAULT_RECIPES,
                    initial_states[seed_idx]["grid"],
                    all_observations.get(seed_idx, []),
                    context, z_est, calibration,
                )
                if submit_prediction(api, round_id, seed_idx, pred, dry_run):
                    log.info(f"  Seed {seed_idx}: OVERWRITTEN with {recipe_name}")
                    if recipe_name == "aggressive":
                        aggressive_count += 1
    except Exception as e:
        log.warning(f"Champion/challenger failed (safe baseline intact): {e}")

    # Record state
    total_queries = sum(len(obs) for obs in all_observations.values())
    st.mark_round_solved(persistence, round_id, {
        "round_number": round_num,
        "queries_used": budget["queries_used"] + total_queries,
        "seeds_submitted": seeds_submitted,
        "strategy": "champion_challenger_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    log.info(f"=== Round #{round_num} done: {seeds_submitted} seeds, {aggressive_count} aggressive ===")


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
    """Download ground truth from all completed rounds for calibration.

    When new GT is found: recalibrate and consult Codex.
    """
    try:
        my_rounds = api.get_my_rounds()
    except Exception as e:
        log.error(f"Failed to get my rounds: {e}")
        return

    new_gt_rounds = set()
    for r in my_rounds:
        if r["status"] != "completed":
            continue
        round_id = r["id"]
        round_num = r["round_number"]
        score = r.get("round_score")
        seeds = r.get("seeds_count", 5)

        for seed_idx in range(seeds):
            gt_file = GT_DIR / f"round_{round_num}_seed_{seed_idx}.json"
            if gt_file.exists():
                continue
            try:
                analysis = api.get_analysis(round_id, seed_idx)
                gt_file.write_text(json.dumps(analysis))
                log.info(f"  Saved ground truth: round {round_num} seed {seed_idx} (score={analysis.get('score')})")
                new_gt_rounds.add((round_num, score))
            except Exception as e:
                log.info(f"  No analysis for round {round_num} seed {seed_idx}: {e}")

    # If new GT was harvested: recalibrate and consult Codex
    if new_gt_rounds:
        log.info("New GT found — recalibrating...")
        try:
            from calibrate import calibrate
            calibrate()
            # Consult Codex on calibration
            cal = load_calibration()
            round_z = cal.get("round_z", {}) if cal else {}
            codex_advisor.on_calibration_update(
                len(list(GT_DIR.glob("round_*.json"))), round_z
            )
        except Exception as e:
            log.error(f"Recalibration failed: {e}")

        # Consult Codex on each newly scored round
        for round_num, score in new_gt_rounds:
            if score is not None:
                try:
                    cal = load_calibration()
                    rz = float(cal["round_z"].get(str(round_num), 0)) if cal else None
                    codex_advisor.on_round_scored(round_num, score, rz)
                except Exception as e:
                    log.warning(f"Codex consult failed: {e}")


def update_morning_report(api: AstarAPI, event: str = ""):
    """Keep MORNING_REPORT.md live-updated for the user."""
    report_file = Path(__file__).parent / "MORNING_REPORT.md"
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    persistence = st.load()
    solved = persistence.get("solved_rounds", {})

    # Build rounds table
    try:
        my_rounds = api.get_my_rounds()
    except Exception:
        my_rounds = []

    rounds_table = ""
    for r in sorted(my_rounds, key=lambda x: x.get("round_number", 0)):
        rn = r["round_number"]
        status = r["status"]
        score = r.get("round_score")
        score_str = f"{score:.1f}" if score is not None else "pending"
        seeds = r.get("seeds_submitted", 0)
        queries = f"{r.get('queries_used', 0)}/{r.get('queries_max', 50)}"
        rid = r["id"]
        strategy = solved.get(rid, {}).get("strategy", "—")
        rank = r.get("rank")
        rank_str = f" (rank #{rank})" if rank else ""
        rounds_table += f"| {rn} | {status} | {score_str}{rank_str} | {queries} | {seeds}/5 seeds | {strategy} |\n"

    # Leaderboard
    lb_table = ""
    try:
        lb = api.get_leaderboard()
        for entry in lb[:10]:
            lb_table += f"| {entry['rank']} | {entry['team_name']} | {entry['weighted_score']:.1f} |\n"
    except Exception:
        lb_table = "| ? | unavailable | ? |\n"

    # Our position
    our_team = ""
    try:
        for entry in lb:
            if "meinhold" in entry.get("team_name", "").lower() or entry.get("team_slug", "") == "":
                our_team = f"**#{entry['rank']}** — {entry['team_name']}: {entry['weighted_score']:.1f}"
                break
    except Exception:
        pass

    # GT files count
    gt_count = len(list(GT_DIR.glob("round_*.json")))

    # Actions log — append to existing
    actions_section = ""
    if event:
        try:
            existing = report_file.read_text()
            marker = "## Actions Taken Overnight"
            if marker in existing:
                before, after = existing.split(marker, 1)
                # Find next ## section
                lines = after.strip().split("\n")
                action_lines = []
                rest_lines = []
                in_actions = True
                for line in lines:
                    if line.startswith("## ") and in_actions:
                        in_actions = False
                    if in_actions:
                        action_lines.append(line)
                    else:
                        rest_lines.append(line)
                action_lines.append(f"- [{now}] {event}")
                actions_section = "\n".join(action_lines)
        except Exception:
            actions_section = f"- [{now}] {event}"
    else:
        try:
            existing = report_file.read_text()
            marker = "## Actions Taken Overnight"
            if marker in existing:
                after = existing.split(marker, 1)[1]
                lines = after.strip().split("\n")
                action_lines = []
                for line in lines:
                    if line.startswith("## "):
                        break
                    action_lines.append(line)
                actions_section = "\n".join(action_lines)
        except Exception:
            actions_section = "- (waiting for events...)"

    report = f"""# Astar Island — Morning Report

> Last updated: {now}

## Round Status

| Round | Status | Score | Queries | Submissions | Strategy |
|-------|--------|-------|---------|-------------|----------|
{rounds_table}
## Leaderboard Top 10
| Rank | Team | Score |
|------|------|-------|
{lb_table}
## Our Position
{our_team if our_team else "Not yet ranked"}

## Ground Truth & Calibration
- GT files: {gt_count}
- Calibration keys: {len(json.loads(Path(__file__).parent.joinpath('calibration.json').read_text()).get('priors', {})) if Path(__file__).parent.joinpath('calibration.json').exists() else 0}

## Actions Taken Overnight
{actions_section}

## Issues / Alerts
- Round 3 predictions degraded by server overwrite (fixed — server poller killed)
"""
    report_file.write_text(report)


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
            print(f"[astar] Round #{active['round_number']} solved. Waiting. closes={active.get('closes_at','?')}")
            return
        log.info(f"NEW ROUND: #{active['round_number']} (closes: {active.get('closes_at', '?')})")
        solve_round(api, active, dry_run)
        update_morning_report(api, f"Solved round #{active['round_number']} with calibrated Dirichlet strategy")
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
        # No active round — update report periodically (every ~10 ticks)
        report_file = Path(__file__).parent / "MORNING_REPORT.md"
        try:
            age = time.time() - report_file.stat().st_mtime
            if age > 600:  # Update every 10 min
                update_morning_report(api)
        except Exception:
            update_morning_report(api)
        print("[astar] No active round.")


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("=== DRY RUN MODE ===")
    check_and_solve(dry_run=dry)
