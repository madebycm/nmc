"""Astar Island solver — safe, stateful, deterministic live path."""

import time
import json
import logging
import fcntl
import numpy as np
from pathlib import Path

from api import AstarAPI
from config import TOKEN, QUERIES_PER_SEED
from strategy import (
    predict_for_seed, floor_and_normalize, extract_round_dynamics,
    load_calibration, compute_context_vector, estimate_z_from_context,
)
import state as st

LOCK_FILE = Path(__file__).parent / ".solver.lock"

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
    """Return full 3x3 tiling in fixed order for unbiased coverage.

    All 9 tiles observed = 100% map coverage, no selection bias in context vector.
    """
    return _full_tiling(map_w, map_h)


def _pick_precision_targets(
    initial_states: list, seeds_count: int, n_queries: int,
    z: float = None, base_observations: dict = None,
    context: np.ndarray = None, initial_grids: list = None,
) -> list[tuple[int, int, int, int, int]]:
    """Pick precision queries by ENTROPY-WEIGHTED INFORMATION GAIN.

    Targets tiles where:
    1. Prediction entropy is HIGH (uncertain cells = highest score impact)
    2. Model disagreement is HIGH (NN vs Dirichlet)
    3. Observation count is LOW (most info gain from new observation)

    This directly optimizes for the competition metric: entropy-weighted KL.
    """
    scored = []
    from strategy import compute_empirical_observations

    # Get current model predictions for each seed
    nn_preds = {}
    dir_preds = {}
    obs_coverage = {}  # track observation coverage per seed
    try:
        import nn_predict
        from strategy import dirichlet_predict, load_calibration
        calibration = load_calibration()
        for seed_idx in range(seeds_count):
            grid = initial_states[seed_idx]["grid"]
            obs = base_observations.get(seed_idx, []) if base_observations else []
            nn_p = nn_predict.predict(grid, z=z, context=context)
            dir_p = dirichlet_predict(grid, obs, calibration, z=z)
            if nn_p is not None:
                nn_preds[seed_idx] = nn_p
                dir_preds[seed_idx] = dir_p
            # Track observation coverage
            _, n_obs = compute_empirical_observations(obs)
            obs_coverage[seed_idx] = n_obs
    except Exception as e:
        log.warning(f"Precision scoring failed, falling back to entropy: {e}")

    # Compute per-seed z for alive gating
    seed_z = {}
    if base_observations and initial_grids:
        from strategy import compute_context_vector, estimate_z_from_context
        for s in range(seeds_count):
            try:
                ctx_s = compute_context_vector({0: base_observations.get(s, [])}, [initial_grids[s]])
                seed_z[s] = estimate_z_from_context(ctx_s)
            except Exception:
                seed_z[s] = z if z else 0.3

    for seed_idx in range(seeds_count):
        grid = np.array(initial_states[seed_idx]["grid"])
        h_map, w_map = grid.shape
        tiles = _full_tiling(w_map, h_map)

        # Alive gate: suppress precision on catastrophic seeds
        zs = seed_z.get(seed_idx, z if z else 0.3)
        if zs < 0.08:
            gate = 0.0
        elif zs < 0.15:
            gate = 0.5
        else:
            gate = 1.0

        for x, y, w, h in tiles:
            if seed_idx in nn_preds and gate > 0:
                nn_tile = nn_preds[seed_idx][y:y+h, x:x+w]
                dir_tile = dir_preds[seed_idx][y:y+h, x:x+w]
                eps = 1e-8

                # 1. Prediction entropy (directly from ensemble mean)
                m_tile = 0.5 * (nn_tile + dir_tile)
                pred_entropy = -(m_tile * np.log(m_tile + eps)).sum(axis=-1)

                # 2. Model disagreement (JS divergence)
                js = 0.5 * (nn_tile * np.log((nn_tile + eps) / (m_tile + eps))).sum(axis=-1) \
                   + 0.5 * (dir_tile * np.log((dir_tile + eps) / (m_tile + eps))).sum(axis=-1)

                # 3. Low-coverage bonus (cells with fewer observations gain more)
                n_obs_tile = obs_coverage.get(seed_idx, np.zeros((h_map, w_map)))[y:y+h, x:x+w]
                coverage_bonus = 1.0 / (n_obs_tile + 1.0)  # 1/1=1.0, 1/2=0.5, 1/3=0.33

                # Combined: entropy-weighted info gain
                # Entropy is the weight in the competition metric, so high-entropy cells
                # are where we need to be most accurate
                info_gain = pred_entropy * (js + 0.1) * coverage_bonus
                score = float(gate * info_gain.mean())
            else:
                region = grid[y:y+h, x:x+w]
                score = float(gate * np.isin(region, [1, 2, 3]).sum())

            scored.append((score, seed_idx, x, y, w, h))

    scored.sort(reverse=True)

    # Spread across seeds: max 2 per seed for 5 queries
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


def _save_round_data(round_num: int, detail: dict, all_observations: dict,
                     base_observations: dict = None):
    """Persist full round data with base/precision split for faithful replay."""
    round_dir = OBS_DIR / f"round_{round_num}"
    round_dir.mkdir(exist_ok=True)

    meta = {
        "round_number": round_num,
        "map_width": detail["map_width"],
        "map_height": detail["map_height"],
        "seeds_count": detail["seeds_count"],
    }
    # Save initial grids
    for seed_idx, state in enumerate(detail["initial_states"]):
        init_file = round_dir / f"initial_seed_{seed_idx}.json"
        if not init_file.exists():
            init_file.write_text(json.dumps(state["grid"]))

    # Save all observations per seed
    for seed_idx, observations in all_observations.items():
        obs_file = round_dir / f"observations_seed_{seed_idx}.json"
        obs_file.write_text(json.dumps(observations))

    # Save base observation counts for faithful replay
    if base_observations:
        base_counts = {str(k): len(v) for k, v in base_observations.items()}
        (round_dir / "base_counts.json").write_text(json.dumps(base_counts))

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
        log.info("No queries left — attempting resubmit from cached observations...")
        round_dir = OBS_DIR / f"round_{round_num}"
        if not round_dir.exists():
            log.warning("No cached observations found, cannot resubmit!")
            return
        # Load cached observations
        initial_grids = [initial_states[i]["grid"] for i in range(seeds_count)]
        all_observations = {}
        base_observations = {}
        for seed_idx in range(seeds_count):
            obs_file = round_dir / f"observations_seed_{seed_idx}.json"
            if obs_file.exists():
                all_observations[seed_idx] = json.loads(obs_file.read_text())
            else:
                all_observations[seed_idx] = []
            # Split base from precision using base_counts
            base_counts_file = round_dir / "base_counts.json"
            if base_counts_file.exists():
                bc = json.loads(base_counts_file.read_text())
                n_base = bc.get(str(seed_idx), len(all_observations[seed_idx]))
            else:
                n_base = len(all_observations[seed_idx])
            base_observations[seed_idx] = all_observations[seed_idx][:n_base]
        context = compute_context_vector(base_observations, initial_grids)
        z_est = estimate_z_from_context(context)
        log.info(f"Resubmit z estimate: {z_est:.3f} (from cached observations)")
        # Submit with current models
        seeds_submitted = 0
        for seed_idx in range(seeds_count):
            pred = predict_for_seed(
                initial_states[seed_idx]["grid"],
                all_observations.get(seed_idx, []),
                context=context,
                observations_by_seed=base_observations,
                initial_grids=initial_grids,
                seed_idx=seed_idx,
            )
            if submit_prediction(api, round_id, seed_idx, pred, dry_run):
                seeds_submitted += 1
        log.info(f"  Resubmitted: {seeds_submitted}/5 seeds (NN models active)")
        st.mark_round_solved(persistence, round_id, {
            "round_number": round_num,
            "queries_used": 50,
            "seeds_submitted": seeds_submitted,
            "strategy": "resubmit_nn",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return

    initial_grids = [initial_states[i]["grid"] for i in range(seeds_count)]
    tiles = _full_tiling(map_w, map_h)  # 9 tiles
    all_observations = {s: [] for s in range(seeds_count)}

    # Phase 1: MANDATORY base scan — interleaved across seeds, retry on failure
    # Do tile 0 for all seeds, then tile 1 for all seeds, etc.
    # This spreads rate-limit pain evenly — no seed is structurally disadvantaged.
    log.info(f"Phase 1: Interleaved base scan ({len(tiles)} tiles × {seeds_count} seeds = {len(tiles)*seeds_count} queries)...")
    base_success = 0
    for tile_idx, (vx, vy, vw, vh) in enumerate(tiles):
        for seed_idx in range(seeds_count):
            if dry_run:
                log.info(f"  [DRY RUN] tile {tile_idx} seed {seed_idx}: ({vx},{vy})")
                continue
            # Mandatory: retry until success — do NOT advance past a failed tile
            for attempt in range(5):
                try:
                    time.sleep(0.8)  # Conservative: 45/45 matters more than speed
                    result = api.simulate(round_id, seed_idx, vx, vy, vw, vh)
                    all_observations[seed_idx].append(result)
                    base_success += 1
                    if tile_idx == 0 or (tile_idx == 8 and seed_idx == seeds_count - 1):
                        log.info(f"  tile {tile_idx} seed {seed_idx}: ({vx},{vy}) — {result['queries_used']}/{result['queries_max']}")
                    break
                except Exception as e:
                    if "429" in str(e):
                        wait = 2.0 + attempt * 1.5  # Increasing backoff
                        log.warning(f"  tile {tile_idx} seed {seed_idx}: 429, retry in {wait:.0f}s (attempt {attempt+1}/5)")
                        time.sleep(wait)
                    else:
                        log.error(f"  tile {tile_idx} seed {seed_idx}: {e}")
                        break
    log.info(f"  Base scan: {base_success}/{len(tiles)*seeds_count} queries successful")

    # Snapshot base observations for unbiased context
    base_observations = {k: list(v) for k, v in all_observations.items()}

    # Compute z from base observations
    context = compute_context_vector(base_observations, initial_grids)
    z_est = estimate_z_from_context(context)
    log.info(f"z estimate: {z_est:.3f} (from {base_success} base queries)")

    # Phase 1b: Precision queries — only AFTER all base tiles succeeded
    budget_after = api.get_budget()
    precision_left = budget_after["queries_max"] - budget_after["queries_used"]
    if precision_left > 0 and not dry_run:
        log.info(f"Phase 1b: {precision_left} precision queries (JS-divergence targeted)...")
        precision_targets = _pick_precision_targets(
            initial_states, seeds_count, precision_left,
            z=z_est, base_observations=base_observations,
            context=context, initial_grids=initial_grids,
        )
        for seed_idx, vx, vy, vw, vh in precision_targets:
            try:
                time.sleep(0.8)
                result = api.simulate(round_id, seed_idx, vx, vy, vw, vh)
                all_observations[seed_idx].append(result)
            except Exception as e:
                if "429" in str(e):
                    time.sleep(3)
                    try:
                        result = api.simulate(round_id, seed_idx, vx, vy, vw, vh)
                        all_observations[seed_idx].append(result)
                    except Exception:
                        pass
                else:
                    log.warning(f"  Precision failed: {e}")
                    break

    # Save all data with base/precision split for faithful replay
    _save_round_data(round_num, detail, all_observations, base_observations)

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

    # Phase 2: SAFE BASELINE — submit for ALL 5 seeds immediately
    # No codex/advisor in critical path — determinism over intelligence
    log.info("Phase 2: Safe baseline — submitting all seeds...")
    seeds_submitted = 0
    for seed_idx in range(seeds_count):
        pred = predict_for_seed(
            initial_states[seed_idx]["grid"],
            all_observations.get(seed_idx, []),
            context=context,
            observations_by_seed=base_observations,
            initial_grids=initial_grids,
            seed_idx=seed_idx,
        )
        if submit_prediction(api, round_id, seed_idx, pred, dry_run):
            seeds_submitted += 1

    log.info(f"  Submitted: {seeds_submitted}/5 seeds")

    # Record state — only mark solved when all 5 seeds submitted
    total_queries = sum(len(obs) for obs in all_observations.values())
    if seeds_submitted >= 5:
        st.mark_round_solved(persistence, round_id, {
            "round_number": round_num,
            "queries_used": budget["queries_used"] + total_queries,
            "seeds_submitted": seeds_submitted,
            "strategy": "doctrine_v1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        log.info(f"=== Round #{round_num} done: {seeds_submitted}/5 seeds ===")
    else:
        # Partial submit — do NOT mark as solved so next cron run retries
        log.warning(f"=== Round #{round_num} PARTIAL: only {seeds_submitted}/5 seeds — will retry ===")


    # observe_seed removed — replaced by interleaved base scan in solve_round


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

    # If new GT was harvested: recalibrate (no external calls in this path)
    if new_gt_rounds:
        log.info("New GT found — recalibrating...")
        try:
            from calibrate import calibrate
            calibrate()
        except Exception as e:
            log.error(f"Recalibration failed: {e}")


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
    """Main entry: check for active round FIRST, then harvest GT.

    Priority order: solve > harvest > report. Nothing non-essential in critical path.
    Filesystem lock prevents concurrent cron overlap.
    """
    # Filesystem lock — prevent concurrent runs (R3-class disaster prevention)
    lock_fd = None
    try:
        lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("[astar] Another solver instance running — skipping.")
        if lock_fd:
            lock_fd.close()
        return

    try:
        api = AstarAPI(TOKEN)

        # PRIORITY 1: Check for active round FIRST — solve before anything else
        active = api.get_active_round()
        if active:
            persistence = st.load()
            if st.is_round_solved(persistence, active["id"]):
                print(f"[astar] Round #{active['round_number']} solved. Waiting. closes={active.get('closes_at','?')}")
            else:
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

        # PRIORITY 2: Harvest GT and recalibrate (AFTER solving, never before)
        harvest_ground_truth(api)

    finally:
        # Release lock
        if lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            try:
                LOCK_FILE.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    if dry:
        log.info("=== DRY RUN MODE ===")
    check_and_solve(dry_run=dry)
