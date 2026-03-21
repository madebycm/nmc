#!/usr/bin/env python3
"""Resubmit a round with updated model/strategy using saved observations."""
import json, sys, logging, numpy as np
from pathlib import Path
from api import AstarAPI
from config import TOKEN
from solver import submit_prediction
from strategy import predict_for_seed, compute_context_vector, estimate_z_from_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def resubmit(round_num: int):
    state = json.load(open("state.json"))
    api = AstarAPI(TOKEN)

    # Find round_id from solved_rounds
    round_id = None
    for rid, info in state["solved_rounds"].items():
        if info["round_number"] == round_num:
            round_id = rid
            break
    if not round_id:
        log.error(f"Round {round_num} not found in solved_rounds")
        return

    # Load saved observations
    obs_dir = Path(f"observations/round_{round_num}")
    seeds_count = 5
    initial_grids = []
    all_observations = {}
    base_observations = {}

    for seed_idx in range(seeds_count):
        grid = json.loads((obs_dir / f"initial_seed_{seed_idx}.json").read_text())
        initial_grids.append(grid)
        obs = json.loads((obs_dir / f"observations_seed_{seed_idx}.json").read_text())
        all_observations[seed_idx] = obs
        # Base observations = first 9 per seed (full 3x3 tiling)
        base_observations[seed_idx] = obs[:9]

    # Compute context from base observations only
    context = compute_context_vector(base_observations, initial_grids)
    z = estimate_z_from_context(context)
    log.info(f"Round {round_num}: z={z:.3f}, resubmitting with updated model...")

    # Predict and submit each seed
    submitted = 0
    for seed_idx in range(seeds_count):
        pred = predict_for_seed(
            initial_grids[seed_idx],
            all_observations[seed_idx],
            context=context,
            observations_by_seed=base_observations,
            initial_grids=initial_grids,
            seed_idx=seed_idx,
        )
        if submit_prediction(api, round_id, seed_idx, pred, dry_run=False):
            submitted += 1

    log.info(f"Resubmitted {submitted}/5 seeds for round {round_num}")

if __name__ == "__main__":
    round_num = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    resubmit(round_num)
