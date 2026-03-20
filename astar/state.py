"""Persistent state management — never lose context, never double-execute."""

import json
from pathlib import Path

STATE_FILE = Path(__file__).parent / "state.json"


def load() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "solved_rounds": {},  # round_id → {queries_used, seeds_submitted, strategy, score}
        "observations": {},   # round_id → {seed_idx → [observations]}
        "calibration": {},    # learned priors from ground truth
    }


def save(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def mark_round_solved(state: dict, round_id: str, info: dict):
    state["solved_rounds"][round_id] = info
    save(state)


def is_round_solved(state: dict, round_id: str) -> bool:
    return round_id in state.get("solved_rounds", {})


def get_round_info(state: dict, round_id: str) -> dict | None:
    return state.get("solved_rounds", {}).get(round_id)
