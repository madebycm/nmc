"""Harvest replay data from POST /replay endpoint.

Fetches N Monte Carlo samples per round/seed config.
Stores raw JSON frames for training and analysis.

Usage:
    python harvest_replays.py              # N=20 quick pass
    python harvest_replays.py --full       # N=100 full harvest
    python harvest_replays.py --round 8    # specific round only
"""

import json
import time
import sys
import logging
from pathlib import Path
import httpx

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

ROOT = Path(__file__).parent
REPLAY_DIR = ROOT / "replays"

# Import config
sys.path.insert(0, str(ROOT))
from config import TOKEN

BASE = "https://api.ainm.no/astar-island"
DELAY = 3.0  # chill — no rush, we have hours
MAX_RETRIES = 5


def fetch_replay(client: httpx.Client, round_id: str, seed_index: int) -> dict | None:
    """Fetch one replay with retry/backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            r = client.post(f"{BASE}/replay", json={
                "round_id": round_id,
                "seed_index": seed_index,
            })
            data = r.json()
            if "frames" in data:
                return data
            if "detail" in data and "rate limit" in data["detail"].lower():
                wait = 10 * (attempt + 1)  # very generous backoff
                log.warning(f"Rate limited, backing off {wait}s...")
                time.sleep(wait)
                continue
            log.error(f"Unexpected response: {data}")
            return None
        except Exception as e:
            log.error(f"Request failed: {e}")
            time.sleep(3)
    return None


def get_completed_rounds(client: httpx.Client) -> list[dict]:
    """Get all completed rounds."""
    r = client.get(f"{BASE}/my-rounds")
    r.raise_for_status()
    rounds = r.json()
    return [rd for rd in rounds if rd.get("status") == "completed"]


def count_existing(round_num: int, seed: int) -> int:
    """Count already-fetched samples for a round/seed."""
    d = REPLAY_DIR / f"round_{round_num}" / f"seed_{seed}"
    if not d.exists():
        return 0
    return len(list(d.glob("sample_*.json")))


def harvest(n_samples: int = 20, target_round: int | None = None):
    """Main harvest loop."""
    client = httpx.Client(
        headers={"Authorization": f"Bearer {TOKEN}"},
        timeout=30.0,
    )

    rounds = get_completed_rounds(client)
    if target_round:
        rounds = [r for r in rounds if r["round_number"] == target_round]

    log.info(f"Harvesting {n_samples} samples per config from {len(rounds)} rounds")

    total_fetched = 0
    total_skipped = 0

    for rd in sorted(rounds, key=lambda x: x["round_number"]):
        rnum = rd["round_number"]
        rid = rd["id"]
        seeds_count = rd.get("seeds_count", 5)

        for seed in range(seeds_count):
            existing = count_existing(rnum, seed)
            needed = max(0, n_samples - existing)

            if needed == 0:
                total_skipped += n_samples
                continue

            out_dir = REPLAY_DIR / f"round_{rnum}" / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"R{rnum} s{seed}: fetching {needed} samples ({existing} existing)")

            for i in range(needed):
                sample_idx = existing + i
                data = fetch_replay(client, rid, seed)
                if data is None:
                    log.error(f"R{rnum} s{seed} sample {sample_idx}: FAILED")
                    continue

                out_path = out_dir / f"sample_{sample_idx:03d}.json"
                out_path.write_text(json.dumps(data))
                total_fetched += 1

                if total_fetched % 10 == 0:
                    log.info(f"  Progress: {total_fetched} fetched, {total_skipped} skipped")

                time.sleep(DELAY)

    log.info(f"DONE: {total_fetched} fetched, {total_skipped} skipped")
    return total_fetched


if __name__ == "__main__":
    full = "--full" in sys.argv
    n = 100 if full else 20

    target = None
    if "--round" in sys.argv:
        idx = sys.argv.index("--round")
        target = int(sys.argv[idx + 1])

    harvest(n_samples=n, target_round=target)
