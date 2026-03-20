"""Polling loop — checks for active rounds and auto-solves them."""

import time
import logging

from api import AstarAPI
from config import TOKEN, POLL_INTERVAL
from solver import solve_round

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def main():
    api = AstarAPI(TOKEN)
    solved_rounds = set()
    log.info(f"Astar Island poller started. Checking every {POLL_INTERVAL}s.")

    while True:
        try:
            active = api.get_active_round()
            if active and active["id"] not in solved_rounds:
                log.info(f"NEW active round detected: #{active['round_number']}")
                solve_round(api, active)
                solved_rounds.add(active["id"])
                log.info(f"Round #{active['round_number']} solved and marked complete.")
            elif active:
                log.info(f"Round #{active['round_number']} already solved, skipping.")
            else:
                log.info("No active round.")
        except Exception as e:
            log.error(f"Poll error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
