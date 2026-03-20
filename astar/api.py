"""Astar Island API client."""

import json
import time
import httpx

BASE = "https://api.ainm.no/astar-island"


class AstarAPI:
    def __init__(self, token: str):
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )

    # ---- Public endpoints ----

    def get_rounds(self) -> list[dict]:
        r = self.client.get(f"{BASE}/rounds")
        r.raise_for_status()
        return r.json()

    def get_active_round(self) -> dict | None:
        rounds = self.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)

    def get_round_detail(self, round_id: str) -> dict:
        r = self.client.get(f"{BASE}/rounds/{round_id}")
        r.raise_for_status()
        return r.json()

    def get_leaderboard(self) -> list[dict]:
        r = self.client.get(f"{BASE}/leaderboard")
        r.raise_for_status()
        return r.json()

    # ---- Team endpoints ----

    def get_budget(self) -> dict:
        r = self.client.get(f"{BASE}/budget")
        r.raise_for_status()
        return r.json()

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int = 0,
        viewport_y: int = 0,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> dict:
        r = self.client.post(f"{BASE}/simulate", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        })
        r.raise_for_status()
        return r.json()

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        r = self.client.post(f"{BASE}/submit", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        })
        r.raise_for_status()
        return r.json()

    def get_my_rounds(self) -> list[dict]:
        r = self.client.get(f"{BASE}/my-rounds")
        r.raise_for_status()
        return r.json()

    def get_my_predictions(self, round_id: str) -> list[dict]:
        r = self.client.get(f"{BASE}/my-predictions/{round_id}")
        r.raise_for_status()
        return r.json()

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        r = self.client.get(f"{BASE}/analysis/{round_id}/{seed_index}")
        r.raise_for_status()
        return r.json()
