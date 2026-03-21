#!/usr/bin/env bash
# NM i AI 2026 - Leaderboard Checker
# Usage: ./lbchecker.sh [overall|tripletex|astar|norgesgruppen|all] [top]

set -euo pipefail

TASK="${1:-all}"
TOP="${2:-10}"
API_BASE="https://api.ainm.no"

overall() {
  echo "=== OVERALL COMPETITION (Top $TOP) ==="
  curl -s "$API_BASE/competitions/nm-i-ai-2026/leaderboard" | python3 -c '
import json
import sys

top = int(sys.argv[1])
rows = json.load(sys.stdin)[:top]
print("{:<5} {:<30} {:>8} {:>5} {:>5}".format("Rank", "Team", "Score", "Tasks", "U23"))
print("-" * 60)
for row in rows:
    u23 = "Y" if row.get("is_u23") else "N"
    print("{:<5} {:<30} {:>8.4f} {:>5} {:>5}".format(
        row["rank"],
        row["team_name"][:29],
        row["total_score"],
        row.get("tasks_completed", 0),
        u23,
    ))
' "$TOP"
}

tripletex() {
  echo "=== TRIPLETEX (Top $TOP) ==="
  curl -s "$API_BASE/tripletex/leaderboard" | python3 -c '
import json
import sys

top = int(sys.argv[1])
rows = json.load(sys.stdin)[:top]
print("{:<5} {:<30} {:>8} {:>5} {:>5} {:>7} {:>7} {:>7}".format(
    "Rank", "Team", "Score", "Tasks", "Subs", "T1", "T2", "T3"
))
print("-" * 80)
for row in rows:
    print("{:<5} {:<30} {:>8.2f} {:>5} {:>5} {:>7.2f} {:>7.2f} {:>7.2f}".format(
        row["rank"],
        row["team_name"][:29],
        row["total_score"],
        row.get("tasks_touched", 0),
        row.get("total_submissions", 0),
        row.get("tier1_score", 0),
        row.get("tier2_score", 0),
        row.get("tier3_score", 0),
    ))
' "$TOP"
}

astar() {
  echo "=== ASTAR ISLAND (Top $TOP) ==="
  curl -s "$API_BASE/astar-island/leaderboard" | python3 -c '
import json
import sys

top = int(sys.argv[1])
rows = json.load(sys.stdin)[:top]
print("{:<5} {:<30} {:>8} {:>6} {:>8} {:>8}".format(
    "Rank", "Team", "Score", "Rounds", "Streak", "Verified"
))
print("-" * 70)
for row in rows:
    verified = "Y" if row.get("is_verified") else "N"
    print("{:<5} {:<30} {:>8.2f} {:>6} {:>8.2f} {:>8}".format(
        row["rank"],
        row["team_name"][:29],
        row["weighted_score"],
        row.get("rounds_participated", 0),
        row.get("hot_streak_score", 0),
        verified,
    ))
' "$TOP"
}

norgesgruppen() {
  echo "=== NORGESGRUPPEN DATA (Top $TOP) ==="
  curl -s "$API_BASE/competitions/nm-i-ai-2026/leaderboard/norgesgruppen-data" | python3 -c '
import json
import sys

top = int(sys.argv[1])
payload = json.load(sys.stdin)
rows = payload.get("rankings", [])[:top]
if not rows:
    print("No submissions yet.")
    raise SystemExit(0)

print("{:<5} {:<30} {:>8} {:>5} {:>8}".format(
    "Rank", "Team", "Score", "Subs", "Verified"
))
print("-" * 62)
for row in rows:
    verified = "Y" if row.get("is_verified") else "N"
    print("{:<5} {:<30} {:>8.4f} {:>5} {:>8}".format(
        row.get("rank", "?"),
        row.get("team_name", "?")[:29],
        row.get("overall_score", 0),
        row.get("total_submissions", 0),
        verified,
    ))
' "$TOP"
}

case "$TASK" in
  overall|o) overall ;;
  tripletex|t) tripletex ;;
  astar|a) astar ;;
  norgesgruppen|ng|n) norgesgruppen ;;
  all|*)
    overall
    echo ""
    tripletex
    echo ""
    astar
    echo ""
    norgesgruppen
    ;;
esac
