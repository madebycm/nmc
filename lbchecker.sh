#!/bin/bash
# NM i AI 2026 - Leaderboard Checker
# Usage: ./lbchecker.sh [tripletex|astar|all]

TASK="${1:-all}"
TOP="${2:-10}"

tripletex() {
  echo "=== TRIPLETEX (Top $TOP) ==="
  curl -s "https://api.ainm.no/tripletex/leaderboard" | python3 -c "
import json,sys
d=json.load(sys.stdin)[:$TOP]
print(f'{'Rank':<5} {'Team':<30} {'Score':>8} {'Tasks':>5} {'Subs':>5} {'T1':>7} {'T2':>7} {'T3':>7}')
print('-'*80)
for t in d:
    print(f'{t[\"rank\"]:<5} {t[\"team_name\"][:29]:<30} {t[\"total_score\"]:>8.2f} {t[\"tasks_touched\"]:>5} {t[\"total_submissions\"]:>5} {t.get(\"tier1_score\",0):>7.2f} {t.get(\"tier2_score\",0):>7.2f} {t.get(\"tier3_score\",0):>7.2f}')
"
}

astar() {
  echo "=== ASTAR ISLAND (Top $TOP) ==="
  curl -s "https://api.ainm.no/astar-island/leaderboard" | python3 -c "
import json,sys
d=json.load(sys.stdin)[:$TOP]
print(f'{'Rank':<5} {'Team':<30} {'Score':>8} {'Rounds':>6} {'Streak':>8} {'Verified':>8}')
print('-'*70)
for t in d:
    v='Y' if t.get('is_verified') else 'N'
    print(f'{t[\"rank\"]:<5} {t[\"team_name\"][:29]:<30} {t[\"weighted_score\"]:>8.2f} {t[\"rounds_participated\"]:>6} {t.get(\"hot_streak_score\",0):>8.2f} {v:>8}')
"
}

norgesgruppen() {
  echo "=== NORGESGRUPPEN DATA (Top $TOP) ==="
  DATA=$(curl -s "https://api.ainm.no/norgesgruppen/leaderboard" 2>/dev/null)
  if echo "$DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); assert isinstance(d,list)" 2>/dev/null; then
    echo "$DATA" | python3 -c "
import json,sys
d=json.load(sys.stdin)[:$TOP]
if not d: print('No submissions yet.'); sys.exit()
print(f'{'Rank':<5} {'Team':<30} {'Score':>8}')
print('-'*45)
for t in d:
    print(f'{t.get(\"rank\",\"?\"):<5} {t.get(\"team_name\",\"?\")[:29]:<30} {t.get(\"score\",t.get(\"weighted_score\",0)):>8.4f}')
"
  else
    echo "No public API available yet."
  fi
}

case "$TASK" in
  tripletex|t) tripletex ;;
  astar|a) astar ;;
  norgesgruppen|ng|n) norgesgruppen ;;
  all|*)
    tripletex
    echo ""
    astar
    echo ""
    norgesgruppen
    ;;
esac
