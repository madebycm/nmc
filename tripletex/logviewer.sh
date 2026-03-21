#!/usr/bin/env bash
# Live log viewer for Tripletex agent on nm.j6x.com
# Streams task_log.jsonl entries as they arrive + uvicorn output

set -euo pipefail

HOST="vps"
LOG="/opt/tripletex/task_log.jsonl"

# Colors
C_RESET='\033[0m'
C_GREEN='\033[32m'
C_RED='\033[31m'
C_YELLOW='\033[33m'
C_CYAN='\033[36m'
C_DIM='\033[2m'
C_BOLD='\033[1m'

pretty_print() {
    while IFS= read -r line; do
        # Skip empty lines
        [[ -z "$line" ]] && continue

        # Try to parse as JSON (task_log.jsonl entries)
        if echo "$line" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
            echo "$line" | python3 -c "
import sys, json
e = json.load(sys.stdin)
ts = e.get('logged_at', e.get('timestamp', '?'))[:19].replace('T', ' ')
outcome = e.get('outcome', '?')
score = e.get('score', '')
prompt = e.get('prompt', '')[:100]
task_id = e.get('id', e.get('task_id', '?'))
errors = e.get('errors', [])
calls = e.get('api_calls', [])
turns = e.get('turns', '?')

# Color code
if outcome == 'success':
    oc = '\033[32m'  # green
elif outcome == 'fail':
    oc = '\033[31m'  # red
else:
    oc = '\033[33m'  # yellow

print()
print(f'\033[1m\033[36m━━━ Case {task_id} ━━━\033[0m')
print(f'\033[2m{ts}\033[0m  {oc}{outcome.upper()}\033[0m  {f\"score: {score}\" if score else \"\"}  turns: {turns}')
print(f'\033[36mPrompt:\033[0m {prompt}')

if calls:
    chain = ' → '.join(f\"{c.get(\"method\",\"?\")} {c.get(\"endpoint\",\"?\")} [{c.get(\"status\",\"?\")}]\" for c in calls[:8])
    print(f'\033[2mAPI:\033[0m {chain}')

if errors:
    for err in errors[:3]:
        print(f'\033[31m  ✗ {err}\033[0m')

print()
"
        else
            # Plain log line from uvicorn/agent
            echo -e "${C_DIM}${line}${C_RESET}"
        fi
    done
}

echo -e "${C_BOLD}${C_CYAN}╔══════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}║  Tripletex Agent — Live Log Viewer       ║${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}║  nm.j6x.com → vps (77.42.85.193)    ║${C_RESET}"
echo -e "${C_BOLD}${C_CYAN}╚══════════════════════════════════════════╝${C_RESET}"
echo

# Show last 3 entries then follow
echo -e "${C_DIM}Connecting to ${HOST}...${C_RESET}"
echo -e "${C_DIM}Tailing ${LOG} + server logs${C_RESET}"
echo

ssh -o ConnectTimeout=5 "$HOST" "tail -n 3 -f ${LOG}" 2>&1 | pretty_print
