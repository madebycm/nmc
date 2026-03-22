#!/usr/bin/env bash
# Live prod watcher — polls task_log.jsonl from VPS, formats nicely locally.
# Prod entries saved to logs/live/prod.jsonl. Smoke tests (X-Smoke-Test: 1 header)
# are shown dimmed but NOT written to prod.jsonl.
#
# Usage: ./prod-watcher.sh

set -euo pipefail
cd "$(dirname "$0")"

LIVE_DIR="logs/live"
mkdir -p "$LIVE_DIR"
PROD_LOG="$LIVE_DIR/prod.jsonl"
REMOTE="vps"
REMOTE_LOG="/opt/tripletex/task_log.jsonl"

echo -e "\033[1mTripletex Prod Watcher\033[0m"
echo -e "\033[2mStreaming from ${REMOTE}:${REMOTE_LOG}\033[0m"
echo -e "\033[2mProd log: ${PROD_LOG}\033[0m"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

OFFSET=$(ssh "$REMOTE" "wc -l < $REMOTE_LOG" 2>/dev/null || echo "0")
echo -e "\033[2mSkipping $OFFSET existing entries, watching for new...\033[0m\n"

while true; do
    CURRENT=$(ssh "$REMOTE" "wc -l < $REMOTE_LOG" 2>/dev/null || echo "$OFFSET")
    if [ "$CURRENT" -gt "$OFFSET" ]; then
        NEW_LINES=$((CURRENT - OFFSET))
        ssh "$REMOTE" "tail -n $NEW_LINES $REMOTE_LOG" 2>/dev/null | python3 -c "
import json, sys

prod_log = sys.argv[1]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except:
        continue

    ip = d.get('ip', d.get('_client_ip', '?'))
    is_smoke = d.get('smoke', False)

    # Only write prod (non-smoke) to file
    if not is_smoke:
        with open(prod_log, 'a') as f:
            f.write(line + '\n')

    ts = d.get('logged_at', '')[:19].replace('T', ' ')
    outcome = d.get('outcome', '?')
    turns = d.get('turns', '-')
    elapsed = d.get('elapsed_s', '-')
    calls = len(d.get('api_calls', []))
    errs = d.get('errors', [])
    prompt = d.get('prompt', '')[:120]
    summary = (d.get('summary') or '')[:120]

    if is_smoke:
        ip_tag = f'\033[2m[SMOKE]\033[0m'
    else:
        ip_tag = f'\033[35m[PROD]\033[0m'

    if outcome == 'completed':
        oc = f'\033[32m{outcome}\033[0m'
    elif outcome == 'crash':
        oc = f'\033[31m{outcome}\033[0m'
    elif outcome == 'no_completion_signal':
        oc = f'\033[33m{outcome}\033[0m'
    else:
        oc = outcome

    print(f'\033[1m━━━ {ts} ━━━\033[0m')
    print(f'  {ip_tag}  Status: {oc}  Turns: {turns}  Calls: {calls}  Time: {elapsed}s')
    print(f'  \033[36mPrompt:\033[0m {prompt}')
    if summary:
        print(f'  \033[32mResult:\033[0m {summary}')
    if errs:
        for e in errs[:3]:
            print(f'  \033[31mERR:\033[0m {e[:150]}')
    print()
    sys.stdout.flush()
" "$PROD_LOG" 2>/dev/null || true
        OFFSET=$CURRENT
    fi
    sleep 3
done
