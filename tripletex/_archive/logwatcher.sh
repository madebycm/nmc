#!/usr/bin/env bash
# Live task watcher — shows last N tasks with real completion stats
# Usage: ./logwatcher.sh [count]  (default: 10)

set -euo pipefail
COUNT=${1:-10}
HOST="vps"
LOG="/opt/tripletex/task_log.jsonl"

C='\033[0m'
G='\033[32m'
R='\033[31m'
Y='\033[33m'
D='\033[2m'
B='\033[1m'
CY='\033[36m'

ssh "$HOST" "tail -${COUNT} ${LOG}" | python3 -c "
import sys, json

tasks = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        tasks.append(json.loads(line))
    except: pass

total = len(tasks)
passed = 0
failed = 0
for d in tasks:
    outcome = d.get('outcome', '?')
    errs = d.get('errors', [])
    turns = d.get('turns', '?')
    elapsed = d.get('elapsed_s', 0)
    ip = d.get('ip', '?')
    t = d.get('logged_at', '')[:19].replace('T', ' ')
    prompt = d.get('prompt', '')[:100]
    summary = d.get('summary', '')[:120]

    # Detect likely failures
    issues = []
    calls = d.get('api_calls', [])
    prompt_l = d['prompt'].lower()

    # Time reg as invoice
    if any(k in prompt_l for k in ['hora', 'timer', 'hours', 'stunden', 'timesheet']):
        if any('order' in c.get('endpoint','') for c in calls):
            issues.append('TIME_REG→INVOICE')

    # Year-end partial
    if 'delvis' in summary.lower() or 'partial' in summary.lower():
        issues.append('PARTIAL')

    # Turn limit
    if outcome in ('forced_completion_at_turn_limit',):
        issues.append('TURN_LIMIT')

    # 403 errors
    n403 = sum(1 for e in errs if '403' in e)
    if n403: issues.append(f'{n403}x403')

    # 422 errors
    n422 = sum(1 for e in errs if '422' in e)
    if n422: issues.append(f'{n422}x422')

    # Zero-amount voucher
    for c in calls:
        if c.get('method') == 'POST' and 'voucher' in c.get('endpoint',''):
            body = c.get('body', {})
            postings = body.get('postings', []) if body else []
            if postings and all(isinstance(p, dict) and p.get('amountGross', 1) == 0 for p in postings):
                issues.append('ZERO_VOUCHER')

    if outcome != 'completed' or issues:
        icon = '❌'
        failed += 1
    else:
        icon = '✅'
        passed += 1

    issue_str = f' ⚠ {\" \".join(issues)}' if issues else ''
    print(f'{icon} {t} [{elapsed:5.1f}s] {turns:>2}t ip={ip:<15}{issue_str}')
    print(f'   {prompt}')
    if issues or outcome != 'completed':
        print(f'   → {summary}')
    print()

print(f'━━━ {passed}/{total} likely correct ━━━')
"
