# Sweep Directive

When user says "sweep.md $ARGUMENTS", immediately launch 5x parallel Explore agents to investigate $ARGUMENTS across:

1. **prompts.py** — Does the system prompt guide the model correctly for this case? Wrong tool names, wrong params, misleading instructions?
2. **tool_router.py** — Are all ref fields mapped? Is the tool in TOOL_MAP? Does routing work?
3. **agent.py _pre_validate** — Does the tool have pre-validation? Missing defaults that cause 422s?
4. **agent.py auto-recovery** — Is there a recovery handler for this error class? What 422 patterns are unhandled?
5. **schema_guard.py + spec_catalog.py** — Is the endpoint in ENDPOINT_SCHEMA_MAP? Will tripletex_api fallback validate it?

## Known Gaps (Fixed)
- `activity_id`/`activityId` missing from REF_FIELDS — timesheet entries failed
- Pattern 28 told model to use `tripletex_api` instead of typed `post_timesheet_entry`
- `sendToCustomer` was clobbered (explicit False overwritten)
- "pre-existing products" false claim in system prompt
- `_ensure_bank_account` ran on every task (wasted writes)
- No voucher preflight validation (VAT lock 422s)

## Deploy Protocol
ALWAYS ask user before deploying. We are in production.

## Quick Debug Commands
```bash
# Last N tasks
ssh vps "tail -N /opt/tripletex/task_log.jsonl" | python3 -c "
import sys, json
for l in sys.stdin:
    d=json.loads(l.strip())
    e=len(d.get('errors',[]))
    s='FAIL' if e>0 else 'OK'
    print(f'[{s}] {d[\"logged_at\"][:19]} | t={d[\"turns\"]} | w={d.get(\"write_calls\",\"?\")}/{d.get(\"write_4xx\",\"?\")} | e={e} | {d[\"prompt\"][:100]}')
"

# Full trace of last task
ssh vps "tail -1 /opt/tripletex/task_log.jsonl" | python3 -c "
import sys,json
d=json.loads(sys.stdin.read().strip())
print(f'PROMPT: {d[\"prompt\"]}')
print(f'OUTCOME: {d[\"outcome\"]} | turns={d[\"turns\"]} | w={d.get(\"write_calls\")}/{d.get(\"write_4xx\")}')
for e in d.get('errors',[]): print(f'ERR: {e[:300]}')
for c in d['api_calls']:
    w=' [W]' if c.get('is_write') else ''
    err=' !!ERR!!' if c.get('error') else ''
    print(f'  {c[\"method\"]} {c[\"endpoint\"]}{w}{err}')
    if c.get('body'): print(f'    body: {json.dumps(c[\"body\"],ensure_ascii=False)[:400]}')
    print(f'    -> {c.get(\"result_snippet\",\"\")[:300]}')
"

# Live journal
ssh vps "journalctl -u tripletex -f"
```

## VPS
- Host: vps (XXx--xx-H100)
- Service: `systemctl restart tripletex`
- Log: `/opt/tripletex/task_log.jsonl`
- Code: `/opt/tripletex/`
