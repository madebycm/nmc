# Tripletex Live Log Query

**VPS**: XXx--xx-H100 (ssh alias: `vps`)
**Log file**: `/opt/tripletex/task_log.jsonl`
**Service**: `tripletex` (systemd)

## Last N tasks (summary)
```bash
ssh vps "tail -N /opt/tripletex/task_log.jsonl" | python3 -c "
import sys, json
for l in sys.stdin:
    d=json.loads(l.strip())
    e=len(d.get('errors',[]))
    s='FAIL' if e>0 else 'OK'
    print(f'[{s}] {d[\"logged_at\"][:19]} | t={d[\"turns\"]} | w={d.get(\"write_calls\",\"?\")}/{d.get(\"write_4xx\",\"?\")} | e={e} | {d[\"prompt\"][:100]}')
"
```
Replace `-N` with number, e.g. `tail -10`.

## Full trace of last task
```bash
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
```

## Live streaming logs
```bash
ssh vps "journalctl -u tripletex -f"
```

## Total task count
```bash
ssh vps "wc -l /opt/tripletex/task_log.jsonl"
```

## Filter only failures
```bash
ssh vps "cat /opt/tripletex/task_log.jsonl" | python3 -c "
import sys, json
for l in sys.stdin:
    d=json.loads(l.strip())
    if d.get('errors'):
        print(f'{d[\"logged_at\"][:19]} | e={len(d[\"errors\"])} | {d[\"prompt\"][:120]}')
        for e in d['errors']: print(f'  {e[:200]}')
        print()
"
```
