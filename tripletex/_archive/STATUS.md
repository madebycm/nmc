# Tripletex Agent — Live Status

**Score: 5/8** (last scored batch, 22:21 UTC Mar 21)
**Server: vps (77.42.85.193)** | **Backend: Gemini 2.5 Pro**

---

## Fixes Deployed (22:33 UTC)

1. **incomingInvoice blocked** — intercepted at tool + API level, forces voucher fallback
2. **Payroll** — plan validator enforces full Norwegian statutory deductions (AGA, feriepenger, skattetrekk)
3. **Depreciation** — simplified account search, fallback to 6000/1200
4. **Plan validator** — no longer false-rejects year-end tasks as "payment reversals"
5. **VAT handling** — prompt specifies which accounts are locked to MVA-kode 0

## What Works

- CRUD: customers, suppliers, departments, employees — near 100%
- Order → invoice → payment flow
- Credit notes, reminders
- Projects, milestones
- Voucher posting, journal entries
- Multi-language (NO/DE/FR/PT/ES/EN)
- Travel expenses
- Payment reversals (negative paidAmount)
- Multi-currency + agio/disagio

## Architecture

```
Platform (34.34.240.x) → POST /solve → nginx → uvicorn → agent.py (Gemini 2.5 Pro)
                                                              ↓
                                                     Tripletex API
```

## Quick Commands

```bash
ssh vps "journalctl -u tripletex -f"                          # Live logs
ssh vps "wc -l /opt/tripletex/task_log.jsonl"                  # Task count
ssh vps "tail -5 /opt/tripletex/task_log.jsonl" | python3 -m json.tool
ssh vps "systemctl restart tripletex"                          # Restart
rsync -avz ~/www/nm/tripletex/*.py vps:/opt/tripletex/ && ssh vps "systemctl restart tripletex"  # Deploy
```
