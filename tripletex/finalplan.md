# Tripletex Agent — Final Plan

## Goal
100% success rate on all 205 captured competition prompts, then resubmit live.

## Assets
- **205 real prompts** dumped from competition platform (`logs/prompts_dump/`)
- **23 task types** across 7 languages, 31 with file attachments
- **33 prior prod runs**: 32 completed, 1 hit turn limit, 4 had mid-run API errors
- **Gemini agent** (`agent.py`) — original implementation, proven in prod
- **Test infrastructure** (`_archive/test_harness.py`, `_archive/test_e2e.py`) with verifiers

## Scoring (from competition docs)
- **Field-by-field verification**: each task has specific checks worth different points
- **Correctness** = points_earned / max_points (0–1)
- **Tier multiplier**: T1 ×1, T2 ×2, T3 ×3
- **Efficiency bonus** (only if correctness=1.0): up to 2× tier score
  - Based on: API call count vs best-known + zero 4xx errors
  - Perfect T3 task: up to **6.0 points**
- **Best score per task kept** — bad runs never hurt
- **30 tasks total**, 56 variants each (7 languages × 8 data sets)

## Architecture Principles

### Building an Effective Agent
1. **Parse the prompt** — extract task type, entity names, field values, relationships
2. **Handle files** — decode base64 PDFs/images, extract relevant data
3. **Map to API calls** — determine endpoints + order, create prerequisites first
4. **Verify work** — query back after creating entities to confirm correct values
5. **Handle errors** — parse Tripletex error messages, retry with corrections

### Optimizing for Efficiency
1. **Plan before calling** — parse prompt fully before any API call
2. **Avoid trial-and-error** — every 4xx reduces efficiency bonus. Validate first
3. **Minimize GET calls** — don't fetch what you already know from POST responses
4. **Batch where possible** — use list endpoints instead of individual calls
5. **Read error messages** — fix in one retry, not several

## Strategy: Option B — Batch by Task Type

### Phase 1: Local Smoke Test
- Run `test_runner.py` against our sandbox with a handful of prompts
- Verify the test infrastructure works end-to-end
- Fix any import/credential/connectivity issues

### Phase 2: Full Sweep by Task Type
- Group 205 prompts into 23 task types
- Run representative prompts from each type sequentially
- Real-time logging: prompt, API calls, errors, outcome → `logs/test_results.jsonl`
- Identify failing task types

### Phase 3: Fix Failures
- For each failing task type, analyze the error pattern
- Fix in agent.py / prompts.py / tool_router.py / schema_guard.py
- Re-run that task type until green

### Phase 4: Full Run
- Run all 205 prompts, verify 100% completion
- Measure: avg turns, avg API calls, error rate per type

### Phase 5: Live Resubmit
- Deploy updated agent to vps (nm.j6x.com)
- Switch endpoint back to `/solve`
- Monitor leaderboard

## Task Type Distribution (205 prompts)

| Type | Count | Files | Tier |
|------|-------|-------|------|
| project_billing | 26 | 2 | T2-T3 |
| create_invoice | 25 | 1 | T2 |
| supplier_invoice | 15 | 4 | T2 |
| create_employee | 14 | 6 | T1 |
| payroll | 13 | 8 | T3 |
| create_customer | 10 | - | T1 |
| cost_analysis | 9 | - | T3 |
| reminder_fee | 9 | - | T2 |
| create_product | 8 | - | T1 |
| monthly_closing | 8 | - | T3 |
| register_payment | 8 | - | T2 |
| year_end_closing | 8 | - | T3 |
| travel_expense | 7 | - | T2 |
| create_project | 6 | - | T1 |
| custom_dimension | 6 | - | T2 |
| ledger_correction | 6 | - | T3 |
| bank_reconciliation | 6 | 6 | T3 |
| credit_note | 4 | - | T2 |
| reverse_payment | 4 | - | T2 |
| create_department | 4 | - | T1 |
| expense_receipt | 4 | 4 | T2 |
| multi_currency_payment | 3 | - | T2 |
| create_order | 2 | - | T2 |

## Key Risk: Sandbox State
Each competition task gets a **fresh sandbox**. Our local sandbox accumulates state.
Mitigation: unique names in prompts should avoid collisions. Monitor for conflicts.

## Files
- `test_runner.py` — local test runner, replays prompts against sandbox
- `logs/test_results.jsonl` — real-time structured results log
- `finalplan.md` — this file
