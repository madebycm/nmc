# Tripletex AI Agent — Problem Analysis & Architecture Review

**Date**: 2026-03-21
**Context**: NM i AI 2026 Competition — Tripletex accounting agent task (33% of total score)
**Current score**: ~50% average across real submissions
**Target**: 95%+
**Time remaining**: ~20 hours

---

## 1. Current Architecture

```
HTTP POST /solve
    │
    ▼
FastAPI server (server.py, 48 lines)
    │  — Always returns {"status": "completed"}, even on failure
    │
    ▼
Gemini 2.5 Pro Agent Loop (agent.py, 755 lines)
    │  — Single monolithic system prompt (223 lines, 21 patterns)
    │  — Max 25 turns, freeform ReAct conversation
    │  — Runtime autofixes (_pre_validate) for known field issues
    │  — Structured response compaction per entity type
    │
    ├── Typed Tools (38 from OpenAPI spec + 3 meta tools)
    │       │
    │       ▼
    │   Tool Router (tool_router.py, 278 lines)
    │       │  — Maps tool names → HTTP method + endpoint
    │       │  — Canonicalizes flat refs → nested objects
    │       │
    │       ▼
    │   Schema Guard (schema_guard.py, 221 lines)
    │       │  — Strips unknown/read-only fields (TOP-LEVEL ONLY)
    │       │  — Does NOT validate nested arrays (postings, orderLines)
    │       │
    │       ▼
    │   Tripletex REST API
    │
    ├── Generic API Tool (tripletex_api) — fallback for uncovered endpoints
    │
    └── Spec Search Tool — searches OpenAPI spec for endpoint discovery
```

---

## 2. Observed Failure Modes (Real Competition Data)

### FM-1: Wrong Debit/Credit Direction on Correction Vouchers
- **Severity**: HIGH — cost us 25% on a ledger correction task
- **Example**: Duplicate voucher reversal posted +1650 to expense (making it worse) instead of -1650 (reversing it)
- **Root cause**: Model has no accounting-semantic guardrail. The prompt teaches patterns but the runtime doesn't verify reversal logic.
- **Category**: Domain validation gap

### FM-2: Flat vs Nested Field Format (accountId vs account:{id})
- **Severity**: HIGH — causes 422 error chains that waste 3-4 turns
- **Example**: Voucher posting with `accountId: 123` instead of `account: {id: 123}`, then `vatTypeId: 1` instead of `vatType: {id: 1}`
- **Root cause**: Prompt teaches camelCase flat fields for incomingInvoice pattern. Router only canonicalizes underscore-style refs (`account_id`). Schema guard only strips top-level fields — nested arrays (postings, orderLines) are not validated.
- **Category**: Schema contract mismatch

### FM-3: Turn Limit Exhaustion → 0% on Complex Tasks
- **Severity**: CRITICAL — complete loss on bank reconciliation (7+ operations)
- **Example**: 25 turns used, 5 payments + 2 vouchers created, but `task_complete` never called. Score: 0% despite 70% of work done.
- **Root cause**: No budget-aware finalization. No forced completion path. No partial-credit checkpoint. The agent treats 25 turns as infinite.
- **Category**: Orchestration failure

### FM-4: Ignores Prompt Patterns / Invents Manual Workarounds
- **Severity**: HIGH — reminder task used product/order/invoice chain instead of `createReminder_invoice`
- **Example**: Task asked for reminder charge → agent created a product, order, invoice, and payment instead of calling the reminder API
- **Root cause**: 223-line monolithic prompt with 21 patterns. Model ignores relevant pattern and freelances. No enforcement that specific task types must use specific tools.
- **Category**: Prompt overload / lack of workflow routing

### FM-5: Swapped Multi-Currency Payment Fields
- **Severity**: HIGH — caused massive overpayment (-130K outstanding)
- **Example**: `paidAmount` (should be NOK) got EUR value; `paidAmountCurrency` (should be EUR) got NOK value
- **Root cause**: Field semantics are confusing. Prompt pattern was added but model still confused the two. No deterministic validator checks the invariant (paidAmount > paidAmountCurrency for NOK→foreign).
- **Category**: Domain validation gap

### FM-6: No Validation Layer
- **Severity**: SYSTEMIC — server returns "completed" regardless of outcome
- **Root cause**: `task_complete` is advisory. No postcondition checks. No accounting invariant validation. No review gate.
- **Category**: Missing architecture layer

---

## 3. Expert Assessment (GPT-5.4 / Codex Review)

### Core Verdict
> "Your main bottleneck is not model intelligence. It is missing execution architecture around the model."

### Key Findings
1. **Prompt iteration is secondary** — failures are contract, control-flow, and validation failures, not "model needs better wording"
2. **Schema guard is shallow** — only strips top-level fields; nested arrays (postings, orderLines) pass through unvalidated
3. **Router naming mismatch** — prompt teaches `accountId` (camelCase), router canonicalizes `account_id` (underscore). Neither catches the other.
4. **No domain validators** — the system validates HTTP syntax but not accounting semantics (balanced postings, correct reversal signs, currency field consistency)
5. **Monolithic prompt is anti-pattern** — 21 patterns competing for attention. Industry standard: retrieve only the relevant workflow card per task.
6. **Freeform loop is wrong abstraction** — complex multi-step tasks need structured plans with per-step budgets, not an unguided 25-turn chat

### Industry Standard Architecture (for comparison)
```
1. Intent classification / workflow selection
2. Structured plan generation (steps, success criteria)
3. Deterministic tool execution with schema normalization
4. Local repair loops per step (bounded)
5. Postcondition validation before marking success
6. Durable state / progress tracking
7. Eval harness built from real failures
```

Our agent has: a monolithic prompt (#1-2 via prompt), tool execution (#3 partial), schema guard (#3 partial), no #4-7.

---

## 4. Proposed Fixes — Priority Order

### P0: Immediate (next 2 hours) — Highest ROI, minimal risk

| Fix | Impact | Effort | Risk |
|-----|--------|--------|------|
| **Increase MAX_TURNS to 40** | Prevents 0% on complex tasks | 1 line | None |
| **Force task_complete at turn limit** | Partial credit instead of 0% | 10 lines | None |
| **Deep schema normalization** — coerce flat fields in nested arrays (postings, orderLines) to nested format | Eliminates 422 chains | 30 lines in schema_guard | Low |
| **Few-shot examples** for top 5 failure patterns (correction vouchers, reminders, FX payments, supplier invoices, reconciliation) | Directly addresses FM-1, FM-4, FM-5 | Prompt additions | Low |

### P1: Short-term (next 4-6 hours) — Structural improvements

| Fix | Impact | Effort | Risk |
|-----|--------|--------|------|
| **Task-type router** — classify intent, inject only relevant workflow card + few-shot trace | Reduces FM-4 (pattern ignoring) | New module, ~100 lines | Medium |
| **Accounting validators** — verify postings balance, reversal sign correctness, currency field ordering | Catches FM-1, FM-5 before completion | ~80 lines | Medium |
| **Budget-aware finalization** — inject "you have N turns left, wrap up" at turn N-3 | Prevents FM-3 | 15 lines | Low |

### P2: Ideal architecture (if time permits)

| Fix | Impact | Effort | Risk |
|-----|--------|--------|------|
| **Structured plan artifact** — model emits step list before execution, loop tracks completion per step | Prevents FM-3, FM-4 | ~150 lines | Medium-High |
| **Completion gating** — `task_complete` only succeeds after validators pass | Prevents FM-6 | ~50 lines | Medium |
| **Review agent** — secondary LLM call checks work against task requirements | Catches fuzzy errors | ~100 lines | Medium (latency) |

---

## 5. Decision Points for Review

### Q1: Do we invest in architecture or keep iterating prompts?
- **Codex recommendation**: Architecture first, prompts second
- **Risk**: Architecture changes in competition time window could introduce regressions
- **Counter**: P0 fixes are low-risk and address the highest-impact failures

### Q2: Validation agent — yes or no?
- **Pro**: Catches errors the actor misses; industry standard is validator before completion
- **Con**: Adds latency (5-10s per task); another LLM call that could itself err
- **Recommendation**: Deterministic validators first (P1), LLM review agent only if time permits (P2)

### Q3: Task-type routing — worth the investment?
- **Pro**: Directly addresses FM-4 (monolithic prompt overload); industry standard
- **Con**: Requires classifying task types reliably; new failure mode if misclassified
- **Recommendation**: Worth it if time allows. Fallback: keep monolithic prompt but add few-shot examples (P0)

### Q4: What's the realistic ceiling with current architecture + P0 fixes only?
- **Estimate**: 70-80% (up from ~50%)
- **With P1**: 85-90%
- **With P2**: 90-95%

---

## 6. Competition Scoring Context

- Each submission sends 1-4 tasks, each with 4-8 checks
- Partial credit per check within a task
- We get unlimited submissions (but each is ONE SHOT per task instance)
- Time remaining: ~20 hours
- Other tasks (NorgesGruppen, Astar) also need attention

**The question is not "what's ideal" — it's "what gets us the most points in 20 hours."**

---

## Appendix: Real Competition Task Log

| # | Task | Lang | Turns | Errors | Score | Failure Mode |
|---|------|------|-------|--------|-------|-------------|
| 1 | Currency invoice + agio | Nynorsk | 8 | 0 | 50% | FM-5 (swapped currency fields) |
| 2 | Overdue invoice + reminder charge | Spanish | 8 | 0 | 50% | FM-4 (ignored reminder API) |
| 3 | Create project | Bokmål | 3 | 0 | 100% | — |
| 4 | Ledger correction (4 errors) | German | 11 | 0 | 75% | FM-1 (wrong reversal direction) |
| 5 | Supplier invoice registration | Spanish | 10 | 4 | ~50% | FM-2 (field format chain) |
| 6 | Bank reconciliation from CSV | Portuguese | 25 | 0 | 0% | FM-3 (turn limit, no completion) |
