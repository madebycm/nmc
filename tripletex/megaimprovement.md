# Megaimprovement Plan — Final Hours

**Created: 2026-03-22 ~04:30 CET | Deadline: 15:00 CET (~10.5h)**
**Guiding principle: free-read preflight validators, one invoice-path bug fix, score-aligned instrumentation.**

---

## Critical Corrections to Our Mental Model

| Wrong assumption | Reality | Impact |
|---|---|---|
| "All API calls hurt efficiency" | **Only WRITE calls + 4xx on writes** count. GETs are free for scoring. | We've been optimizing the wrong proxy. Search cache was wasted effort. |
| "Pre-existing products in sandbox" | Each task starts from a **brand-new empty account**. Must create prerequisites. | System prompt says "pre-existing products" — actively misleading the LLM. |
| `sendToCustomer=True` default is safe | Clobbers explicit `False`. Tripletex default is already `True`. | Bug: we override valid "don't send" intent → wrong behavior on non-send tasks. |
| Hardcoded 7100/7350 VAT recovery | Tripletex exposes `legalVatTypes`, `vatLocked`, `ledgerType` on account metadata | Our fix is brittle folklore. Generic preflight kills the whole error class. |
| Dummy email is low-risk | Email is score-bearing in official examples | Placeholder email = partial credit loss, not free salvage. |
| 45s timer is safe | Official limit is 300s. Token expiry ≠ normal Tripletex behavior. | Fail-fast on first 403, but don't cut the clock artificially. |
| Task types are assigned uniformly | **Weighted toward under-attempted types** | Broad fixes > overfitting the 3 easiest families. |

---

## Priority 1: Invoice Path Bug Fix (HIGH EV, LOW RISK)

### 1A. Fix `sendToCustomer` clobbering
- **Where**: `_pre_validate`, `invoice_order` handler
- **Current**: `if not args.get("sendToCustomer"): args["sendToCustomer"] = True`
- **Problem**: Overwrites explicit `False`. Also, Tripletex already defaults to `True`.
- **Fix**: Remove the override entirely. Only set `sendToCustomer=True` when prompt explicitly mentions sending/emailing. Otherwise let Tripletex default handle it.

### 1B. Remove "pre-existing products" from system prompt
- **Where**: `prompts.py`, rule 3
- **Current**: "Account starts FRESH each time (1 default employee, 1 default department, pre-existing products)"
- **Problem**: LLM assumes products exist and tries to use them → 404 or wrong product.
- **Fix**: Change to: "Account starts FRESH each time. You may have a default employee and department, but DO NOT assume products, customers, suppliers, or other entities exist. Create everything you need."

### 1C. Gate bank-account setup to invoice/payment flows only
- **Where**: `_ensure_bank_account()` call in `solve_task_sync`
- **Current**: Runs on EVERY task (1-2 write calls wasted on non-invoice tasks)
- **Problem**: Not that GETs cost score (they don't), but: latency, failure surface, proxy weirdness risk, and the PUT is a write call that counts.
- **Fix**: Only call `_ensure_bank_account()` when prompt matches invoice/payment/order keywords. The GET to check is free; only the PUT if missing costs a write.

---

## Priority 2: Generic Voucher Preflight (HIGHEST EV PLAY)

### The Insight
Tripletex `GET /ledger/account` returns per-account metadata including:
- `legalVatTypes` — which VAT types are allowed
- `vatLocked` — whether the account is locked to a specific VAT code
- `ledgerType` — whether the account requires customer/supplier/employee ref
- `vatType` — the locked VAT code if applicable

### The Play
Before ANY `post_ledger_voucher` or `post_incomingInvoice` write:
1. Collect all distinct account numbers from postings
2. `GET /ledger/account?number=XXXX` for each unique account (GETs are FREE)
3. For each posting, validate:
   - Is the proposed `vatType` in `legalVatTypes`? If not, use the correct one.
   - Is the account `vatLocked`? If so, force the locked VAT code.
   - Does `ledgerType` require a customer/supplier/employee ref? If so, ensure it's present.
4. Only then POST the voucher with guaranteed-valid postings.

### What This Kills
- ALL "konto X er låst til mva-kode Y" 422 errors (our #1 write-side 4xx class)
- ALL "konto krever kunde/leverandør-referanse" 422 errors
- Replaces brittle hardcoded 7100/7350 recovery with a universal solution
- Zero scoring cost (all preflight is GETs)

### Implementation Shape
- New function: `_preflight_voucher_postings(postings: list) -> list`
- Called from `_pre_validate` when `tool_name == "post_ledger_voucher"`
- Returns corrected postings with valid vatTypes and required refs
- Log all corrections for debugging

---

## Priority 3: Score-Aligned Metrics

### Current Metrics (Wrong Proxies)
- `avg API calls/task` — includes GETs which don't count
- `tasks with errors` — includes GET errors which don't count
- `zero-error rate` — same problem

### New Metrics to Track
| Metric | What it measures | How to compute |
|---|---|---|
| `write_calls_per_task` | Efficiency denominator | Count only POST/PUT/DELETE calls |
| `write_4xx_rate` | Efficiency killer | 4xx responses on POST/PUT/DELETE only |
| `local_perfect_by_family` | Correctness coverage | Verifier pass rate grouped by task type |
| `median_writes_on_perfect` | Best-case efficiency | Median write calls where verifier passed |
| `fatal_blocker_rate` | Unrecoverable failures | Token expiry, crash, max turns exhausted |

### Implementation Shape
- Extend task log entry with: `write_calls`, `write_4xx_count`, `total_writes`, `verifier_result`
- Add to `_exec_api`: tag each call as read/write based on HTTP method
- Dashboard script or log parser that computes the 5 metrics above

---

## Priority 4: Remaining Fixes (Medium EV)

### 4A. Dummy email — last resort only
- **Current**: Agent sometimes creates employees without email → 422, or generates placeholder
- **Fix order**: (1) PDF text extraction, (2) regex email scan on extracted text, (3) prompt scan, (4) domain synthesis from company name in document, (5) ONLY THEN `firstname.lastname@company.no` placeholder
- **Log separately**: flag `email_source=placeholder` so we know when this fires

### 4B. Image attachment forwarding
- **Check**: Do any live tasks include JPG/PNG attachments (not just PDFs)?
- **If yes**: Forward image bytes directly to Gemini as inline_data parts (Gemini is multimodal)
- **Current gap**: We only extract text from PDFs. Receipt images would be completely missed.

### 4C. Fail-fast on proxy 403
- **Current**: Agent keeps trying after first 403, wasting all remaining turns
- **Fix**: On first 403 "Invalid or expired proxy token", immediately call `task_complete` with best partial result
- **Don't**: Set artificial 45s timer. Official limit is 300s.

### 4D. Remove auto-balance voucher logic (if any)
- **CTO suggested**: Auto-balance by adjusting largest posting
- **Reviewer HARD REJECT**: This creates field-level wrong accounting
- **Policy**: Only auto-balance when delta is a deterministic rounding artifact from our own VAT/FX math. Otherwise fail fast or recalculate from source values.

---

## Priority 5: Prompt Cleanup (Low Risk, Moderate EV)

### Changes to system prompt
1. Remove "pre-existing products" claim (P1B above)
2. Add: "Before creating a voucher, the system will validate your account/VAT choices automatically. Focus on getting the right accounts and amounts."
3. Add: "GETs are free — search liberally to find existing entities and verify account properties."
4. Remove any instruction that discourages searching (we were optimizing for fewer total calls, but GETs don't count)
5. Add for invoice tasks: "Only set sendToCustomer=true if the task explicitly says to send/email the invoice."

---

## What We Are NOT Doing

| Rejected idea | Why |
|---|---|
| Switch LLM | Calibration risk with 200 submissions left. Failures are domain-constraint, not model-intelligence. |
| Broad tool expansion | 50 typed tools + generic fallback covers our needs. Adding tools adds failure surface. |
| Deterministic regex pipeline | 30 types × 7 languages × 56 variants = too brittle for finals. |
| Auto-balance voucher postings | Field-level wrong accounting = correctness killer. |
| Regenerate full OpenAPI stack | Destabilization risk. Targeted diff of used endpoints only if time permits. |
| 45-second timer | Official limit is 300s. Fail-fast on 403 is the right move, not clock-cutting. |

---

## Execution Order

```
Phase 1 (SHIP FIRST — ~2h)
├── 1A. Fix sendToCustomer clobbering
├── 1B. Remove "pre-existing products" from prompt
├── 1C. Gate bank-account to invoice flows only
└── Deploy + run test suite

Phase 2 (HIGHEST EV — ~3h)
├── 2. Generic voucher preflight (_preflight_voucher_postings)
├── Remove hardcoded 7100/7350 recovery (replaced by generic)
└── Deploy + run test suite

Phase 3 (INSTRUMENTATION — ~1h)
├── 3. Score-aligned metrics in task log
├── Write/read classification in _exec_api
└── Deploy

Phase 4 (REMAINING FIXES — ~2h)
├── 4A. Email extraction cascade (last-resort placeholder)
├── 4B. Image attachment forwarding check
├── 4C. Fail-fast on proxy 403
└── Deploy + monitor

Phase 5 (PROMPT POLISH — ~30m)
├── 5. System prompt cleanup
└── Final deploy

Buffer: ~2h for monitoring, hotfixes, and unexpected issues
```

---

## Success Criteria

After all phases:
- `write_4xx_rate` < 5% (currently ~25% including GETs)
- `local_perfect_by_family` coverage ≥ 25/30 task types
- `median_writes_on_perfect` ≤ 3 for simple tasks, ≤ 6 for complex
- Zero hardcoded account-number recovery patterns (all replaced by generic preflight)
- System prompt contains no false assumptions about sandbox state
