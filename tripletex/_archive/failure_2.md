# Failure Analysis 2 — Competition Submissions (2026-03-21)

## SUBMISSION 3: 100% (22/22) ✓
**Time**: ~15:30
**Fixes applied**: removed invalid `dueDate` from invoice SEARCH_FIELDS, improved reminder pattern 19, added combo pattern 20

## SUBMISSION 2: 77% (17/22)
**Time**: ~15:00
**Previous**: 7/15 (47%) → **+30pp improvement**

---

## Summary of 5 Failed Checks

The 5 failures fall into 3 root causes. Cross-referencing task_log.jsonl entries against competition timing:

---

## FAILURE A: `createReminder_invoice` — Date Validation (HIGH IMPACT)

**Entries**: 81, 161 (both FR: "Trouvez la facture impayée...")

**Symptom**: `PUT /invoice/{id}/:createReminder` → 422 validation on `date` field

**What happens**:
1. Agent searches for unpaid invoices — finds none (in sandbox, all paid off by prior tests)
2. Agent creates a NEW customer + order + invoice TODAY
3. Tries `createReminder_invoice(date=today)` on the just-created invoice
4. **REJECTED**: Tripletex requires reminder date to be STRICTLY AFTER the invoice due date
5. Agent tries backdating invoice to 2026-02-21, creates reminder with date=2026-03-21 → sometimes works
6. Takes 10-16 turns, huge efficiency penalty

**In Competition**: Task gives a sandbox with a PRE-EXISTING overdue invoice. Agent just needs to find it and send reminder. But when `search_invoice` returns results with `amountOutstanding > 0`, the agent should use it directly — NOT create a new invoice.

**Root Cause**: System prompt pattern 19 says "find invoice where amountOutstanding > 0" but the agent:
1. Fails to find existing invoices (possible search field issue)
2. Falls back to creating a new one (unnecessary)
3. Then can't send reminder on same-day invoice

**Fix Required**:
- Pattern 19 must emphasize: NEVER create a new invoice for reminder tasks
- Search fields for invoice need to include `dueDate` so agent can identify overdue ones
- If the sandbox has no overdue invoices, agent should report this, not create artificial ones
- Pre-validate reminder date: must be >= invoice due date (or today if invoice is overdue)

---

## FAILURE B: Combo Task — Dimension + Voucher (MEDIUM IMPACT)

**Entry**: 36 (ES: "Cree una dimensión contable...Luego registre un asiento en la cuenta 7000 por 14550 NOK, vinculado al valor de dimensión 'Forskning'")

**Symptom**: `no_completion_signal` after only 4 turns, zero API calls

**What happens**: Task has TWO parts:
1. Create dimension "Prosjekttype" with values "Forskning" and "Utvikling"
2. Create a voucher on account 7000 for 14550 NOK linked to dimension value "Forskning"

Agent ran 4 turns but made zero API calls — likely Gemini produced only text responses and hit the turn limit.

**Root Cause**: Combo tasks combining two different task types confuse the model. It doesn't know how to:
- Create the dimension + values first (pattern 18)
- Then link a voucher posting to a dimension value

**Fix Required**:
- Add system prompt pattern for "dimension + voucher" combo
- Add pattern for linking dimension values to voucher postings (the posting needs a dimension value reference)
- Increase nudge/force on Gemini when it gives text-only responses (existing rule 10)

---

## FAILURE C: Supplier Invoice Voucher Posting Validation (MEDIUM IMPACT)

**Entry**: 7 (EN: "We have received invoice INV-2026-9075 from the supplier Brightstone Ltd...for 59800 NOK including VAT")

**Symptom**: `post_ledger_voucher` → 422 on `postings` field

**What happens**:
1. Agent creates customer (not supplier!) with `isSupplier=true`
2. Searches for accounts 6300, 1401, 2400
3. Creates voucher with postings — fails with posting validation
4. Retries — fails again

**Root Cause**: Two issues:
1. Agent uses `post_customer` instead of `post_supplier` for suppliers
2. Voucher postings have validation errors (likely missing `supplier_id` on the 2400 posting, or wrong field format)

**Details**: Entry 7 was from early testing (01:59 timestamp) and shows an old failure pattern. Later entries (15, 24, 32, 41) show supplier invoices completing successfully. **This may already be fixed** — the system prompt pattern 14 was updated to explicitly mention supplier_id on the 2400 posting.

**Verify**: Check if competition task 5 (supplier invoice type) is passing now.

---

## EFFICIENCY PENALTIES (Secondary Impact)

Several "completed" tasks had excessive turns due to recoverable errors:

| Pattern | Turns | Error | Impact |
|---------|-------|-------|--------|
| `post_employee` 422 duplicate | 3-5 | Employee already exists → retry with unique email | -1 turn per employee task |
| `post_product` 422 duplicate | 3 | Product number exists → search instead | -1 turn per product |
| `post_order` product_id mapping | 3 | Gemini sends `product_id` in orderLines → 3 retries | -3 turns on order tasks |
| `post_employee_employment` 422 | 2-3 | Employment already exists → retry | -2 turns |
| `post_ledger_accountingDimensionName` 422 | 2-3 | Dimension name already exists (max 3) | -2 turns |

In competition (fresh sandbox), many of these won't occur. But the `product_id` mapping issue costs 3 turns even on fresh sandboxes if Gemini sends the flat format.

**product_id in orderLines**: Our `_canonicalize_nested_item` handles this correctly, but early log entries (3, 5) show it wasn't working. Need to verify the deployed version handles it. Later entries (22, 25, 33) show orders completing without product_id errors — canonicalization is working on deployed code.

---

## COMPETITION TASK TYPES SEEN (from real submission)

Based on log entries with proxy token errors + known competition prompts:

| Type | Languages Seen | Status |
|------|---------------|--------|
| Department | nb | PASS |
| Customer | nb, nn, es, de, pt, fr | PASS |
| Employee | nb, en, fr, es | PASS |
| Product | nb | PASS |
| Order + Invoice | nb, de | PASS |
| Invoice + Payment | nb | PASS |
| Supplier Invoice | nb, en, de, es | PASS |
| Travel Expense | nb, pt | PASS (if costCategory fixed) |
| Project | nb, fr | PASS |
| Voucher | nb, de | PASS |
| Credit Note | nb, en | PASS |
| Employee Admin | nb | PASS |
| Employee Contract | nb, de, en | PASS (with retries) |
| Custom Dimension | nb, es, en | PASS |
| Reminder | nb, fr | **FLAKY** — fails on sandbox, may work on competition |
| Dimension + Voucher combo | es | **FAIL** — no_completion_signal |
| Supplier + Invoice combo | nb | PASS (after fixes) |

---

## PRIORITY FIXES

### P0: Reminder task robustness (Failure A)
1. Update pattern 19: search for EXISTING overdue invoices first, NEVER create new ones
2. Add `dueDate` to invoice SEARCH_FIELDS
3. Reminder date logic: use today if invoice is overdue, else dueDate + 1
4. Pre-validate: if searching for unpaid invoices and finding none, call task_complete with "no overdue invoices found"

### P1: Combo task support (Failure B)
1. Add system prompt pattern 20: "DIMENSION + VOUCHER LINKED TO DIMENSION"
2. Show how to get dimensionIndex from created dimension, then reference it in voucher posting
3. May need a new field in postings for dimension values

### P2: Efficiency — reduce wasted turns
1. `_pre_validate` for `post_employee`: auto-add `dateOfBirth` if missing
2. `_pre_validate` for `post_order`: verify product_id canonicalization working
3. Verify agent recovers from duplicate-entity 422s quickly

---

## SCORE PROJECTION

| Fix | Checks Recovered | Confidence |
|-----|-----------------|------------|
| Reminder P0 | +2 | 70% (depends on competition having overdue invoices) |
| Combo tasks P1 | +1-2 | 60% (new pattern, untested in competition) |
| Already-fixed (supplier) | +1 | 90% (later test entries show it works) |
| **Total possible** | **+3-5** | → **20-22/22 (91-100%)** |
