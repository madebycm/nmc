# Failure Analysis — 2026-03-21 ~21:03 Batch

## Scores: 5.5/8 and 7/10

---

## BATCH 1: Month-End Closing (5.5/8)

**Prompt**: Perform month-end closing for March 2026. Post accrual reversal (6850 NOK from 1700 to expense). Depreciation 228600/9yr straight-line to account 6010. Verify trial balance. Salary accrual (debit 5000, credit 2900).

### What succeeded:
- ✓ Accrual reversal 6850 NOK (1700 → expense)
- ✓ Salary accrual posted (5000 debit, 2900 credit)

### What failed:
1. **Depreciation SKIPPED entirely** — agent searched 12× by NAME for accumulated depreciation contra-account ("Akkumulerte avskrivninger", "Akk. avskrivning maskiner", "transportmidler") but never found it. Should have used number 1029 or similar standard contra-account.
   - Root cause: Agent searches accounts by NAME instead of NUMBER. The prompt gave account 6010 for depreciation expense but agent couldn't figure out the contra-account (balance sheet side).
   - Fix: Prompt pattern should tell agent standard depreciation pairs (6010↔1029, 6020↔1039, etc.) OR tell it to search by number range.

2. **Salary accrual amount = 30,000 NOK (assumed)** — prompt didn't specify amount. Agent guessed. May lose check if specific amount expected.

3. **Trial balance verification** — unclear if performed.

### Depreciation math:
- 228600 / 9 / 12 = **2116.67 NOK/month**
- Debit 6010 (depreciation expense), Credit 10xx (accumulated depreciation)

---

## BATCH 2: Travel Expense (7/10)

**Prompt**: Travel expense for Sofia Ferreira, "Conferência Drammen", 4 days, per diem 800 NOK/day. Expenses: flight 7050 NOK, taxi 450 NOK.

### What succeeded:
- ✓ Employee found
- ✓ Travel expense created
- ✓ Flight cost 7050 NOK posted
- ✓ Taxi cost 450 NOK posted

### What failed:
- **Per diem (ajudas de custo)** — 4 × 800 = 3200 NOK. Agent searched for per diem cost category ("Diett") but may not have posted it as a separate cost line. Only 2 cost items visible in API calls.
- Total should be 10700 NOK (3200 + 7050 + 450). Agent reported 10700 in summary but unclear if per diem was actually posted correctly.

---

## BATCH 3: Earlier failures (from 20:33 batch)

### Dimension linking (17 turns, 5 errors)
- Created dimension + values OK
- **Could not link dimension to voucher posting** — 422 errors on freeAccountingDimension fields via tripletex_api

### Bank recon CSV (t=1, 0 calls)
- **Total failure** — forced_completion after 1 turn, 0 API calls
- This was the pre-fix crash (mode="ANY" schema too complex)

---

## Root Causes & Fixes Needed

| # | Issue | Fix |
|---|-------|-----|
| 1 | Agent searches accounts by NAME not NUMBER | Add pattern: "When prompt gives account numbers, use search_ledger_account(number=X). Standard depreciation pairs: 6010↔1029, 6020↔1039, 6030↔1049" |
| 2 | Agent can't find contra-accounts for depreciation | Add depreciation pattern with standard Norwegian chart of accounts pairs |
| 3 | Per diem cost category lookup fails/skips | Verify travelExpense costCategory for per diem — may need explicit category ID guidance |
| 4 | Dimension→voucher linking still broken | freeAccountingDimension1/2/3 fields cause 422 via tripletex_api — need to debug exact format |
