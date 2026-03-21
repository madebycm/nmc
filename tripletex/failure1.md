# Failure Analysis — Tripletex Agent (2026-03-21)

**Submission score**: 7/15 checks passed, 8/15 failed
**Previous submission**: 0/24 (3 submissions)

---

## Official API Documentation

- **Swagger UI**: https://tripletex.no/v2-docs/
- **Developer Portal**: https://developer.tripletex.no/
- **Test environment**: https://api-test.tripletex.tech/v2-docs/
- **Our spec source**: `kkpqfuj-amager.tripletex.dev/v2/openapi.json` (extracted 2026-03-21)
- **Coverage**: 379/546 endpoints extracted, 1035/2167 schemas — **our gen_tools.py only exposes 38 tools**

---

## Task Breakdown (Latest Submission)

| Check | Task | Status | Root Cause |
|-------|------|--------|------------|
| 1-7 | Tasks 1-3 (unknown types) | PASSED | — |
| 8-15 | Tasks 4-6 (3 tasks, ~8 checks) | FAILED | See below |

---

## FAILURE 1: Custom Accounting Dimensions (Task 4 — Spanish)

**Prompt**: "Cree una dimensión contable personalizada 'Prosjekttype' con los valores 'Intern' y 'Ekstern'"

**Result**: `no_completion_signal`, 0 API calls. Gemini gave up entirely — no tools available.

### Root Cause
We have **zero tools** for custom dimensions. The API supports them fully:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ledger/accountingDimensionName` | POST | Create dimension (name, description) |
| `/ledger/accountingDimensionName` | GET | Search/list dimensions |
| `/ledger/accountingDimensionName/{id}` | PUT | Update dimension |
| `/ledger/accountingDimensionValue` | POST | Create dimension value (displayName, dimensionIndex, number) |
| `/ledger/accountingDimensionValue/search` | GET | Search dimension values |

**Schema — AccountingDimensionName**:
- `dimensionName` (string) — name of the dimension
- `description` (string) — description
- `dimensionIndex` (int, readOnly) — assigned automatically (1, 2, or 3; max 3 free dimensions)
- `active` (boolean)

**Schema — AccountingDimensionValue**:
- `displayName` (string) — value name
- `dimensionIndex` (int) — which dimension this value belongs to
- `number` (string) — alphanumeric identifier
- `active` (boolean)
- `showInVoucherRegistration` (boolean)
- `position` (int) — sort order

### Fix Required
1. Add `post_accountingDimensionName` tool (body: `dimensionName`, `description`, `active`)
2. Add `search_accountingDimensionName` tool (GET, query params)
3. Add `post_accountingDimensionValue` tool (body: `displayName`, `dimensionIndex`, `number`, `active`)
4. Add `search_accountingDimensionValue` tool (GET `/ledger/accountingDimensionValue/search`, query params)
5. Add TOOL_MAP entries + SEARCH_FIELDS
6. Add pattern 17 to system prompt: dimension creation flow
7. Gemini needs to know: create dimension first → get `dimensionIndex` from response → create values with that index

---

## FAILURE 2: Token Expiry / Sequential Blocking (Task 5 — French)

**Prompt**: "L'un de vos clients a une facture en retard. Trouvez la facture impayée, ajoutez des frais de retard, et envoyez un rappel."

**Result**: All API calls returned 403 Forbidden.

### Root Cause
Before the ThreadPoolExecutor fix, tasks were processed **sequentially** in the async event loop. Each task's proxy token expires within ~60 seconds. Tasks queued behind a slow task (Gemini takes 20-40s per turn) had their tokens expire before execution began.

### Status: PARTIALLY FIXED
ThreadPoolExecutor(max_workers=8) now processes tasks concurrently. But this task also has a **second failure mode**:

**Complex multi-step task we don't fully support:**
1. Search for overdue/unpaid invoices → we can do this (`search_invoice`)
2. Add late fee (purregebyr) → **unclear how** — may need a new invoice line, a voucher, or a dedicated endpoint
3. Send reminder → **no tool for this** — `/reminder` endpoint not in our tool set

### API endpoints we're missing for reminders:
- `/reminder` — GET/POST for payment reminders
- `/invoice/:sendPaymentReminder` — if it exists
- Need to investigate the full `/reminder` endpoint in the API spec

---

## FAILURE 3: Employee Employment Details (Task 6 — German)

**Prompt**: "Sie haben einen Arbeitsvertrag erhalten (siehe beigefügte PDF). Erstellen Sie den Mitarbeiter mit allen Details aus dem Vertrag."

The PDF contained an employment contract with fields like:
- Employee name, email, department
- **Berufsschlüssel** (occupation code)
- **Gehalt** (annual salary)
- **Beschäftigungsprozentsatz** (employment percentage)
- **Startdatum** (start date)
- **Arbeitszeitmodell** (working hours scheme)

**Result**: Employee created (7 checks passed) but 8 checks failed — all the employment-specific fields.

### Root Cause
`post_employee` only creates the base employee record. Employment details require **two separate API calls** on different endpoints:

**Step 1: POST /employee/employment** — creates the employment relationship
```
Required fields:
- employee: {id: N}          ← ref to created employee
- startDate: "YYYY-MM-DD"    ← from contract
- employmentId: "string"     ← optional, external ID
- division: {id: N}          ← optional
- isMainEmployer: true       ← default true
```

**Step 2: POST /employee/employment/details** — creates salary/occupation details
```
Required fields:
- employment: {id: N}                    ← ref to employment from step 1
- date: "YYYY-MM-DD"                     ← effective date (= startDate)
- employmentType: "ORDINARY"             ← enum
- remunerationType: "MONTHLY_WAGE"       ← enum
- workingHoursScheme: "NOT_SHIFT"        ← enum
- occupationCode: {id: N}               ← need to search first
- percentageOfFullTimeEquivalent: 100.0  ← from contract
- annualSalary: 650000                   ← from contract
```

**Step 0 (prerequisite): GET /employee/employment/occupationCode** — find occupation code ID
```
Query params: code="string" or nameNO="string"
Returns: id, code, nameNO
Example: code="2310" → "Lektor og universitetslektor"
```

### Missing Tools (must add)
| Tool Name | Method | Endpoint | Param Type |
|-----------|--------|----------|------------|
| `search_employment` | GET | `/employee/employment` | query |
| `post_employment` | POST | `/employee/employment` | body |
| `put_employment` | PUT | `/employee/employment/{id}` | path_body |
| `search_employment_details` | GET | `/employee/employment/details` | query |
| `post_employment_details` | POST | `/employee/employment/details` | body |
| `put_employment_details` | PUT | `/employee/employment/details/{id}` | path_body |
| `search_occupationCode` | GET | `/employee/employment/occupationCode` | query |

### System Prompt Addition Needed
```
Pattern 17: CREATE EMPLOYEE WITH EMPLOYMENT CONTRACT
→ search_department
→ post_employee(firstName, lastName, email, department_id, dateOfBirth)
→ post_employment(employee_id, startDate, isMainEmployer=true)
→ search_occupationCode(code="XXXX" or nameNO="...")
→ post_employment_details(employment_id, date=startDate,
    occupationCode_id, annualSalary, percentageOfFullTimeEquivalent=100,
    employmentType="ORDINARY", remunerationType="MONTHLY_WAGE",
    workingHoursScheme="NOT_SHIFT")
```

---

## EARLIER FAILURES (First 3 Submissions — 0/24)

These were from pre-fix submissions. Root causes already addressed:

| Issue | Fix Applied |
|-------|-------------|
| Voucher `row=0` → "systemgenererte" error | `_pre_validate` autofix: row ≥ 1 |
| Missing `amountGrossCurrency` on postings | `_pre_validate` autofix |
| Wrong VAT types (5 vs 1) | Corrected in prompts.py |
| No bank account → invoice 422 | `_ensure_bank_account()` on startup |
| `invoice_order` response typed as "order" | Fixed `_guess_entity_type` |
| Travel expense `amountCurrencyIncVat` missing | `_pre_validate` autofix |
| Sequential HTTP blocking | ThreadPoolExecutor(max_workers=8) |
| Gemini text-only responses | 5-turn nudge + rules 10-11 |

---

## SYSTEMIC GAPS

### 1. Tool Coverage: 38/546 endpoints (7%)
We only expose 38 tools. The API has 546 endpoints. Key missing categories:

| Category | Endpoints | Impact |
|----------|-----------|--------|
| Employment | 7 endpoints | **HIGH** — employee contracts fail |
| Accounting Dimensions | 5 endpoints | **HIGH** — custom dimensions fail |
| Reminder/Dunning | ~3 endpoints | **MEDIUM** — overdue invoice flows |
| Bank/Payment | ~10 endpoints | **LOW** — basic payments work |
| Salary/Payslip | ~15 endpoints | **LOW** — unlikely in competition |
| Address | ~3 endpoints | **LOW** — inline on customer/employee |

### 2. File/PDF Processing
Task 6 included a PDF attachment. Our agent passes it as `inlineData` to Gemini, which can read PDFs/images. This part works. The failure is in missing tools, not PDF handling.

### 3. Language Coverage
All 7 languages (NO bokmål, NO nynorsk, EN, ES, PT, DE, FR) appear in competition tasks. Our language key in the system prompt covers basic terms but may miss domain-specific terms in less common languages.

### 4. Error Recovery Limitations
- No retry with modified params on 422 (only consecutive-same-error detection)
- No fallback strategies per task type
- No tool suggestion system ("you tried X which failed, try Y instead")

---

## TEST HARNESS PLAN

### Existing test_harness.py
We have a basic test harness but it only covers the 9 original task types. Must expand to cover ALL failure modes.

### New Test Cases to Add

```python
TEST_CASES = [
    # === PASSING (regression guard) ===
    {"id": "T01", "type": "department", "lang": "nb",
     "prompt": "Opprett en avdeling som heter 'Logistikk'"},
    {"id": "T02", "type": "employee", "lang": "nb",
     "prompt": "Opprett ansatt Kari Nordmann (kari@test.no) i avdeling 1"},
    {"id": "T03", "type": "customer", "lang": "nb",
     "prompt": "Opprett kunden Bølgekraft AS med org.nr 812297848, e-post post@test.no"},
    {"id": "T04", "type": "order_invoice_payment", "lang": "nb",
     "prompt": "Opprett en ordre med produkt 1874 (2 stk) til kunde 1, fakturer og registrer full betaling"},
    {"id": "T05", "type": "travel_expense", "lang": "nb",
     "prompt": "Registrer en reiseregning for ansatt 1 'Konferanse Oslo' fra 2026-03-10 til 2026-03-12, med kostnad 'Fly' 3500 kr"},
    {"id": "T06", "type": "supplier_invoice", "lang": "nb",
     "prompt": "Registrer en leverandørfaktura fra Kontorservice AS (org.nr 999888777) på 12500 kr inkl. MVA for kontorrekvisita"},
    {"id": "T07", "type": "voucher", "lang": "nb",
     "prompt": "Opprett et bilag: debet konto 6300 10000 kr, kredit konto 1920 10000 kr, beskrivelse 'Husleie mars'"},
    {"id": "T08", "type": "project", "lang": "nb",
     "prompt": "Opprett prosjekt 'Nettbutikk' med ansatt 1 som prosjektleder og kunde 1"},
    {"id": "T09", "type": "credit_note", "lang": "nb",
     "prompt": "Opprett en kreditnota for faktura 1"},

    # === NEW: Failed task types ===

    # Custom dimensions (Task 4 failure)
    {"id": "T10", "type": "custom_dimension", "lang": "es",
     "prompt": "Cree una dimensión contable personalizada 'Prosjekttype' con los valores 'Intern' y 'Ekstern'",
     "checks": ["dimension_created", "value_intern_created", "value_ekstern_created"]},
    {"id": "T11", "type": "custom_dimension", "lang": "nb",
     "prompt": "Opprett en egendefinert regnskapsdimensjon 'Region' med verdiene 'Nord', 'Sør', 'Øst' og 'Vest'",
     "checks": ["dimension_created", "4_values_created"]},

    # Employee with employment contract (Task 6 failure)
    {"id": "T12", "type": "employee_contract", "lang": "de",
     "prompt": "Erstellen Sie einen Mitarbeiter: Max Müller, max@test.de, Abteilung 1. Startdatum 2026-04-01, Gehalt 600000 NOK, Beschäftigungsprozentsatz 100%, Berufsschlüssel 2310",
     "checks": ["employee_created", "employment_created", "salary_set", "occupation_code_set", "percentage_set"]},
    {"id": "T13", "type": "employee_contract", "lang": "nb",
     "prompt": "Opprett ansatt Ola Hansen (ola@test.no). Startdato 2026-04-01, årslønn 550000, stillingsprosent 80%, yrkeskode 3112",
     "checks": ["employee_created", "employment_created", "salary_set", "percentage_80"]},

    # Overdue invoice + reminder (Task 5 failure)
    {"id": "T14", "type": "overdue_reminder", "lang": "fr",
     "prompt": "Trouvez la facture impayée, ajoutez des frais de retard de 100 NOK, et envoyez un rappel",
     "checks": ["invoice_found", "late_fee_added", "reminder_sent"]},
    {"id": "T15", "type": "overdue_reminder", "lang": "nb",
     "prompt": "Finn den ubetalte fakturaen, legg til purregebyr på 50 kr, og send purring",
     "checks": ["invoice_found", "late_fee_added", "reminder_sent"]},

    # Multi-language regression
    {"id": "T16", "type": "customer", "lang": "pt",
     "prompt": "Crie o cliente Oceano Azul Ltda com e-mail oceano@test.pt e telefone 912345678"},
    {"id": "T17", "type": "order_invoice", "lang": "de",
     "prompt": "Erstellen Sie eine Rechnung für Kunde 1 über Produkt 1874 (3 Stück) und senden Sie sie"},
    {"id": "T18", "type": "employee_admin", "lang": "en",
     "prompt": "Create employee John Smith (john@test.com) in department 1 as administrator with all privileges"},

    # Edge cases
    {"id": "T19", "type": "multi_product_order", "lang": "nb",
     "prompt": "Opprett en ordre med produkt 1874 (2 stk à 500 kr) og produkt 1875 (1 stk à 1200 kr) til kunde 1, fakturer og send"},
    {"id": "T20", "type": "supplier_multi_vat", "lang": "nb",
     "prompt": "Registrer leverandørfaktura fra Mat AS (org.nr 111222333) med 2 linjer: mat 5000 kr inkl 15% MVA, kontorutstyr 3000 kr inkl 25% MVA"},
]
```

### Validation Strategy
For each test case:
1. Run against live sandbox
2. After completion, query all relevant entities via GET
3. Verify each `check` field matches expected state
4. Log: turns used, errors encountered, total time, fields present/missing

### Priority Order for Fixes
1. **Employment tools** (7 new tools) — directly fixes Task 6 (8 checks)
2. **Dimension tools** (4 new tools) — directly fixes Task 4
3. **Reminder/dunning tools** — fixes Task 5 (investigate `/reminder` endpoint first)
4. **System prompt patterns** — add patterns 17-19 for new task types
5. **gen_tools.py update** — regenerate tool definitions including new endpoints

---

## ESTIMATED SCORE IMPROVEMENT

| Fix | Checks Recovered | New Total |
|-----|-------------------|-----------|
| Employment tools | +8 (Task 6) | 15/15 possible |
| Dimension tools | +? (Task 4, unknown check count) | — |
| Reminder tools | +? (Task 5, unknown check count) | — |
| **Conservative estimate** | **+8** | **15/15** |

---

## TIMELINE RISK

**Deadline**: March 22 15:00 CET (~20 hours remaining)

Each new tool requires:
1. Add to TOOL_MAP in tool_router.py (~2 min)
2. Generate tool definition via gen_tools.py or manually (~5 min)
3. Add REF_FIELDS if needed (~1 min)
4. Add SEARCH_FIELDS (~1 min)
5. Update system prompt pattern (~3 min)
6. Test against sandbox (~5 min)

**Estimated total**: ~2 hours for all fixes + testing
