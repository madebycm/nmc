# Tripletex Pivot Plan — Saturday 2026-03-21

**Directive**: 100% quality. Efficiency bonus matters.
**Deadline**: March 22 15:00 CET (~18 hours)
**Current score**: 7/15 checks on best submission

---

## Architecture: Hybrid Typed + Spec-Grounded Fallback

### Layer 1: Typed tools (38 existing + ~11 new)
Hot-path tasks with optimal efficiency (fewest API calls, zero errors).
These earn efficiency bonus multiplier on Tier 1-2 tasks.

### Layer 2: `search_tripletex_spec` tool
Gemini searches our extracted OpenAPI catalog before calling unknown endpoints.
Returns: method, path, summary, required params, body fields, enums.
Prevents hallucinated endpoints.

### Layer 3: `tripletex_api` tool with recursive spec validation
Generic executor for any endpoint found via spec search.
Validates: required fields, enums, nested arrays, readOnly stripping, ref nesting.
`dry_run=true` mode for Gemini to check before executing.

---

## Execution Order

### Phase 1: New typed tools for confirmed failures (~1h)
**Goal**: Fix the 8 known failed checks immediately.

#### 1a. Employment tools (7 tools)
```
search_employment        GET  /employee/employment           query
post_employment          POST /employee/employment           body
put_employment           PUT  /employee/employment/{id}      path_body
search_employment_details GET  /employee/employment/details   query
post_employment_details  POST /employee/employment/details   body
put_employment_details   PUT  /employee/employment/details/{id} path_body
search_occupationCode    GET  /employee/employment/occupationCode query
```

REF_FIELDS to add:
- `employment_id` → `employment`
- `occupationCode_id` → `occupationCode`
- `division_id` → `division`

SEARCH_FIELDS:
- search_employment: `id,employee(id,firstName,lastName),startDate,endDate,employmentId`
- search_employment_details: `id,employment(id),date,annualSalary,percentageOfFullTimeEquivalent,occupationCode(id,code,nameNO),employmentType,remunerationType,workingHoursScheme`
- search_occupationCode: `id,code,nameNO`

System prompt pattern 17:
```
17. CREATE EMPLOYEE WITH EMPLOYMENT CONTRACT:
    → search_department
    → post_employee(firstName, lastName, email, department_id, dateOfBirth="1990-01-01")
    → post_employment(employee_id, startDate, isMainEmployer=true)
    → search_occupationCode(code="XXXX" or nameNO="...")
    → post_employment_details(employment_id, date=startDate,
        occupationCode_id, annualSalary, percentageOfFullTimeEquivalent,
        employmentType="ORDINARY", remunerationType="MONTHLY_WAGE",
        workingHoursScheme="NOT_SHIFT")
```

#### 1b. Dimension tools (4 tools)
```
post_accountingDimensionName    POST /ledger/accountingDimensionName        body
search_accountingDimensionName  GET  /ledger/accountingDimensionName/search query
post_accountingDimensionValue   POST /ledger/accountingDimensionValue       body
search_accountingDimensionValue GET  /ledger/accountingDimensionValue/search query
```

SEARCH_FIELDS:
- search_accountingDimensionName: `id,dimensionName,description,dimensionIndex,active`
- search_accountingDimensionValue: `id,displayName,dimensionIndex,number,active`

System prompt pattern 18:
```
18. CREATE CUSTOM ACCOUNTING DIMENSION:
    → post_accountingDimensionName(dimensionName="...", description="...", active=true)
    → read dimensionIndex from response (auto-assigned, 1-3)
    → post_accountingDimensionValue(displayName="...", dimensionIndex=N, number="01", active=true)
    → repeat for each value
```

#### 1c. Reminder tool (1 tool) — CONFIRMED EXISTS
```
createReminder_invoice   PUT  /invoice/{id}/:createReminder  path_query
```
Parameters: `id` (invoice ID), `type` (SOFT_REMINDER|REMINDER|NOTICE_OF_DEBT_COLLECTION),
`date`, `dispatchType` (EMAIL|SMS|NETS_PRINT), `charge` (fee amount), `interestRate`

System prompt pattern 19:
```
19. OVERDUE INVOICE + REMINDER:
    → search_invoice(invoiceDateFrom="2020-01-01", invoiceDateTo=today)
    → find invoice where amountOutstanding > 0
    → createReminder_invoice(id=invoiceId, type="SOFT_REMINDER",
        date=today, dispatchType="EMAIL")
    NOTE: charge (purregebyr) and interestRate set via the reminder.
    For "late fee" tasks, the reminder itself carries the fee.
```

#### 1d. gen_tools.py update
Regenerate gen_tools_output.json including the 11 new tools.
Or manually add tool declarations (faster).

#### 1d. Deploy + test confirmed fixes
```bash
scp *.py vps:/opt/tripletex/ && ssh vps "systemctl restart tripletex"
```

---

### Phase 2: Spec-grounded fallback (~2-3h)
**Goal**: Zero dead ends for unknown task types.

#### 2a. Build spec catalog (`spec_catalog.py`)
Pre-process `api_spec_extract.json` into a compact searchable index:
```python
CATALOG = [
    {
        "method": "POST",
        "path": "/employee/employment",
        "summary": "Create employment.",
        "tags": ["employee"],
        "required_body": ["employee", "startDate"],
        "optional_body": ["employmentId", "endDate", "division", "isMainEmployer"],
        "enums": {"employmentEndReason": ["EMPLOYMENT_END_EXPIRED", ...]},
        "is_beta": False,
    },
    ...
]
```
Target: ~200-300 entries (writable + searchable endpoints).

#### 2b. `search_tripletex_spec` tool
Gemini tool declaration:
```
search_tripletex_spec(query: str, method_filter: str = None, limit: int = 10)
```
Returns top matching operations from catalog via keyword search on path + summary + tags.
Output: compact JSON array of candidates.

#### 2c. `tripletex_api` tool with recursive validation
```
tripletex_api(method: str, path: str, query_params: dict = None, body: dict = None)
```
Before executing:
1. Look up endpoint in catalog
2. Validate required fields present
3. Strip readOnly fields
4. Validate enum values
5. Unflatten nested refs (reuse existing _unflatten_refs logic)
6. Canonicalize arrays
7. Execute HTTP request
8. Return compacted response

#### 2d. System prompt additions
```
FALLBACK PATTERN:
If you don't have a specific tool for a task, use these two tools:
1. search_tripletex_spec(query="...") — find the right endpoint
2. tripletex_api(method, path, query_params, body) — execute it

Always search first, then execute. Never guess endpoint paths.
```

#### 2e. Deploy + submission grinding begins

---

### Phase 3: Recursive schema_guard (~1-2h)
**Goal**: Prevent nested field validation errors.

Current schema_guard only validates top-level fields.
Extend to:
- Validate nested objects (orderLines, postings, travelDetails)
- Validate enum values against spec
- Strip unknown nested fields
- Validate required nested fields

---

### Phase 4: Submission grinding + log-driven fixes (ongoing)
**Goal**: Discover remaining task types, fix from real data.

1. Submit continuously (3 concurrent)
2. Monitor task_log.jsonl for new task types
3. For each new failure:
   - Analyze prompt + errors
   - Add typed tool if it's a hot path
   - Otherwise verify fallback handles it
4. Redeploy, resubmit

Expected: see all 30 task types within ~50 submissions (~2 hours)

---

## Files to modify

| File | Changes |
|------|---------|
| `tool_router.py` | Add 11 new TOOL_MAP entries, SEARCH_FIELDS, REF_FIELDS |
| `gen_tools_output.json` | Add 11 new tool declarations (or regenerate) |
| `prompts.py` | Add patterns 17-18, fallback pattern |
| `agent.py` | Add search_tripletex_spec + tripletex_api handling in agent loop |
| `spec_catalog.py` | NEW — searchable endpoint catalog from api_spec_extract.json |
| `schema_guard.py` | Extend to recursive nested validation |
| `_pre_validate` in agent.py | Add autofixes for new tools if needed |

## Success criteria

- [ ] Employment tools working (employee contract tasks pass)
- [ ] Dimension tools working (custom dimension tasks pass)
- [ ] Spec search + generic API tool deployed
- [ ] No zero-tool dead ends for any task type
- [ ] Recursive schema validation prevents nested field errors
- [ ] Submission grinding started, logs being monitored
- [ ] All discovered task types handled or fallback-covered

## Score target

| Tier | Tasks | Target | Points |
|------|-------|--------|--------|
| Tier 1 | ~10 | 100% correctness + efficiency | ~10-20 |
| Tier 2 | ~10 | 100% correctness | ~20-40 |
| Tier 3 | ~10 | >50% correctness via fallback | ~15-30 |
| **Total** | **30** | | **45-90** |

Max theoretical: 30 × 6.0 = 180 points.
Realistic target: 60-90 points (top quartile).
