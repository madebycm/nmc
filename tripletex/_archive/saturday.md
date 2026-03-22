# Saturday — Tripletex Agent (2026-03-21)

**Deadline**: March 22 15:00 CET
**Previous score**: 0/24 across 3 submissions
**Current status**: All 4 units shipped. 9/9 task types passing. 0 errors.

---

## Completed

### Unit 1: agent.py — Structured responses + runtime autofixes ✅
- Compact JSON responses per entity type (≤3000 chars vs 15K raw)
- `_pre_validate()` autofixes:
  - `post_order`: auto-fill `deliveryDate`
  - `search_product`: reject comma-separated productNumber
  - `post_ledger_voucher`: auto-fill `row ≥ 1`, `amountGrossCurrency`, validate balance, fill posting dates
  - `post_travelExpense_cost`: auto-fill `amountCurrencyIncVat = rate * count`
  - `post_employee`: default `userType = "STANDARD"`
  - `search_invoice`: bump `invoiceDateTo += 1` when equal to `invoiceDateFrom` (exclusive end)
  - `invoice_order`: default `sendType = "EMAIL"`
- Per-tool `SEARCH_FIELDS` (no more `fields="*"`)
- Error loop detection: bail after 3 identical errors, inject guidance
- 401 auth failure: immediate bail

### Unit 2: tool_router.py — Query+body routing ✅
- `body_query` param type: sendToLedger → query, rest → body
- `_canonicalize_nested_item()`: accepts both `product_id: N` and `product: {id: N}`
- `path_body` routing keeps `id` in body for PUT endpoints (Tripletex optimistic locking)

### Unit 3: Supplier + incoming invoice tools ✅
- 38 typed tools (added `search_supplier`, `post_supplier`, `post_incomingInvoice`)
- `/incomingInvoice` returns 403 (BETA-only) → fallback to voucher pattern
- Supplier invoice via `post_ledger_voucher`: expense posting (vatType=1) + AP posting (acct 2400 with supplier_id)

### Unit 4: Validation + Codex review ✅
- SEARCH_FIELDS validated against API spec — removed invalid fields (`number` on CostCategory, `status` on TravelExpense)
- Codex review: fixed `invoice_order` entity type (was "order" → now "invoice", preserves `amountOutstanding`)
- Corrected VAT types: 1 = 25% input (purchases), 3 = 25% output (sales)

---

## Test Results (live endpoint, 2026-03-21 14:25 UTC)

| Task Type | Turns | Errors | Time |
|-----------|-------|--------|------|
| Department creation | 2 | 0 | 5s |
| Employee creation | 5 | 1 (self-corrected) | 22s |
| Customer creation | 2 | 0 | 5s |
| Order + invoice + full payment | 8 | 0 | 22s |
| Travel expense + cost | 9 | 0 | 39s |
| Supplier invoice (voucher) | 6 | 0 | 20s |
| Voucher / journal entry | 3 | 0 | 13s |
| German: invoice + payment | 7 | 0 | 19s |
| Project creation | 4 | 0 | 16s |

**9/9 passing. 1 self-corrected error total.**

---

## Key Files

| File | Lines | Role |
|------|-------|------|
| `agent.py` | ~480 | Gemini 2.5 Pro agent loop + autofixes + compact responses |
| `prompts.py` | ~107 | System prompt: 16 patterns, VAT ref, account ref, language key |
| `tool_router.py` | ~252 | TOOL_MAP (38 tools), SEARCH_FIELDS, REF_FIELDS, canonicalization |
| `schema_guard.py` | ~222 | OpenAPI spec validation, strips read-only/unknown fields |
| `gen_tools.py` | ~400 | Generates typed tool definitions from api_spec_extract.json |
| `server.py` | ~40 | FastAPI `/solve` endpoint |

## Deployment

- **Server**: vps (XXx--xx-VPS2) → `/opt/tripletex/`
- **Service**: `systemctl restart tripletex`
- **Endpoint**: `https://nm.j6x.com/solve`
- **Deploy**: `scp *.py vps:/opt/tripletex/ && ssh vps "systemctl restart tripletex"`

## Remaining Risks

1. Competition sandbox may have different account IDs / products than test sandbox
2. Multi-VAT-rate orders (25%/15%/0% mixed) not tested with real products
3. `post_incomingInvoice` may work on competition sandbox (BETA might be enabled)
4. Nested schema validation is top-level only — wrong nested field names still possible
5. Tasks requiring employee updates (`put_employee`) not tested
