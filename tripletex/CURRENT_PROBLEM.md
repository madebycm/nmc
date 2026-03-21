# Tripletex Agent — CURRENT PROBLEM (2026-03-21 03:00)

## Status: 0/8 on ALL THREE real competition submissions

| Task | Prompt | Root Causes | Errors |
|------|--------|------------|--------|
| Vestfjord (Nynorsk) | Order+invoice+full payment, 2 products | Comma-separated product search, missing deliveryDate, manual payment calc | 2 × 422 |
| Nordlicht (German) | Invoice with 3 VAT rates (25/15/0%) | `product_id` not unflattened in orderLines, bank account missing | 4 × 422 |
| Brightstone (English) | Register supplier invoice 59800 NOK with 25% input VAT | No supplier/incoming invoice tools, wrong accounts, voucher postings rejected | 2 × 422 |

---

## 5 Root Causes (Codex-confirmed)

### 1. MISSING TOOLS — No supplier invoice support
- **gen_tools.py TARGETS** omits `/supplier` (GET, POST) and `/incomingInvoice` (POST [BETA])
- **tool_router.py TOOL_MAP** has no entries for supplier or incoming invoice
- **prompts.py** has no supplier invoice pattern
- **schema_guard.py ENDPOINT_SCHEMA_MAP** has no supplier mapping
- The API spec HAS these endpoints: `api_spec_extract.json:10225` (`/incomingInvoice`) and `:34431` (`/supplier`)
- `/incomingInvoice` uses `IncomingInvoiceAggregateExternalWrite` schema with `invoiceHeader` (vendorId, invoiceAmount, invoiceDate, invoiceNumber) and `orderLines` (accountId, amountInclVat, vatTypeId)
- Result: Gemini forced into manual voucher postings — wrong abstraction, wrong accounts

### 2. NESTED ARRAY SCHEMA CONFUSION — `product_id` vs `product: {id: N}`
- Tool schemas declare flat `product_id: INTEGER` inside orderLines/postings items
- But Gemini sometimes emits nested `product: {id: N}` or `account: {id: N}` instead
- Router's `_unflatten_nested_refs` only converts `_id` suffix keys — nested format passes through unchanged
- When Gemini uses flat `product_id`, it gets unflattened correctly
- When Gemini uses nested `product: {id: N}`, it passes through — BUT Tripletex sometimes rejects this with "Feltet eksisterer ikke"
- **Real issue**: tool schema is ambiguous. Items say `product_id` but descriptions say "ID of Product", and the item description says "full object for creation"

### 3. ROUTER CANNOT DO QUERY + BODY
- `tool_router.py:15-18`: param_type is one of `query`, `body`, `path_query`, `path_body`
- No `body_query` or `path_body_query` option
- `/incomingInvoice?sendTo=ledger` needs query param + JSON body
- `/ledger/voucher?sendToLedger=true` also needs query + body
- This must be fixed before adding incoming invoice tools

### 4. RESPONSES AS OPAQUE STRINGS — amountOutstanding unreadable
- `agent.py:122`: `_exec_api` stringifies JSON responses, truncates at 15000 chars
- `agent.py:288`: passes string back as tool response to Gemini
- Gemini must parse `amountOutstanding` from a string blob — unreliable
- Task 1 (Vestfjord): Gemini manually calculated 61375 instead of reading amountOutstanding
- Fix: return structured response or at minimum extract key fields

### 5. NO RUNTIME VALIDATION — prompt-only guardrails fail
- Prompt says "always set deliveryDate" → Gemini still forgot it (Task 1)
- Prompt says "search ONE product at a time" → Gemini still comma-separated (Task 1)
- Prompt says "read amountOutstanding" → Gemini still calculated manually (Task 1)
- `schema_guard.py` only strips top-level unknown fields, does NOT:
  - Validate required fields
  - Validate nested objects (orderLines, postings)
  - Auto-fill deliveryDate
  - Reject comma in productNumber
- Need: deterministic runtime fixes BEFORE sending to API

---

## Minimum Fix Plan (priority order)

### A. Runtime autofixes (agent.py) — prevents 422s on known fields
1. Auto-fill `deliveryDate = orderDate` if missing on post_order
2. Split comma-separated productNumber searches into individual calls
3. Extract amountOutstanding from invoice responses and inject into context
4. Validate postings balance before sending voucher

### B. Add query+body routing (tool_router.py)
- New param_type `body_query`: specified fields go to query, rest to body
- Or: hardcode known query params per endpoint (sendToLedger, sendTo, sendToCustomer)

### C. Add supplier/incoming invoice tools
1. gen_tools.py: add to TARGETS: `/supplier` (GET, POST), `/incomingInvoice` (POST)
2. Handle `invoiceHeader` as inline object (like travelDetails)
3. tool_router.py: add TOOL_MAP entries
4. schema_guard.py: add ENDPOINT_SCHEMA_MAP entries
5. prompts.py: add supplier invoice pattern

### D. Tighten nested array schemas (gen_tools.py)
- Add `required` on orderLine items: `product_id`, `count`
- Add `required` on posting items: `account_id`, `amountGross`
- Remove ambiguous "full object for creation" wording from item descriptions

### E. Structured tool responses (agent.py)
- For invoice responses: extract `id`, `amountOutstanding`, `invoiceNumber` into compact JSON
- For search responses: extract just `values[].id` and key fields
- Reduces token usage AND makes field extraction reliable

---

## What our smoke tests missed

| We tested | Competition sends | Gap |
|-----------|-------------------|-----|
| Create dept, customer (simple POST) | Multi-step workflows | No order/invoice/payment e2e test |
| Direct sandbox URL | Proxy URL with fresh sandbox | Bank account not pre-set |
| Norwegian only | 7 languages (German, English, etc.) | Language handling |
| No supplier tasks | Supplier invoice registration | Missing tools entirely |
| No voucher tasks | Journal entries with VAT | Posting format unknown |
| No multi-VAT | 3 different VAT rates on one invoice | vatType handling |

## Files to modify
- `agent.py` — runtime autofixes, structured responses
- `tool_router.py` — query+body support, new TOOL_MAP entries
- `gen_tools.py` — new TARGETS, tighter schemas
- `prompts.py` — supplier invoice pattern, better examples
- `schema_guard.py` — new ENDPOINT_SCHEMA_MAP entries
