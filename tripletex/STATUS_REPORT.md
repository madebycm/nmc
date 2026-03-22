# Tripletex Agent — Full Status Report for 3rd Party Review
**Generated: 2026-03-22 ~04:00 CET | Competition ends: 15:00 CET (~11h remaining)**
**Current rank: ~#141 | Score: 53.41 | #1: 89.03 | Submissions left: ~260**

---

## 1. Competition Scoring System

- 30 task types, each scored independently (best score kept across submissions)
- Score = correctness (0.0–1.0) × tier multiplier (T1=×1, T2=×2, T3=×3)
- **ONLY if correctness = 1.0**: efficiency bonus up to 2× (based on fewest API calls + zero 4xx errors vs best-known solutions)
- Efficiency benchmarks recalculated every 12 hours
- Total = sum of best per task type
- Maximum possible: 30 tasks × 3 tier × 2 efficiency = 180 (theoretical)

**Critical implication**: A task scoring 0.9 correctness gets ZERO efficiency bonus. Only 1.0 correctness tasks earn the bonus. Near-misses are worth far less than perfect scores.

---

## 2. Architecture Overview

```
Request Flow:
  Competition → POST /solve → server.py → agent.py::solve_task_sync()
                                              ↓
                                    prompt hints injected
                                              ↓
                                    Gemini 2.5-pro loop (max 25 turns)
                                              ↓
                                    Tool calls → _pre_validate() → route_tool_call()
                                              ↓
                                    _exec_api() → schema_guard → Tripletex API
                                              ↓
                                    Auto-recovery on errors
                                              ↓
                                    _compact_response() → back to Gemini
                                              ↓
                                    task_complete → log → return
```

### Components
| File | Lines | Purpose |
|------|-------|---------|
| `server.py` | 96 | FastAPI endpoint, thread pool executor |
| `agent.py` | 1416 | Main agent loop, pre-validation, auto-recovery, compaction |
| `prompts.py` | 432 | System prompt with 28 task patterns |
| `tool_router.py` | 367 | Maps 50 typed tools to HTTP method + endpoint |
| `schema_guard.py` | 222 | Validates POST/PUT bodies against OpenAPI spec |
| `spec_catalog.py` | 188 | Searchable API catalog for generic fallback |
| `verifier.py` | 635 | Local test verification (22 task type verifiers) |
| `gen_tools_output.json` | ~3060 | Tool definitions sent to Gemini (56 tools total) |

### Key Design Decisions
- **Gemini 2.5-pro** as LLM backend (temperature 0.1 for consistency)
- **Three-layer defense**: (1) prompt hints before model sees task, (2) `_pre_validate` deterministic fixes before API, (3) auto-recovery after API errors
- **Search cache**: per-task GET deduplication
- **Response compaction**: structured extraction per entity type (~3000 char limit)
- **Bank account setup**: runs before every task to prevent 422 on invoice tasks

---

## 3. Aggregate Statistics (767 tasks logged)

| Metric | Value |
|--------|-------|
| Total tasks | 767 |
| Avg API calls/task | 3.7 |
| Zero-error tasks | 577/767 (75%) |
| Tasks with errors | 190/767 (25%) |
| Local/deployed sync | ✅ MD5 match |

---

## 4. Recent Errors (Last 30 Tasks)

### Error 1: Bank account not registered (invoice task)
```
PROMPT: Opprett en faktura til kunden Havbris AS med tre produktlinjer...
ERR: invoice_order → 422 "Faktura kan ikke opprettes før selskapet har registrert et bankkontonummer"
```
**Root cause**: `_ensure_bank_account()` ran but bank account was not properly set on THIS company's ledger account 1920. The competition creates fresh Tripletex instances per task with different company setups.
**Impact**: Any invoice/payment task on a company where 1920 has no bank number → 0%.

### Error 2: Account locked to MVA 0 (supplier invoice with VAT)
```
PROMPT: Rechnung INV-2026-2399 vom Lieferanten Waldstein GmbH über 5550 NOK einschließlich MwSt
ERR: post_ledger_voucher → 422 "Kontoen 7100 Bilgodtgjørelse oppgavepliktig er låst til mva-kode 0"
```
**Root cause**: Account 7100 is locked to MVA code 0 (no VAT). Agent tried vatType_id=1 (25% VAT). Auto-recovery exists but may not have triggered for this specific case.
**Status**: Auto-recovery pattern added — rebuilds with manual 3-posting (expense excl VAT + separate 2710 VAT line + supplier credit).

### Error 3: Token expiration mid-task
```
PROMPT: Precisamos da despesa de Kaffemøte deste recibo registada no departamento Utvikling
ERR: 8x HTTP 403 "Invalid or expired proxy token"
```
**Root cause**: Competition proxy token expired during task execution. All 8 remaining calls failed.
**Impact**: Unrecoverable — agent correctly detected auth failure but all work was lost.

### Error 4: Supplier invoice on account 7100 (same locked-VAT pattern)
```
PROMPT: faktura INV-2026-9559 frå leverandøren Elvdal AS på 36600 kr inklusiv MVA... konto 7100
ERR: post_ledger_voucher → 422 "Kontoen 7100 er låst til mva-kode 0"
```
**Same pattern as Error 2.** Both tasks use account 7100 which is locked.

### Error 5: Employee email required
```
PROMPT: You received an offer letter (see attached PDF) for a new employee
ERR: post_employee → 422 "email: Må angis for Tripletex-brukere"
```
**Root cause**: PDF didn't contain an email address (or model failed to extract one), but Tripletex requires email for creating users.
**Impact**: Employee creation fails → entire onboarding chain fails → 0%.

---

## 5. Known Failure Patterns

| # | Pattern | Impact | Fix Status |
|---|---------|--------|------------|
| 1 | Bank account missing on fresh company | 0% on all invoice tasks | `_ensure_bank_account` runs — but may fail silently |
| 2 | Account locked to MVA 0 (7100, 7350) | 0% on supplier invoices using these accounts | Auto-recovery added: manual 3-posting split |
| 3 | Token expiration mid-task | 0% — unrecoverable | Auth detection + early bail (reduces wasted calls) |
| 4 | Missing email from PDF extraction | 0% on employee-from-PDF | No fallback — agent should generate placeholder email |
| 5 | FX agio wrong base amount | Partial score on FX tasks | Prompt hint rewritten to use invoice amountOutstanding |
| 6 | 4-digit STYRK code returns 0 results | Partial on employee tasks | Auto-broadening search added |
| 7 | Wrong costCategory for diett | Partial on travel expense | Prompt hint for 'Kost' category added |
| 8 | No-VAT tasks getting vatType=3 | 0% on tax-exempt invoices | Code enforcement + prompt hint added |

---

## 6. Complete Source Code

### 6.1 server.py (Entry Point)

```python
import asyncio
import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "gemini")  # "gemini" or "claude"

if AGENT_BACKEND == "claude":
    from agent_claude import solve_task_sync
else:
    from agent import solve_task_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tripletex")

app = FastAPI()

API_KEY = os.environ.get("TRIPLETEX_API_KEY", "")
TASK_LOG_FILE = os.environ.get("TASK_LOG_FILE", "/opt/tripletex/task_log.jsonl")

# Thread pool for concurrent task execution (sync requests blocks event loop)
_executor = ThreadPoolExecutor(max_workers=8)


@app.post("/solve")
async def solve(request: Request):
    # Optional bearer token auth
    if API_KEY:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {API_KEY}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    body = await request.json()
    prompt = body.get("prompt", "")
    client_ip = request.client.host if request.client else "unknown"
    body["_client_ip"] = client_ip
    body["_smoke"] = request.headers.get("x-smoke-test", "") == "1"
    log.info("Task received [%s]%s: %s", client_ip, " (SMOKE)" if body["_smoke"] else "", prompt[:120])
    t0 = time.time()

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, solve_task_sync, body)
        log.info("Task completed successfully")
    except Exception:
        log.error("Task failed:\n%s", traceback.format_exc())
        try:
            crash_entry = {
                "prompt": prompt,
                "ip": client_ip,
                "outcome": "crash",
                "error": traceback.format_exc()[-500:],
                "elapsed_s": round(time.time() - t0, 1),
                "logged_at": datetime.utcnow().isoformat() + "Z",
            }
            with open(TASK_LOG_FILE, "a") as f:
                f.write(json.dumps(crash_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # Always return completed — partial work still earns partial credit
    return JSONResponse({"status": "completed"})


@app.post("/solve2")
async def solve2(request: Request):
    """Dump-only endpoint: stores the incoming prompt and returns immediately."""
    body = await request.json()
    prompt = body.get("prompt", "")
    client_ip = request.client.host if request.client else "unknown"
    log.info("solve2 DUMP [%s]: %s", client_ip, prompt[:120])

    entry = {
        "prompt": prompt,
        "body": body,
        "ip": client_ip,
        "logged_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(TASK_LOG_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return JSONResponse({"status": "completed"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 6.2 agent.py (Main Agent — 1416 lines)

```python
"""Gemini-powered Tripletex agent with typed tools from OpenAPI spec."""

import base64
import io
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta

import requests

from prompts import SYSTEM_PROMPT
from tool_router import route_tool_call
from schema_guard import validate_and_sanitize
from spec_catalog import search_spec, validate_generic_call

log = logging.getLogger("tripletex.agent")


# ─── Per-task search cache ───

class _SearchCache:
    """Cache GET results within a single task to avoid duplicate API calls."""
    def __init__(self):
        self._store = {}

    def key(self, method, endpoint, params):
        if method != "GET":
            return None
        p = json.dumps(params, sort_keys=True) if params else ""
        return f"{endpoint}|{p}"

    def get(self, method, endpoint, params):
        k = self.key(method, endpoint, params)
        return self._store.get(k) if k else None

    def put(self, method, endpoint, params, result):
        k = self.key(method, endpoint, params)
        if k and "ERROR" not in result:
            self._store[k] = result


# ─── PDF text extraction ───

def _extract_pdf_text(b64_data: str) -> str:
    """Extract text from a base64-encoded PDF. Returns empty string on failure."""
    try:
        pdf_bytes = base64.b64decode(b64_data)
    except Exception:
        return ""
    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return "\n".join(texts)
    except Exception:
        pass
    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n".join(texts)
    except Exception:
        pass
    return ""


# ─── Field syntax fix for tripletex_api ───

def _fix_fields_syntax(fields_str: str) -> str:
    """Fix invalid {*} field expansion in Tripletex API queries."""
    if not fields_str:
        return fields_str
    fixed = re.sub(r'(\w+)\{\*\}', r'\1(*)', fields_str)
    fixed = re.sub(r',+', ',', fixed).strip(',')
    return fixed or "*"


TASK_LOG_FILE = os.environ.get("TASK_LOG_FILE", "/opt/tripletex/task_log.jsonl")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

MAX_TURNS = 25

# Module-level task prompt — set at start of each task for _pre_validate access
_current_task_prompt = ""

# No-VAT keywords in multiple languages
_NO_VAT_KEYWORDS = [
    "sem iva", "sin iva", "sans tva", "ohne mwst", "ohne mehrwertsteuer",
    "uten mva", "avgiftsfri", "without vat", "no vat", "vat free",
    "0% mva", "0% vat", "0% iva", "0% tva", "mva-fri", "momsfri",
    "exento de iva", "isento de iva", "exonéré de tva",
]


class _ApiResult(str):
    """String subclass that carries raw API response data for verifiers."""
    def __new__(cls, compact: str, raw_data=None):
        obj = str.__new__(cls, compact)
        obj.raw_data = raw_data
        return obj


def _prompt_says_no_vat(prompt: str) -> bool:
    """Check if task prompt explicitly says no VAT / exempt from VAT."""
    lower = prompt.lower()
    return any(kw in lower for kw in _NO_VAT_KEYWORDS)


# ─── Runtime autofixes ───

def _pre_validate(tool_name: str, args: dict) -> dict:
    """Deterministic fixes BEFORE routing to API. Returns corrected args."""
    args = dict(args)

    if tool_name == "post_order":
        if "deliveryDate" not in args and "orderDate" in args:
            args["deliveryDate"] = args["orderDate"]
            log.info("AUTOFIX: set deliveryDate = %s", args["deliveryDate"])
        elif "deliveryDate" not in args:
            args["deliveryDate"] = date.today().isoformat()
            args.setdefault("orderDate", args["deliveryDate"])
            log.info("AUTOFIX: set deliveryDate + orderDate = %s", args["deliveryDate"])

    elif tool_name == "post_product":
        if _prompt_says_no_vat(_current_task_prompt):
            vat = args.get("vatType")
            vat_id = vat.get("id") if isinstance(vat, dict) else vat
            if vat_id != 6:
                args["vatType"] = {"id": 6}
                log.info("AUTOFIX: task says no-VAT → forced vatType={id:6} (was %s)", vat_id)

    elif tool_name == "search_product":
        pn = args.get("productNumber", "")
        if isinstance(pn, str) and "," in pn:
            first = pn.split(",")[0].strip()
            log.warning("AUTOFIX: comma in productNumber '%s' → using '%s'", pn, first)
            args["productNumber"] = first

    elif tool_name == "post_ledger_voucher":
        if not args.get("sendToLedger"):
            args["sendToLedger"] = True
            log.info("AUTOFIX: set sendToLedger=true")
        postings = args.get("postings", [])
        if not postings:
            log.warning("AUTOFIX: voucher has no postings")
        elif isinstance(postings, list):
            total = sum(p.get("amountGross", 0) for p in postings if isinstance(p, dict))
            if abs(total) > 0.01:
                log.warning("AUTOFIX: voucher postings unbalanced (sum=%.2f)", total)
            voucher_date = args.get("date", date.today().isoformat())
            for i, p in enumerate(postings):
                if not isinstance(p, dict):
                    continue
                if "date" not in p:
                    p["date"] = voucher_date
                if "row" not in p or p.get("row", 0) == 0:
                    p["row"] = i + 1
                    log.info("AUTOFIX: set row=%d on posting %d", i + 1, i)
                if "amountGross" in p and "amountGrossCurrency" not in p:
                    if not p.get("currency") and not p.get("currency_id"):
                        p["amountGrossCurrency"] = p["amountGross"]
                        log.info("AUTOFIX: set amountGrossCurrency=%s on posting %d (NOK)", p["amountGross"], i)

    elif tool_name == "post_incomingInvoice":
        order_lines = args.get("orderLines", [])
        if isinstance(order_lines, list):
            for i, line in enumerate(order_lines):
                if isinstance(line, dict) and "externalId" not in line:
                    line["externalId"] = f"line{i+1}"
                    log.info("AUTOFIX: set externalId='line%d' on orderLine %d", i+1, i)

    elif tool_name == "post_travelExpense_cost":
        if "amountCurrencyIncVat" not in args:
            rate = args.get("rate") or args.get("amount", 0)
            count = args.get("count", 1) or 1
            amount = rate * count if rate else 0
            if amount:
                args["amountCurrencyIncVat"] = amount
                log.info("AUTOFIX: set amountCurrencyIncVat=%s", amount)

    elif tool_name == "post_travelExpense":
        td = args.get("travelDetails")
        if isinstance(td, dict):
            td.setdefault("isForeignTravel", False)
            td.setdefault("isDayTrip", False)
            td.setdefault("departureFrom", "Kontoret")
            args["travelDetails"] = td

    elif tool_name == "post_supplier":
        email = args.get("email", "")
        if email and not args.get("invoiceEmail"):
            args["invoiceEmail"] = email
            log.info("AUTOFIX: set invoiceEmail=%s", email)

    elif tool_name == "post_customer":
        email = args.get("email", "")
        if email and not args.get("invoiceEmail"):
            args["invoiceEmail"] = email
            log.info("AUTOFIX: set invoiceEmail=%s", email)

    elif tool_name == "post_employee":
        ut = args.get("userType", "")
        if not ut or ut == "0":
            args["userType"] = "STANDARD"
            log.info("AUTOFIX: set userType=STANDARD")

    elif tool_name == "search_invoice":
        date_from = args.get("invoiceDateFrom", "")
        date_to = args.get("invoiceDateTo", "")
        if date_from and date_to and date_from >= date_to:
            try:
                dt = date.fromisoformat(date_to)
                args["invoiceDateTo"] = (dt + timedelta(days=1)).isoformat()
                log.info("AUTOFIX: invoiceDateTo bumped to %s", args["invoiceDateTo"])
            except ValueError:
                pass

    elif tool_name == "createReminder_invoice":
        if "sendType" in args and "dispatchType" not in args:
            args["dispatchType"] = args.pop("sendType")
            log.info("AUTOFIX: renamed sendType → dispatchType")
        if not args.get("dispatchType"):
            args["dispatchType"] = "EMAIL"
            log.info("AUTOFIX: set dispatchType=EMAIL")

    elif tool_name == "invoice_order":
        if not args.get("sendToCustomer"):
            args["sendToCustomer"] = True
        if not args.get("invoiceDate"):
            args["invoiceDate"] = date.today().isoformat()

    elif tool_name == "post_project":
        if not args.get("startDate"):
            args["startDate"] = date.today().isoformat()
            log.info("AUTOFIX: set project startDate = %s", args["startDate"])

    elif tool_name == "post_employee_employment":
        if "isMainEmployer" not in args:
            args["isMainEmployer"] = True
            log.info("AUTOFIX: set isMainEmployer=true")
        if not args.get("taxDeductionCode"):
            args["taxDeductionCode"] = "loennFraHovedarbeidsgiver"
            log.info("AUTOFIX: set taxDeductionCode=loennFraHovedarbeidsgiver")

    elif tool_name == "post_employee_employment_details":
        if "employmentType" not in args or not args.get("employmentType"):
            args["employmentType"] = "ORDINARY"
            log.info("AUTOFIX: set employmentType=ORDINARY")
        if "remunerationType" not in args or not args.get("remunerationType"):
            args["remunerationType"] = "MONTHLY_WAGE"
            log.info("AUTOFIX: set remunerationType=MONTHLY_WAGE")
        if "workingHoursScheme" not in args or not args.get("workingHoursScheme"):
            args["workingHoursScheme"] = "NOT_SHIFT"
            log.info("AUTOFIX: set workingHoursScheme=NOT_SHIFT")

    return args


# ─── Response compaction (entity field extraction, ~3000 char limit) ───

_ENTITY_FIELDS = {
    "customer": ["id", "name", "organizationNumber", "customerNumber", "supplierNumber", "email"],
    "employee": ["id", "firstName", "lastName", "email", "employeeNumber"],
    "department": ["id", "name", "departmentNumber"],
    "product": ["id", "name", "number", "priceExcludingVatCurrency", "vatType"],
    "order": ["id", "number", "customer", "orderDate", "deliveryDate"],
    "invoice": ["id", "invoiceNumber", "invoiceDate", "customer", "amountOutstanding",
                 "amountOutstandingTotal", "amount", "amountCurrency",
                 "amountExcludingVat", "amountExcludingVatCurrency",
                 "isCredited", "isCreditNote", "isApproved"],
    "travelExpense": ["id", "title", "employee", "state", "amount", "travelDetails"],
    "project": ["id", "name", "number", "version", "isFixedPrice", "fixedprice",
                 "projectManager", "customer", "startDate", "endDate", "isInternal"],
    "voucher": ["id", "number", "date", "description", "postings"],
    "supplier": ["id", "name", "supplierNumber", "organizationNumber"],
    "contact": ["id", "firstName", "lastName", "email"],
    "paymentType": ["id", "description", "displayName"],
    "costCategory": ["id", "description", "displayName"],
    "cost": ["id", "travelExpense", "costCategory", "paymentType", "rate", "count",
             "amountCurrencyIncVat", "amountNOKInclVAT", "date", "comments"],
    "account": ["id", "number", "name"],
    "employment": ["id", "employee", "startDate", "endDate", "employmentId"],
    "employmentDetails": ["id", "employment", "date", "annualSalary", "percentageOfFullTimeEquivalent",
                          "occupationCode", "employmentType", "remunerationType", "workingHoursScheme"],
    "occupationCode": ["id", "code", "nameNO"],
    "accountingDimensionName": ["id", "dimensionName", "description", "dimensionIndex", "active"],
    "accountingDimensionValue": ["id", "displayName", "dimensionIndex", "number", "active"],
}

# [_extract_fields, _guess_entity_type, _post_filter_results, _compact_response, _log_task]
# These are helper functions for response compaction — see full source on VPS at /opt/tripletex/agent.py

# ─── Load typed tools from gen_tools_output.json ───
# Loads 56 tools: 50 typed tools + task_complete + search_tripletex_spec +
# get_account_balances + tripletex_api (generic fallback)

# ─── Gemini API call ───
# _call_gemini: POST to Gemini 2.5-pro with system prompt, contents, tools
# Retries 3x on 429/5xx, timeout=180s, temperature=0.1

# ─── Tripletex API execution ───
# _exec_api: schema guard → HTTP request → compact response
# Returns _ApiResult (str subclass with raw_data attached)

# ─── Bank account setup ───
# _ensure_bank_account: GET /ledger/account?number=1920 → PUT bankAccountNumber if empty
# CRITICAL: runs before every task, costs 1-2 proxy calls

# ─── FX controller ───
# _parse_fx_from_prompt: regex EUR amount, old rate, new rate
# _compute_fx_amounts: deterministic NOK/EUR/agio calculation

# ─── CSV bank statement parser ───
# _parse_csv_bank_statement: extract structured {amount, ref, date} from CSV files

# ─── Agent loop (solve_task_sync) ───
# This is the main entry point. Key flow:
# 1. Set _current_task_prompt for _pre_validate access
# 2. Extract PDF text from attachments
# 3. Pre-parse CSV bank statements
# 4. Pre-compute FX amounts
# 5. Inject prompt hints (no-VAT, travel expense, receipt VAT)
# 6. Gemini loop: up to 25 turns
#    - For each tool call: _pre_validate → route_tool_call → _exec_api
#    - Auto-recovery patterns after errors
#    - Cache hits for duplicate GETs
# 7. Log task record

# ─── Auto-recovery patterns in agent loop ───
# 1. costCategory search empty → list all categories
# 2. occupation code short (4-digit) → broaden with count=100, then 2-digit prefix
# 3. voucher locked to MVA 0 → rebuild with manual 3-posting split
# 4. product number conflict → search existing product
# 5. employee email conflict → search by email
# 6. project missing projectManager → find default employee and retry

# ─── Prompt hint injections ───
# 1. FX pre-computation hint (exact payment + agio amounts)
# 2. No-VAT detection → force vatType={id:6}
# 3. Travel expense diett → search costCategory 'Kost'
# 4. Receipt/kvittering → VAT handling reminder
# 5. CSV bank statement → structured parsed data
```

### 6.3 tool_router.py (Full Source)

```python
"""Routes typed tool calls back to HTTP API requests."""

import logging

log = logging.getLogger("tripletex.router")

TOOL_MAP = {
    # Employee
    "search_employee":      ("GET",    "/employee",           "query"),
    "post_employee":        ("POST",   "/employee",           "body"),
    "put_employee":         ("PUT",    "/employee/{id}",      "path_body"),
    # Department
    "search_department":    ("GET",    "/department",          "query"),
    "post_department":      ("POST",   "/department",          "body"),
    # Customer
    "search_customer":      ("GET",    "/customer",            "query"),
    "post_customer":        ("POST",   "/customer",            "body"),
    "put_customer":         ("PUT",    "/customer/{id}",       "path_body"),
    # Contact
    "search_contact":       ("GET",    "/contact",             "query"),
    "post_contact":         ("POST",   "/contact",             "body"),
    # Product
    "search_product":       ("GET",    "/product",             "query"),
    "post_product":         ("POST",   "/product",             "body"),
    # Order
    "post_order":           ("POST",   "/order",               "body"),
    "invoice_order":        ("PUT",    "/order/{id}/:invoice", "path_query"),
    # Invoice
    "search_invoice":       ("GET",    "/invoice",             "query"),
    "payment_invoice":      ("PUT",    "/invoice/{id}/:payment",        "path_query"),
    "createCreditNote_invoice": ("PUT", "/invoice/{id}/:createCreditNote", "path_query"),
    "send_invoice":         ("PUT",    "/invoice/{id}/:send",  "path_query"),
    "search_invoice_paymentType": ("GET", "/invoice/paymentType", "query"),
    # Supplier
    "search_supplier":      ("GET",    "/supplier",            "query"),
    "post_supplier":        ("POST",   "/supplier",            "body"),
    # Incoming Invoice (supplier invoices)
    "post_incomingInvoice": ("POST",   "/incomingInvoice",     "body_query"),
    # Travel Expense
    "search_travelExpense": ("GET",    "/travelExpense",       "query"),
    "post_travelExpense":   ("POST",   "/travelExpense",       "body"),
    "delete_travelExpense": ("DELETE", "/travelExpense/{id}",  "path_query"),
    "post_travelExpense_cost": ("POST", "/travelExpense/cost", "body"),
    "delete_travelExpense_cost": ("DELETE", "/travelExpense/cost/{id}", "path_query"),
    "search_travelExpense_paymentType": ("GET", "/travelExpense/paymentType", "query"),
    "search_travelExpense_costCategory": ("GET", "/travelExpense/costCategory", "query"),
    # Project
    "search_project":       ("GET",    "/project",             "query"),
    "post_project":         ("POST",   "/project",             "body"),
    "put_project":          ("PUT",    "/project/{id}",        "path_body"),
    # Ledger
    "search_ledger_voucher": ("GET",   "/ledger/voucher",      "query"),
    "post_ledger_voucher":  ("POST",   "/ledger/voucher",      "body_query"),
    "delete_ledger_voucher": ("DELETE", "/ledger/voucher/{id}", "path_query"),
    "search_ledger_account": ("GET",   "/ledger/account",      "query"),
    # Company
    "get_company":          ("GET",    "/company/{id}",        "path_query"),
    "get_company_withLoginAccess": ("GET", "/company/withLoginAccess", "query"),
    # Entitlement
    "grantEntitlementsByTemplate_employee_entitlement": ("PUT", "/employee/entitlement/:grantEntitlementsByTemplate", "query"),
    # Employment
    "search_employee_employment":      ("GET",    "/employee/employment",           "query"),
    "post_employee_employment":        ("POST",   "/employee/employment",           "body"),
    "put_employee_employment":         ("PUT",    "/employee/employment/{id}",      "path_body"),
    "search_employee_employment_details": ("GET", "/employee/employment/details",   "query"),
    "post_employee_employment_details": ("POST",  "/employee/employment/details",   "body"),
    "put_employee_employment_details": ("PUT",    "/employee/employment/details/{id}", "path_body"),
    "search_employee_employment_occupationCode": ("GET", "/employee/employment/occupationCode", "query"),
    # Accounting Dimensions
    "post_ledger_accountingDimensionName":  ("POST", "/ledger/accountingDimensionName",        "body"),
    "search_ledger_accountingDimensionName": ("GET", "/ledger/accountingDimensionName/search", "query"),
    "post_ledger_accountingDimensionValue": ("POST", "/ledger/accountingDimensionValue",       "body"),
    "search_ledger_accountingDimensionValue": ("GET", "/ledger/accountingDimensionValue/search", "query"),
    # Invoice Reminder
    "createReminder_invoice":          ("PUT",    "/invoice/{id}/:createReminder",  "path_query"),
    # Timesheet
    "search_activity":                 ("GET",    "/activity",                       "query"),
    "search_timesheet_entry":          ("GET",    "/timesheet/entry",                "query"),
    "post_timesheet_entry":            ("POST",   "/timesheet/entry",                "body"),
    # Salary
    "search_salary_type":              ("GET",    "/salary/type",                    "query"),
    "post_salary_transaction":         ("POST",   "/salary/transaction",             "body"),
    # Supplier Invoice
    "search_supplierInvoice":          ("GET",    "/supplierInvoice",                "query"),
    "addPayment_supplierInvoice":      ("POST",   "/supplierInvoice/{invoiceId}/:addPayment", "path_query"),
}

QUERY_FIELDS = {
    "post_ledger_voucher": {"sendToLedger"},
    "post_incomingInvoice": {"sendTo"},
}

SEARCH_FIELDS = {
    "search_employee": "id,firstName,lastName,email,employeeNumber,dateOfBirth,department(id,name)",
    "search_customer": "id,name,organizationNumber,customerNumber,email,phoneNumber,supplierNumber",
    "search_product": "id,name,number,priceExcludingVatCurrency,vatType(id,name)",
    "search_invoice": "id,invoiceNumber,invoiceDate,invoiceDueDate,customer(id,name),currency(id,code),amountOutstanding,amountOutstandingTotal,amount,amountCurrency,amountExcludingVat,amountExcludingVatCurrency,isCredited,isCreditNote,isApproved",
    "search_department": "id,name,departmentNumber",
    "search_contact": "id,firstName,lastName,email,phoneNumberMobile,phoneNumberWork,customer(id,name)",
    "search_project": "id,name,number,version,isFixedPrice,fixedprice,isInternal,projectManager(id,firstName,lastName),customer(id,name),startDate,endDate,description",
    "search_travelExpense": "id,title,state,amount,employee(id,firstName,lastName),travelDetails(departureDate,returnDate)",
    "search_travelExpense_paymentType": "id,description,displayName",
    "search_travelExpense_costCategory": "id,description,displayName",
    "search_invoice_paymentType": "id,description,displayName,debitAccount(id,number)",
    "search_ledger_voucher": "id,number,date,description,year",
    "search_ledger_account": "id,number,name",
    "search_supplier": "id,name,supplierNumber,organizationNumber,email,phoneNumber",
    "search_supplierInvoice": "id,invoiceNumber,invoiceDate,invoiceDueDate,supplier(id,name),amount,amountCurrency,outstandingAmount,currency(id,code),voucher(id)",
    "search_employee_employment": "id,employee(id,firstName,lastName),startDate,endDate,employmentId",
    "search_employee_employment_details": "id,employment(id),date,annualSalary,percentageOfFullTimeEquivalent,occupationCode(id,code,nameNO),employmentType,remunerationType,workingHoursScheme",
    "search_employee_employment_occupationCode": "id,code,nameNO",
    "search_ledger_accountingDimensionName": "id,dimensionName,description,dimensionIndex,active",
    "search_ledger_accountingDimensionValue": "id,displayName,dimensionIndex,number,active",
    "search_activity": "id,name,number,isProjectActivity",
    "search_timesheet_entry": "id,employee(id,firstName,lastName),project(id,name),activity(id,name),date,hours,comment",
}

# REF_FIELDS: 67 entries mapping flat _id params to nested API format
# _CAMEL_REF_FIELDS: 20 entries mapping camelCase refs (accountId → account)
# _canonicalize_nested_item: accepts all 4 ref formats from model
# _canonicalize_all_arrays: canonicalize refs inside all array-of-dict values
# _flatten_for_incoming_invoice: ensures incomingInvoice orderLines use flat format
# route_tool_call: main routing function → (method, endpoint, params, body)
```

### 6.4 prompts.py (System Prompt — 432 lines, 28 task patterns)

```python
SYSTEM_PROMPT = """You are an expert AI accounting agent for Tripletex (Norwegian accounting software).
You receive task prompts in multiple languages (Norwegian Bokmål, Nynorsk, English, Spanish, Portuguese, German, French) and must execute the correct API calls to complete them.

RULES:
1. Execute API calls IMMEDIATELY — every extra turn costs efficiency points.
2. Plan carefully to get calls right on the first try.
3. Account starts FRESH each time (1 default employee, 1 default department, pre-existing products).
4. Use IDs from responses — never query for something you just created.
5. Call task_complete when done.
6. ALWAYS make tool calls — NEVER respond with only text.
7. If no specific tools, use search_tripletex_spec → tripletex_api.
8. MINIMIZE API calls. Every call counts against efficiency.
9. SET EVERY FIELD mentioned in the task prompt.
10. INLINE ADDRESSES: use tripletex_api directly with nested postalAddress objects.

ERROR RECOVERY:
- 422: fix the field and retry
- 403: module disabled, use fallback pattern
- 404: wrong entity ID, search again

28 TASK PATTERNS:
1. CREATE CUSTOMER — post_customer with address via tripletex_api
2. CREATE EMPLOYEE — search_department → post_employee
3. CREATE EMPLOYEE AS ADMIN — + grantEntitlementsByTemplate
4. CREATE DEPARTMENT
5. CREATE PRODUCT
6. CREATE ORDER + INVOICE — post_order → invoice_order
7. ORDER + INVOICE + FULL PAYMENT — + payment_invoice
8. REGISTER PAYMENT ON EXISTING INVOICE
9. CREATE CREDIT NOTE
10. CREATE TRAVEL EXPENSE — with cost lines
11. DELETE TRAVEL EXPENSE
12. CREATE PROJECT
13. CREATE VOUCHER — balanced postings
14. REGISTER SUPPLIER INVOICE — ALWAYS use post_ledger_voucher (post_incomingInvoice returns 403)
15. CREATE SUPPLIER
16. SEND INVOICE
17. EMPLOYEE WITH EMPLOYMENT CONTRACT — from PDF
18. CUSTOM ACCOUNTING DIMENSION
19. OVERDUE INVOICE + REMINDER FEE
20. CUSTOM DIMENSION + VOUCHER LINKED TO DIMENSION VALUE
21. MULTI-CURRENCY PAYMENT + AGIO
22. REVERSE / CANCEL PAYMENT
23. SALARY / PAYROLL
24. PROJECT FIXED PRICE + MILESTONE INVOICING
25. BANK RECONCILIATION (from CSV)
26. YEAR-END CLOSING
27. MONTHLY CLOSING
28. TIMESHEET / HOUR REGISTRATION

VAT TYPE REFERENCE:
- 1 = 25% input, 3 = 25% output, 5 = 0% within law, 6 = 0% outside law
- 11 = 15% input (food), 12 = 12% input, 31 = 15% output, 32 = 12% output

NORWEGIAN ACCOUNTING ACCOUNTS:
- 1920=Bank, 2400=AP, 2710=Input VAT, 4000=Purchases, 6300=Rent
- 6800=Office, 7100=Travel, 1500=AR, 8060=FX gain, 8160=FX loss
"""
```

### 6.5 schema_guard.py (Full Source)

```python
"""Schema guard: validates and sanitizes API calls against the real OpenAPI spec."""

import json
import logging
import os
import re

log = logging.getLogger("tripletex.guard")

_SPEC = None
_SCHEMAS = None

def _load_spec():
    global _SPEC, _SCHEMAS
    if _SPEC is not None:
        return
    spec_path = os.environ.get("API_SPEC_PATH",
        os.path.join(os.path.dirname(__file__), "api_spec_extract.json"))
    with open(spec_path) as f:
        data = json.load(f)
    _SPEC = data["endpoints"]
    _SCHEMAS = data["schemas"]
    log.info("Schema guard loaded: %d endpoints, %d schemas", len(_SPEC), len(_SCHEMAS))

def _match_endpoint(method, endpoint):
    """Match concrete endpoint to spec pattern."""
    _load_spec()
    if endpoint in _SPEC:
        ep = _SPEC[endpoint]
        if method in ep:
            return endpoint, ep[method]
    for spec_path in _SPEC:
        spec_parts = spec_path.strip("/").split("/")
        parts = endpoint.strip("/").split("/")
        if len(spec_parts) != len(parts):
            continue
        if all(sp == pp or (sp.startswith("{") and sp.endswith("}")) for sp, pp in zip(spec_parts, parts)):
            ep = _SPEC[spec_path]
            if method in ep:
                return spec_path, ep[method]
    return None, None

ENDPOINT_SCHEMA_MAP = {
    "/employee": "Employee", "/employee/{id}": "Employee",
    "/department": "Department", "/customer": "Customer",
    "/contact": "Contact", "/product": "Product",
    "/order": "Order", "/travelExpense": "TravelExpense",
    "/travelExpense/cost": "Cost", "/project": "Project",
    "/ledger/voucher": "Voucher", "/ledger/account": "Account",
    "/supplier": "Supplier",
    "/incomingInvoice": "IncomingInvoiceAggregateExternalWrite",
    # + more mapped endpoints
}

def validate_and_sanitize(method, endpoint, params, body):
    """Validate POST/PUT body. Strips unknown + read-only fields. Does NOT block."""
    warnings = []
    if method not in ("POST", "PUT") or not body:
        return body, warnings
    spec_path, spec_info = _match_endpoint(method, endpoint)
    schema_name = ENDPOINT_SCHEMA_MAP.get(spec_path) if spec_path else None
    if not schema_name:
        return body, warnings
    # Get all fields and readonly fields from schema
    schema = _SCHEMAS.get(schema_name, {})
    all_fields = set(schema.get("properties", {}).keys())
    readonly = {n for n, s in schema.get("properties", {}).items() if s.get("readOnly")}
    sanitized = {}
    for key, value in body.items():
        if key in readonly:
            warnings.append(f"STRIPPED read-only '{key}'")
        elif key not in all_fields:
            warnings.append(f"STRIPPED unknown '{key}'")
        else:
            sanitized[key] = value
    return sanitized, warnings
```

### 6.6 spec_catalog.py (Full Source)

```python
"""Searchable API catalog from OpenAPI spec + generic validated executor."""

import json, logging, os, re

log = logging.getLogger("tripletex.catalog")
_CATALOG = None

def _load_catalog():
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    spec_path = os.path.join(os.path.dirname(__file__), "api_spec_extract.json")
    with open(spec_path) as f:
        spec = json.load(f)
    catalog = []
    for path, methods in spec["endpoints"].items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            summary = details.get("summary", "")
            tags = details.get("tags", [])
            params = details.get("parameters", [])
            param_names = [p["name"] for p in params if isinstance(p, dict) and p.get("name")]
            body_fields = []
            body_enums = {}
            rb = details.get("requestBody", {})
            if rb:
                content = rb.get("application/json; charset=utf-8", rb)
                props = content.get("properties", {})
                for fname, fdef in props.items():
                    if fdef.get("readOnly") or fname in ("changes", "url"):
                        continue
                    body_fields.append(fname)
                    if "enum" in fdef:
                        body_enums[fname] = fdef["enum"]
            search_text = f"{method} {path} {summary} {' '.join(tags)} {' '.join(param_names)} {' '.join(body_fields)}".lower()
            catalog.append({
                "method": method.upper(), "path": path,
                "summary": summary, "tags": tags,
                "query_params": param_names, "body_fields": body_fields,
                "body_enums": body_enums,
                "_search": search_text,
            })
    _CATALOG = catalog
    log.info("Loaded API catalog: %d operations", len(catalog))
    return catalog

def search_spec(query, method_filter=None, limit=8):
    """Keyword search. Returns compact list of matching operations."""
    catalog = _load_catalog()
    keywords = query.lower().split()
    scored = []
    for entry in catalog:
        score = sum(1 + 2 * (kw in entry["path"].lower()) for kw in keywords if kw in entry["_search"])
        if method_filter and entry["method"] != method_filter.upper():
            continue
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [{"method": e["method"], "path": e["path"], "summary": e["summary"][:120],
             **({"body_fields": e["body_fields"]} if e["body_fields"] else {}),
             **({"enums": e["body_enums"]} if e["body_enums"] else {})}
            for _, e in scored[:limit]]

def validate_generic_call(method, path, query_params=None, body=None):
    """Validate generic API call. Returns (is_valid, warnings, cleaned_body)."""
    catalog = _load_catalog()
    clean_path = re.sub(r"/\d+", "/{id}", path)
    match = next((e for e in catalog if e["method"] == method.upper() and e["path"] == clean_path), None)
    if not match:
        return False, [f"Unknown: {method} {clean_path}"], body
    warnings = []
    if body and match["body_fields"]:
        known = set(match["body_fields"]) | {"id", "version"}
        body = {k: v for k, v in body.items() if k in known or warnings.append(f"Stripped '{k}'") is None}
    return True, warnings, body
```

### 6.7 verifier.py (Full Source — 635 lines)

22 task-type verifiers that check:
- Entity was created (POST success)
- Entity can be re-fetched by ID
- Key fields match (org number, amounts, etc.)
- Employment chain was completed
- Voucher postings exist
- Payments were applied (amountOutstanding → 0)

**Types covered**: create_customer, create_employee, create_department, create_product, create_invoice, create_order, create_project, supplier_invoice, travel_expense, custom_dimension, register_payment, credit_note, reverse_payment, multi_currency_payment, reminder_fee, payroll, expense_receipt, cost_analysis, ledger_correction, monthly_closing, year_end_closing, bank_reconciliation, project_billing

---

## 7. Questions for Reviewer

1. **Bank account setup**: `_ensure_bank_account` runs on every task, costing 1-2 proxy calls. Is there a way to detect if the company already has a bank account without the extra call? Or should we only run it for invoice/payment tasks?

2. **Efficiency vs correctness trade-off**: With 3.7 avg calls/task, are we close to competitive? Top teams likely achieve 2-3 calls for simple tasks. Our auto-recovery patterns add extra calls on failure paths.

3. **The 25% error rate**: 190/767 tasks had at least one error. Common causes: locked MVA accounts, token expiration, missing fields from PDF extraction. What else could drive this down?

4. **Should we switch LLM?**: Currently using Gemini 2.5-pro. Would Claude Sonnet 4 or GPT-4o be more reliable for structured tool calling? The model needs to: (a) extract exact values from PDFs, (b) choose correct tools, (c) never miscalculate amounts.

5. **Deterministic pipeline**: For the ~10 most common task types (customer, supplier, invoice, payment), should we bypass the LLM entirely and use regex+rules? This would guarantee 100% correctness + minimal calls.

6. **Missing tool definitions**: Are there Tripletex API endpoints we're not covering that the competition tests? We have 50 typed tools but the API has 551 operations.

7. **The competition proxy counts ALL HTTP calls** including our bank account setup and auto-recovery retries. These hidden calls hurt efficiency. How aggressively should we cut them?

8. **Token expiration**: 403 "Invalid or expired proxy token" is unrecoverable. Is there a way to detect this before wasting turns? Should we validate the token at task start?

9. **PDF extraction quality**: We use pdfplumber → PyPDF2 fallback. When extraction fails, the model halluccinates values. Should we add a confidence check?

10. **Voucher validation gap**: Our `_pre_validate` checks postings balance but doesn't validate that the right accounts are used. Should we add account-number validation for common patterns (depreciation, salary accrual)?
