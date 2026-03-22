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
    """Fix invalid {*} field expansion in Tripletex API queries.

    Model emits postings{*} / voucher{*} which Tripletex rejects.
    Strip the invalid patterns; postings must be fetched separately.
    """
    if not fields_str:
        return fields_str
    # Convert braces to parentheses: postings{*} → postings(*), voucher{*} → voucher(*)
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

# No-VAT keywords in multiple languages (Portuguese, Spanish, French, German, Norwegian, English)
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
        # Auto-fill deliveryDate = orderDate if missing
        if "deliveryDate" not in args and "orderDate" in args:
            args["deliveryDate"] = args["orderDate"]
            log.info("AUTOFIX: set deliveryDate = %s", args["deliveryDate"])
        elif "deliveryDate" not in args:
            args["deliveryDate"] = date.today().isoformat()
            args.setdefault("orderDate", args["deliveryDate"])
            log.info("AUTOFIX: set deliveryDate + orderDate = %s", args["deliveryDate"])

    elif tool_name == "post_product":
        # CRITICAL: If task says "no VAT" → force vatType to id=6 (outside MVA)
        if _prompt_says_no_vat(_current_task_prompt):
            vat = args.get("vatType")
            vat_id = vat.get("id") if isinstance(vat, dict) else vat
            if vat_id != 6:
                args["vatType"] = {"id": 6}
                log.info("AUTOFIX: task says no-VAT → forced vatType={id:6} (was %s)", vat_id)

    elif tool_name == "search_product":
        # Reject comma-separated productNumber — only first value
        pn = args.get("productNumber", "")
        if isinstance(pn, str) and "," in pn:
            first = pn.split(",")[0].strip()
            log.warning("AUTOFIX: comma in productNumber '%s' → using '%s'", pn, first)
            args["productNumber"] = first

    elif tool_name == "post_ledger_voucher":
        # CRITICAL: Always send to ledger — without this, voucher exists but never posts
        if not args.get("sendToLedger"):
            args["sendToLedger"] = True
            log.info("AUTOFIX: set sendToLedger=true")
        # Validate postings exist and balance
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
                # Ensure each posting has date
                if "date" not in p:
                    p["date"] = voucher_date
                # CRITICAL: row must be >= 1 (row 0 = system-generated, causes rejection)
                if "row" not in p or p.get("row", 0) == 0:
                    p["row"] = i + 1
                    log.info("AUTOFIX: set row=%d on posting %d", i + 1, i)
                # amountGrossCurrency = amountGross ONLY when no foreign currency on posting
                if "amountGross" in p and "amountGrossCurrency" not in p:
                    if not p.get("currency") and not p.get("currency_id"):
                        p["amountGrossCurrency"] = p["amountGross"]
                        log.info("AUTOFIX: set amountGrossCurrency=%s on posting %d (NOK)",
                                 p["amountGross"], i)

    elif tool_name == "post_incomingInvoice":
        # Auto-fill externalId on orderLines if missing
        order_lines = args.get("orderLines", [])
        if isinstance(order_lines, list):
            for i, line in enumerate(order_lines):
                if isinstance(line, dict) and "externalId" not in line:
                    line["externalId"] = f"line{i+1}"
                    log.info("AUTOFIX: set externalId='line%d' on orderLine %d", i+1, i)

    elif tool_name == "post_travelExpense_cost":
        # Ensure amountCurrencyIncVat is set (API requires it, model often uses 'rate' instead)
        if "amountCurrencyIncVat" not in args:
            rate = args.get("rate") or args.get("amount", 0)
            count = args.get("count", 1) or 1
            amount = rate * count if rate else 0
            if amount:
                args["amountCurrencyIncVat"] = amount
                log.info("AUTOFIX: set amountCurrencyIncVat=%s (rate=%s * count=%s)", amount, rate, count)

    elif tool_name == "post_travelExpense":
        # Ensure travelDetails has required defaults
        td = args.get("travelDetails")
        if isinstance(td, dict):
            td.setdefault("isForeignTravel", False)
            td.setdefault("isDayTrip", False)
            td.setdefault("departureFrom", "Kontoret")
            args["travelDetails"] = td

    elif tool_name == "post_supplier":
        # Auto-copy email to invoiceEmail — competition checks both
        email = args.get("email", "")
        if email and not args.get("invoiceEmail"):
            args["invoiceEmail"] = email
            log.info("AUTOFIX: set invoiceEmail=%s", email)

    elif tool_name == "post_customer":
        # Auto-copy email to invoiceEmail — competition checks both
        email = args.get("email", "")
        if email and not args.get("invoiceEmail"):
            args["invoiceEmail"] = email
            log.info("AUTOFIX: set invoiceEmail=%s", email)

    elif tool_name == "post_employee":
        # Ensure valid userType — default "STANDARD" if missing/empty
        ut = args.get("userType", "")
        if not ut or ut == "0":
            args["userType"] = "STANDARD"
            log.info("AUTOFIX: set userType=STANDARD")

    elif tool_name == "search_invoice":
        # Fix: invoiceDateTo must be > invoiceDateFrom (exclusive end date)
        date_from = args.get("invoiceDateFrom", "")
        date_to = args.get("invoiceDateTo", "")
        if date_from and date_to and date_from >= date_to:
            try:
                dt = date.fromisoformat(date_to)
                args["invoiceDateTo"] = (dt + timedelta(days=1)).isoformat()
                log.info("AUTOFIX: invoiceDateTo bumped to %s (exclusive)", args["invoiceDateTo"])
            except ValueError:
                pass

    elif tool_name == "createReminder_invoice":
        # CRITICAL: API field is dispatchType, NOT sendType (prompt had wrong name)
        if "sendType" in args and "dispatchType" not in args:
            args["dispatchType"] = args.pop("sendType")
            log.info("AUTOFIX: renamed sendType → dispatchType")
        if not args.get("dispatchType"):
            args["dispatchType"] = "EMAIL"
            log.info("AUTOFIX: set dispatchType=EMAIL")

    elif tool_name == "invoice_order":
        # CRITICAL: always send to customer — tasks expect invoices to be delivered
        if not args.get("sendToCustomer"):
            args["sendToCustomer"] = True
        if not args.get("invoiceDate"):
            args["invoiceDate"] = date.today().isoformat()

    elif tool_name == "post_project":
        # CRITICAL: startDate is required, model often sends null
        if not args.get("startDate"):
            args["startDate"] = date.today().isoformat()
            log.info("AUTOFIX: set project startDate = %s", args["startDate"])

    elif tool_name == "post_employee_employment":
        # Default isMainEmployer and taxDeductionCode if not set
        if "isMainEmployer" not in args:
            args["isMainEmployer"] = True
            log.info("AUTOFIX: set isMainEmployer=true")
        if not args.get("taxDeductionCode"):
            args["taxDeductionCode"] = "loennFraHovedarbeidsgiver"
            log.info("AUTOFIX: set taxDeductionCode=loennFraHovedarbeidsgiver")

    elif tool_name == "post_employee_employment_details":
        # Default employmentType to ORDINARY if not set
        if "employmentType" not in args or not args.get("employmentType"):
            args["employmentType"] = "ORDINARY"
            log.info("AUTOFIX: set employmentType=ORDINARY")
        # Default remunerationType
        if "remunerationType" not in args or not args.get("remunerationType"):
            args["remunerationType"] = "MONTHLY_WAGE"
            log.info("AUTOFIX: set remunerationType=MONTHLY_WAGE")
        # Default workingHoursScheme
        if "workingHoursScheme" not in args or not args.get("workingHoursScheme"):
            args["workingHoursScheme"] = "NOT_SHIFT"
            log.info("AUTOFIX: set workingHoursScheme=NOT_SHIFT")

    return args


# ─── Structured response compaction ───

# Key fields to extract per entity type
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


def _extract_fields(obj: dict, fields: list) -> dict:
    """Extract specified fields from an object, including nested refs."""
    result = {}
    for f in fields:
        if f in obj:
            val = obj[f]
            # Compact nested refs to just id + name/displayName
            if isinstance(val, dict) and "id" in val:
                compact = {"id": val["id"]}
                for k in ("name", "displayName", "number", "firstName", "lastName"):
                    if k in val:
                        compact[k] = val[k]
                val = compact
            result[f] = val
    return result


def _guess_entity_type(endpoint: str) -> str | None:
    """Guess entity type from endpoint path."""
    # Action endpoints that return a DIFFERENT entity type than their base path
    if "/:invoice" in endpoint:
        return "invoice"  # /order/{id}/:invoice returns Invoice
    if "/:payment" in endpoint or "/:createCreditNote" in endpoint or "/:send" in endpoint:
        return "invoice"  # /invoice/{id}/:payment etc. returns Invoice

    # Strip path params and action suffixes
    clean = re.sub(r'/\d+', '', endpoint)
    clean = re.sub(r'/:[^/]+', '', clean)
    clean = clean.strip("/")
    # Map endpoint segments to entity types
    mapping = {
        "customer": "customer", "employee": "employee", "department": "department",
        "product": "product", "order": "order", "invoice": "invoice",
        "travelExpense": "travelExpense", "travelExpense/cost": "cost",
        "travelExpense/paymentType": "paymentType",
        "travelExpense/costCategory": "costCategory",
        "project": "project", "ledger/voucher": "voucher",
        "ledger/account": "account", "contact": "contact",
        "invoice/paymentType": "paymentType",
        "supplier": "supplier",
        "incomingInvoice": "invoice",
        "employee/employment": "employment",
        "employee/employment/details": "employmentDetails",
        "employee/employment/occupationCode": "occupationCode",
        "ledger/accountingDimensionName": "accountingDimensionName",
        "ledger/accountingDimensionName/search": "accountingDimensionName",
        "ledger/accountingDimensionValue": "accountingDimensionValue",
        "ledger/accountingDimensionValue/search": "accountingDimensionValue",
    }
    return mapping.get(clean)


def _post_filter_results(endpoint: str, params: dict | None, data: dict) -> dict:
    """Smart post-filtering: when a search returns too many results, auto-pick the best match."""
    if not isinstance(data, dict) or "values" not in data:
        return data
    values = data.get("values", [])
    if not values or len(values) <= 5:
        return data  # small result set, no filtering needed

    # Occupation code: match by code prefix or nameNO keyword
    if "occupationCode" in endpoint:
        code_q = (params or {}).get("code", "")
        name_q = (params or {}).get("nameNO", "").upper()
        if code_q:
            # Prefer exact prefix match on code
            exact = [v for v in values if isinstance(v, dict) and v.get("code", "").startswith(code_q)]
            if exact:
                data = dict(data)
                data["values"] = exact[:5]
                data["_filtered"] = f"Filtered {len(values)}→{len(exact)} by code prefix '{code_q}'"
                return data
        if name_q:
            # Keyword match on nameNO
            matched = [v for v in values if isinstance(v, dict) and name_q in v.get("nameNO", "").upper()]
            if matched:
                data = dict(data)
                data["values"] = matched[:5]
                data["_filtered"] = f"Filtered {len(values)}→{len(matched)} by nameNO '{name_q}'"
                return data
        # Still too many — just return first 5 with a hint
        data = dict(data)
        data["values"] = values[:5]
        data["_hint"] = f"Showing 5 of {len(values)}. Use more specific code or nameNO to narrow."
        return data

    return data


def _compact_response(endpoint: str, method: str, data, params: dict | None = None) -> str:
    """Convert API response to compact structured JSON for the model."""
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)[:2000]

    # Smart filtering for large result sets
    data = _post_filter_results(endpoint, params, data)

    entity_type = _guess_entity_type(endpoint)
    fields = _ENTITY_FIELDS.get(entity_type, [])

    # Single value response (POST/PUT)
    if "value" in data and isinstance(data["value"], dict):
        val = data["value"]
        if fields:
            compact = _extract_fields(val, fields)
        else:
            compact = val
        result = {"value": compact}
        # Allow more space for vouchers with postings (audit tasks need them)
        limit = 5000 if "postings" in (compact or {}) else 3000
        return json.dumps(result, ensure_ascii=False)[:limit]

    # List response (GET search)
    if "values" in data and isinstance(data["values"], list):
        values = data["values"]
        count = data.get("fullResultSize", len(values))
        if fields:
            compact_values = [_extract_fields(v, fields) for v in values[:20]]
        else:
            compact_values = values[:10]
        result = {"count": count, "values": compact_values}
        return json.dumps(result, ensure_ascii=False)[:3000]

    # Fallback
    raw = json.dumps(data, ensure_ascii=False)
    return raw[:2000]


def _log_task(entry: dict):
    try:
        entry["logged_at"] = datetime.utcnow().isoformat() + "Z"
        with open(TASK_LOG_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("Failed to write task log: %s", e)


# ─── Load typed tools from gen_tools_output.json ───

def _load_tools():
    tools_path = os.path.join(os.path.dirname(__file__), "gen_tools_output.json")
    with open(tools_path) as f:
        data = json.load(f)
    tools = data["tools"]
    # Add task_complete tool
    tools.append({
        "name": "task_complete",
        "description": "Signal that all API operations for this task are done.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "summary": {
                    "type": "STRING",
                    "description": "Brief summary of what was accomplished",
                }
            },
            "required": ["summary"],
        },
    })
    # Spec search tool
    tools.append({
        "name": "search_tripletex_spec",
        "description": "Search the Tripletex API specification to find available endpoints. Use this BEFORE tripletex_api to discover the correct endpoint path, method, parameters, and enum values. Returns matching operations from the OpenAPI spec.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Search keywords (e.g. 'employee employment salary', 'reminder invoice', 'accounting dimension')",
                },
                "method_filter": {
                    "type": "STRING",
                    "description": "Optional: filter by HTTP method (GET, POST, PUT, DELETE)",
                },
            },
            "required": ["query"],
        },
    })
    # Trial balance / account balances tool
    tools.append({
        "name": "get_account_balances",
        "description": "Get trial balance: sum of all postings per account for a date range. Use to calculate taxable income, verify posted amounts, or check account balances. Returns totals sorted by account number with revenue/expense summary.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "dateFrom": {
                    "type": "STRING",
                    "description": "Start date (YYYY-MM-DD)",
                },
                "dateTo": {
                    "type": "STRING",
                    "description": "End date (YYYY-MM-DD)",
                },
            },
            "required": ["dateFrom", "dateTo"],
        },
    })
    # Generic API tool
    tools.append({
        "name": "tripletex_api",
        "description": "Execute any Tripletex API call directly. Use search_tripletex_spec first to find the correct endpoint. This tool validates parameters against the OpenAPI spec before executing.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "method": {
                    "type": "STRING",
                    "description": "HTTP method: GET, POST, PUT, or DELETE",
                },
                "path": {
                    "type": "STRING",
                    "description": "API endpoint path (e.g. '/employee/employment', '/ledger/accountingDimensionName'). Include numeric IDs directly in path for specific resources (e.g. '/employee/employment/123').",
                },
                "query_params": {
                    "type": "OBJECT",
                    "description": "Query parameters as key-value pairs",
                },
                "body": {
                    "type": "OBJECT",
                    "description": "Request body for POST/PUT calls",
                },
            },
            "required": ["method", "path"],
        },
    })
    return tools


TOOLS = _load_tools()
log.info("Loaded %d typed tools", len(TOOLS))


# ─── Gemini API call ───

def _call_gemini(contents: list, api_key: str) -> dict:
    url = GEMINI_URL.format(GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "tools": [{"functionDeclarations": TOOLS}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
        },
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            if resp.status_code == 429 or resp.status_code >= 500:
                log.warning("Gemini %d on attempt %d, retrying...", resp.status_code, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                log.error("Gemini API error %d: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            log.warning("Gemini timeout on attempt %d", attempt + 1)
            if attempt == 2:
                raise
    raise RuntimeError("Gemini API failed after 3 attempts")


# ─── Tripletex API execution ───

def _exec_api(base_url: str, auth: tuple, method: str, endpoint: str,
              params: dict | None = None, body: dict | None = None) -> str:
    # Schema guard: validate body
    if body and method in ("POST", "PUT"):
        body, warnings = validate_and_sanitize(method, endpoint, params, body)
        for w in warnings:
            log.warning("GUARD: %s", w)

    url = f"{base_url}{endpoint}"
    log.info("%s %s params=%s body=%s", method, endpoint, params,
             json.dumps(body, ensure_ascii=False)[:200] if body else None)

    try:
        resp = requests.request(
            method, url, auth=auth,
            params=params,
            json=body if method in ("POST", "PUT") else None,
            timeout=60,
        )
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            data = resp.text

        if status >= 400:
            raw = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
            log.warning("API error %d: %s", status, raw[:500])
            return f"HTTP {status} ERROR: {raw[:2000]}"

        # Compact structured response
        result = _compact_response(endpoint, method, data, params)
        log.info("API %d OK (%d chars compact)", status, len(result))
        # Stash raw response for verifier (attached to the string)
        result = _ApiResult(result, data)
        return result
    except Exception as e:
        log.error("Request failed: %s", e)
        return f"REQUEST FAILED: {e}"


# ─── Bank account setup ───

def _ensure_bank_account(base_url: str, auth: tuple):
    """Ensure the company has a bank account on ledger account 1920.

    Many tasks (invoicing, payments) fail silently without this.
    """
    try:
        resp = requests.get(
            f"{base_url}/ledger/account",
            auth=auth,
            params={"number": "1920", "fields": "id,bankAccountNumber"},
            timeout=30,
        )
        if resp.status_code != 200:
            log.warning("Bank account lookup failed: %d", resp.status_code)
            return
        data = resp.json()
        values = data.get("values", [])
        if not values:
            log.warning("Ledger account 1920 not found")
            return
        acct = values[0]
        acct_id = acct.get("id")
        existing = acct.get("bankAccountNumber", "")
        if existing:
            log.info("Bank account 1920 already has number: %s", existing)
            return
        # Set a bank account number
        put_resp = requests.put(
            f"{base_url}/ledger/account/{acct_id}",
            auth=auth,
            json={"id": acct_id, "bankAccountNumber": "12345678903"},
            timeout=30,
        )
        if put_resp.status_code < 300:
            log.info("Set bank account number on 1920 (id=%s)", acct_id)
        else:
            log.warning("Failed to set bank account: %d %s", put_resp.status_code, put_resp.text[:200])
    except Exception as e:
        log.warning("Bank account setup error: %s", e)


# ─── Deterministic FX controller ───

def _parse_fx_from_prompt(prompt: str) -> dict | None:
    """Parse multi-currency payment details from prompt. Returns dict or None."""
    import re
    result = {}

    # Extract EUR amount — "1690 EUR", "1 690 EUR"
    m = re.search(r'(\d[\d\s]*\d)\s*EUR', prompt)
    if m:
        result["eur_amount"] = float(m.group(1).replace(" ", ""))
    else:
        return None  # Not an FX task

    # Extract old rate — "kursen var 11.66", "rate was 11.66", "taux était 11,66"
    m = re.search(r'(?:var|was|était|era|war|fue)\s+(\d+[.,]\d+)\s*(?:NOK|kr)', prompt, re.I)
    if m:
        result["old_rate"] = float(m.group(1).replace(",", "."))

    # Extract new rate — "er nå 11.00", "is now 11.00", "est maintenant"
    m = re.search(r'(?:nå|now|maintenant|agora|ahora|jetzt|faktisk)\s+(\d+[.,]\d+)\s*(?:NOK|kr)', prompt, re.I)
    if not m:
        m = re.search(r'(?:men|but|mais|pero|però|aber).*?(\d+[.,]\d+)\s*NOK', prompt, re.I)
    if m:
        result["new_rate"] = float(m.group(1).replace(",", "."))

    if "old_rate" in result and "new_rate" in result:
        return result
    return None


def _compute_fx_amounts(fx: dict, invoice_amount_outstanding: float, invoice_amount_currency: float) -> dict:
    """Compute exact FX payment amounts. No LLM math."""
    eur = fx["eur_amount"]
    new_rate = fx["new_rate"]
    old_rate = fx["old_rate"]

    paid_nok = eur * new_rate  # What we actually receive in NOK
    paid_currency = eur  # What the customer pays in EUR

    # Agio = rate difference × EUR amount (including VAT from invoice)
    # Use invoice_amount_currency for the full amount including VAT
    agio_amount = abs((new_rate - old_rate) * invoice_amount_currency)
    is_gain = new_rate > old_rate  # gain = agio, loss = disagio

    return {
        "paid_amount_nok": round(paid_nok, 2),
        "paid_amount_currency": round(paid_currency, 2),
        "agio_amount": round(agio_amount, 2),
        "is_gain": is_gain,
    }


# ─── Bank reconciliation controller ───

def _parse_csv_bank_statement(files: list) -> list | None:
    """Extract bank statement lines from CSV file attachment."""
    import csv
    for f in files:
        mime = f.get("mime_type", "")
        fname = f.get("filename", "")
        if not (mime.startswith("text/") or fname.endswith(".csv")):
            continue
        try:
            text = base64.b64decode(f["content_base64"]).decode("utf-8", errors="replace")
            lines = []
            reader = csv.DictReader(io.StringIO(text), delimiter=";")
            if not reader.fieldnames:
                reader = csv.DictReader(io.StringIO(text), delimiter=",")
            for row in reader:
                # Normalize keys
                entry = {}
                for k, v in row.items():
                    kl = (k or "").strip().lower()
                    entry[kl] = (v or "").strip()
                # Extract amount and reference
                amount = None
                # Try direct amount columns first
                for ak in ("beløp", "belop", "amount", "montant", "betrag", "importe", "valor"):
                    if ak in entry:
                        try:
                            amount = float(entry[ak].replace(",", ".").replace(" ", ""))
                        except ValueError:
                            pass
                        break
                # Try Inn/Ut columns (Norwegian bank statements)
                if amount is None:
                    for ak in ("inn", "ut", "in", "out", "credit", "debit", "crédit", "débit"):
                        if ak in entry and entry[ak]:
                            try:
                                val = float(entry[ak].replace(",", ".").replace(" ", ""))
                                if val != 0:
                                    # "Ut" (out) = negative, "Inn" (in) = positive
                                    amount = val if ak in ("inn", "in", "credit", "crédit") else -abs(val)
                                    break
                            except ValueError:
                                pass
                ref = ""
                for rk in ("referanse", "reference", "ref", "beskrivelse", "description", "tekst", "text"):
                    if rk in entry and entry[rk]:
                        ref = entry[rk]
                        break
                dt = ""
                for dk in ("dato", "date", "datum", "fecha", "data"):
                    if dk in entry and entry[dk]:
                        dt = entry[dk]
                        break
                if amount is not None:
                    lines.append({"amount": amount, "ref": ref, "date": dt, "raw": entry})
            if lines:
                return lines
        except Exception as e:
            log.warning("CSV parse error: %s", e)
    return None


# ─── Agent loop ───

def solve_task_sync(body: dict):
    global _current_task_prompt
    prompt = body.get("prompt", "")
    if not prompt:
        log.warning("Empty or missing prompt — returning early")
        return
    _current_task_prompt = prompt
    files = body.get("files", [])
    creds = body.get("tripletex_credentials") or body.get("credentials", {})
    if not creds:
        log.warning("Missing credentials — returning early")
        return
    base_url = creds.get("base_url", "")
    token = creds.get("session_token", "")
    auth = ("0", token)
    api_key = GOOGLE_API_KEY

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    # Bank account setup — required for invoice/payment tasks (422 without it)
    # Costs 1-2 proxy calls but prevents 0% on all invoice tasks
    _ensure_bank_account(base_url, auth)

    # Per-task search cache
    cache = _SearchCache()

    task_record = {
        "prompt": prompt,
        "files": [f.get("filename", "unknown") for f in files],
        "base_url": base_url,
        "api_calls": [],
        "errors": [],
        "turns": 0,
        "outcome": "unknown",
        "smoke": body.get("_smoke", False),
    }
    t0 = time.time()

    # Build user message
    parts = []
    for f in files:
        mime = f.get("mime_type", "application/octet-stream")
        if mime.startswith("image/") or mime == "application/pdf":
            parts.append({"inlineData": {"mimeType": mime, "data": f["content_base64"]}})
            # Extract text from PDFs so model has explicit data
            if mime == "application/pdf":
                pdf_text = _extract_pdf_text(f["content_base64"])
                if pdf_text and len(pdf_text) > 30:
                    parts.append({"text": f"[Extracted text from {f['filename']}]:\n{pdf_text[:5000]}\n\nIMPORTANT: Use the EXACT values from this document (amounts, dates, names, org numbers). Do NOT guess."})
                    log.info("PDF text extracted: %d chars from %s", len(pdf_text), f['filename'])
                else:
                    parts.append({"text": f"[Attached PDF: {f['filename']}] — Read ALL values from this document before making API calls."})
                    log.info("PDF text extraction returned little/no text for %s", f['filename'])
            else:
                parts.append({"text": f"[Attached file: {f['filename']}]"})
        else:
            try:
                text = base64.b64decode(f["content_base64"]).decode("utf-8", errors="replace")
                parts.append({"text": f"[File: {f['filename']}]\n{text[:5000]}"})
            except Exception:
                parts.append({"text": f"[Attached binary file: {f['filename']} ({mime})]"})

    today = date.today().isoformat()

    # Pre-parse CSV bank statements so the model has structured data
    csv_lines = _parse_csv_bank_statement(files)
    if csv_lines:
        csv_summary = f"[PARSED BANK STATEMENT: {len(csv_lines)} transactions]\n"
        for i, line in enumerate(csv_lines):
            sign = "IN" if line["amount"] > 0 else "OUT"
            csv_summary += f"  {i+1}. {sign} {abs(line['amount']):.2f} | date={line['date']} | ref={line['ref']}\n"
        csv_summary += "\nMatch IN (positive) amounts to customer invoices, OUT (negative) to supplier invoices.\n"
        parts.append({"text": csv_summary})
        log.info("CSV pre-parsed: %d bank statement lines", len(csv_lines))

    # Pre-compute FX amounts for multi-currency tasks (LLM gets math wrong 64% of the time)
    fx = _parse_fx_from_prompt(prompt)
    if fx:
        rate_diff = abs(fx['new_rate'] - fx['old_rate'])
        direction = "gain (agio → credit 8060)" if fx['new_rate'] > fx['old_rate'] else "loss (disagio → debit 8160)"
        fx_hint = (f"[SYSTEM: PRE-COMPUTED FX VALUES — USE THESE EXACT NUMBERS]\n"
                   f"Foreign currency amount from prompt: {fx['eur_amount']} EUR (excl VAT)\n"
                   f"Old exchange rate: {fx['old_rate']} NOK/EUR\n"
                   f"New exchange rate: {fx['new_rate']} NOK/EUR\n"
                   f"Rate difference: {rate_diff:.4f} NOK/EUR → {direction}\n\n"
                   f"PAYMENT STEP:\n"
                   f"→ First search_invoice to find the actual invoice. Read amountOutstanding from response.\n"
                   f"→ paidAmountCurrency = amountOutstanding (the full invoice amount in invoice currency)\n"
                   f"→ paidAmount = amountOutstanding × {fx['new_rate']} (convert to NOK at new rate)\n"
                   f"→ If invoice currency is already NOK: just use paidAmount = amountOutstanding (no conversion)\n\n"
                   f"AGIO STEP:\n"
                   f"→ agioAmount = {rate_diff:.4f} × amountOutstanding_in_foreign_currency\n"
                   f"→ The amountOutstanding from the invoice response IS the foreign currency total (incl VAT)\n"
                   f"→ Do NOT use {fx['eur_amount']} (that's excl VAT from prompt). Use the invoice's actual total.\n"
                   f"→ Post voucher: debit 1500 (with customer_id), credit 8060 (or debit 8160 for loss)")
        parts.append({"text": fx_hint})
        log.info("FX pre-computed: EUR=%s old=%s new=%s", fx['eur_amount'], fx['old_rate'], fx['new_rate'])

    # Detect no-VAT keywords and inject explicit instruction
    if _prompt_says_no_vat(prompt):
        parts.append({"text": (
            "[SYSTEM: NO-VAT DETECTED]\n"
            "The task explicitly says this is WITHOUT VAT / tax-exempt.\n"
            "When creating the product, use vatType: {id: 6} (Utenfor avgiftsområdet / outside MVA).\n"
            "Do NOT use vatType id 3 (25% MVA). The invoice amount must equal the stated amount EXACTLY, with zero VAT.\n"
        )})
        log.info("NO-VAT hint injected for prompt: %s", prompt[:80])

    # Detect travel expense with diett/per diem — inject costCategory hint
    prompt_lower = prompt.lower()
    _diett_keywords = ["diett", "per diem", "dagsats", "dieta", "ajuda de custo", "tagesgeld",
                        "indemnité journalière", "dietas"]
    if any(kw in prompt_lower for kw in ["reiseregning", "travel expense", "reisekosten",
                                          "despesa de viagem", "gastos de viaje", "frais de voyage"]):
        diett_hint_parts = []
        if any(kw in prompt_lower for kw in _diett_keywords):
            diett_hint_parts.append(
                "DIETT/PER DIEM: Search costCategory with description='Kost' (NOT 'Reisekostnad'). "
                "Create a SEPARATE cost line for diett with the 'Kost' category."
            )
        diett_hint_parts.append(
            "TRAVEL EXPENSE EFFICIENCY: Search costCategory ONCE for each type (e.g. 'Fly', 'Taxi', 'Kost'). "
            "Search paymentType ONCE with description='Privat'. Do NOT repeat searches."
        )
        parts.append({"text": "[SYSTEM: TRAVEL EXPENSE HINTS]\n" + "\n".join(diett_hint_parts)})
        log.info("Travel expense hint injected")

    # Detect receipt/kvittering voucher tasks — inject VAT handling reminder
    if any(kw in prompt_lower for kw in ["quittung", "kvittering", "receipt", "recibo", "reçu"]):
        parts.append({"text": (
            "[SYSTEM: RECEIPT/EXPENSE VOUCHER]\n"
            "When posting a voucher from a receipt, check if VAT handling is required.\n"
            "If the task says 'correct VAT treatment' (korrekte MwSt/MVA), you MUST set vatType_id on the expense posting.\n"
            "For food/catering: vatType_id=11 (15% input VAT). For most other expenses: vatType_id=1 (25% input VAT).\n"
            "When vatType_id is set, use the GROSS amount (including VAT) as amountGross. The system auto-generates the VAT line.\n"
            "If the account is locked to MVA 0, try a different expense account or post without vatType and add a manual VAT line on 2710.\n"
        )})
        log.info("Receipt VAT hint injected")

    parts.append({"text": f"Today's date: {today}\n\nComplete this accounting task:\n\n{prompt}"})
    contents = [{"role": "user", "parts": parts}]

    # Agent loop
    consecutive_errors = 0
    last_error_sig = None
    auth_failed = False

    for turn in range(MAX_TURNS):
        log.info("Agent turn %d/%d", turn + 1, MAX_TURNS)
        result = _call_gemini(contents, api_key)

        candidates = result.get("candidates", [])
        if not candidates:
            log.error("No candidates: %s", json.dumps(result)[:500])
            if turn < 2:
                # Retry once — Gemini sometimes returns empty on first call
                contents.append({"role": "user", "parts": [{"text": "Please begin executing the task using the available tools."}]})
                continue
            break

        content = candidates[0].get("content", {})
        model_parts = content.get("parts", [])
        if not model_parts:
            if turn < 2:
                contents.append({"role": "user", "parts": [{"text": "Please begin executing the task using the available tools."}]})
                continue
            break

        contents.append({"role": "model", "parts": model_parts})

        fn_calls = [p for p in model_parts if "functionCall" in p]
        if not fn_calls:
            if turn < 4:
                nudge = ("You MUST use tools. Do NOT respond with only text. "
                         "If you cannot do the full task, do as much as possible — "
                         "create vouchers, search accounts, create entities. "
                         "Partial work earns partial credit. Use tools NOW.")
                contents.append({"role": "user", "parts": [{"text": nudge}]})
                continue
            break

        fn_responses = []
        done = False
        turn_had_error = False

        for fc_part in fn_calls:
            fc = fc_part["functionCall"]
            tool_name = fc["name"]
            args = fc.get("args", {})

            if tool_name == "task_complete":
                log.info("Task complete: %s", args.get("summary", ""))
                done = True
                task_record["outcome"] = "completed"
                task_record["summary"] = args.get("summary", "")
                fn_responses.append({
                    "functionResponse": {"name": tool_name, "response": {"result": "OK"}}
                })
                continue

            if tool_name == "submit_plan":
                log.info("Plan submitted: %s — %s", args.get("task_type", ""), args.get("steps", "")[:200])
                fn_responses.append({
                    "functionResponse": {"name": tool_name, "response": {"result": "Plan approved. Execute now."}}
                })
                continue

            if tool_name == "search_tripletex_spec":
                query = args.get("query", "")
                method_filter = args.get("method_filter")
                results = search_spec(query, method_filter)
                api_result = json.dumps(results, ensure_ascii=False)[:3000]
                fn_responses.append({
                    "functionResponse": {"name": tool_name, "response": {"result": api_result}}
                })
                continue

            if tool_name == "get_account_balances":
                date_from = args.get("dateFrom", "2025-01-01")
                date_to = args.get("dateTo", date.today().isoformat())
                log.info("get_account_balances %s to %s", date_from, date_to)
                postings_resp = _exec_api(
                    base_url, auth, "GET", "/ledger/posting",
                    {"dateFrom": date_from, "dateTo": date_to, "count": "10000",
                     "fields": "account(id,number,name),amountGross"},
                    None,
                )
                if "ERROR" in postings_resp:
                    api_result = postings_resp
                else:
                    try:
                        raw_data = getattr(postings_resp, 'raw_data', None)
                        if raw_data and isinstance(raw_data, dict):
                            values = raw_data.get("values", [])
                        else:
                            values = json.loads(postings_resp).get("values", [])
                        balances = {}
                        for v in values:
                            acct = v.get("account", {})
                            num = acct.get("number", 0)
                            name = acct.get("name", "")
                            key = (num, name)
                            balances[key] = balances.get(key, 0) + (v.get("amountGross", 0) or 0)
                        sorted_bals = sorted(balances.items(), key=lambda x: x[0][0])
                        lines = [f"Trial balance ({date_from} to {date_to}):"]
                        total_revenue = 0
                        total_expense = 0
                        for (num, name), bal in sorted_bals:
                            if abs(bal) > 0.01:
                                lines.append(f"  {num} {name}: {bal:.2f}")
                                if 3000 <= num < 4000:
                                    total_revenue += bal  # negative = credit
                                elif 4000 <= num < 9000:
                                    total_expense += bal  # positive = debit
                        taxable_income = (-total_revenue) - total_expense
                        lines.append(f"\nRevenue accounts (3xxx) net: {total_revenue:.2f} (credit=negative)")
                        lines.append(f"Expense accounts (4xxx-8xxx) net: {total_expense:.2f} (debit=positive)")
                        lines.append(f"Taxable income (revenue - expenses): {taxable_income:.2f}")
                        lines.append(f"Tax at 22%: {taxable_income * 0.22:.2f}")
                        api_result = "\n".join(lines)
                    except Exception as e:
                        log.error("Balance computation error: %s", e)
                        api_result = postings_resp
                task_record["api_calls"].append({
                    "tool": tool_name, "method": "GET", "endpoint": "/ledger/posting",
                    "params": {"dateFrom": date_from, "dateTo": date_to},
                    "body": None, "result_snippet": api_result[:300],
                    "result_full": None, "raw_response": None,
                })
                fn_responses.append({
                    "functionResponse": {"name": tool_name, "response": {"result": api_result}}
                })
                continue

            if tool_name == "tripletex_api":
                method = args.get("method", "GET")
                path = args.get("path", "")
                qp = args.get("query_params") or {}
                req_body = args.get("body")

                # Fix invalid {*} field expansion syntax
                if isinstance(qp, dict) and "fields" in qp:
                    original = qp["fields"]
                    qp["fields"] = _fix_fields_syntax(original)
                    if qp["fields"] != original:
                        log.info("AUTOFIX: fields '%s' → '%s'", original, qp["fields"])

                # Auto-fix activity creation: activityType is required
                if method == "POST" and path == "/activity" and isinstance(req_body, dict):
                    if "activityType" not in req_body:
                        req_body["activityType"] = "PROJECT_GENERAL_ACTIVITY"
                        log.info("AUTOFIX: set activityType=PROJECT_GENERAL_ACTIVITY")

                # Validate against spec
                is_valid, warnings, req_body = validate_generic_call(method, path, qp, req_body)
                for w in warnings:
                    log.warning("GENERIC_API: %s", w)

                if not is_valid:
                    api_result = f"VALIDATION ERROR: {'; '.join(warnings)}"
                else:
                    # Check cache for GET requests
                    cached = cache.get(method, path, qp)
                    if cached is not None:
                        api_result = cached
                        log.info("CACHE HIT: %s %s", method, path)
                    else:
                        api_result = _exec_api(base_url, auth, method, path, qp if qp else None, req_body)
                        cache.put(method, path, qp, api_result)

                raw = getattr(api_result, 'raw_data', None)
                call_log = {
                    "tool": tool_name, "method": method, "endpoint": path,
                    "params": qp, "body": req_body,
                    "result_snippet": api_result[:300],
                    "result_full": api_result if "ERROR" in api_result else None,
                    "raw_response": raw,
                }
                if "ERROR" in api_result:
                    call_log["error"] = True
                    task_record["errors"].append(f"tripletex_api → {method} {path}: {api_result}")
                    turn_had_error = True
                    if "HTTP 401" in api_result or ("HTTP 403" in api_result and "expired" in api_result.lower()):
                        auth_failed = True
                    error_sig = f"tripletex_api:{path}:{api_result[:30]}"
                    if error_sig == last_error_sig:
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 1
                        last_error_sig = error_sig
                else:
                    consecutive_errors = 0
                    last_error_sig = None

                task_record["api_calls"].append(call_log)
                fn_responses.append({
                    "functionResponse": {"name": tool_name, "response": {"result": api_result}}
                })
                continue

            # Runtime autofixes then route
            args = _pre_validate(tool_name, args)
            method, endpoint, params, req_body = route_tool_call(tool_name, args)
            if method is None:
                fn_responses.append({
                    "functionResponse": {"name": tool_name, "response": {"error": f"Unknown tool: {tool_name}"}}
                })
                continue

            # Check cache for GET requests
            cached = cache.get(method, endpoint, params)
            if cached is not None:
                api_result = cached
                log.info("CACHE HIT: %s %s", method, endpoint)
            else:
                api_result = _exec_api(base_url, auth, method, endpoint, params, req_body)
                cache.put(method, endpoint, params, api_result)

            # Auto-recovery: cost category search returned 0 → list all categories
            if (tool_name == "search_travelExpense_costCategory"
                    and "ERROR" not in api_result
                    and '"count": 0' in api_result):
                log.info("AUTOFIX: costCategory search empty — listing all categories")
                all_cats = _exec_api(
                    base_url, auth, "GET", "/travelExpense/costCategory",
                    {"fields": "id,description,displayName", "count": "50"},
                    None,
                )
                if "ERROR" not in all_cats:
                    api_result = all_cats
                    log.info("AUTOFIX: returning all %s categories for model to pick", all_cats[:50])


            # Auto-recovery: occupation code search returned 0 → broaden search
            if (tool_name == "search_employee_employment_occupationCode"
                    and "ERROR" not in api_result
                    and '"count": 0' in api_result):
                code_q = (params or {}).get("code", "")
                name_q = (params or {}).get("nameNO", "")
                if code_q and len(code_q) <= 4:
                    # 4-digit STYRK is just a category — try with wildcard by searching broader
                    log.info("AUTOFIX: occupation code '%s' (short) returned 0 — broadening with count=100", code_q)
                    broad = _exec_api(
                        base_url, auth, "GET", "/employee/employment/occupationCode",
                        {"code": code_q, "fields": "id,code,nameNO", "count": "100"},
                        None,
                    )
                    if "ERROR" not in broad and '"count": 0' not in broad:
                        api_result = broad
                    else:
                        # Still 0 — try just first 2 digits
                        log.info("AUTOFIX: still 0 — trying 2-digit prefix '%s'", code_q[:2])
                        broad2 = _exec_api(
                            base_url, auth, "GET", "/employee/employment/occupationCode",
                            {"code": code_q[:2], "fields": "id,code,nameNO", "count": "50"},
                            None,
                        )
                        if "ERROR" not in broad2 and '"count": 0' not in broad2:
                            api_result = broad2
                elif not code_q and name_q:
                    # nameNO search returned 0, try broader
                    log.info("AUTOFIX: occupation nameNO '%s' returned 0 — trying count=100", name_q)
                    broad = _exec_api(
                        base_url, auth, "GET", "/employee/employment/occupationCode",
                        {"nameNO": name_q, "fields": "id,code,nameNO", "count": "100"},
                        None,
                    )
                    if "ERROR" not in broad and '"count": 0' not in broad:
                        api_result = broad

            # Auto-recovery: voucher rejected because account locked to vatType 0
            # → rebuild with manual 3-posting approach (separate VAT line)
            if (tool_name == "post_ledger_voucher"
                    and "HTTP 422" in api_result
                    and "låst til mva-kode 0" in api_result):
                log.info("AUTOFIX: account locked to vatType 0 — rebuilding with manual VAT postings")
                postings = (req_body or {}).get("postings", [])
                if postings:
                    new_postings = []
                    row = 1
                    for p in postings:
                        if not isinstance(p, dict):
                            continue
                        vat = p.get("vatType")
                        vat_id = vat.get("id") if isinstance(vat, dict) else vat
                        amt = p.get("amountGross", 0)
                        # If this posting has VAT and positive (expense), split into expense + VAT
                        if vat_id in (1, 11) and amt > 0:
                            # 25% incoming VAT — split: expense=amt/1.25, vat=amt-expense
                            excl = round(amt / 1.25, 2)
                            vat_amt = round(amt - excl, 2)
                            # Expense posting without VAT
                            exp_posting = dict(p)
                            exp_posting["amountGross"] = excl
                            exp_posting["amountGrossCurrency"] = excl
                            exp_posting.pop("vatType", None)
                            exp_posting["row"] = row
                            new_postings.append(exp_posting)
                            row += 1
                            # VAT posting on account 2710
                            vat_posting = {
                                "account": {"id": None},  # Will be filled
                                "amountGross": vat_amt,
                                "amountGrossCurrency": vat_amt,
                                "date": p.get("date", req_body.get("date")),
                                "row": row,
                            }
                            # Find 2710 account
                            acct_2710 = _exec_api(base_url, auth, "GET", "/ledger/account",
                                                  {"number": "2710", "fields": "id,number"}, None)
                            if "ERROR" not in acct_2710:
                                try:
                                    raw_2710 = getattr(acct_2710, 'raw_data', None)
                                    vals = (raw_2710 or json.loads(acct_2710)).get("values", [])
                                    if vals:
                                        vat_posting["account"] = {"id": vals[0]["id"]}
                                except Exception:
                                    pass
                            if vat_posting["account"]["id"]:
                                new_postings.append(vat_posting)
                                row += 1
                            else:
                                log.warning("AUTOFIX: could not find account 2710 for VAT posting")
                        else:
                            # Credit posting (supplier) — adjust to full amount incl VAT
                            cp = dict(p)
                            cp.pop("vatType", None)
                            cp["row"] = row
                            new_postings.append(cp)
                            row += 1
                    if new_postings:
                        new_body = dict(req_body)
                        new_body["postings"] = new_postings
                        log.info("AUTOFIX: retrying voucher with %d manual postings", len(new_postings))
                        retry_result = _exec_api(base_url, auth, "POST", "/ledger/voucher",
                                                 {"sendToLedger": "true"}, new_body)
                        if "ERROR" not in retry_result:
                            api_result = retry_result
                            log.info("AUTOFIX: voucher succeeded with manual VAT split")

            # Auto-recovery: product number conflict → search existing product
            if (tool_name == "post_product"
                    and "HTTP 422" in api_result
                    and "er i bruk" in api_result):
                number = (req_body or {}).get("number", "")
                if number:
                    log.info("AUTOFIX: product number '%s' in use — searching existing", number)
                    search_result = _exec_api(
                        base_url, auth, "GET", "/product",
                        {"number": str(number), "fields": "id,name,number,priceExcludingVatCurrency,vatType(id,name)"},
                        None,
                    )
                    if "ERROR" not in search_result:
                        api_result = search_result
                        log.info("AUTOFIX: found existing product by number")

            # Auto-recovery: employee email conflict → search by that email
            if (tool_name == "post_employee"
                    and "HTTP 422" in api_result
                    and "allerede en bruker med denne e-postadressen" in api_result):
                email = (req_body or {}).get("email", "")
                if email:
                    log.info("AUTOFIX: email conflict for '%s' — searching existing employee", email)
                    search_result = _exec_api(
                        base_url, auth, "GET", "/employee",
                        {"email": email, "fields": "id,firstName,lastName,email,employeeNumber,dateOfBirth,department(id,name)"},
                        None,
                    )
                    if "ERROR" not in search_result:
                        api_result = search_result
                        log.info("AUTOFIX: found existing employee by email")

            # Auto-recovery: post_project missing projectManager → find default employee and retry
            if (tool_name == "post_project"
                    and "HTTP 422" in api_result
                    and ("Prosjektleder" in api_result or "projectManager" in api_result.lower())):
                log.info("AUTOFIX: project missing projectManager — searching for default employee")
                emp_result = _exec_api(
                    base_url, auth, "GET", "/employee",
                    {"count": "1", "fields": "id,firstName,lastName"},
                    None,
                )
                if "ERROR" not in emp_result:
                    try:
                        raw_emp = getattr(emp_result, 'raw_data', None)
                        if raw_emp:
                            emp_values = raw_emp.get("values", [])
                        else:
                            emp_values = json.loads(emp_result).get("values", [])
                        if emp_values:
                            emp_id = emp_values[0]["id"]
                            req_body = req_body or {}
                            req_body["projectManager"] = {"id": emp_id}
                            if "startDate" not in req_body:
                                req_body["startDate"] = date.today().isoformat()
                            log.info("AUTOFIX: retrying post_project with projectManager=%s", emp_id)
                            retry_result = _exec_api(base_url, auth, "POST", "/project", None, req_body)
                            if "ERROR" not in retry_result:
                                api_result = retry_result
                                log.info("AUTOFIX: project created successfully on retry")
                    except Exception as e:
                        log.warning("AUTOFIX: project recovery failed: %s", e)

            raw = getattr(api_result, 'raw_data', None)
            call_log = {
                "tool": tool_name, "method": method, "endpoint": endpoint,
                "params": params, "body": req_body,
                "result_snippet": api_result[:300],
                "result_full": api_result if "ERROR" in api_result else None,
                "raw_response": raw,
            }
            if "ERROR" in api_result:
                call_log["error"] = True
                task_record["errors"].append(f"{tool_name} → {method} {endpoint}: {api_result}")
                turn_had_error = True

                # Detect 401 auth failure — unrecoverable
                if "HTTP 401" in api_result:
                    auth_failed = True

                # Detect repeated errors (same tool+endpoint+status)
                error_sig = f"{tool_name}:{endpoint}:{api_result[:30]}"
                if error_sig == last_error_sig:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 1
                    last_error_sig = error_sig
            else:
                consecutive_errors = 0
                last_error_sig = None

            task_record["api_calls"].append(call_log)

            fn_responses.append({
                "functionResponse": {"name": tool_name, "response": {"result": api_result}}
            })

        contents.append({"role": "user", "parts": fn_responses})

        # Force execution if stuck in spec-search/planning loop
        if turn >= 4 and len(task_record["api_calls"]) == 0:
            log.warning("Turn %d with 0 API calls — forcing execution", turn)
            contents.append({"role": "user", "parts": [
                {"text": "SYSTEM: You have made ZERO Tripletex API calls after multiple turns. STOP searching specs and planning. START executing API calls NOW. Use the typed tools (post_customer, post_employee, search_invoice, etc.) to make real API calls. Every turn without an API call is wasted."}
            ]})

        # Bail on unrecoverable auth failure
        if auth_failed:
            log.error("Auth failure (401) — aborting agent loop")
            contents.append({"role": "user", "parts": [
                {"text": "SYSTEM: Authentication failed (401 Unauthorized). The API credentials are invalid. Call task_complete now and report the auth failure."}
            ]})
            # Give Gemini one more turn to call task_complete
            result = _call_gemini(contents, api_key)
            break

        # Bail on repeated identical errors (likely stuck in a loop)
        if consecutive_errors >= 2:
            log.warning("Same error repeated %d times — injecting guidance", consecutive_errors)
            contents.append({"role": "user", "parts": [
                {"text": f"SYSTEM: The same API error has occurred {consecutive_errors} times in a row. This approach is not working. Either try a completely different approach or call task_complete to report partial progress. Do NOT retry the same call."}
            ]})
            consecutive_errors = 0  # Reset so we don't inject every turn

        if done:
            break

    task_record["turns"] = min(turn + 1, MAX_TURNS)
    task_record["elapsed_s"] = round(time.time() - t0, 1)
    if task_record["outcome"] == "unknown":
        task_record["outcome"] = "no_completion_signal"
    _log_task(task_record)
    log.info("Agent finished after %d turns (%.1fs)", task_record["turns"], task_record["elapsed_s"])
    return task_record
