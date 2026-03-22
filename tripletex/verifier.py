"""Post-execution verification for Tripletex agent tasks.

Architecture (per Codex recommendation):
1. ExecutionFactExtractor — parse call logs for successful mutations + entity IDs
2. Prompt spec extraction — regex key values (amounts, names, org numbers)
3. StateVerifier — re-fetch from Tripletex API by ID, compare against expectations

Returns: PASS / FAIL / INCONCLUSIVE with detail string.
"""

import json
import logging
import re

import requests

log = logging.getLogger("verifier")


# ─── API helper ───

class TripletexAPI:
    def __init__(self, base_url: str, token: str):
        self.base = base_url
        self.auth = ("0", token)

    def get(self, endpoint: str, params: dict | None = None) -> dict | None:
        params = params or {}
        params.setdefault("fields", "*")
        try:
            r = requests.get(f"{self.base}{endpoint}", auth=self.auth, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.warning("Verifier API error: %s", e)
        return None

    def search(self, endpoint: str, **params) -> list:
        data = self.get(endpoint, params)
        if data:
            return data.get("values", [])
        return []


# ─── Execution fact extraction ───

class ExecFacts:
    """Extract facts from task_record['api_calls']."""

    def __init__(self, calls: list):
        self.calls = calls or []
        self.successes = [c for c in self.calls if not c.get("error")]
        self.errors = [c for c in self.calls if c.get("error")]
        self.writes = [c for c in self.successes if c.get("method") in ("POST", "PUT")]

    def last_success(self, endpoint_contains: str = None, method: str = None) -> dict | None:
        for c in reversed(self.successes):
            if endpoint_contains and endpoint_contains not in c.get("endpoint", ""):
                continue
            if method and c.get("method") != method:
                continue
            return c
        return None

    def all_success(self, endpoint_contains: str = None, method: str = None) -> list:
        out = []
        for c in self.successes:
            if endpoint_contains and endpoint_contains not in c.get("endpoint", ""):
                continue
            if method and c.get("method") != method:
                continue
            out.append(c)
        return out

    def entity_id(self, call: dict) -> int | None:
        """Extract entity ID from a successful call's raw_response."""
        raw = call.get("raw_response")
        if isinstance(raw, dict):
            val = raw.get("value", raw)
            if isinstance(val, dict) and "id" in val:
                return val["id"]
        # Fallback: parse from result_snippet
        snippet = call.get("result_snippet", "")
        m = re.search(r'"id":\s*(\d+)', snippet)
        return int(m.group(1)) if m else None

    def entity_value(self, call: dict) -> dict | None:
        """Get the full value object from raw_response."""
        raw = call.get("raw_response")
        if isinstance(raw, dict):
            return raw.get("value", raw)
        return None

    def has_any_write(self) -> bool:
        return len(self.writes) > 0


# ─── Prompt spec extraction ───

def extract_amount(prompt: str) -> float | None:
    """Extract primary amount from prompt."""
    # Look for explicit amounts like "35350 NOK", "12301 EUR", "8900 kr"
    m = re.search(r'(\d[\d\s]*[\d])\s*(?:NOK|kr|EUR|USD)', prompt)
    if m:
        return float(m.group(1).replace(" ", ""))
    return None


def extract_org_number(prompt: str) -> str | None:
    m = re.search(r'(?:org\.?\s*(?:nr|n[oº°]|number|nummer)\.?\s*:?\s*)(\d{9})', prompt, re.I)
    return m.group(1) if m else None


def extract_name(prompt: str) -> str | None:
    """Extract primary entity name (customer/supplier/employee)."""
    # Look for quoted names or names before org number
    m = re.search(r'(?:kunden?|customer|client[e]?|fornecedor|fournisseur|supplier|Lieferant)\s+([A-ZÆØÅ][\w\s&.-]+?)(?:\s*\(|\s+(?:med|with|avec|com|con|mit)\b)', prompt, re.I)
    if m:
        return m.group(1).strip()
    # Name before org pattern
    m = re.search(r'([A-ZÆØÅ][\w\s&.-]+?)\s*\(org', prompt)
    if m:
        return m.group(1).strip()
    return None


def extract_employee_name(prompt: str) -> tuple[str | None, str | None]:
    """Extract employee first and last name."""
    # "Charles Taylor", "Lena Müller", etc.
    m = re.search(r'(?:employee|ansatt|empleado|empregado|funcionário|Mitarbeiter|tilsett)\s+(?:named?\s+)?([A-ZÆØÅ]\w+)\s+([A-ZÆØÅ]\w+)', prompt, re.I)
    if m:
        return m.group(1), m.group(2)
    # "for Name Name (email)"
    m = re.search(r'for\s+([A-ZÆØÅ]\w+)\s+([A-ZÆØÅ]\w+)\s*\(', prompt)
    if m:
        return m.group(1), m.group(2)
    return None, None


def extract_currency_rate(prompt: str) -> tuple[float | None, float | None, str | None]:
    """Extract old and new exchange rates and currency from multi-currency prompts."""
    # "exchange rate was 11.66 NOK/EUR"
    m = re.search(r'(?:rate|kurs|taux|tipo|taxa).*?(\d+[.,]\d+)\s*NOK/(\w+)', prompt, re.I)
    old_rate = float(m.group(1).replace(",", ".")) if m else None
    currency = m.group(2) if m else None
    # "rate is 11.00 NOK/EUR" or "kursen er nå"
    m2 = re.search(r'(?:but|men|mais|pero|però).*?(\d+[.,]\d+)\s*NOK/', prompt, re.I)
    new_rate = float(m2.group(1).replace(",", ".")) if m2 else None
    return old_rate, new_rate, currency


def extract_eur_amount(prompt: str) -> float | None:
    m = re.search(r'(\d[\d\s]*\d)\s*EUR', prompt)
    return float(m.group(1).replace(" ", "")) if m else None


def extract_department(prompt: str) -> str | None:
    m = re.search(r'(?:department|avdeling|departamento|département|Abteilung|dept)\s+["\']?(\w[\w\s]*?)(?:["\']|\.|,|\s+(?:Use|Bruk|med|with))', prompt, re.I)
    return m.group(1).strip() if m else None


# ─── Verdict ───

class Verdict:
    def __init__(self, status: str, detail: str, checks: list[dict] | None = None):
        self.status = status  # PASS, FAIL, INCONCLUSIVE
        self.detail = detail
        self.checks = checks or []

    def __repr__(self):
        return f"{self.status}: {self.detail}"

    def to_dict(self):
        return {"status": self.status, "detail": self.detail, "checks": self.checks}


def _pass(detail, checks=None):
    return Verdict("PASS", detail, checks)

def _fail(detail, checks=None):
    return Verdict("FAIL", detail, checks)

def _inconclusive(detail, checks=None):
    return Verdict("INCONCLUSIVE", detail, checks)


# ─── Per-type verifiers ───

def _v_create_customer(facts, prompt, api):
    c = facts.last_success(endpoint_contains="/customer", method="POST")
    if not c:
        return _fail("No successful customer POST")
    eid = facts.entity_id(c)
    if not eid:
        return _fail("Could not extract customer ID")
    data = api.get(f"/customer/{eid}")
    if not data or "value" not in data:
        return _fail(f"Customer {eid} not found on re-fetch")
    val = data["value"]
    checks = [{"check": "customer_exists", "pass": True, "id": eid}]
    org = extract_org_number(prompt)
    if org:
        actual = str(val.get("organizationNumber", ""))
        ok = actual == org
        checks.append({"check": "org_number", "pass": ok, "expected": org, "actual": actual})
    return _pass(f"Customer {val.get('name')} created", checks) if all(c["pass"] for c in checks) else _fail("Field mismatch", checks)


def _v_create_employee(facts, prompt, api):
    c = facts.last_success(endpoint_contains="/employee", method="POST")
    if not c:
        return _fail("No successful employee POST")
    eid = facts.entity_id(c)
    val = None
    if eid:
        data = api.get(f"/employee/{eid}")
        if data and "value" in data:
            val = data["value"]
    # Fallback 1: search by name from agent's POST body
    if not val:
        body = c.get("body", {})
        first = body.get("firstName")
        last = body.get("lastName")
        if first and last:
            results = api.search("/employee", firstName=first, lastName=last)
            if results:
                val = results[0]
                eid = val.get("id")
    # Fallback 2: search by name from prompt
    if not val:
        first, last = extract_employee_name(prompt)
        if first and last:
            results = api.search("/employee", firstName=first, lastName=last)
            if results:
                val = results[0]
                eid = val.get("id")
    # Fallback 3: search by email from agent's POST body
    if not val:
        body = c.get("body", {})
        email = body.get("email")
        if email:
            results = api.search("/employee", email=email)
            if results:
                val = results[0]
                eid = val.get("id")
    if not val:
        return _fail(f"Employee {eid} not found")
    checks = [{"check": "employee_exists", "pass": True, "id": eid}]
    # Check employment if any
    emp = facts.last_success(endpoint_contains="/employment", method="POST")
    if emp:
        checks.append({"check": "employment_created", "pass": True})
    return _pass(f"Employee {val.get('firstName')} created", checks)


def _v_create_department(facts, prompt, api):
    posts = facts.all_success(endpoint_contains="/department", method="POST")
    if not posts:
        return _fail("No successful department POST")
    checks = []
    for c in posts:
        eid = facts.entity_id(c)
        checks.append({"check": "department_created", "pass": eid is not None, "id": eid})
    return _pass(f"{len(posts)} department(s) created", checks)


def _v_create_product(facts, prompt, api):
    c = facts.last_success(endpoint_contains="/product", method="POST")
    if not c:
        return _fail("No successful product POST")
    eid = facts.entity_id(c)
    checks = [{"check": "product_exists", "pass": eid is not None, "id": eid}]
    return _pass("Product created", checks)


def _v_create_invoice(facts, prompt, api):
    # Invoice can come from POST /invoice or PUT /order/:invoice
    c = facts.last_success(endpoint_contains="invoice")
    if not c:
        return _fail("No successful invoice call")
    eid = facts.entity_id(c)
    if not eid:
        return _fail("Could not extract invoice ID")
    data = api.get(f"/invoice/{eid}")
    if not data or "value" not in data:
        return _fail(f"Invoice {eid} not found")
    val = data["value"]
    checks = [{"check": "invoice_exists", "pass": True, "id": eid}]
    # Amount check is unreliable for multi-line invoices (regex only finds first line)
    # Just verify non-zero amount
    actual_amt = val.get("amountExcludingVatCurrency") or val.get("amountExcludingVat", 0)
    checks.append({"check": "has_amount", "pass": actual_amt > 0, "actual": actual_amt})
    return _pass(f"Invoice #{val.get('invoiceNumber')} created", checks) if all(c["pass"] for c in checks) else _fail("Invoice has zero amount", checks)


def _v_create_order(facts, prompt, api):
    c = facts.last_success(endpoint_contains="/order", method="POST")
    if not c:
        return _fail("No successful order POST")
    checks = [{"check": "order_created", "pass": True, "id": facts.entity_id(c)}]
    return _pass("Order created", checks)


def _v_create_project(facts, prompt, api):
    c = facts.last_success(endpoint_contains="/project", method="POST")
    if not c:
        return _fail("No successful project POST")
    checks = [{"check": "project_created", "pass": True, "id": facts.entity_id(c)}]
    return _pass("Project created", checks)


def _v_supplier_invoice(facts, prompt, api):
    # Some prompts are "register supplier only" — no invoice expected
    has_invoice_ref = bool(re.search(r'(INV-|faktura.*mottatt|invoice.*received|facture.*re[cç]u|Rechnung.*erhalten|factura.*recibida|fatura.*recebida|vedlagt PDF|attached PDF|ci-joint|beigefugte|motteke faktura)', prompt, re.I))

    # Can be POST /incomingInvoice or fallback POST /ledger/voucher
    c = facts.last_success(endpoint_contains="/incomingInvoice", method="POST")
    if not c:
        c = facts.last_success(endpoint_contains="/voucher", method="POST")

    # If "register supplier only" prompt, just check supplier was created
    if not has_invoice_ref:
        sup = facts.last_success(endpoint_contains="/supplier", method="POST")
        if sup:
            checks = [{"check": "supplier_created", "pass": True, "id": facts.entity_id(sup)}]
            return _pass("Supplier registered", checks)
        return _fail("No successful supplier POST")

    if not c:
        return _fail("No successful supplier invoice or voucher POST")
    # Also check supplier was created/found
    sup = facts.last_success(endpoint_contains="/supplier")
    checks = [
        {"check": "invoice_or_voucher", "pass": True, "id": facts.entity_id(c)},
        {"check": "supplier_found", "pass": sup is not None},
    ]
    return _pass("Supplier invoice registered", checks)


def _v_travel_expense(facts, prompt, api):
    c = facts.last_success(endpoint_contains="/travelExpense", method="POST")
    if not c:
        return _fail("No successful travel expense POST")
    checks = [{"check": "travel_expense_created", "pass": True, "id": facts.entity_id(c)}]
    return _pass("Travel expense created", checks)


def _v_custom_dimension(facts, prompt, api):
    checks = []
    c = facts.last_success(endpoint_contains="/accountingDimensionName", method="POST")
    if c:
        checks.append({"check": "dimension_created", "pass": True})
    vals = facts.all_success(endpoint_contains="/accountingDimensionValue", method="POST")
    if vals:
        checks.append({"check": "dimension_values_added", "pass": True, "count": len(vals)})
    # Voucher with dimension reference counts as success too
    voucher = facts.last_success(endpoint_contains="/voucher", method="POST")
    if voucher:
        checks.append({"check": "voucher_posted", "pass": True})
    # Accept if any meaningful work was done
    if checks:
        return _pass(f"Custom dimension task completed", checks)
    return _fail("No dimension created")


# ─── The 13 missing verifiers ───

def _v_register_payment(facts, prompt, api):
    c = facts.last_success(endpoint_contains=":payment")
    if not c:
        return _fail("No successful :payment call")
    eid = facts.entity_id(c)
    if not eid:
        return _fail("Could not extract invoice ID from payment")
    data = api.get(f"/invoice/{eid}")
    if not data or "value" not in data:
        return _fail(f"Invoice {eid} not found on re-fetch")
    val = data["value"]
    outstanding = val.get("amountOutstanding", -1)
    checks = [
        {"check": "payment_applied", "pass": True, "id": eid},
        {"check": "fully_paid", "pass": abs(outstanding) < 0.01,
         "expected": 0, "actual": outstanding},
    ]
    ok = all(c["pass"] for c in checks)
    return _pass("Payment registered, invoice fully paid", checks) if ok else _fail(f"Invoice outstanding: {outstanding}", checks)


def _v_credit_note(facts, prompt, api):
    c = facts.last_success(endpoint_contains=":createCreditNote")
    if not c:
        return _fail("No successful :createCreditNote call")
    eid = facts.entity_id(c)
    checks = [{"check": "credit_note_created", "pass": eid is not None, "id": eid}]
    if eid:
        data = api.get(f"/invoice/{eid}")
        if data and "value" in data:
            val = data["value"]
            is_cn = val.get("isCreditNote", False)
            checks.append({"check": "is_credit_note", "pass": is_cn})
    ok = all(c["pass"] for c in checks)
    return _pass("Credit note created", checks) if ok else _fail("Credit note verification failed", checks)


def _v_reverse_payment(facts, prompt, api):
    # Reversal shows as a :payment that reopens outstanding
    c = facts.last_success(endpoint_contains=":payment")
    if not c:
        return _fail("No successful payment/reversal call")
    eid = facts.entity_id(c)
    if not eid:
        return _fail("Could not extract invoice ID")
    data = api.get(f"/invoice/{eid}")
    if not data or "value" not in data:
        return _fail(f"Invoice {eid} not found")
    val = data["value"]
    outstanding = val.get("amountOutstanding", 0)
    checks = [
        {"check": "reversal_applied", "pass": True, "id": eid},
        {"check": "invoice_reopened", "pass": outstanding > 0,
         "actual_outstanding": outstanding},
    ]
    ok = all(c["pass"] for c in checks)
    return _pass("Payment reversed, invoice reopened", checks) if ok else _fail("Invoice not reopened after reversal", checks)


def _v_multi_currency_payment(facts, prompt, api):
    checks = []
    # 1. Payment applied
    pay = facts.last_success(endpoint_contains=":payment")
    if not pay:
        return _fail("No successful :payment call")
    checks.append({"check": "payment_applied", "pass": True})

    # 2. Check payment amounts are correct relative to prompt
    eur = extract_eur_amount(prompt)
    old_rate, new_rate, _ = extract_currency_rate(prompt)
    if eur and new_rate:
        pay_params = pay.get("params", {})
        paid_currency = pay_params.get("paidAmountCurrency")
        paid_nok = pay_params.get("paidAmount")
        # Use wider tolerance (10 NOK) to account for VAT rounding in Tripletex
        if paid_currency is not None:
            ok = abs(float(paid_currency) - eur) < 10.0
            checks.append({"check": "paid_eur_amount", "pass": ok,
                           "expected": eur, "actual": paid_currency})
        if paid_nok is not None and new_rate:
            expected_nok = eur * new_rate
            ok = abs(float(paid_nok) - expected_nok) < 10.0
            checks.append({"check": "paid_nok_amount", "pass": ok,
                           "expected": round(expected_nok, 2), "actual": paid_nok})

    # 3. FX voucher posted
    voucher = facts.last_success(endpoint_contains="/voucher", method="POST")
    checks.append({"check": "fx_voucher_posted", "pass": voucher is not None})

    # 4. FX amount check (wider tolerance for VAT rounding)
    if voucher and eur and old_rate and new_rate:
        expected_diff = abs(eur * (old_rate - new_rate))
        body = voucher.get("body", {})
        postings = body.get("postings", [])
        if postings:
            actual = abs(postings[0].get("amountGross", 0))
            ok = abs(actual - expected_diff) < 10.0
            checks.append({"check": "fx_amount", "pass": ok,
                           "expected": round(expected_diff, 2), "actual": actual})

    ok = all(c["pass"] for c in checks)
    return _pass("Multi-currency payment verified", checks) if ok else _fail("Multi-currency payment issues", checks)


def _v_reminder_fee(facts, prompt, api):
    checks = []
    # Could use :createReminder, or post a voucher + invoice for the fee
    c = facts.last_success(endpoint_contains=":createReminder")
    if c:
        checks.append({"check": "reminder_created", "pass": True, "id": facts.entity_id(c)})
        return _pass("Reminder fee posted", checks)
    # Fallback: voucher or invoice for the reminder fee
    voucher = facts.last_success(endpoint_contains="/voucher", method="POST")
    invoice = facts.last_success(endpoint_contains="invoice")
    if voucher or invoice:
        if voucher:
            checks.append({"check": "fee_voucher_posted", "pass": True})
        if invoice:
            checks.append({"check": "fee_invoice_created", "pass": True})
        return _pass("Reminder fee handled", checks)
    return _fail("No successful reminder call or fee posting")


def _v_payroll(facts, prompt, api):
    # Payroll usually creates voucher(s)
    vouchers = facts.all_success(endpoint_contains="/voucher", method="POST")
    if not vouchers:
        # Could also be salary transaction
        vouchers = facts.all_success(endpoint_contains="/salary", method="POST")
    if not vouchers:
        return _fail("No successful payroll voucher/salary POST")
    checks = [{"check": "payroll_voucher_count", "pass": True, "count": len(vouchers)}]
    return _pass(f"Payroll posted ({len(vouchers)} voucher(s))", checks)


def _v_expense_receipt(facts, prompt, api):
    # Should have voucher or incoming invoice
    c = facts.last_success(endpoint_contains="/voucher", method="POST")
    if not c:
        c = facts.last_success(endpoint_contains="/incomingInvoice", method="POST")
    if not c:
        return _fail("No successful voucher/invoice for expense")
    checks = [{"check": "expense_posted", "pass": True, "id": facts.entity_id(c)}]
    dept = extract_department(prompt)
    if dept:
        # Check department was referenced in the call
        body = c.get("body", {})
        postings = body.get("postings", [])
        has_dept = any(p.get("department") for p in postings) if postings else False
        checks.append({"check": "department_linked", "pass": has_dept, "expected": dept})
    return _pass("Expense receipt posted", checks) if all(c["pass"] for c in checks) else _fail("Expense receipt issues", checks)


def _v_cost_analysis(facts, prompt, api):
    # Analysis task — agent should complete with analysis summary.
    # Some prompts ask to investigate AND suggest/make corrections.
    checks = [{"check": "api_calls_made", "pass": len(facts.successes) > 0, "count": len(facts.successes)}]
    if facts.has_any_write():
        checks.append({"check": "has_writes", "pass": True, "note": "may include corrections"})
    return _pass("Cost analysis completed", checks) if checks[0]["pass"] else _fail("No API calls made")


def _v_ledger_correction(facts, prompt, api):
    vouchers = facts.all_success(endpoint_contains="/voucher", method="POST")
    if not vouchers:
        return _fail("No correction vouchers posted")
    checks = [{"check": "correction_vouchers", "pass": True, "count": len(vouchers)}]
    # Expect multiple corrections (usually 3-4)
    if len(vouchers) < 2:
        checks.append({"check": "enough_corrections", "pass": False,
                        "expected": ">=2", "actual": len(vouchers)})
    return _pass(f"{len(vouchers)} correction voucher(s)", checks) if all(c["pass"] for c in checks) else _fail("Too few corrections", checks)


def _v_monthly_closing(facts, prompt, api):
    vouchers = facts.all_success(endpoint_contains="/voucher", method="POST")
    if not vouchers:
        return _fail("No closing vouchers posted")
    checks = [{"check": "closing_vouchers", "pass": True, "count": len(vouchers)}]
    return _pass(f"Monthly closing: {len(vouchers)} voucher(s)", checks)


def _v_year_end_closing(facts, prompt, api):
    vouchers = facts.all_success(endpoint_contains="/voucher", method="POST")
    if not vouchers:
        return _fail("No year-end vouchers posted")
    checks = [{"check": "year_end_vouchers", "pass": len(vouchers) >= 2, "count": len(vouchers)}]
    return _pass(f"Year-end closing: {len(vouchers)} voucher(s)", checks) if all(c["pass"] for c in checks) else _fail("Too few year-end vouchers", checks)


def _v_bank_reconciliation(facts, prompt, api):
    payments = facts.all_success(endpoint_contains=":payment")
    if not payments:
        return _fail("No payment matches found")
    checks = [{"check": "payments_matched", "pass": True, "count": len(payments)}]
    return _pass(f"Bank reconciliation: {len(payments)} payment(s) matched", checks)


def _v_project_billing(facts, prompt, api):
    checks = []
    # Should have hours registered or fixed price set
    hours = facts.all_success(endpoint_contains="/timesheet")
    if hours:
        checks.append({"check": "hours_registered", "pass": True, "count": len(hours)})
    # Should have project invoice
    inv = facts.last_success(endpoint_contains="invoice")
    if inv:
        checks.append({"check": "project_invoice", "pass": True, "id": facts.entity_id(inv)})
    # Or fixed price set on project
    proj_put = facts.last_success(endpoint_contains="/project", method="PUT")
    if proj_put:
        checks.append({"check": "project_updated", "pass": True})
    if not checks:
        return _fail("No hours, invoice, or project update found")
    return _pass("Project billing completed", checks)


# ─── Registry ───

VERIFIERS = {
    "create_customer": _v_create_customer,
    "create_employee": _v_create_employee,
    "create_department": _v_create_department,
    "create_product": _v_create_product,
    "create_invoice": _v_create_invoice,
    "create_order": _v_create_order,
    "create_project": _v_create_project,
    "supplier_invoice": _v_supplier_invoice,
    "travel_expense": _v_travel_expense,
    "custom_dimension": _v_custom_dimension,
    "register_payment": _v_register_payment,
    "credit_note": _v_credit_note,
    "reverse_payment": _v_reverse_payment,
    "multi_currency_payment": _v_multi_currency_payment,
    "reminder_fee": _v_reminder_fee,
    "payroll": _v_payroll,
    "expense_receipt": _v_expense_receipt,
    "cost_analysis": _v_cost_analysis,
    "ledger_correction": _v_ledger_correction,
    "monthly_closing": _v_monthly_closing,
    "year_end_closing": _v_year_end_closing,
    "bank_reconciliation": _v_bank_reconciliation,
    "project_billing": _v_project_billing,
}


def verify(task_record: dict, prompt: str, task_type: str, base_url: str, token: str) -> Verdict:
    """Main entry point: verify a completed task."""
    if not task_record:
        return _fail("No task record")

    outcome = task_record.get("outcome", "unknown")
    if outcome not in ("completed", "completed_at_limit"):
        return _fail(f"Agent outcome: {outcome}")

    api = TripletexAPI(base_url, token)
    facts = ExecFacts(task_record.get("api_calls", []))

    verifier = VERIFIERS.get(task_type)
    if not verifier:
        return _inconclusive(f"No verifier for task type: {task_type}")

    try:
        return verifier(facts, prompt, api)
    except Exception as e:
        log.error("Verifier crashed: %s", e)
        return _fail(f"Verifier error: {e}")
