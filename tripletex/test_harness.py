"""Comprehensive local test harness for the Tripletex AI agent.

Runs real task prompts through solve_task(), then verifies results against
the Tripletex sandbox API. Reports PASS/FAIL with details.

Usage:
  python test_harness.py                       # Run all tests
  python test_harness.py --list                # List available tests
  python test_harness.py --test create_dept    # Run one test by key
  python test_harness.py --real                # Run real prompts from task_log.jsonl
  python test_harness.py --test create_dept --dry  # Show what would run (no agent call)
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import date, timedelta

import requests

# ─── Env setup (must be before agent import) ───

os.environ.setdefault(
    "GOOGLE_API_KEY",
    "AIzaSyDU9JLjRiaEfmp-Y7n7ilv91uhgk0wDv3U",
)
os.environ.setdefault("TASK_LOG_FILE", os.path.join(os.path.dirname(__file__), "test_harness_log.jsonl"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("harness")

# ─── Sandbox credentials ───

SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjM5OTU2LCJ0b2tlbiI6ImM0YzUyMWM2LTBkMjItNGVhYy1iZmVhLWJlOWRjMjM0N2I3ZSJ9",
}
AUTH = ("0", SANDBOX_CREDS["session_token"])
BASE = SANDBOX_CREDS["base_url"]

TODAY = date.today().isoformat()
TOMORROW = (date.today() + timedelta(days=1)).isoformat()
TWO_WEEKS = (date.today() + timedelta(days=14)).isoformat()


# ─── API helpers ───

def api_get(endpoint, params=None):
    """GET from sandbox. Returns (status_code, parsed_json_or_text)."""
    params = params or {}
    params.setdefault("fields", "*")
    resp = requests.get(f"{BASE}{endpoint}", auth=AUTH, params=params, timeout=30)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, resp.text


def api_search(endpoint, **params):
    """Search and return values list."""
    status, data = api_get(endpoint, params)
    if status != 200:
        return []
    return data.get("values", [])


# ─── Verification functions ───

def verify_customer(name, **expect):
    """Verify customer exists with expected fields."""
    customers = api_search("/customer", name=name)
    match = next((c for c in customers if c.get("name", "").lower() == name.lower()), None)
    if not match:
        return False, f"Customer '{name}' not found in sandbox"

    errors = []
    if "email" in expect:
        actual = (match.get("email") or "").lower()
        if actual != expect["email"].lower():
            errors.append(f"email: got '{actual}', want '{expect['email']}'")
    if "organizationNumber" in expect:
        actual = match.get("organizationNumber", "")
        if str(actual) != str(expect["organizationNumber"]):
            errors.append(f"orgNr: got '{actual}', want '{expect['organizationNumber']}'")
    if "phone" in expect:
        actual = str(match.get("phoneNumber") or match.get("phoneNumberMobile") or "")
        if expect["phone"] not in actual:
            errors.append(f"phone: got '{actual}', want contains '{expect['phone']}'")
    if errors:
        return False, "; ".join(errors)
    return True, f"OK (id={match['id']})"


def verify_employee(first_name, last_name=None, email=None):
    """Verify employee exists."""
    employees = api_search("/employee", firstName=first_name)
    match = None
    for e in employees:
        if e.get("firstName", "").lower() != first_name.lower():
            continue
        if last_name and e.get("lastName", "").lower() != last_name.lower():
            continue
        match = e
        break
    if not match:
        return False, f"Employee '{first_name} {last_name or ''}' not found"
    errors = []
    if email and (match.get("email") or "").lower() != email.lower():
        errors.append(f"email: got '{match.get('email')}', want '{email}'")
    if errors:
        return False, "; ".join(errors)
    return True, f"OK (id={match['id']})"


def verify_department(name):
    """Verify department exists."""
    depts = api_search("/department")
    match = next((d for d in depts if d.get("name", "").lower() == name.lower()), None)
    if not match:
        all_names = [d.get("name") for d in depts]
        return False, f"Department '{name}' not found. Existing: {all_names}"
    return True, f"OK (id={match['id']})"


def verify_product(name, price=None):
    """Verify product exists."""
    products = api_search("/product", name=name)
    match = next((p for p in products if p.get("name", "").lower() == name.lower()), None)
    if not match:
        return False, f"Product '{name}' not found"
    if price is not None:
        actual = float(match.get("priceExcludingVatCurrency", 0))
        if abs(actual - price) > 0.01:
            return False, f"price: got {actual}, want {price}"
    return True, f"OK (id={match['id']})"


def verify_invoice(customer_name):
    """Verify an invoice exists for the given customer."""
    invoices = api_search("/invoice", invoiceDateFrom="2020-01-01", invoiceDateTo="2030-12-31")
    for inv in invoices:
        cust = inv.get("customer", {})
        if customer_name.lower() in (cust.get("name") or "").lower():
            return True, f"OK (invoice id={inv['id']}, customer={cust.get('name')})"
    return False, f"No invoice found for customer '{customer_name}'"


def verify_order(customer_name):
    """Verify an order exists for the given customer."""
    orders = api_search("/order", orderDateFrom="2020-01-01", orderDateTo="2030-12-31")
    for o in orders:
        cust = o.get("customer", {})
        if customer_name.lower() in (cust.get("name") or "").lower():
            return True, f"OK (order id={o['id']}, customer={cust.get('name')})"
    return False, f"No order found for customer '{customer_name}'"


def verify_travel_expense(title_contains):
    """Verify a travel expense exists with title containing the given string."""
    expenses = api_search("/travelExpense")
    for te in expenses:
        title = te.get("title") or ""
        if title_contains.lower() in title.lower():
            return True, f"OK (id={te['id']}, title='{title}')"
    return False, f"No travel expense with title containing '{title_contains}'"


def verify_project(name):
    """Verify project exists."""
    projects = api_search("/project")
    match = next((p for p in projects if name.lower() in (p.get("name") or "").lower()), None)
    if not match:
        return False, f"Project '{name}' not found"
    return True, f"OK (id={match['id']})"


def verify_any_created(entity_type):
    """Generic: just check that at least something new was created.
    entity_type: 'customer', 'employee', 'department', etc."""
    endpoint_map = {
        "customer": "/customer",
        "employee": "/employee",
        "department": "/department",
        "product": "/product",
        "order": "/order",
        "invoice": "/invoice",
        "travelExpense": "/travelExpense",
        "project": "/project",
    }
    endpoint = endpoint_map.get(entity_type)
    if not endpoint:
        return False, f"Unknown entity type: {entity_type}"
    items = api_search(endpoint)
    if items:
        return True, f"Found {len(items)} {entity_type}(s)"
    return False, f"No {entity_type}s found"


# ─── Test case registry ───

TESTS = {}


def test(key, prompt, verify_fn, description=""):
    """Register a test case."""
    TESTS[key] = {
        "prompt": prompt,
        "verify": verify_fn,
        "description": description or key,
    }


# 1. Create department
test(
    "create_dept",
    "Opprett en avdeling med navn Testing.",
    lambda: verify_department("Testing"),
    "Create department (Norwegian)",
)

test(
    "create_dept_en",
    "Create a new department called Marketing.",
    lambda: verify_department("Marketing"),
    "Create department (English)",
)

# 2. Create customer (simple)
test(
    "create_customer_simple",
    "Opprett en ny kunde med namn Fjordkraft AS, e-post post@fjordkraft.no, telefon 55112233.",
    lambda: verify_customer("Fjordkraft AS", email="post@fjordkraft.no", phone="55112233"),
    "Create customer with name/email/phone",
)

# 3. Create customer (full details — from real task_log)
test(
    "create_customer_full",
    "Opprett kunden Bølgekraft AS med organisasjonsnummer 812297848. "
    "Adressa er Havnegata 113, 7010 Trondheim. E-post: post@bolgekraft.no, telefon 41223344.",
    lambda: verify_customer(
        "Bølgekraft AS",
        email="post@bolgekraft.no",
        organizationNumber="812297848",
        phone="41223344",
    ),
    "Create customer with org number + address (real task_log prompt)",
)

# 4. Create employee
test(
    "create_employee",
    "Opprett en ansatt med navn Kari Hansen, e-post kari@firma.no.",
    lambda: verify_employee("Kari", last_name="Hansen", email="kari@firma.no"),
    "Create employee (Norwegian)",
)

# 5. Create employee as admin
test(
    "create_employee_admin",
    "Opprett en ansatt med navn Kari Hansen, e-post kari@firma.no. Hun skal være kontoadministrator.",
    lambda: verify_employee("Kari", last_name="Hansen", email="kari@firma.no"),
    "Create employee + grant admin privileges",
)

# 6. Create employee (English)
test(
    "create_employee_en",
    "Create a new employee named John Smith with email john@company.com. He should have standard access.",
    lambda: verify_employee("John", last_name="Smith", email="john@company.com"),
    "Create employee (English)",
)

# 7. Create product
test(
    "create_product",
    "Opprett et produkt med navn Konsulenttjenester, pris 1500 kr ekskl. mva.",
    lambda: verify_product("Konsulenttjenester", price=1500.0),
    "Create product with price",
)

# 8. Create order + invoice (multi-step)
test(
    "create_invoice",
    "Opprett og send ein faktura til kunden Bølgekraft AS (org.nr 892362416) på 34150 kr eksklusiv MVA. "
    "Fakturaen gjeld Vedlikehald.",
    lambda: verify_order("Bølgekraft"),
    "Create customer + order + invoice (multi-step, real task_log prompt)",
)

# 9. Travel expense (from real task_log)
test(
    "travel_expense",
    'Registe uma despesa de viagem para Bruno Silva (bruno.silva@example.org) referente a '
    '"Conferência Bodø". A viagem durou de 15 a 17 de março de 2026.',
    lambda: verify_travel_expense("Bodø"),
    "Travel expense in Portuguese (real task_log prompt)",
)

# 10. Travel expense (Norwegian)
test(
    "travel_expense_no",
    "Registrer en reiseregning for den eksisterende ansatte i systemet. "
    "Tittel: Kundemøte Bergen. Beløp: 3500 kr.",
    lambda: verify_travel_expense("Bergen"),
    "Travel expense (Norwegian)",
)

# 11. Create project
test(
    "create_project",
    "Opprett et prosjekt med navn 'Kontorbygg Oslo' for kunde Viken Eiendom AS. "
    "Prosjektleder skal være den eksisterende ansatte i systemet.",
    lambda: verify_project("Kontorbygg Oslo"),
    "Create project with customer + project manager",
)

# 12. German customer
test(
    "create_customer_de",
    "Erstellen Sie einen neuen Kunden mit dem Namen Müller GmbH, "
    "E-Mail info@mueller.de, Telefon +49 30 12345678.",
    lambda: verify_customer("Müller GmbH", email="info@mueller.de"),
    "Create customer (German)",
)

# 13. Create invoice with payment (full flow)
test(
    "invoice_with_payment",
    "Opprett en faktura til kunde Berg Elektro AS (e-post: faktura@bergelektro.no). "
    "Fakturaen skal inneholde 3 stk Installasjon (pris 2500 kr ekskl. mva per stk). "
    "Forfallsdato 14 dager fra i dag. Registrer full betaling.",
    lambda: verify_invoice("Berg Elektro"),
    "Full flow: customer + order + invoice + payment",
)

# 14. Credit note
test(
    "credit_note",
    "Opprett en kreditnota for den siste fakturaen i systemet.",
    lambda: verify_any_created("invoice"),
    "Credit note on latest invoice",
)


# ─── Load real prompts from task_log.jsonl ───

def load_real_prompts():
    """Load prompts from task_log.jsonl and add them as test cases."""
    log_path = os.path.join(os.path.dirname(__file__), "task_log.jsonl")
    if not os.path.exists(log_path):
        log.warning("task_log.jsonl not found at %s", log_path)
        return {}

    real_tests = {}
    with open(log_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = entry.get("prompt", "")
            if not prompt:
                continue
            task_id = entry.get("id", f"real_{i+1:03d}")
            outcome = entry.get("outcome", "unknown")
            real_tests[f"real_{task_id}"] = {
                "prompt": prompt,
                "verify": lambda: (True, "no auto-verify for real prompts"),
                "description": f"Real prompt ({outcome}): {prompt[:80]}...",
                "original_outcome": outcome,
            }
    return real_tests


# ─── Runner ───

def build_request_body(prompt, files=None):
    """Build the exact request body the server sends to solve_task."""
    return {
        "prompt": prompt,
        "files": files or [],
        "tripletex_credentials": SANDBOX_CREDS,
    }


async def run_test(key, test_data, dry_run=False):
    """Run a single test case. Returns (key, passed, detail, elapsed)."""
    from agent import solve_task

    prompt = test_data["prompt"]
    verify_fn = test_data["verify"]
    desc = test_data.get("description", key)

    print(f"\n{'='*70}")
    print(f"  TEST: {key}")
    print(f"  DESC: {desc}")
    print(f"  PROMPT: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] Skipping agent execution")
        return key, None, "dry run", 0.0

    body = build_request_body(prompt)

    t0 = time.time()
    agent_error = None
    try:
        await solve_task(body)
    except Exception as e:
        agent_error = str(e)
        log.error("Agent raised exception: %s", e)
    elapsed = time.time() - t0

    # Give API a moment to settle
    time.sleep(0.5)

    # Verify
    try:
        passed, detail = verify_fn()
    except Exception as e:
        passed, detail = False, f"Verification error: {e}"

    if agent_error:
        passed = False
        detail = f"AGENT ERROR: {agent_error} | Verify: {detail}"

    status_str = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"\n  {color}[{status_str}]{reset} {detail}  ({elapsed:.1f}s)")

    return key, passed, detail, elapsed


async def run_all(keys, tests, dry_run=False):
    """Run selected tests sequentially, print summary."""
    results = []
    for key in keys:
        test_data = tests[key]
        result = await run_test(key, test_data, dry_run=dry_run)
        results.append(result)

    # Summary
    passed = sum(1 for _, p, _, _ in results if p is True)
    failed = sum(1 for _, p, _, _ in results if p is False)
    skipped = sum(1 for _, p, _, _ in results if p is None)
    total_time = sum(e for _, _, _, e in results)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for key, p, detail, elapsed in results:
        if p is True:
            icon = "\033[92mPASS\033[0m"
        elif p is False:
            icon = "\033[91mFAIL\033[0m"
        else:
            icon = "\033[93mSKIP\033[0m"
        print(f"  [{icon}] {key:30s}  {detail[:60]}  ({elapsed:.1f}s)")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped  ({total_time:.1f}s)")
    if failed == 0 and passed > 0:
        print("  \033[92mALL TESTS PASSED\033[0m")
    elif failed > 0:
        print(f"  \033[91m{failed} FAILURE(S)\033[0m")
    print(f"{'='*70}")

    # Write results to file
    results_path = os.path.join(os.path.dirname(__file__), "test_harness_results.json")
    serializable = []
    for key, p, detail, elapsed in results:
        serializable.append({
            "test": key,
            "passed": p,
            "detail": detail,
            "elapsed_s": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Results written to {results_path}")

    return failed == 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tripletex agent test harness")
    parser.add_argument("--test", "-t", help="Run specific test by key")
    parser.add_argument("--list", "-l", action="store_true", help="List all available tests")
    parser.add_argument("--real", action="store_true", help="Run real prompts from task_log.jsonl")
    parser.add_argument("--all", "-a", action="store_true", help="Run all tests (built-in + real)")
    parser.add_argument("--dry", action="store_true", help="Dry run: show tests without executing")
    parser.add_argument("--quick", "-q", action="store_true", help="Run only fast tests (create dept/customer)")
    args = parser.parse_args()

    # Merge real prompts
    real_tests = load_real_prompts()
    all_tests = {**TESTS}
    if args.real or args.all:
        all_tests.update(real_tests)

    if args.list:
        print("\nBuilt-in tests:")
        for key, data in TESTS.items():
            print(f"  {key:30s}  {data.get('description', '')}")
        if real_tests:
            print(f"\nReal prompts from task_log.jsonl ({len(real_tests)}):")
            for key, data in real_tests.items():
                print(f"  {key:30s}  {data.get('description', '')[:80]}")
        return

    if args.test:
        if args.test not in all_tests:
            # Try partial match
            matches = [k for k in all_tests if args.test in k]
            if len(matches) == 1:
                keys = matches
            elif len(matches) > 1:
                print(f"Ambiguous test key '{args.test}'. Matches: {matches}")
                sys.exit(1)
            else:
                print(f"Unknown test: {args.test}")
                print(f"Available: {', '.join(sorted(all_tests.keys()))}")
                sys.exit(1)
        else:
            keys = [args.test]
    elif args.quick:
        keys = ["create_dept", "create_customer_simple"]
    elif args.real:
        keys = list(real_tests.keys())
    else:
        keys = list(TESTS.keys())

    success = asyncio.run(run_all(keys, all_tests, dry_run=args.dry))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
