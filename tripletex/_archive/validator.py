"""Self-validator: tests agent against sandbox and verifies correctness.

Usage:
  python validator.py                    # Run all test cases
  python validator.py test_cases.jsonl   # Run specific test file
  python validator.py --case create_customer_basic
"""

import asyncio
import json
import logging
import sys
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validator")

# Sandbox credentials (safe to use for testing)
SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "XXx--xx-REDACTED-SESSION",
}

AUTH = ("0", SANDBOX_CREDS["session_token"])
BASE = SANDBOX_CREDS["base_url"]


def api_get(endpoint, params=None):
    """GET from Tripletex sandbox."""
    resp = requests.get(f"{BASE}{endpoint}", auth=AUTH, params=params, timeout=30)
    return resp.status_code, resp.json() if resp.status_code == 200 else resp.text


def find_entity(endpoint, **filters):
    """Search for entity matching filters."""
    status, data = api_get(endpoint, {"fields": ",".join(["id"] + list(filters.keys()))})
    if status != 200:
        return None
    for item in data.get("values", []):
        if all(str(item.get(k, "")).lower() == str(v).lower() for k, v in filters.items()):
            return item
    return None


# ─── Test case definitions ───

TEST_CASES = {
    "create_customer_basic": {
        "prompt": "Opprett kunden Bølgekraft AS med organisasjonsnummer 812297848. "
                  "Adressa er Havnegata 113, 7010 Trondheim. E-post: post@bolgekraft.no, telefon 41223344.",
        "verify": lambda: verify_customer(
            name="Bølgekraft AS",
            organizationNumber="812297848",
            email="post@bolgekraft.no",
            phone="41223344",
            address_line="Havnegata 113",
            postal_code="7010",
            city="Trondheim",
        ),
    },
    "create_customer_simple": {
        "prompt": "Registrer en ny kunde med navn Nordvik Bygg AS, "
                  "e-post post@nordvikbygg.no, telefon 99887766.",
        "verify": lambda: verify_customer(
            name="Nordvik Bygg AS",
            email="post@nordvikbygg.no",
        ),
    },
    "create_employee_admin": {
        "prompt": "Opprett en ansatt med navn Kari Hansen, e-post kari@firma.no. "
                  "Hun skal være kontoadministrator.",
        "verify": lambda: verify_employee(
            firstName="Kari",
            lastName="Hansen",
            email="kari@firma.no",
        ),
    },
    "create_department": {
        "prompt": "Opprett en ny avdeling med navn Salg.",
        "verify": lambda: verify_department(name="Salg"),
    },
    "create_product": {
        "prompt": "Opprett et produkt med navn Konsulenttjenester, pris 1500 kr ekskl. mva.",
        "verify": lambda: verify_product(
            name="Konsulenttjenester",
            price=1500.0,
        ),
    },
    "create_employee_english": {
        "prompt": "Create a new employee named John Smith with email john@company.com. "
                  "He should have standard access.",
        "verify": lambda: verify_employee(
            firstName="John",
            lastName="Smith",
            email="john@company.com",
        ),
    },
    "create_customer_german": {
        "prompt": "Erstellen Sie einen neuen Kunden mit dem Namen Müller GmbH, "
                  "E-Mail info@mueller.de, Telefon +49 30 12345678.",
        "verify": lambda: verify_customer(
            name="Müller GmbH",
            email="info@mueller.de",
        ),
    },
}


# ─── Verification functions ───

def verify_customer(name, **expected):
    """Check if customer was created with expected fields."""
    status, data = api_get("/customer", {"name": name, "fields": "id,name,email,phoneNumber,organizationNumber,postalAddress"})
    if status != 200:
        return False, f"GET /customer failed: {status}"

    values = data.get("values", [])
    match = None
    for v in values:
        if v.get("name", "").lower() == name.lower():
            match = v
            break

    if not match:
        return False, f"Customer '{name}' not found"

    errors = []
    if "email" in expected and match.get("email", "").lower() != expected["email"].lower():
        errors.append(f"email: got '{match.get('email')}', expected '{expected['email']}'")
    if "organizationNumber" in expected and match.get("organizationNumber") != expected["organizationNumber"]:
        errors.append(f"orgNumber: got '{match.get('organizationNumber')}', expected '{expected['organizationNumber']}'")
    if "phone" in expected:
        actual_phone = match.get("phoneNumber", "")
        if expected["phone"] not in str(actual_phone):
            errors.append(f"phone: got '{actual_phone}', expected '{expected['phone']}'")

    if errors:
        return False, "; ".join(errors)
    return True, f"OK (id={match['id']})"


def verify_employee(firstName, lastName=None, email=None):
    """Check if employee was created."""
    params = {"fields": "id,firstName,lastName,email"}
    if firstName:
        params["firstName"] = firstName
    status, data = api_get("/employee", params)
    if status != 200:
        return False, f"GET /employee failed: {status}"

    values = data.get("values", [])
    match = None
    for v in values:
        if v.get("firstName", "").lower() == firstName.lower():
            if lastName and v.get("lastName", "").lower() != lastName.lower():
                continue
            match = v
            break

    if not match:
        return False, f"Employee '{firstName}' not found"

    errors = []
    if email and match.get("email", "").lower() != email.lower():
        errors.append(f"email: got '{match.get('email')}', expected '{email}'")

    if errors:
        return False, "; ".join(errors)
    return True, f"OK (id={match['id']})"


def verify_department(name):
    """Check if department was created."""
    status, data = api_get("/department", {"fields": "id,name"})
    if status != 200:
        return False, f"GET /department failed: {status}"

    values = data.get("values", [])
    for v in values:
        if v.get("name", "").lower() == name.lower():
            return True, f"OK (id={v['id']})"

    return False, f"Department '{name}' not found"


def verify_product(name, price=None):
    """Check if product was created."""
    status, data = api_get("/product", {"fields": "id,name,priceExcludingVatCurrency"})
    if status != 200:
        return False, f"GET /product failed: {status}"

    values = data.get("values", [])
    for v in values:
        if v.get("name", "").lower() == name.lower():
            if price and abs(float(v.get("priceExcludingVatCurrency", 0)) - price) > 0.01:
                return False, f"price: got {v.get('priceExcludingVatCurrency')}, expected {price}"
            return True, f"OK (id={v['id']})"

    return False, f"Product '{name}' not found"


# ─── Runner ───

async def run_case(case_name, case_data):
    """Run a single test case through the agent and verify."""
    from agent import solve_task

    prompt = case_data["prompt"]
    verify_fn = case_data["verify"]

    print(f"\n{'='*60}")
    print(f"CASE: {case_name}")
    print(f"PROMPT: {prompt[:100]}...")
    print(f"{'='*60}")

    body = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": SANDBOX_CREDS,
    }

    t0 = time.time()
    try:
        await solve_task(body)
    except Exception as e:
        print(f"  AGENT ERROR: {e}")
        return False
    elapsed = time.time() - t0

    # Verify
    success, detail = verify_fn()
    status = "PASS" if success else "FAIL"
    print(f"  [{status}] {detail} ({elapsed:.1f}s)")

    # Log result
    result = {
        "case": case_name,
        "status": status,
        "detail": detail,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open("validation_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    return success


async def main():
    cases_to_run = TEST_CASES

    if len(sys.argv) > 1 and sys.argv[1].startswith("--case"):
        case_name = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].split("=")[1]
        if case_name in TEST_CASES:
            cases_to_run = {case_name: TEST_CASES[case_name]}
        else:
            print(f"Unknown case: {case_name}")
            print(f"Available: {', '.join(TEST_CASES.keys())}")
            sys.exit(1)

    passed = 0
    failed = 0

    for name, data in cases_to_run.items():
        success = await run_case(name, data)
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("ALL TESTS PASSED — safe to deploy")
    else:
        print("FAILURES DETECTED — DO NOT DEPLOY")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
