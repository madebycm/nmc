#!/usr/bin/env python3
"""E2E test suite: 30+ prompts in 7 languages against deployed server.

Sends requests to the live endpoint, then verifies results via Tripletex sandbox API.

Usage:
  python test_e2e.py              # Run all tests
  python test_e2e.py --quick      # Run first 10 only
  python test_e2e.py --test T05   # Run specific test
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

# ─── Config ───

ENDPOINT = os.environ.get("SOLVE_ENDPOINT", "https://nm.j6x.com/solve")
SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjM5OTU2LCJ0b2tlbiI6ImM0YzUyMWM2LTBkMjItNGVhYy1iZmVhLWJlOWRjMjM0N2I3ZSJ9",
}
AUTH = ("0", SANDBOX_CREDS["session_token"])
BASE = SANDBOX_CREDS["base_url"]
TODAY = date.today().isoformat()

# ─── API helpers ───

def api_get(endpoint, params=None):
    params = params or {}
    params.setdefault("fields", "*")
    resp = requests.get(f"{BASE}{endpoint}", auth=AUTH, params=params, timeout=30)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, resp.text

def api_search(endpoint, **params):
    status, data = api_get(endpoint, params)
    if status != 200:
        return []
    return data.get("values", [])

# ─── Verifiers ───

def v_customer(name, **kw):
    custs = api_search("/customer", name=name)
    m = next((c for c in custs if name.lower() in (c.get("name") or "").lower()), None)
    if not m:
        return False, f"Customer '{name}' not found"
    errs = []
    if "email" in kw and (m.get("email") or "").lower() != kw["email"].lower():
        errs.append(f"email: {m.get('email')} != {kw['email']}")
    if "orgNr" in kw and str(m.get("organizationNumber", "")) != str(kw["orgNr"]):
        errs.append(f"orgNr: {m.get('organizationNumber')} != {kw['orgNr']}")
    return (False, "; ".join(errs)) if errs else (True, f"OK id={m['id']}")

def v_employee(first, last=None, **kw):
    emps = api_search("/employee", firstName=first)
    m = None
    for e in emps:
        if e.get("firstName", "").lower() != first.lower(): continue
        if last and e.get("lastName", "").lower() != last.lower(): continue
        m = e; break
    if not m:
        return False, f"Employee '{first} {last or ''}' not found"
    errs = []
    if "email" in kw and (m.get("email") or "").lower() != kw["email"].lower():
        errs.append(f"email mismatch")
    return (False, "; ".join(errs)) if errs else (True, f"OK id={m['id']}")

def v_dept(name):
    depts = api_search("/department")
    m = next((d for d in depts if (d.get("name") or "").lower() == name.lower()), None)
    return (True, f"OK id={m['id']}") if m else (False, f"Dept '{name}' not found")

def v_invoice(cust_name):
    # Must request customer(*) to get nested name field
    status, data = api_get("/invoice", {
        "invoiceDateFrom": "2020-01-01",
        "invoiceDateTo": "2030-12-31",
        "fields": "id,customer(id,name),amountOutstanding",
    })
    if status != 200:
        return False, f"Invoice search HTTP {status}"
    invs = data.get("values", [])
    for i in invs:
        c = i.get("customer", {})
        if cust_name.lower() in (c.get("name") or "").lower():
            return True, f"OK invoice id={i['id']}"
    return False, f"No invoice for '{cust_name}' (searched {len(invs)} invoices)"

def v_travel(title_part):
    tes = api_search("/travelExpense")
    for t in tes:
        if title_part.lower() in (t.get("title") or "").lower():
            return True, f"OK id={t['id']}"
    return False, f"No travel expense with '{title_part}'"

def v_project(name):
    projs = api_search("/project")
    m = next((p for p in projs if name.lower() in (p.get("name") or "").lower()), None)
    return (True, f"OK id={m['id']}") if m else (False, f"Project '{name}' not found")

def v_supplier(name):
    sups = api_search("/supplier", name=name)
    m = next((s for s in sups if name.lower() in (s.get("name") or "").lower()), None)
    return (True, f"OK id={m['id']}") if m else (False, f"Supplier '{name}' not found")

def v_voucher(desc_part):
    vs = api_search("/ledger/voucher", dateFrom="2020-01-01", dateTo="2030-12-31")
    for v in vs:
        if desc_part.lower() in (v.get("description") or "").lower():
            return True, f"OK id={v['id']}"
    return False, f"No voucher with '{desc_part}'"

def v_dimension(dim_name):
    dims = api_search("/ledger/accountingDimensionName")
    # The GET endpoint may not support name filter — check all
    for d in dims:
        if dim_name.lower() in (d.get("dimensionName") or "").lower():
            return True, f"OK id={d['id']} index={d.get('dimensionIndex')}"
    return False, f"Dimension '{dim_name}' not found"

def v_employment(first, last=None):
    emps = api_search("/employee", firstName=first)
    m = None
    for e in emps:
        if e.get("firstName", "").lower() != first.lower(): continue
        if last and e.get("lastName", "").lower() != last.lower(): continue
        m = e; break
    if not m:
        return False, f"Employee '{first}' not found"
    emp_id = m["id"]
    # Check employment exists
    employments = api_search("/employee/employment", employeeId=str(emp_id))
    if not employments:
        return False, f"Employee found (id={emp_id}) but no employment record"
    return True, f"OK employee={emp_id}, employments={len(employments)}"

def v_noop():
    return True, "OK (no verification)"


# ─── Test definitions: 35 tests across 7 languages ───

TESTS = [
    # --- TIER 1: Foundational (Norwegian Bokmål) ---
    {"id": "T01", "lang": "nb", "type": "department",
     "prompt": "Opprett en avdeling med navn Markedsføring.",
     "verify": lambda: v_dept("Markedsføring")},

    {"id": "T02", "lang": "nb", "type": "customer",
     "prompt": "Opprett kunden Solvik Handel AS med e-post post@solvik.no og organisasjonsnummer 987654321.",
     "verify": lambda: v_customer("Solvik Handel AS", email="post@solvik.no")},

    {"id": "T03", "lang": "nb", "type": "employee",
     "prompt": "Opprett en ansatt med navn Erik Johansen, e-post erik@johansen.no.",
     "verify": lambda: v_employee("Erik", "Johansen", email="erik@johansen.no")},

    {"id": "T04", "lang": "nb", "type": "product",
     "prompt": "Opprett et produkt med navn Rådgivningstjenester, produktnummer 9901, pris 1800 kr ekskl. mva.",
     "verify": lambda: v_noop()},

    # --- TIER 1: Other languages ---
    {"id": "T05", "lang": "nn", "type": "customer",
     "prompt": "Opprett ein ny kunde med namn Havbris AS, e-post post@havbris.no, telefon 92334455.",
     "verify": lambda: v_customer("Havbris AS", email="post@havbris.no")},

    {"id": "T06", "lang": "en", "type": "employee",
     "prompt": "Create a new employee named Sarah Johnson with email sarah@company.com in department 1.",
     "verify": lambda: v_employee("Sarah", "Johnson", email="sarah@company.com")},

    {"id": "T07", "lang": "es", "type": "customer",
     "prompt": "Cree un nuevo cliente con el nombre Industrias del Mar SL, correo electrónico contacto@mar.es.",
     "verify": lambda: v_customer("Industrias del Mar SL", email="contacto@mar.es")},

    {"id": "T08", "lang": "pt", "type": "department",
     "prompt": "Crie um departamento chamado Recursos Humanos.",
     "verify": lambda: v_dept("Recursos Humanos")},

    {"id": "T09", "lang": "de", "type": "customer",
     "prompt": "Erstellen Sie einen neuen Kunden: Berliner Technik GmbH, E-Mail info@berliner-technik.de, Organisationsnummer 112233445.",
     "verify": lambda: v_customer("Berliner Technik GmbH", email="info@berliner-technik.de")},

    {"id": "T10", "lang": "fr", "type": "employee",
     "prompt": "Créez un nouvel employé nommé Pierre Dupont avec l'adresse e-mail pierre@dupont.fr.",
     "verify": lambda: v_employee("Pierre", "Dupont", email="pierre@dupont.fr")},

    # --- TIER 2: Multi-step workflows ---
    {"id": "T11", "lang": "nb", "type": "order_invoice",
     "prompt": "Opprett en faktura til kunden Fjelltopp AS (e-post faktura@fjelltopp.no). Fakturaen skal inneholde 2 stk Konsulenttjenester (pris 2000 kr ekskl. mva per stk). Send fakturaen.",
     "verify": lambda: v_invoice("Fjelltopp")},

    {"id": "T12", "lang": "nb", "type": "invoice_payment",
     "prompt": "Opprett og send ein faktura til kunden Kystbygg AS (e-post post@kystbygg.no) på 15000 kr ekskl. mva for Prosjektledelse. Registrer full betaling.",
     "verify": lambda: v_invoice("Kystbygg")},

    {"id": "T13", "lang": "de", "type": "order_invoice",
     "prompt": "Erstellen Sie eine Rechnung für den Kunden Alpenwerk AG (E-Mail rechnung@alpenwerk.ch) über 3 Stück Beratungsleistung (Preis 1500 NOK pro Stück exkl. MwSt.). Senden Sie die Rechnung.",
     "verify": lambda: v_invoice("Alpenwerk")},

    {"id": "T14", "lang": "nb", "type": "travel_expense",
     "prompt": "Registrer en reiseregning for den eksisterende ansatte. Tittel: Kundemøte Tromsø. Reisen varte fra 2026-03-10 til 2026-03-12. Legg til en kostnad for fly på 4500 kr.",
     "verify": lambda: v_travel("Tromsø")},

    {"id": "T15", "lang": "pt", "type": "travel_expense",
     "prompt": "Registre uma despesa de viagem para o funcionário existente. Título: Conferência Lisboa. A viagem durou de 10 a 12 de março de 2026. Adicione um custo de voo de 5000 NOK.",
     "verify": lambda: v_travel("Lisboa")},

    {"id": "T16", "lang": "nb", "type": "project",
     "prompt": "Opprett et prosjekt med namn Havneutbygging for kunden Solvik Handel AS. Prosjektleder er den eksisterende ansatte.",
     "verify": lambda: v_project("Havneutbygging")},

    {"id": "T17", "lang": "nb", "type": "supplier_invoice",
     "prompt": "Registrer en leverandørfaktura fra Kontorservice AS (org.nr 999888777) på 12500 kr inkl. MVA for kontorrekvisita (konto 6800).",
     "verify": lambda: v_supplier("Kontorservice")},

    {"id": "T18", "lang": "nb", "type": "voucher",
     "prompt": "Opprett et bilag: debet konto 6300 på 8000 kr, kredit konto 1920 på 8000 kr. Beskrivelse: Husleie april.",
     "verify": lambda: v_voucher("Husleie")},

    {"id": "T19", "lang": "nb", "type": "credit_note",
     "prompt": "Opprett en kreditnota for den siste fakturaen som ble opprettet.",
     "verify": lambda: v_noop()},

    {"id": "T20", "lang": "nb", "type": "employee_admin",
     "prompt": "Opprett en ansatt med navn Anna Berg, e-post anna@firma.no. Hun skal være kontoadministrator med alle tilganger.",
     "verify": lambda: v_employee("Anna", "Berg", email="anna@firma.no")},

    # --- NEW: Employment contract tasks ---
    {"id": "T21", "lang": "nb", "type": "employee_contract",
     "prompt": "Opprett ansatt Lars Olsen (lars@firma.no) i avdeling 1. Startdato 2026-04-01, årslønn 550000, stillingsprosent 100%, yrkeskode 2310. Arbeidsforholdet er fast.",
     "verify": lambda: v_employment("Lars", "Olsen")},

    {"id": "T22", "lang": "de", "type": "employee_contract",
     "prompt": "Erstellen Sie einen Mitarbeiter: Max Weber, max@weber.de, Abteilung 1. Startdatum 2026-04-01, Gehalt 620000 NOK, Beschäftigungsprozentsatz 100%, Berufsschlüssel 2310, Arbeitstyp: fest angestellt.",
     "verify": lambda: v_employment("Max", "Weber")},

    {"id": "T23", "lang": "en", "type": "employee_contract",
     "prompt": "Create employee Jane Doe (jane@example.com) in department 1. Start date 2026-05-01, annual salary 500000 NOK, 80% position, occupation code 3112. Employment type: ordinary, permanent.",
     "verify": lambda: v_employment("Jane", "Doe")},

    # --- NEW: Custom dimensions ---
    {"id": "T24", "lang": "nb", "type": "custom_dimension",
     "prompt": "Opprett en egendefinert regnskapsdimensjon med navn 'Prosjekttype'. Legg til verdiene 'Intern' og 'Ekstern'.",
     "verify": lambda: v_dimension("Prosjekttype")},

    {"id": "T25", "lang": "es", "type": "custom_dimension",
     "prompt": "Cree una dimensión contable personalizada llamada 'Región' con los valores 'Norte', 'Sur' y 'Centro'.",
     "verify": lambda: v_dimension("Región")},

    {"id": "T26", "lang": "en", "type": "custom_dimension",
     "prompt": "Create a custom accounting dimension named 'CostCenter' with values 'HQ', 'Branch1', and 'Branch2'.",
     "verify": lambda: v_dimension("CostCenter")},

    # --- NEW: Reminder ---
    {"id": "T27", "lang": "nb", "type": "reminder",
     "prompt": "Finn den ubetalte fakturaen i systemet og send en purring (myk purring) med purregebyr.",
     "verify": lambda: v_noop()},

    {"id": "T28", "lang": "fr", "type": "reminder",
     "prompt": "Trouvez la facture impayée dans le système et envoyez un rappel de paiement par e-mail.",
     "verify": lambda: v_noop()},

    # --- Fallback / unknown task types (tests generic API) ---
    {"id": "T29", "lang": "nb", "type": "supplier_create",
     "prompt": "Opprett leverandøren Bygg og Betong AS med organisasjonsnummer 888777666 og e-post post@byggbetong.no.",
     "verify": lambda: v_supplier("Bygg og Betong")},

    {"id": "T30", "lang": "en", "type": "delete_travel",
     "prompt": "Delete all travel expenses in the system.",
     "verify": lambda: v_noop()},

    # --- Multi-language regression ---
    {"id": "T31", "lang": "fr", "type": "customer",
     "prompt": "Créez un nouveau client nommé Océan Bleu SARL avec l'adresse e-mail contact@oceanbleu.fr.",
     "verify": lambda: v_customer("Océan Bleu SARL", email="contact@oceanbleu.fr")},

    {"id": "T32", "lang": "es", "type": "employee",
     "prompt": "Cree un nuevo empleado llamado Carlos García con correo carlos@empresa.es.",
     "verify": lambda: v_employee("Carlos", "García", email="carlos@empresa.es")},

    {"id": "T33", "lang": "pt", "type": "customer",
     "prompt": "Crie o cliente Oceano Azul Ltda com e-mail oceano@azul.pt e número de organização 556677889.",
     "verify": lambda: v_customer("Oceano Azul Ltda", email="oceano@azul.pt")},

    {"id": "T34", "lang": "nn", "type": "voucher",
     "prompt": "Opprett eit bilag: debet konto 7100 på 3500 kr, kredit konto 1920 på 3500 kr. Beskriving: Drivstoff mars.",
     "verify": lambda: v_voucher("Drivstoff")},

    {"id": "T35", "lang": "de", "type": "supplier_invoice",
     "prompt": "Registrieren Sie eine Lieferantenrechnung von Office Supply GmbH (Org.-Nr. 444555666) über 8000 NOK inkl. MwSt. für Bürobedarf (Konto 6800).",
     "verify": lambda: v_supplier("Office Supply")},
]


# ─── Runner ───

def send_task(test):
    """Send a task to the deployed endpoint."""
    body = {
        "prompt": test["prompt"],
        "files": [],
        "tripletex_credentials": SANDBOX_CREDS,
    }
    t0 = time.time()
    try:
        resp = requests.post(ENDPOINT, json=body, timeout=300)
        elapsed = time.time() - t0
        return resp.status_code, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return -1, elapsed


def run_test(test):
    """Run a single test: send to endpoint, wait, verify."""
    tid = test["id"]
    print(f"  [{tid}] {test['lang'].upper()} {test['type']:25s} sending...", end="", flush=True)

    status, elapsed = send_task(test)

    if status != 200:
        print(f" HTTP {status} ({elapsed:.0f}s)")
        return {"id": tid, "passed": False, "detail": f"HTTP {status}", "elapsed": elapsed, "lang": test["lang"], "type": test["type"]}

    # Brief pause for API to settle
    time.sleep(1)

    # Verify
    try:
        passed, detail = test["verify"]()
    except Exception as e:
        passed, detail = False, f"Verify error: {e}"

    icon = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f" [{icon}] {detail[:60]} ({elapsed:.0f}s)")

    return {"id": tid, "passed": passed, "detail": detail, "elapsed": round(elapsed, 1),
            "lang": test["lang"], "type": test["type"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", help="Run specific test by ID (e.g. T05)")
    parser.add_argument("--quick", "-q", action="store_true", help="Run first 10 only")
    parser.add_argument("--type", help="Filter by task type")
    parser.add_argument("--lang", help="Filter by language")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Concurrent tests (default 1)")
    args = parser.parse_args()

    tests = TESTS
    if args.test:
        tests = [t for t in TESTS if t["id"] == args.test.upper()]
        if not tests:
            print(f"Unknown test ID: {args.test}")
            sys.exit(1)
    elif args.type:
        tests = [t for t in TESTS if args.type in t["type"]]
    elif args.lang:
        tests = [t for t in TESTS if t["lang"] == args.lang]
    elif args.quick:
        tests = TESTS[:10]

    print(f"\n{'='*70}")
    print(f"  E2E Test Suite — {len(tests)} tests")
    print(f"  Endpoint: {ENDPOINT}")
    print(f"  Sandbox: {BASE}")
    print(f"{'='*70}\n")

    results = []
    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(run_test, t): t for t in tests}
            for f in as_completed(futures):
                results.append(f.result())
        results.sort(key=lambda r: r["id"])
    else:
        for t in tests:
            results.append(run_test(t))

    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total_time = sum(r["elapsed"] for r in results)

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {passed}/{len(results)} passed, {failed} failed ({total_time:.0f}s total)")
    print(f"{'='*70}")

    # By type
    types = {}
    for r in results:
        types.setdefault(r["type"], []).append(r)
    for tp, rs in sorted(types.items()):
        p = sum(1 for r in rs if r["passed"])
        icon = "\033[92m" if p == len(rs) else "\033[91m"
        print(f"  {icon}{tp:25s} {p}/{len(rs)}\033[0m")

    # By language
    print()
    langs = {}
    for r in results:
        langs.setdefault(r["lang"], []).append(r)
    for la, rs in sorted(langs.items()):
        p = sum(1 for r in rs if r["passed"])
        icon = "\033[92m" if p == len(rs) else "\033[91m"
        print(f"  {icon}{la:5s} {p}/{len(rs)}\033[0m")

    # Failed details
    if failed:
        print(f"\n  Failed tests:")
        for r in results:
            if not r["passed"]:
                print(f"    [{r['id']}] {r['lang']} {r['type']}: {r['detail']}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "test_e2e_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results → {out_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
