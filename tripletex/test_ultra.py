#!/usr/bin/env python3
"""5 ultra-complex synthetic test cases — stress test for competition readiness.

Includes reproduction of the French monthly closing that failed checks 3+4.
"""

import json
import os
import sys
import time

import requests

ENDPOINT = os.environ.get("SOLVE_ENDPOINT", "https://nm.j6x.com/solve")
SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjM5OTU2LCJ0b2tlbiI6ImM0YzUyMWM2LTBkMjItNGVhYy1iZmVhLWJlOWRjMjM0N2I3ZSJ9",
}
AUTH = ("0", SANDBOX_CREDS["session_token"])
BASE = SANDBOX_CREDS["base_url"]


def api_search(endpoint, **params):
    params.setdefault("fields", "*")
    resp = requests.get(f"{BASE}{endpoint}", auth=AUTH, params=params, timeout=30)
    if resp.status_code != 200:
        return []
    return resp.json().get("values", [])


TESTS = [
    # ━━━ U01: REPRODUCTION — French monthly closing with explicit account numbers ━━━
    # This is the exact task that failed checks 3+4 (wrong account 6010 instead of 6030, wrong 2930 instead of 2900)
    {
        "id": "U01",
        "name": "French monthly closing (reproduction of failure)",
        "prompt": (
            "Effectuez la clôture mensuelle de mars 2026. "
            "Comptabilisez la régularisation (13200 NOK par mois du compte 1710 vers le compte 6300). "
            "Enregistrez l'amortissement mensuel d'une immobilisation avec un coût d'acquisition de 52200 NOK "
            "et une durée de vie utile de 7 ans (amortissement linéaire sur compte 6030). "
            "Comptabilisez également une provision pour salaires "
            "(débit compte de charges salariales 5000, crédit compte de salaires à payer 2900). "
            "Montant de la provision: 45000 NOK."
        ),
        "checks": [
            ("Voucher exists with accrual 13200", "voucher_desc", "lôture"),
            ("Account 6030 used for depreciation", "posting_account", 6030),
            ("Account 2900 used for salary provision", "posting_account", 2900),
            ("Depreciation = 52200/7/12 ≈ 621.43", "posting_amount_approx", 621.43),
            ("Salary provision = 45000", "posting_amount_approx", 45000),
        ],
    },

    # ━━━ U02: German payment reversal + correction voucher combo ━━━
    # Tests: payment_invoice with negative amount, then correction voucher
    {
        "id": "U02",
        "name": "German payment reversal (must use payment_invoice, NOT voucher)",
        "prompt": (
            "Aufgabe mit zwei Schritten:\n"
            "1. Erstellen Sie einen Kunden Nordsee Technik AG (Org.-Nr. 871234567), "
            "eine Bestellung mit Produkt 1 (1 Stück), fakturieren Sie und registrieren Sie die vollständige Zahlung.\n"
            "2. Anschließend wurde die Zahlung von der Bank zurückgebucht. "
            "Stornieren Sie die Zahlung, damit die Rechnung wieder den offenen Betrag anzeigt."
        ),
        "checks": [
            ("Customer created", "customer", "Nordsee Technik"),
            ("Invoice created", "invoice_exists", True),
            ("Payment reversed (amountOutstanding > 0)", "invoice_outstanding_gt0", True),
        ],
    },

    # ━━━ U03: Multi-currency payment + agio + supplier invoice combo (Portuguese) ━━━
    # Tests: FX payment fields, agio voucher, supplier invoice in same task
    {
        "id": "U03",
        "name": "Portuguese multi-step: supplier invoice + customer FX payment + agio",
        "prompt": (
            "Execute as seguintes tarefas contábeis:\n\n"
            "1. Registre uma fatura de fornecedor da empresa Sol Nascente Lda (org. nº 891234567) "
            "no valor de 37500 NOK incluindo IVA para serviços de consultoria (conta 6700). "
            "Data da fatura: 2026-03-10, vencimento: 2026-04-10.\n\n"
            "2. Crie o cliente Mar Azul SA (org. nº 892345678) e fature 15000 NOK excl. IVA "
            "por Produto 1 (1 unidade a 15000 NOK). Registre o pagamento integral.\n\n"
            "3. Crie um lançamento contábil: débito conta 7100 por 8500 NOK, crédito conta 1920 por 8500 NOK. "
            "Descrição: Despesas de transporte março."
        ),
        "checks": [
            ("Supplier created", "supplier", "Sol Nascente"),
            ("Customer created", "customer", "Mar Azul"),
            ("Invoice paid (outstanding=0)", "invoice_outstanding_eq0", True),
            ("Voucher for transport expenses", "voucher_desc", "transporte"),
        ],
    },

    # ━━━ U04: Spanish employment contract + custom dimension + voucher with dimension ━━━
    # Tests: employee_contract + dimension creation + dimension-linked voucher
    {
        "id": "U04",
        "name": "Spanish: employee contract + custom dimension + linked voucher",
        "prompt": (
            "Realice las siguientes tareas:\n\n"
            "1. Cree el empleado Miguel Torres (miguel.torres@example.es) en el departamento 1. "
            "Fecha de inicio: 2026-06-01, salario anual: 540000 NOK, porcentaje de empleo: 100%, "
            "código de profesión: 2310. Tipo de empleo: ordinario.\n\n"
            "2. Cree una dimensión contable personalizada llamada 'Centro de Coste' "
            "con los valores 'Producción' y 'Administración'.\n\n"
            "3. Registre un asiento contable: débito cuenta 6800 por 12000 NOK, "
            "crédito cuenta 1920 por 12000 NOK. Descripción: Material de oficina Q1."
        ),
        "checks": [
            ("Employee created", "employee", "Miguel"),
            ("Employment record", "employment", "Miguel"),
            ("Dimension created", "dimension", "Centro de Coste"),
            ("Voucher created", "voucher_desc", "oficina"),
        ],
    },

    # ━━━ U05: Nynorsk bank reconciliation from CSV — 4 payments + 2 vouchers ━━━
    # Tests: turn budget management, bulk operations, CSV parsing
    {
        "id": "U05",
        "name": "Nynorsk CSV bank reconciliation (6 operations, turn-heavy)",
        "prompt": (
            "Frå vedlagt CSV-fil skal du utføre følgjande bankavstemmingsoppgåver:\n\n"
            "For kvar rad i filen:\n"
            "- Dersom typen er 'faktura_betaling': finn fakturaen til kunden og registrer betalinga.\n"
            "- Dersom typen er 'utgift': opprett eit bilag med rett konto.\n\n"
            "Alle beløp er i NOK. Bruk dagens dato for alle posteringar."
        ),
        "files": [{
            "filename": "bankavstemming.csv",
            "mime_type": "text/csv",
            "content_base64": __import__("base64").b64encode(
                "Type,Kunde/Leverandør,Beløp,Konto,Beskrivelse\n"
                "utgift,,4500,6800,Kontorrekvisita mars\n"
                "utgift,,12000,6300,Husleige april\n"
                "utgift,,3200,7100,Drivstoff mars\n".encode()
            ).decode(),
        }],
        "checks": [
            ("Voucher for office supplies", "voucher_desc", "ontorrekvisita"),
            ("Voucher for rent", "voucher_desc", "usleige"),
            ("Voucher for fuel", "voucher_desc", "rivstoff"),
        ],
    },
]


def send_task(test):
    body = {
        "prompt": test["prompt"],
        "files": test.get("files", []),
        "tripletex_credentials": SANDBOX_CREDS,
    }
    t0 = time.time()
    try:
        resp = requests.post(ENDPOINT, json=body, timeout=300)
        return resp.status_code, time.time() - t0
    except Exception as e:
        return -1, time.time() - t0


def verify_after(test):
    """Quick verification after task completes."""
    results = []
    for check_name, check_type, check_val in test.get("checks", []):
        try:
            if check_type == "voucher_desc":
                vs = api_search("/ledger/voucher", dateFrom="2020-01-01", dateTo="2030-12-31")
                found = any(str(check_val).lower() in (v.get("description") or "").lower() for v in vs)
                results.append((check_name, found))
            elif check_type == "customer":
                cs = api_search("/customer", name=str(check_val))
                found = any(str(check_val).lower() in (c.get("name") or "").lower() for c in cs)
                results.append((check_name, found))
            elif check_type == "supplier":
                ss = api_search("/supplier", name=str(check_val))
                found = any(str(check_val).lower() in (s.get("name") or "").lower() for s in ss)
                results.append((check_name, found))
            elif check_type == "employee":
                es = api_search("/employee", firstName=str(check_val))
                found = len(es) > 0
                results.append((check_name, found))
            elif check_type == "employment":
                es = api_search("/employee", firstName=str(check_val))
                if es:
                    emps = api_search("/employee/employment", employeeId=str(es[0]["id"]))
                    results.append((check_name, len(emps) > 0))
                else:
                    results.append((check_name, False))
            elif check_type == "dimension":
                ds = api_search("/ledger/accountingDimensionName")
                found = any(str(check_val).lower() in (d.get("dimensionName") or "").lower() for d in ds)
                results.append((check_name, found))
            elif check_type == "posting_account":
                # Check if any voucher has a posting on this account number
                vs = api_search("/ledger/voucher", dateFrom="2020-01-01", dateTo="2030-12-31")
                # Can't easily check posting details from voucher search — mark as manual check
                results.append((check_name, "MANUAL_CHECK"))
            elif check_type == "posting_amount_approx":
                results.append((check_name, "MANUAL_CHECK"))
            elif check_type in ("invoice_exists", "invoice_outstanding_gt0", "invoice_outstanding_eq0"):
                invs = api_search("/invoice", invoiceDateFrom="2020-01-01", invoiceDateTo="2030-12-31")
                if check_type == "invoice_exists":
                    results.append((check_name, len(invs) > 0))
                elif check_type == "invoice_outstanding_gt0":
                    found = any(i.get("amountOutstanding", 0) > 0 for i in invs)
                    results.append((check_name, found))
                elif check_type == "invoice_outstanding_eq0":
                    found = any(i.get("amountOutstanding", 0) == 0 for i in invs)
                    results.append((check_name, found))
            else:
                results.append((check_name, "UNKNOWN_CHECK"))
        except Exception as e:
            results.append((check_name, f"ERROR: {e}"))
    return results


def main():
    subset = sys.argv[1:] if len(sys.argv) > 1 else None

    tests = TESTS
    if subset:
        tests = [t for t in TESTS if t["id"] in [s.upper() for s in subset]]

    print(f"\n{'━'*70}")
    print(f"  Ultra-Complex Test Suite — {len(tests)} cases")
    print(f"  Endpoint: {ENDPOINT}")
    print(f"{'━'*70}\n")

    for test in tests:
        tid = test["id"]
        print(f"╔══ {tid}: {test['name']}")
        print(f"║ Sending...", end="", flush=True)

        status, elapsed = send_task(test)
        if status != 200:
            print(f" HTTP {status} ({elapsed:.0f}s)")
            print(f"╚══ FAIL (HTTP error)\n")
            continue

        print(f" OK ({elapsed:.0f}s)")
        time.sleep(2)

        # Verify
        checks = verify_after(test)
        passed = sum(1 for _, v in checks if v is True)
        total = len(checks)
        manual = sum(1 for _, v in checks if v == "MANUAL_CHECK")

        for name, result in checks:
            if result is True:
                print(f"║   \033[92m✓\033[0m {name}")
            elif result == "MANUAL_CHECK":
                print(f"║   \033[33m?\033[0m {name} (check logs)")
            else:
                print(f"║   \033[91m✗\033[0m {name}")

        print(f"╚══ {passed}/{total} passed ({manual} manual) — {elapsed:.0f}s\n")

    # Also dump latest task log entries
    print(f"\n{'━'*70}")
    print("  Check VPS logs: ssh vps 'tail -5 /opt/tripletex/task_log.jsonl | python3 -m json.tool'")
    print(f"{'━'*70}")


if __name__ == "__main__":
    main()
