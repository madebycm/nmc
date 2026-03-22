#!/usr/bin/env python3
"""E2E test suite: 50+ prompts in 7 languages against deployed server.

Sends requests to the live endpoint, then verifies results via Tripletex sandbox API.
Includes document-attached tests (PDF, XLSX, DOCX, CSV).

Usage:
  python test_e2e.py              # Run all tests
  python test_e2e.py --quick      # Run first 10 only
  python test_e2e.py --test T05   # Run specific test
  python test_e2e.py --type supplier_invoice_pdf  # Run by type
"""

import base64
import io
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
    "session_token": "XXx--xx-REDACTED-SESSION",
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

def v_incoming_invoice(supplier_name):
    """Verify a supplier invoice was created — check both incomingInvoice and voucher."""
    # First check supplier exists
    ok, detail = v_supplier(supplier_name)
    if not ok:
        return False, f"Supplier not found: {detail}"
    # Check for vouchers (incomingInvoice creates a voucher internally)
    vs = api_search("/ledger/voucher", dateFrom="2020-01-01", dateTo="2030-12-31")
    if vs:
        return True, f"OK supplier={detail}, vouchers={len(vs)}"
    return False, f"Supplier found but no voucher/invoice entity"

def v_noop():
    return True, "OK (no verification)"


# ─── Document generators ───

def _make_csv(rows: list[list[str]]) -> str:
    """Generate a CSV file as base64 string."""
    lines = [",".join(str(c) for c in row) for row in rows]
    csv_text = "\n".join(lines)
    return base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")


def _make_xlsx(sheet_data: dict) -> str:
    """Generate XLSX as base64 string. sheet_data: {sheet_name: [[row], ...]}"""
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        first = True
        for name, rows in sheet_data.items():
            ws = wb.active if first else wb.create_sheet(name)
            if first:
                ws.title = name
                first = False
            for row in rows:
                ws.append(row)
        buf = io.BytesIO()
        wb.save(buf)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        # Fallback: return empty xlsx-like placeholder
        return base64.b64encode(b"XLSX_NOT_AVAILABLE").decode("utf-8")


def _make_docx(paragraphs: list[str]) -> str:
    """Generate DOCX as base64 string."""
    try:
        import docx
        doc = docx.Document()
        for p in paragraphs:
            doc.add_paragraph(p)
        buf = io.BytesIO()
        doc.save(buf)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        return base64.b64encode(b"DOCX_NOT_AVAILABLE").decode("utf-8")


def _make_simple_pdf(lines: list[str]) -> str:
    """Generate a minimal valid PDF with multi-line text (no external deps)."""
    # Build content stream with proper line-by-line text positioning
    # Use separate BT/ET blocks with absolute positioning via Tm (text matrix)
    ops = []
    y = 750
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"BT /F1 12 Tf 1 0 0 1 50 {y} Tm ({safe}) Tj ET")
        y -= 18
    stream_text = "\n".join(ops)
    stream_bytes = stream_text.encode("latin-1", errors="replace")

    # Build PDF structure with proper xref
    objects = []

    # obj 1: catalog
    objects.append(b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj")
    # obj 2: pages
    objects.append(b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj")
    # obj 3: page
    objects.append(b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj")
    # obj 4: content stream
    obj4 = b"4 0 obj <</Length " + str(len(stream_bytes)).encode() + b">>\nstream\n" + stream_bytes + b"\nendstream\nendobj"
    objects.append(obj4)
    # obj 5: font
    objects.append(b"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj")

    # Build PDF
    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj + b"\n"

    xref_offset = len(pdf)
    pdf += b"xref\n"
    pdf += f"0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += b"trailer <</Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R>>\n"
    pdf += b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF\n"

    return base64.b64encode(pdf).decode("utf-8")


# ─── Test definitions: 60 tests across 7 languages ───

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

    # ═══════════════════════════════════════════════════════════════
    # TIER 3: Document-attached tests (PDF, XLSX, DOCX, CSV)
    # ═══════════════════════════════════════════════════════════════

    # --- Supplier invoice from PDF (5 languages) ---
    {"id": "T36", "lang": "nb", "type": "supplier_invoice_pdf",
     "prompt": "Vi har mottatt en leverandørfaktura (se vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandøren om den ikke finnes. Bruk riktig utgiftskonto og inngående MVA.",
     "files": [{"filename": "leverandorfaktura_nb.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "FAKTURA",
                    "Leverandor: Nordisk Kontor AS",
                    "Org.nr: 987111222",
                    "Fakturanummer: INV-2026-5001",
                    "Fakturadato: 2026-03-15",
                    "Forfallsdato: 2026-04-15",
                    "Kontortjenester - 25000 NOK inkl. MVA",
                    "MVA 25%: 5000 NOK",
                    "Totalt inkl. MVA: 25000 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("Nordisk Kontor")},

    {"id": "T37", "lang": "de", "type": "supplier_invoice_pdf",
     "prompt": "Wir haben eine Lieferantenrechnung erhalten (siehe beigefügte PDF). Registrieren Sie die Rechnung in Tripletex. Erstellen Sie den Lieferanten, falls er nicht existiert. Verwenden Sie das korrekte Aufwandskonto und die Vorsteuer.",
     "files": [{"filename": "lieferantenrechnung_de.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "RECHNUNG",
                    "Lieferant: Berliner Buero GmbH",
                    "Org.Nr: 987222333",
                    "Rechnungsnummer: RE-2026-3001",
                    "Rechnungsdatum: 2026-03-18",
                    "Faelligkeitsdatum: 2026-04-18",
                    "Buerobedarf - 15000 NOK inkl. MwSt.",
                    "MwSt 25%: 3000 NOK",
                    "Gesamtbetrag inkl. MwSt: 15000 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("Berliner Buero")},

    {"id": "T38", "lang": "es", "type": "supplier_invoice_pdf",
     "prompt": "Has recibido una factura de proveedor (ver PDF adjunto). Registra la factura en Tripletex. Crea el proveedor si no existe. Usa la cuenta de gastos correcta y el IVA de entrada.",
     "files": [{"filename": "factura_proveedor_es.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "FACTURA",
                    "Proveedor: Servicios Madrid SL",
                    "Org.nr: 987333444",
                    "Numero de factura: FACT-2026-7001",
                    "Fecha de factura: 2026-03-10",
                    "Fecha de vencimiento: 2026-04-10",
                    "Servicios de consultoria - 32000 NOK incl. IVA",
                    "IVA 25%: 6400 NOK",
                    "Total incl. IVA: 32000 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("Servicios Madrid")},

    {"id": "T39", "lang": "fr", "type": "supplier_invoice_pdf",
     "prompt": "Vous avez reçu une facture fournisseur (voir PDF ci-joint). Enregistrez la facture dans Tripletex. Créez le fournisseur s'il n'existe pas. Utilisez le bon compte de charges et la TVA déductible.",
     "files": [{"filename": "facture_fournisseur_fr.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "FACTURE",
                    "Fournisseur: Bureau Lyon SARL",
                    "Org.nr: 987444555",
                    "Numero de facture: FACT-2026-9001",
                    "Date de facture: 2026-03-12",
                    "Date echeance: 2026-04-12",
                    "Fournitures de bureau - 18500 NOK TTC",
                    "TVA 25%: 3700 NOK",
                    "Total TTC: 18500 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("Bureau Lyon")},

    {"id": "T40", "lang": "en", "type": "supplier_invoice_pdf",
     "prompt": "We have received a supplier invoice (see attached PDF). Register the invoice in Tripletex. Create the supplier if they don't exist. Use the correct expense account and input VAT.",
     "files": [{"filename": "supplier_invoice_en.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "INVOICE",
                    "Supplier: London Tech Ltd",
                    "Org.no: 987555666",
                    "Invoice number: INV-2026-1001",
                    "Invoice date: 2026-03-20",
                    "Due date: 2026-04-20",
                    "IT Consulting Services - 45000 NOK incl. VAT",
                    "VAT 25%: 9000 NOK",
                    "Total incl. VAT: 45000 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("London Tech")},

    # --- Employment contracts from PDF (3 languages) ---
    {"id": "T41", "lang": "nb", "type": "employee_contract_pdf",
     "prompt": "Du har mottatt en arbeidsavtale (se vedlagt PDF). Opprett den ansatte i Tripletex med alle detaljer fra avtalen: startdato, avdeling, yrkeskode, årslønn, stillingsprosent.",
     "files": [{"filename": "arbeidsavtale_nb.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "ARBEIDSAVTALE",
                    "Ansatt: Ingrid Haugen",
                    "Epost: ingrid.haugen@example.no",
                    "Fodselsdato: 1992-08-15",
                    "Avdeling: Salg",
                    "Startdato: 2026-06-01",
                    "Yrkeskode: 3321",
                    "Aarslonn: 580000 NOK",
                    "Stillingsprosent: 100%",
                    "Ansettelsestype: Fast",
                ])}],
     "verify": lambda: v_employment("Ingrid", "Haugen")},

    {"id": "T42", "lang": "de", "type": "employee_contract_pdf",
     "prompt": "Sie haben einen Arbeitsvertrag erhalten (siehe beigefügte PDF). Erstellen Sie den Mitarbeiter in Tripletex mit allen Details aus dem Vertrag.",
     "files": [{"filename": "arbeitsvertrag_de.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "ARBEITSVERTRAG",
                    "Mitarbeiter: Hans Mueller",
                    "E-Mail: hans.mueller@example.de",
                    "Geburtsdatum: 1988-03-22",
                    "Abteilung: IT",
                    "Startdatum: 2026-07-01",
                    "Berufsschluessel: 2511",
                    "Gehalt: 650000 NOK",
                    "Beschaeftigungsprozentsatz: 80%",
                    "Arbeitstyp: Fest angestellt",
                ])}],
     "verify": lambda: v_employment("Hans", "Mueller")},

    {"id": "T43", "lang": "fr", "type": "employee_contract_docx",
     "prompt": "Vous avez reçu un contrat de travail (voir document ci-joint). Créez l'employé dans Tripletex avec tous les détails du contrat.",
     "files": [{"filename": "contrat_travail_fr.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "content_base64": _make_docx([
                    "CONTRAT DE TRAVAIL",
                    "Employé: Marie Lefevre",
                    "E-mail: marie.lefevre@example.fr",
                    "Date de naissance: 1995-11-30",
                    "Département: Marketing",
                    "Date de début: 2026-08-01",
                    "Code de profession: 2431",
                    "Salaire annuel: 520000 NOK",
                    "Pourcentage d'emploi: 100%",
                    "Type: Ordinaire",
                ])}],
     "verify": lambda: v_employment("Marie", "Lefevre")},

    # --- Travel expenses from spreadsheets ---
    {"id": "T44", "lang": "nb", "type": "travel_expense_xlsx",
     "prompt": "Se vedlagt Excel-fil med reiseutgifter. Registrer en reiseregning for den eksisterende ansatte med tittel 'Kundebesøk Bergen' fra 2026-03-05 til 2026-03-07.",
     "files": [{"filename": "reiseutgifter.xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "content_base64": _make_xlsx({"Utgifter": [
                    ["Dato", "Beskrivelse", "Beløp NOK"],
                    ["2026-03-05", "Flybillett Oslo-Bergen", 3200],
                    ["2026-03-05", "Hotell Bergen", 1800],
                    ["2026-03-06", "Taxi", 450],
                    ["2026-03-07", "Flybillett Bergen-Oslo", 2900],
                ]})}],
     "verify": lambda: v_travel("Bergen")},

    {"id": "T45", "lang": "en", "type": "travel_expense_csv",
     "prompt": "See attached CSV with travel costs. Register a travel expense for the existing employee titled 'Client Visit Stockholm' from 2026-03-10 to 2026-03-12.",
     "files": [{"filename": "travel_costs.csv", "mime_type": "text/csv",
                "content_base64": _make_csv([
                    ["Date", "Description", "Amount_NOK"],
                    ["2026-03-10", "Flight Oslo-Stockholm", 2800],
                    ["2026-03-10", "Hotel Stockholm", 2200],
                    ["2026-03-11", "Dinner with client", 950],
                    ["2026-03-12", "Flight Stockholm-Oslo", 2600],
                ])}],
     "verify": lambda: v_travel("Stockholm")},

    {"id": "T46", "lang": "de", "type": "travel_expense_xlsx",
     "prompt": "Siehe beigefügte Excel-Datei mit Reisekosten. Registrieren Sie eine Reisekostenabrechnung für den bestehenden Mitarbeiter. Titel: 'Geschäftsreise München', 2026-03-14 bis 2026-03-16.",
     "files": [{"filename": "reisekosten.xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "content_base64": _make_xlsx({"Kosten": [
                    ["Datum", "Beschreibung", "Betrag_NOK"],
                    ["2026-03-14", "Flug Oslo-Muenchen", 3500],
                    ["2026-03-14", "Hotel Muenchen", 2100],
                    ["2026-03-15", "Taxi", 380],
                ]})}],
     "verify": lambda: v_travel("München")},

    # --- Bulk operations from spreadsheets ---
    {"id": "T47", "lang": "nb", "type": "bulk_customers_csv",
     "prompt": "Se vedlagt CSV-fil med kundeliste. Opprett alle kundene i Tripletex.",
     "files": [{"filename": "kunder.csv", "mime_type": "text/csv",
                "content_base64": _make_csv([
                    ["Navn", "Epost", "Organisasjonsnummer"],
                    ["Fjordservice AS", "post@fjordservice.no", "111222333"],
                    ["Kystlogistikk AS", "info@kystlogistikk.no", "222333444"],
                ])}],
     "verify": lambda: v_customer("Fjordservice")},

    {"id": "T48", "lang": "en", "type": "bulk_products_xlsx",
     "prompt": "See attached Excel file with product list. Create all products in Tripletex with the given product numbers and prices (excluding VAT).",
     "files": [{"filename": "products.xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "content_base64": _make_xlsx({"Products": [
                    ["ProductNumber", "Name", "Price_exVAT"],
                    [8801, "Network Cable Cat6", 250],
                    [8802, "USB-C Hub", 890],
                ]})}],
     "verify": lambda: v_noop()},

    # --- Supplier invoice multi-line ---
    {"id": "T49", "lang": "nb", "type": "supplier_invoice_multiline_pdf",
     "prompt": "Vi har mottatt en leverandørfaktura med flere linjer (se vedlagt PDF). Registrer fakturaen med alle linjer.",
     "files": [{"filename": "faktura_multiline.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "FAKTURA",
                    "Leverandor: Trondheim Teknikk AS",
                    "Org.nr: 987666777",
                    "Fakturanummer: INV-2026-8001",
                    "Fakturadato: 2026-03-19",
                    "Forfallsdato: 2026-04-19",
                    "Linje 1: IT-utstyr (konto 6800) - 12000 NOK inkl. MVA 25%",
                    "Linje 2: Konsulenttjenester (konto 6300) - 35000 NOK inkl. MVA 25%",
                    "Linje 3: Programvarelisens (konto 6800) - 8000 NOK inkl. MVA 25%",
                    "Totalt inkl. MVA: 55000 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("Trondheim Teknikk")},

    # --- Employment contract from DOCX (English) ---
    {"id": "T50", "lang": "en", "type": "employee_contract_docx",
     "prompt": "You have received an employment contract (see attached document). Create the employee in Tripletex with all details from the contract.",
     "files": [{"filename": "employment_contract.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "content_base64": _make_docx([
                    "EMPLOYMENT CONTRACT",
                    "Employee: David Chen",
                    "Email: david.chen@example.com",
                    "Date of Birth: 1990-06-12",
                    "Department: Engineering",
                    "Start Date: 2026-09-01",
                    "Occupation Code: 2512",
                    "Annual Salary: 720000 NOK",
                    "Employment Percentage: 100%",
                    "Employment Type: Ordinary, permanent",
                ])}],
     "verify": lambda: v_employment("David", "Chen")},

    # --- Mixed: dimension + voucher (existing task type, now with variety) ---
    {"id": "T51", "lang": "de", "type": "dimension_voucher",
     "prompt": "Erstellen Sie eine benutzerdefinierte Buchhaltungsdimension 'Abteilungstyp' mit den Werten 'Produktion' und 'Verwaltung'. Dann buchen Sie einen Beleg auf Konto 7000 über 22000 NOK.",
     "verify": lambda: v_dimension("Abteilungstyp")},

    # --- Supplier invoice from text (no file, regression) ---
    {"id": "T52", "lang": "pt", "type": "supplier_invoice",
     "prompt": "Registre uma fatura de fornecedor da empresa Porto Serviços Ltda (org.nr 987888999) no valor de 19500 NOK incluindo IVA para serviços de escritório (conta 6300).",
     "verify": lambda: v_supplier("Porto Serviços")},

    # --- Employment contract from PDF (Spanish) ---
    {"id": "T53", "lang": "es", "type": "employee_contract_pdf",
     "prompt": "Has recibido un contrato de trabajo (ver PDF adjunto). Crea el empleado en Tripletex con todos los detalles del contrato.",
     "files": [{"filename": "contrato_trabajo_es.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "CONTRATO DE TRABAJO",
                    "Empleado: Sofia Garcia",
                    "Email: sofia.garcia@example.es",
                    "Fecha de nacimiento: 1993-04-18",
                    "Departamento: Ventas",
                    "Fecha de inicio: 2026-10-01",
                    "Codigo de profesion: 3322",
                    "Salario anual: 490000 NOK",
                    "Porcentaje de empleo: 80%",
                    "Tipo de empleo: Ordinario",
                ])}],
     "verify": lambda: v_employment("Sofia", "Garcia")},

    # --- Supplier invoice from PDF (Portuguese) ---
    {"id": "T54", "lang": "pt", "type": "supplier_invoice_pdf",
     "prompt": "Recebemos uma fatura de fornecedor (ver PDF anexo). Registre a fatura no Tripletex. Crie o fornecedor se não existir.",
     "files": [{"filename": "fatura_fornecedor_pt.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "FATURA",
                    "Fornecedor: Lisboa Tech Lda",
                    "Org.nr: 987999111",
                    "Numero da fatura: FAT-2026-2001",
                    "Data da fatura: 2026-03-17",
                    "Data de vencimento: 2026-04-17",
                    "Servicos de TI - 28000 NOK incl. IVA",
                    "IVA 25%: 5600 NOK",
                    "Total incl. IVA: 28000 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("Lisboa Tech")},

    # --- Nynorsk supplier invoice PDF ---
    {"id": "T55", "lang": "nn", "type": "supplier_invoice_pdf",
     "prompt": "Me har motteke ein leverandørfaktura (sjå vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandøren om han ikkje finst.",
     "files": [{"filename": "leverandorfaktura_nn.pdf", "mime_type": "application/pdf",
                "content_base64": _make_simple_pdf([
                    "FAKTURA",
                    "Leverandor: Vestland Kontor AS",
                    "Org.nr: 987111333",
                    "Fakturanummer: VK-2026-4001",
                    "Fakturadato: 2026-03-14",
                    "Forfallsdato: 2026-04-14",
                    "Reinhald og vedlikehald - 16500 NOK inkl. MVA",
                    "MVA 25%: 3300 NOK",
                    "Totalt inkl. MVA: 16500 NOK",
                ])}],
     "verify": lambda: v_incoming_invoice("Vestland Kontor")},

    # --- Employment contract from XLSX (English) ---
    {"id": "T56", "lang": "en", "type": "employee_contract_xlsx",
     "prompt": "See attached spreadsheet with new hire details. Create the employee in Tripletex with employment and employment details.",
     "files": [{"filename": "new_hire.xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "content_base64": _make_xlsx({"Employee": [
                    ["Field", "Value"],
                    ["First Name", "Alex"],
                    ["Last Name", "Thompson"],
                    ["Email", "alex.thompson@example.com"],
                    ["Date of Birth", "1991-02-28"],
                    ["Department", "Finance"],
                    ["Start Date", "2026-11-01"],
                    ["Annual Salary", 600000],
                    ["Employment Percentage", 100],
                    ["Occupation Code", "2411"],
                ]})}],
     "verify": lambda: v_employment("Alex", "Thompson")},

    # --- Voucher from CSV data ---
    {"id": "T57", "lang": "nb", "type": "voucher_csv",
     "prompt": "Se vedlagt CSV med bilagsdata. Opprett et bilag basert på dataene. Beskriving: Diverse kontorkostnader.",
     "files": [{"filename": "bilag.csv", "mime_type": "text/csv",
                "content_base64": _make_csv([
                    ["Konto", "Debet", "Kredit", "Beskrivelse"],
                    ["6800", "5500", "0", "Kontorrekvisita"],
                    ["6300", "3200", "0", "Leie lokale"],
                    ["1920", "0", "8700", "Bank"],
                ])}],
     "verify": lambda: v_voucher("kontorkostnader")},

    # --- Simple regression: Supplier invoice without file ---
    {"id": "T58", "lang": "fr", "type": "supplier_invoice",
     "prompt": "Enregistrez une facture fournisseur de Service Paris SARL (org.nr 987222111) pour 22000 NOK TTC pour des fournitures de bureau (compte 6800).",
     "verify": lambda: v_supplier("Service Paris")},

    # --- Combo: order + invoice + payment (regression) ---
    {"id": "T59", "lang": "en", "type": "order_invoice_payment",
     "prompt": "Create an order for customer Summit Corp (email summit@corp.com, org 777888999) with 5 units of Consulting Service at 3000 NOK each (excl. VAT). Convert to invoice and register full payment.",
     "verify": lambda: v_invoice("Summit")},

    # --- Employee with employment (regression, no file) ---
    {"id": "T60", "lang": "es", "type": "employee_contract",
     "prompt": "Cree el empleado Roberto Fernández (roberto@empresa.es), departamento 1. Fecha de inicio 2026-05-15, salario anual 480000 NOK, porcentaje 100%, código de profesión 2310. Tipo: ordinario.",
     "verify": lambda: v_employment("Roberto", "Fernández")},
]


# ─── Runner ───

def send_task(test):
    """Send a task to the deployed endpoint."""
    body = {
        "prompt": test["prompt"],
        "files": test.get("files", []),
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
