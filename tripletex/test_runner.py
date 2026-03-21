#!/usr/bin/env python3
"""Replay captured competition prompts against local agent + sandbox.

Runs prompts from logs/prompts_dump/, calls solve_task_sync locally,
logs structured results in real-time to logs/test_results.jsonl.

Usage:
  python test_runner.py                     # Run all 205
  python test_runner.py --count 5           # Run first 5 only
  python test_runner.py --type payroll      # Run only payroll prompts
  python test_runner.py --id 3              # Run prompt #3 only
  python test_runner.py --list              # List all prompts with types
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime

# ─── Env setup (before agent import) ───

os.environ.setdefault("GOOGLE_API_KEY", "AIzaSyDU9JLjRiaEfmp-Y7n7ilv91uhgk0wDv3U")
os.environ.setdefault("TASK_LOG_FILE", os.path.join(os.path.dirname(__file__), "logs", "test_task_log.jsonl"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("test_runner")

# ─── Sandbox credentials (our persistent dev sandbox) ───

SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjM5OTU2LCJ0b2tlbiI6ImM0YzUyMWM2LTBkMjItNGVhYy1iZmVhLWJlOWRjMjM0N2I3ZSJ9",
}

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "logs", "prompts_dump")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "logs", "test_results.jsonl")


# ─── Task type classifier ───

TYPE_PATTERNS = [
    ("bank_reconciliation", r"(reconcili|bankavsteming|bankavstemmming|extrato banc|rapprochement|concilia|bankutskrift|CSV.*faktura|bank.*statement)"),
    ("year_end_closing", r"(year.?end|årsoppgj|arsoppgj|cierre.*(anual|ejercicio)|cl[oô]ture.*(annuelle|exercice)|encerramento.*anual|forenkla.*arsoppgjer|simplified.*year|vereinfachten.*Jahresabschluss)"),
    ("monthly_closing", r"(monthly.?clos|måneds.*avslut|maneds.*avslut|månavslutn|cierre.*mens|cl[oô]ture.*mens|encerramento.*mens|periodereg|periodis|Monatsabschluss|Rechnungsabgrenzung)"),
    ("ledger_correction", r"(ledger.*error|hovedbok.*feil|grand.*livre.*erreur|libro.*mayor.*error|razão.*erro|korrig|errors.*general.*ledger|erreurs.*grand.*livre|errores.*libro)"),
    ("cost_analysis", r"(cost.*analys|kostnads.*analys|kostn.*stieg|análisis.*cost|analyse.*co[uû]t|análise.*cust|deutlich gestiegen|aumentaron significativamente|augmenté.*significat|costs.*increased.*significantly|kosten.*gestiegen)"),
    ("payroll", r"(payroll|l[oø]nnskj[oø]r|l[oø]nn.*kj[oø]r|nómina|bulletin.*paie|folha.*pagamento|Gehaltsabrechnung|run payroll|exécutez la paie|ejecute la nómina|[Kk][oø]yr l[oø]n|[Kk]j[oø]r l[oø]nn)"),
    ("project_billing", r"(prosjektfaktura|project.*invoice|factur.*projet|factura.*proyecto|fatura.*projeto|timer.*registrer|hour.*register|registrer.*timer|registre.*horas|enregistrez.*heures|Register.*hours|Registrer.*timer|fastpris|fixed.*price.*project|Festpreis)"),
    ("travel_expense", r"(travel.*expense|reiseregning|note.*frais.*d[eé]placement|nota.*viaje|despesa.*viagem|Reisekosten|frais.*deplacement)"),
    ("expense_receipt", r"(receipt.*post|kvittering|re[cç]u.*enregistr|recibo.*registr|Quittung|depense.*re[cç]u|despesa.*recibo|expense.*from.*receipt|Togbillett.*expense)"),
    ("supplier_invoice", r"(supplier.*invoice|leverandørfaktura|facture.*fournisseur|factura.*proveedor|fatura.*fornecedor|Lieferantenrechnung|incoming.?invoice|fournisseur.*enregistr|registrer.*leverand|register.*supplier|enregistrez.*fournisseur|registre.*fornecedor)"),
    ("credit_note", r"(credit.*note|kreditnota|kreditert|note.*cr[eé]dit|nota.*cr[eé]dito|Gutschrift|reklamert|r[eé]clam[eé])"),
    ("reverse_payment", r"(reverse.*payment|tilbakef[oø]r.*betal|annuler.*paiement|revertir.*pago|reverter.*pagamento|Zahlung.*r[uü]ckg|storniert|storno|payment.*reversed|Betalingen fra.*tilbake|Betalinga fr[aå].*tilbake|[Rr]everser betalin)"),
    ("reminder_fee", r"(reminder.*fee|purring|rappel|recordatorio|lembrete|Mahnung|overdue.*invoice|forfait|factura.*pendiente|facture.*impay|faktura.*vencida|unbezahlt|forfallen)"),
    ("multi_currency_payment", r"(currency.*exchange|valutakurs|taux.*change|tipo.*cambio|taxa.*c[aâ]mbio|Wechselkurs|NOK/EUR|EUR.*kurs|kursen var)"),
    ("register_payment", r"(register.*payment|registrer.*betal|enregistr.*paiement|registrar.*pago|registar.*pagamento|Zahlung.*registr|betalt.*faktura|impay[eé]e|vencida|unbezahlt)"),
    ("custom_dimension", r"(dimension|dimensjon|Produktlinje|accounting.*dimension)"),
    ("create_invoice", r"(create.*invoice|opprett.*faktura|cr[eé]e[rz].*factur|crear.*factura|crie.*fatura|Rechnung.*erstell|send.*faktura|envie.*fatura|envoy.*facture|erstellen.*senden.*Rechnung|lag.*faktura|Erstellen Sie eine Rechnung|env[ií]a una factura|Crea y env)"),
    ("create_order", r"(create.*order|opprett.*bestilling|cr[eé]e[rz].*commande|crear.*pedido|crie.*pedido|Bestellung.*erstell|Auftrag.*erstell|Crea un pedido)"),
    ("create_project", r"(create.*project|opprett.*prosjekt|cr[eé]e[rz].*projet|crear.*proyecto|crie.*projeto|Projekt.*erstell|vinculado|linked.*customer|pre[cç]o fixo|Defina.*pre[cç]o)"),
    ("create_employee", r"(create.*employee|opprett.*ansatt|ny.*tilsett|cr[eé]e[rz].*employ|crear.*empleado|crie.*empregado|Mitarbeiter.*erstell|new.*employee|ansett|nuevo.*empleado|nouvel.*employ|Arbeitsvertrag|employment.*contract|offer.*letter|lettre.*offre|carta.*oferta|tilbudsbrev|novo funcion[aá]rio|Crie-o como func)"),
    ("create_customer", r"(create.*customer|opprett.*kunde|cr[eé]e[rz].*client|crear.*cliente|crie.*cliente|Kunden.*erstell|Crea el cliente)"),
    ("create_product", r"(create.*product|opprett.*produkt|cr[eé]e[rz].*produit|crear.*producto|crie.*produto|Produkt.*erstell|Crea el producto|Erstellen Sie.*Produkt)"),
    ("create_department", r"(create.*department|opprett.*avdeling|cr[eé]e[rz].*d[eé]partement|crear.*departamento|crie.*departamento|Abteilung.*erstell|Erstellen Sie drei Abteilungen)"),
]


def classify_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for task_type, pattern in TYPE_PATTERNS:
        if re.search(pattern, prompt_lower, re.IGNORECASE):
            return task_type
    return "unknown"


# ─── Load prompts ───

def load_prompts():
    """Load all prompts from dump directory, sorted by timestamp."""
    prompts = []
    for fname in sorted(os.listdir(PROMPTS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(PROMPTS_DIR, fname)) as f:
            data = json.load(f)
        body = data.get("body", data)
        prompt_text = body.get("prompt", "")
        files = body.get("files", [])
        prompts.append({
            "file": fname,
            "prompt": prompt_text,
            "files": files,
            "type": classify_prompt(prompt_text),
            "timestamp": data.get("logged_at", fname),
        })
    return prompts


# ─── Run one prompt ───

def run_prompt(idx: int, entry: dict) -> dict:
    """Run a single prompt through the agent. Returns structured result."""
    prompt = entry["prompt"]
    task_type = entry["type"]
    log.info("=" * 70)
    log.info("[%d] TYPE: %s", idx, task_type)
    log.info("[%d] PROMPT: %s", idx, prompt[:120])

    body = {
        "prompt": prompt,
        "files": entry.get("files", []),
        "tripletex_credentials": SANDBOX_CREDS,
    }

    result = {
        "idx": idx,
        "file": entry["file"],
        "type": task_type,
        "prompt": prompt[:200],
        "has_files": len(entry.get("files", [])) > 0,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    t0 = time.time()
    try:
        from agent import solve_task_sync
        solve_task_sync(body)
        elapsed = round(time.time() - t0, 1)
        result["status"] = "completed"
        result["elapsed_s"] = elapsed
        log.info("[%d] COMPLETED in %.1fs", idx, elapsed)

    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        result["status"] = "crashed"
        result["error"] = str(e)[-500:]
        result["elapsed_s"] = elapsed
        log.error("[%d] CRASHED after %.1fs: %s", idx, elapsed, str(e)[:200])

    result["finished_at"] = datetime.utcnow().isoformat() + "Z"

    # Append to results log in real-time
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


# ─── Summary ───

C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_YELLOW = "\033[33m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RESET = "\033[0m"


def print_summary(results: list):
    total = len(results)
    completed = sum(1 for r in results if r["status"] == "completed")
    crashed = sum(1 for r in results if r["status"] == "crashed")

    print(f"\n{'=' * 70}")
    print(f"{C_BOLD}RESULTS: {completed}/{total} completed, {crashed} crashed{C_RESET}")
    print(f"{'=' * 70}")

    # By type
    by_type = {}
    for r in results:
        t = r["type"]
        by_type.setdefault(t, {"ok": 0, "fail": 0, "times": []})
        if r["status"] == "completed":
            by_type[t]["ok"] += 1
        else:
            by_type[t]["fail"] += 1
        by_type[t]["times"].append(r.get("elapsed_s", 0))

    print(f"\n{'Type':<28} {'OK':>4} {'FAIL':>4} {'Avg(s)':>7}")
    print("-" * 50)
    for t in sorted(by_type.keys()):
        d = by_type[t]
        avg = sum(d["times"]) / len(d["times"]) if d["times"] else 0
        color = C_GREEN if d["fail"] == 0 else C_RED
        print(f"{color}{t:<28} {d['ok']:>4} {d['fail']:>4} {avg:>7.1f}{C_RESET}")

    # Failures detail
    failures = [r for r in results if r["status"] != "completed"]
    if failures:
        print(f"\n{C_RED}{C_BOLD}FAILURES:{C_RESET}")
        for r in failures:
            print(f"  [{r['idx']}] {r['type']} — {r.get('error', '?')[:120]}")

    total_time = sum(r.get("elapsed_s", 0) for r in results)
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Replay competition prompts locally")
    parser.add_argument("--count", type=int, help="Run first N prompts only")
    parser.add_argument("--type", type=str, help="Run only this task type")
    parser.add_argument("--id", type=int, help="Run only prompt #N (1-indexed)")
    parser.add_argument("--list", action="store_true", help="List all prompts and exit")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N prompts")
    args = parser.parse_args()

    prompts = load_prompts()
    log.info("Loaded %d prompts from %s", len(prompts), PROMPTS_DIR)

    if args.list:
        for i, p in enumerate(prompts, 1):
            files = " [+FILES]" if p.get("files") else ""
            print(f"{i:3d}. [{p['type']:<24}] {p['prompt'][:90]}{files}")
        print(f"\nTotal: {len(prompts)}")
        # Type summary
        types = {}
        for p in prompts:
            types[p["type"]] = types.get(p["type"], 0) + 1
        print(f"\nBy type:")
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {t:<28} {c}")
        return

    # Filter
    if args.id:
        prompts = [prompts[args.id - 1]]
    elif args.type:
        prompts = [p for p in prompts if p["type"] == args.type]
        log.info("Filtered to %d prompts of type '%s'", len(prompts), args.type)

    if args.offset:
        prompts = prompts[args.offset:]

    if args.count:
        prompts = prompts[:args.count]

    if not prompts:
        print("No prompts matched. Use --list to see available prompts.")
        return

    log.info("Running %d prompts", len(prompts))

    results = []
    for i, entry in enumerate(prompts, 1):
        result = run_prompt(i, entry)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
