"""Test the agent locally against the sandbox."""

import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Test prompts matching competition task types
TEST_PROMPTS = {
    "create_employee": (
        "Opprett en ansatt med navn Kari Hansen, e-post kari@firma.no. "
        "Hun skal være kontoadministrator."
    ),
    "create_customer": (
        "Registrer en ny kunde med navn Nordvik Bygg AS, "
        "e-post post@nordvikbygg.no, telefon 99887766."
    ),
    "create_department": (
        "Opprett en ny avdeling med navn Salg."
    ),
    "create_product": (
        "Opprett et produkt med navn Konsulenttjenester, pris 1500 kr ekskl. mva."
    ),
    "create_invoice": (
        "Opprett en faktura til kunde Berg Elektro AS (e-post: faktura@bergelektro.no). "
        "Fakturaen skal inneholde 3 stk Installasjon (pris 2500 kr ekskl. mva per stk). "
        "Forfallsdato 14 dager fra i dag."
    ),
    "create_project": (
        "Opprett et prosjekt med navn 'Kontorbygg Oslo' for kunde Viken Eiendom AS. "
        "Prosjektleder skal være den eksisterende ansatte i systemet."
    ),
    "travel_expense": (
        "Registrer en reiseregning for den eksisterende ansatte i systemet. "
        "Tittel: Kundemøte Bergen. Beløp: 3500 kr."
    ),
    "english_task": (
        "Create a new employee named John Smith with email john@company.com. "
        "He should have standard access."
    ),
    "german_task": (
        "Erstellen Sie einen neuen Kunden mit dem Namen Müller GmbH, "
        "E-Mail info@mueller.de, Telefon +49 30 12345678."
    ),
}

SANDBOX_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "eyJ0b2tlbklkIjoyMTQ3NjM5OTU2LCJ0b2tlbiI6ImM0YzUyMWM2LTBkMjItNGVhYy1iZmVhLWJlOWRjMjM0N2I3ZSJ9",
}


async def test(task_name: str):
    from agent import solve_task

    prompt = TEST_PROMPTS.get(task_name)
    if not prompt:
        print(f"Unknown task: {task_name}")
        print(f"Available: {', '.join(TEST_PROMPTS.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Testing: {task_name}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    body = {
        "prompt": prompt,
        "files": [],
        "tripletex_credentials": SANDBOX_CREDS,
    }

    await solve_task(body)
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "create_customer"
    asyncio.run(test(task))
