# Failure Analysis 3 — Full Competition Log (2026-03-21)

## Submission History

| # | Time | Score | Tasks | Notes |
|---|------|-------|-------|-------|
| 1 | ~00:30 | 0/? | 3 (entries 3,5,7) | Pre-fix: product_id, voucher, bank account |
| 2 | ~14:00 | 17/22 | 4 (entries 36,37,43,162) | Token expiry, no_completion, missing details |
| 3 | ~15:00 | 22/22 | ? | Internal test prompts — 100% |
| 4 | ~17:06 | 8/13 | 2 (entries 241,242) | Employee 7/7, Supplier invoice 1/6 |

---

## ALL 9 COMPETITION ENTRIES (by proxy URL)

### Entry 3 — Order+Invoice+Payment (Nynorsk, ~00:28)
**Prompt**: "Opprett ein ordre for kunden Vestfjord AS (org.nr 960144015) med produkta Konsulenttimar (1874) til 34750 kr og Nettverksteneste (9344) til 14350 kr. Konverter ordren til faktura og registrer full betaling."

**Result**: Completed in 11 turns (84s), 2 errors
- Tried to CREATE product 1874 instead of SEARCHING first → 422
- Missing deliveryDate on first order attempt → 422
- Eventually succeeded: order created, invoiced, payment registered
- **Likely partial pass** — order+invoice worked but inefficient (wasted turns)

**Root cause**: Agent tried `post_product` before `search_product`. System prompt says "Products referenced by number ALREADY EXIST" but Gemini ignored it.

---

### Entry 5 — Multi-VAT Invoice (German, ~00:43)
**Prompt**: "Erstellen Sie eine Rechnung für den Kunden Nordlicht GmbH (Org.-Nr. 855854171) mit drei Produktzeilen: Netzwerkdienst (2450) zu 28650 NOK mit 25% MwSt., Cloud-Speicher (6871) zu 13750 NOK mit 15% MwSt., und Wartung (2881) zu 18000 NOK mit 0% MwSt."

**Result**: Completed in 8 turns (66s), 4 errors
- `product_id` in orderLines NOT unflattened → 3x 422 "Request mapping failed"
- Gemini dropped product refs, used description-only orderLines → order created
- `invoice_order` → 422 "Faktura kan ikke opprettes før selskapet har registrert et bankkontonummer"
- **FAILED**: No bank account on competition sandbox (pre-_ensure_bank_account fix)
- **Now fixed**: `_ensure_bank_account()` runs per-task

**Root cause**: Two issues — (1) product_id not canonicalized in orderLines on OLD code, (2) no bank account. Both now fixed.

---

### Entry 7 — Supplier Invoice (English, ~01:59)
**Prompt**: "We have received invoice INV-2026-9075 from the supplier Brightstone Ltd (org no. 890932991) for 59800 NOK including VAT. The amount relates to office services (account 6300). Register the supplier invoice with the correct input VAT (25%)."

**Result**: Completed in 4 turns (45s), 2 errors
- Used `post_customer(isSupplier=true)` instead of `post_supplier` ✗
- Voucher postings failed — manually split gross/VAT amounts (47840 + 11960) instead of using vatType_id=1 auto-split ✗
- **FAILED**: Wrong entity type (customer vs supplier), wrong voucher structure

**Root cause**: Agent didn't use `post_supplier`, and manually calculated VAT instead of letting vatType handle it. System prompt pattern 14 now covers this correctly.

---

### Entry 36 — Dimension + Voucher Combo (Spanish, ~14:42)
**Prompt**: "Cree una dimensión contable personalizada 'Prosjekttype' con los valores 'Forskning' y 'Utvikling'. Luego registre un asiento en la cuenta 7000 por 14550 NOK, vinculado al valor de dimensión 'Forskning'."

**Result**: `no_completion_signal`, 4 turns (25s), 0 API calls
- Agent produced text-only responses — never made any tool calls
- **TOTAL FAILURE**: Nothing created

**Root cause**: Complex combo task confused the model. Now has pattern 20 in system prompt. Retested and works.

---

### Entry 37 — Overdue Invoice + Reminder (French, ~14:42)
**Prompt**: "L'un de vos clients a une facture en retard. Trouvez la facture en retard et enregistrez des frais de rappel de 60 NOK. Debit creances clients (1500), credit revenus de rappel (3400). Créez également une facture pour les frais de rappel au client et envoyez-la. De plus, enregistrez un paiement partiel de 5000 NOK sur la facture en retard."

**Result**: Completed in 5 turns (29s), 4 errors — ALL 403 "Invalid or expired proxy token"
- Every API call returned 403
- Token had expired before this task started processing
- **TOTAL FAILURE**: Nothing created

**Root cause**: Token expiry. This task was queued behind other slow tasks. ThreadPoolExecutor(max_workers=8) should prevent this now, but if competition sends tasks sequentially (one after another), this can still happen if a prior task takes too long.

---

### Entry 43 — Employment Contract from PDF (German, submission 2)
**Prompt**: "Sie haben einen Arbeitsvertrag erhalten (siehe beigefügte PDF). Erstellen Sie den Mitarbeiter in Tripletex mit allen Details aus dem Vertrag: Personalnummer, Geburtsdatum, Abteilung, Berufsschlüssel, Gehalt, Beschäftigungsprozentsatz und Startdatum."

**Result**: Completed in 4 turns (26s), 0 errors
- Created department "Kvalitetskontroll"
- Created employee Maximilian Fischer with DOB, national ID, bank account, email
- **DID NOT** create employment or employment_details
- Missing: Berufsschlüssel (occupation code), Gehalt (salary), Beschäftigungsprozentsatz (80%), Startdatum

**Root cause**: Agent completed too early — created employee but stopped before creating employment + details. The prompt explicitly requested these fields. Agent called `task_complete` after only creating the employee record.

**Likely check results**: Employee name/email/DOB ✓, but salary/occupation/start date/percentage ✗

---

### Entry 162 — Employment Contract from PDF (French, submission 2)
**Prompt**: "Vous avez reçu un contrat de travail (voir PDF ci-joint). Créez l'employé dans Tripletex avec tous les détails du contrat : numéro d'identité nationale, date de naissance, département, code de profession, salaire, pourcentage d'emploi et date de début."

**Result**: Completed in 8 turns (61s), 0 errors
- Created department "Produksjon"
- Created employee Arthur Moreau with national ID, bank account, DOB, email
- Created employment (startDate 2026-08-10)
- Searched occupation codes, found matching one
- Created employment details (salary 710000, 80%, occupation code, ORDINARY, MONTHLY_WAGE, NOT_SHIFT)
- **FULL SUCCESS**: All employment details created

**Why this worked but entry 43 didn't**: Entry 162 is from AFTER our employment tools were added and the system prompt was updated with pattern 17. Entry 43 was from an earlier submission where the tools existed but maybe the agent wasn't prompted strongly enough to use them all.

---

### Entry 241 — Simple Employee (Norwegian, latest submission) ✓ 7/7
**Prompt**: "Vi har en ny ansatt som heter Astrid Strand, født 4. May 1986. Opprett vedkommende som ansatt med e-post astrid.strand@example.org og startdato 21. December 2026."

**Result**: Completed in 4 turns (20s), 0 errors
- Created employee (Astrid Strand, DOB 1986-05-04, email astrid.strand@example.org)
- Created employment (startDate 2026-12-21, isMainEmployer=true)
- **ALL 7 CHECKS PASSED**

---

### Entry 242 — Supplier Invoice from PDF (Spanish, latest submission) ✗ 1/6
**Prompt**: "Has recibido una factura de proveedor (ver PDF adjunto). Registra la factura en Tripletex. Crea el proveedor si no existe. Usa la cuenta de gastos correcta y el IVA de entrada."
**File**: `leverandorfaktura_es_01.pdf`

**Result**: Completed in 6 turns (42s), 0 errors
- Created supplier Luna SL (org 966941901)
- Found account 6340 (Lys, varme) and 2400 (Leverandørgjeld)
- Created voucher: date=2026-03-13, INV-2026-7337
  - Posting 1: 48625 on 6340 with vatType 1 (25% input VAT)
  - Posting 2: -48625 on 2400 with supplier ref

**Check 1 PASSED**: Supplier created correctly
**Checks 2-6 FAILED**: Unknown — we cannot see the PDF content

**Possible failure causes**:
1. **Wrong amount**: Gemini extracted 48625 from PDF but actual amount differs
2. **Wrong account**: PDF may specify a different account than 6340
3. **Wrong date**: "2026-03-13" may not match invoice date in PDF
4. **Wrong invoice number**: "INV-2026-7337" may not match
5. **Voucher vs incoming invoice**: Checker may expect a proper incoming invoice entity, not a manual voucher. The `/supplierInvoice` or `/incomingInvoice` endpoint might be the expected flow.
6. **Missing description field**: The voucher description might need a specific format

**Most likely root cause**: The checker expects the supplier invoice to be registered via the `/supplierInvoice` or `/purchaseOrder` endpoint, NOT as a manual ledger voucher. Our fallback approach of creating vouchers for supplier invoices creates the correct accounting entries but may not create the proper "incoming invoice" entity that the checker looks for.

---

## AGGREGATE ERROR PATTERNS (All 242 entries)

| Error Type | Count | Impact | Status |
|-----------|-------|--------|--------|
| `post_employee` 422 (duplicate/validation) | 41 | LOW — sandbox state, recovers | OK in competition |
| `post_ledger_accountingDimensionValue` 422 | 28 | LOW — sandbox max 3 dims | OK in competition |
| `post_supplier` 401 | 25 | MEDIUM — token expiry | Fixed |
| `post_employee_employment` 422 | 14 | LOW — sandbox state | OK in competition |
| `post_ledger_accountingDimensionName` 422 | 14 | LOW — sandbox state | OK in competition |
| `search_invoice` 422/400/403 | 10 | MEDIUM — mixed causes | Mostly fixed |
| `post_product` 422 (duplicate) | 7 | LOW — sandbox state | OK in competition |
| `post_ledger_voucher` 422 | 7 | MEDIUM — posting validation | Partially fixed |
| `createReminder_invoice` 422 | 6 | MEDIUM — date validation | Fixed |
| `post_employee_employment_details` 422 | 6 | LOW — sandbox state | OK in competition |

**Key insight**: Most errors (>80%) are from sandbox state pollution (duplicate entities). In competition (fresh sandbox per task), these don't occur.

---

## CRITICAL FINDING: Supplier Invoice Registration Method

The failing task (entry 242) uses `post_ledger_voucher` to register the supplier invoice. This creates correct accounting entries but may NOT satisfy the checker.

The competition may expect one of:
1. **`/supplierInvoice`** — dedicated supplier invoice endpoint
2. **`/incomingInvoice`** — incoming invoice module (requires module activation)
3. **`/purchaseOrder`** — purchase order workflow

Our current approach (pattern 14 in system prompt) registers supplier invoices as manual vouchers. This works for the accounting but doesn't create a trackable invoice entity.

**Action needed**: Investigate if the competition sandbox has `/supplierInvoice` or `/incomingInvoice` enabled, and if the checker looks for those entities.

---

## PRIORITY FIXES FOR NEXT SUBMISSION

### P0: Supplier invoice entity (5 checks at stake)
- Test if `/supplierInvoice` endpoint is available on competition sandbox
- If available, add `post_supplierInvoice` tool and update pattern 14
- This alone could recover 5 checks → 13/13 (100%)

### P1: Employment contract completeness
- Entry 43 shows agent stopping after employee creation without employment details
- Strengthen pattern 17 to emphasize: ALWAYS create employment + employment_details when contract info is provided
- Add to system prompt: "If the task mentions salary, occupation code, start date, or employment percentage, you MUST create employment AND employment_details records"

### P2: Efficiency reduction
- Many tasks take 5-11 turns due to recoverable errors
- Each wasted turn reduces efficiency bonus (up to 2x multiplier at stake)
- Focus: reduce first-attempt failures for common patterns

---

## SCORE TRAJECTORY

| Submission | Score | Rate | Blocked By |
|-----------|-------|------|------------|
| 1 (early) | ~0/? | 0% | Bank account, product_id, voucher structure |
| 2 (14:00) | 17/22 | 77% | Token expiry, no_completion, missing employment details |
| 3 (15:00) | 22/22 | 100% | Internal test only |
| 4 (17:06) | 8/13 | 62% | Supplier invoice method (5 checks) |

**If supplier invoice fix lands**: 13/13 → 100% on next submission
