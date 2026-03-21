# Pivot: Universal Attachment Pipeline + Supplier Invoice Entity Fix

**Date**: 2026-03-21
**Deadline**: 2026-03-22 15:00 CET (~18 hours)
**Goal**: 100% on all competition tasks including document-attached ones

---

## Failure Root Cause (Entry 242 — grounded)

Entry 242: Supplier invoice from PDF (`leverandorfaktura_es_01.pdf`), Spanish prompt.

**What happened**:
1. Gemini correctly read the PDF → extracted supplier Luna SL (org 966941901), amount 48625, account 6340
2. Agent created supplier via `post_supplier` → **Check 1 PASSED**
3. Agent used `post_ledger_voucher` (manual journal entry) with postings on 6340 + 2400
4. **Checks 2-6 FAILED** — checker queries `/incomingInvoice` or `/supplierInvoice` entity, not raw vouchers

**Diagnosis**: Correct accounting, wrong entity type. The `post_incomingInvoice` tool already exists in TOOL_MAP and gen_tools_output.json but pattern 14 in prompts.py directs Gemini to use `post_ledger_voucher` instead.

**Secondary risk**: If PDF extraction was wrong (amount, date, invoice number), checks would also fail. We cannot verify this from logs alone.

---

## Current File Handling Gap (agent.py:466-478)

```
MIME type                           → What Gemini sees
─────────────────────────────────────────────────────────
image/*                             → inlineData (native vision) ✓
application/pdf                     → inlineData (native vision) ✓
text/csv                            → UTF-8 decode attempt (may work) ~
application/vnd.ms-excel (.xls)     → binary fallback "[Attached binary file]" ✗
application/vnd.openxmlformats...   → binary fallback "[Attached binary file]" ✗
  (.xlsx)
application/msword (.doc)           → binary fallback "[Attached binary file]" ✗
application/vnd.openxmlformats...   → binary fallback "[Attached binary file]" ✗
  (.docx)
text/plain                          → UTF-8 decode (works) ✓
```

**Impact**: If competition sends .xlsx/.docx, Gemini gets zero content. Total blindness.

---

## Implementation Plan

### Phase 1: Fix supplier invoice entity (30 min) — 5 POINTS AT STAKE

1. **Update pattern 14** in `prompts.py`:
   - PRIMARY path: `post_incomingInvoice` with `sendTo="ledger"`
   - FALLBACK only if 403: `post_ledger_voucher` (current approach)

2. **Route fix in tool_router.py**:
   - `post_incomingInvoice` body_query routing already correct
   - `_unflatten_refs` won't corrupt nested `invoiceHeader`/`orderLines` (camelCase IDs, not `_id` format)
   - Verify: `sendTo` goes to query params, rest to body ✓

3. **Pattern 14 rewrite**:
```
14. REGISTER SUPPLIER INVOICE (leverandørfaktura):
    → post_supplier(name, organizationNumber) OR search_supplier to find existing
    → search_ledger_account (find expense account, e.g. 6300)
    → post_incomingInvoice(
        sendTo="ledger",
        invoiceHeader={
          vendorId: SUPPLIER_ID,
          invoiceDate: "YYYY-MM-DD",
          dueDate: "YYYY-MM-DD" (30 days from invoice date if not specified),
          currencyId: 1,
          invoiceAmount: TOTAL_INCL_VAT,
          invoiceNumber: "INV-XXXX",
          description: "..."
        },
        orderLines=[{
          externalId: "line1",
          accountId: EXPENSE_ACCOUNT_ID,
          vatTypeId: 5 (25% input VAT),
          amountInclVat: AMOUNT_INCL_VAT,
          description: "...",
          count: 1
        }]
      )
    FALLBACK: If post_incomingInvoice returns 403 (module not enabled), use post_ledger_voucher pattern.
```

**NOTE on vatTypeId**: The incoming invoice uses different vatType IDs than the voucher system.
From gen_tools_output.json: `vatTypeId: 5=25% incoming, 6=exempt, 31=15% food`
This differs from voucher vatType_id=1. Need to verify — this is a CRITICAL detail.

### Phase 2: Document pipeline (45 min)

**Dependencies** (lightweight, no pandas):
```
openpyxl>=3.1.0    # ~4MB — xlsx read
python-docx>=0.8.11 # ~1MB — docx read
```

**New function `_parse_attachment()`** in agent.py:

```python
def _parse_attachment(f: dict) -> list[dict]:
    """Convert attachment to Gemini-compatible parts.

    Returns list of Gemini content parts (inlineData or text).
    Routes by MIME type + extension for maximum coverage.
    """
    mime = f.get("mime_type", "application/octet-stream")
    filename = f.get("filename", "unknown")
    file_data = base64.b64decode(f["content_base64"])
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # 1. VISUAL: PDF, images → native Gemini vision
    if mime.startswith("image/") or mime == "application/pdf":
        return [
            {"inlineData": {"mimeType": mime, "data": f["content_base64"]}},
            {"text": f"[Attached: {filename}] — Extract ALL amounts, dates, names, and numbers exactly as shown."}
        ]

    # 2. EXCEL: .xlsx, .xls → parse to markdown table
    if ext in ("xlsx", "xls") or "spreadsheet" in mime or "excel" in mime:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(file_data), read_only=True, data_only=True)
            tables = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                rows = list(ws.iter_rows(values_only=True))
                if not rows:
                    continue
                # Build markdown table
                headers = [str(c) if c is not None else "" for c in rows[0]]
                md = "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in rows[1:]:
                    cells = [str(c) if c is not None else "" for c in row]
                    md += "| " + " | ".join(cells) + " |\n"
                tables.append(f"### Sheet: {sheet}\n{md}")
            wb.close()
            text = "\n".join(tables)
            return [{"text": f"[Spreadsheet: {filename}]\n{text}"}]
        except Exception as e:
            log.error("Failed to parse %s: %s", filename, e)
            return [{"text": f"[Spreadsheet: {filename}] — Could not parse file."}]

    # 3. DOCX → extract text
    if ext == "docx" or "wordprocessingml" in mime:
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_data))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [{"text": f"[Document: {filename}]\n{text[:8000]}"}]
        except Exception as e:
            log.error("Failed to parse %s: %s", filename, e)
            return [{"text": f"[Document: {filename}] — Could not parse file."}]

    # 4. CSV → read as text (already works but add header)
    if ext == "csv" or mime == "text/csv":
        try:
            text = file_data.decode("utf-8", errors="replace")
            return [{"text": f"[CSV Data: {filename}]\n{text[:8000]}"}]
        except Exception:
            return [{"text": f"[CSV: {filename}] — Could not parse file."}]

    # 5. Plain text fallback
    try:
        text = file_data.decode("utf-8", errors="replace")
        return [{"text": f"[File: {filename}]\n{text[:8000]}"}]
    except Exception:
        return [{"text": f"[Attached: {filename} ({mime})] — Binary file, cannot read."}]
```

### Phase 3: Prompt hardening for documents (15 min)

Add to SYSTEM_PROMPT (compact, not a new section — fold into existing rules):

```
DOCUMENT HANDLING:
- If a file is attached, it is the SOURCE OF TRUTH for amounts, dates, names, and numbers.
- For supplier invoices from PDF: use EXACT total amount including VAT from the document. Never recalculate.
- For employment contracts: you MUST extract AND create employment + employment_details (salary, start date, occupation code, percentage).
- For spreadsheets: process ALL rows in the data.
```

### Phase 4: Comprehensive testing (60 min)

Generate 25 test prompts covering:

| # | Type | Language | Attachment | Expected Entity |
|---|------|----------|------------|-----------------|
| 1 | Supplier invoice | nb | PDF | incomingInvoice + supplier |
| 2 | Supplier invoice | de | PDF | incomingInvoice + supplier |
| 3 | Supplier invoice | es | PDF | incomingInvoice + supplier |
| 4 | Supplier invoice | fr | PDF | incomingInvoice + supplier |
| 5 | Supplier invoice | en | PDF | incomingInvoice + supplier |
| 6 | Employment contract | nb | PDF | employee + employment + details |
| 7 | Employment contract | de | PDF | employee + employment + details |
| 8 | Employment contract | fr | DOCX | employee + employment + details |
| 9 | Employment contract | en | PDF | employee + employment + details |
| 10 | Employment contract | es | PDF | employee + employment + details |
| 11 | Travel expenses | nb | XLSX | travelExpense + costs |
| 12 | Travel expenses | en | CSV | travelExpense + costs |
| 13 | Travel expenses | de | XLSX | travelExpense + costs |
| 14 | Bulk customers | nb | CSV | multiple customers |
| 15 | Bulk products | en | XLSX | multiple products |
| 16 | Order+Invoice | nn | none | order + invoice |
| 17 | Order+Invoice+Payment | nb | none | order + invoice + payment |
| 18 | Credit note | en | none | credit note |
| 19 | Reminder | fr | none | reminder on overdue invoice |
| 20 | Dimension+Voucher | es | none | dimension + voucher |
| 21 | Voucher | de | none | voucher |
| 22 | Employee admin | nb | none | employee + entitlements |
| 23 | Project | pt | none | project |
| 24 | Supplier invoice multi-line | nb | PDF | incomingInvoice, 3 lines |
| 25 | Supplier invoice + payment | en | PDF | incomingInvoice + partial payment |

**Test generates**: fake PDFs (reportlab/fpdf or raw), fake XLSX (openpyxl), fake DOCX (python-docx), fake CSV (string).
Each test verifies: correct entity created, field values match fixture.

---

## Critical Risk: vatTypeId mapping

The `post_incomingInvoice` orderLines use `vatTypeId` directly (5, 6, 31) — NOT the same IDs as `post_ledger_voucher` postings (1, 3, etc.).

From gen_tools_output.json line 2263:
> `vatTypeId: 5=25% incoming/inngående MVA, 6=exempt, 31=15% food`

But this might be WRONG in the tool description. The actual API spec says it's just "ID of the VAT type" without specifying which IDs map to what. We MUST test this with the actual API.

**Safest approach**: Use the same vatType IDs as the voucher system (1=25% input, 3=25% output) since those are verified working. Test both.

---

## Deployment Checklist

1. [ ] Install openpyxl, python-docx on VPS
2. [ ] Update agent.py: replace file handling block with _parse_attachment()
3. [ ] Update prompts.py: pattern 14 (incomingInvoice primary), document handling rules
4. [ ] Deploy to VPS, restart service
5. [ ] Run 25-prompt test suite
6. [ ] Verify: supplier invoice creates incomingInvoice entity (not voucher)
7. [ ] Verify: XLSX/DOCX/CSV parsed correctly
8. [ ] Submit to competition

---

## Success Criteria

- Supplier invoice tasks: 6/6 checks (not 1/6)
- Employment from PDF: 7/7 checks
- XLSX/CSV tasks: parsed and processed (no binary fallback)
- All existing passing tests still pass (regression)
- Total competition score: 100%
