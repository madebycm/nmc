#!/usr/bin/env python3
"""
Generate Gemini function-calling tool declarations from the Tripletex OpenAPI spec extract.

One tool per operation with exact typed parameters — no more hallucinated field names.

Usage: python3 gen_tools.py
Output: gen_tools_output.json
"""

import json
from pathlib import Path

SPEC_PATH = Path(__file__).parent / "api_spec_extract.json"
OUTPUT_PATH = Path(__file__).parent / "gen_tools_output.json"

# ---------------------------------------------------------------------------
# Target endpoints: (path, method) pairs
# ---------------------------------------------------------------------------
TARGETS = [
    ("/employee", "GET"),
    ("/employee", "POST"),
    ("/employee/{id}", "PUT"),
    ("/department", "GET"),
    ("/department", "POST"),
    ("/customer", "GET"),
    ("/customer", "POST"),
    ("/customer/{id}", "PUT"),
    ("/contact", "GET"),
    ("/contact", "POST"),
    ("/product", "GET"),
    ("/product", "POST"),
    ("/order", "POST"),
    ("/order/{id}/:invoice", "PUT"),
    ("/invoice", "GET"),
    ("/invoice/{id}/:payment", "PUT"),
    ("/invoice/{id}/:createCreditNote", "PUT"),
    ("/invoice/{id}/:send", "PUT"),
    ("/travelExpense", "GET"),
    ("/travelExpense", "POST"),
    ("/travelExpense/{id}", "DELETE"),
    ("/travelExpense/cost", "POST"),
    ("/travelExpense/cost/{id}", "DELETE"),
    ("/travelExpense/paymentType", "GET"),
    ("/travelExpense/costCategory", "GET"),
    ("/project", "GET"),
    ("/project", "POST"),
    ("/ledger/voucher", "GET"),
    ("/ledger/voucher", "POST"),
    ("/ledger/voucher/{id}", "DELETE"),
    ("/ledger/account", "GET"),
    ("/company/{id}", "GET"),
    ("/company/>withLoginAccess", "GET"),
    ("/invoice/paymentType", "GET"),
    ("/employee/entitlement/:grantEntitlementsByTemplate", "PUT"),
    # Supplier
    ("/supplier", "GET"),
    ("/supplier", "POST"),
    # Incoming Invoice (supplier invoices)
    ("/incomingInvoice", "POST"),
]

# Meta fields to always skip from POST bodies
META_FIELDS = {"changes", "url"}

# Paging/sorting params — skip from tool declarations (handled by framework)
PAGING_PARAMS = {"from", "count", "sorting", "fields"}

# Schemas that are inline value objects (no id), not entity references.
# These get expanded as nested OBJECT params instead of flattened to _id.
INLINE_SCHEMAS = {
    "TravelDetails", "InternationalId", "AttestationStep",
    "IncomingInvoiceHeaderExternalWrite",
    "IncomingOrderLineExternalWrite",
}

# Array fields where we provide structured item schemas instead of bare OBJECT.
# Maps (tool_name, field_name) -> items schema definition.
ARRAY_ITEM_SCHEMAS = {
    ("post_ledger_voucher", "postings"): {
        "type": "OBJECT",
        "description": "A single posting/journal line within the voucher",
        "properties": {
            "account_id": {"type": "INTEGER", "description": "Ledger account ID (required)"},
            "amountGross": {"type": "NUMBER", "description": "Gross amount incl. VAT in NOK (use positive for debit, negative for credit)"},
            "amountGrossCurrency": {"type": "NUMBER", "description": "Gross amount in foreign currency (if applicable)"},
            "currency_id": {"type": "INTEGER", "description": "Currency ID (omit for NOK)"},
            "vatType_id": {"type": "INTEGER", "description": "VAT type ID"},
            "description": {"type": "STRING", "description": "Posting description/text"},
            "date": {"type": "STRING", "description": "Posting date (YYYY-MM-DD), defaults to voucher date"},
            "customer_id": {"type": "INTEGER", "description": "Customer ID (for receivable postings)"},
            "supplier_id": {"type": "INTEGER", "description": "Supplier ID (for payable postings)"},
            "employee_id": {"type": "INTEGER", "description": "Employee ID"},
            "project_id": {"type": "INTEGER", "description": "Project ID"},
            "department_id": {"type": "INTEGER", "description": "Department ID"},
            "product_id": {"type": "INTEGER", "description": "Product ID"},
        },
    },
    ("post_incomingInvoice", "orderLines"): {
        "type": "OBJECT",
        "description": "A single line on the supplier invoice",
        "properties": {
            "externalId": {"type": "STRING", "description": "REQUIRED. Unique external ID for this line (use 'line1', 'line2', etc.)"},
            "accountId": {"type": "INTEGER", "description": "Ledger account ID (e.g. 6300 for office services)"},
            "vatTypeId": {"type": "INTEGER", "description": "VAT type ID (5=25% incoming/inngående MVA, 6=exempt, 31=15% food)"},
            "amountInclVat": {"type": "NUMBER", "description": "Line amount INCLUDING VAT in invoice currency"},
            "description": {"type": "STRING", "description": "Line description"},
            "count": {"type": "NUMBER", "description": "Quantity (default 1)"},
            "departmentId": {"type": "INTEGER", "description": "Department ID (optional)"},
            "projectId": {"type": "INTEGER", "description": "Project ID (optional)"},
            "productId": {"type": "INTEGER", "description": "Product ID (optional)"},
        },
    },
    ("post_order", "orderLines"): {
        "type": "OBJECT",
        "description": "A single order line within the order",
        "properties": {
            "product_id": {"type": "INTEGER", "description": "Product ID (optional, can use description instead)"},
            "description": {"type": "STRING", "description": "Line description"},
            "count": {"type": "NUMBER", "description": "Quantity"},
            "unitPriceExcludingVatCurrency": {"type": "NUMBER", "description": "Unit price excl. VAT in order currency"},
            "unitPriceIncludingVatCurrency": {"type": "NUMBER", "description": "Unit price incl. VAT in order currency"},
            "vatType_id": {"type": "INTEGER", "description": "VAT type ID"},
            "discount": {"type": "NUMBER", "description": "Discount percentage"},
            "markup": {"type": "NUMBER", "description": "Markup percentage"},
            "currency_id": {"type": "INTEGER", "description": "Currency ID"},
        },
    },
}

# Additional required fields that the spec doesn't mark but the API enforces at runtime.
EXTRA_REQUIRED = {
    "post_order": ["deliveryDate", "orderDate"],
    "post_customer": ["name"],
    "post_travelExpense_cost": ["travelExpense_id"],
}

# Type overrides: fix spec types that Gemini rejects (e.g. ARRAY without items).
TYPE_OVERRIDES = {
    ("search_product", "productNumber"): {"type": "STRING", "description": "Product number to search for (single value)"},
    ("search_product", "number"): {"type": "STRING", "description": "Deprecated, use productNumber instead"},
}

# Per-endpoint body field overrides: only include these fields (if set).
# Keeps tools focused on the fields the LLM actually needs.
BODY_FIELD_ALLOWLIST = {
    "POST /employee": {
        "firstName", "lastName", "employeeNumber", "dateOfBirth", "email",
        "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
        "nationalIdentityNumber", "dnumber", "bankAccountNumber",
        "iban", "bic", "creditorBankCountryId", "usesAbroadPayment",
        "userType", "isContact", "comments", "address", "department",
        "employeeCategory",
    },
    "PUT /employee/{id}": {
        "id", "version",  # needed for PUT targeting
        "firstName", "lastName", "employeeNumber", "dateOfBirth", "email",
        "phoneNumberMobile", "phoneNumberHome", "phoneNumberWork",
        "nationalIdentityNumber", "dnumber", "bankAccountNumber",
        "iban", "bic", "creditorBankCountryId", "usesAbroadPayment",
        "userType", "isContact", "comments", "address", "department",
        "employeeCategory",
    },
    "POST /customer": {
        "name", "organizationNumber", "supplierNumber", "customerNumber",
        "isSupplier", "isInactive", "accountManager", "department",
        "email", "invoiceEmail", "overdueNoticeEmail",
        "bankAccounts", "phoneNumber", "phoneNumberMobile",
        "description", "language", "isPrivateIndividual",
        "singleCustomerInvoice", "invoiceSendMethod",
        "postalAddress", "physicalAddress", "deliveryAddress",
        "invoicesDueIn", "invoicesDueInType", "currency",
    },
    "PUT /customer/{id}": {
        "id", "version",
        "name", "organizationNumber", "supplierNumber", "customerNumber",
        "isSupplier", "isInactive", "accountManager", "department",
        "email", "invoiceEmail", "overdueNoticeEmail",
        "bankAccounts", "phoneNumber", "phoneNumberMobile",
        "description", "language", "isPrivateIndividual",
        "singleCustomerInvoice", "invoiceSendMethod",
        "postalAddress", "physicalAddress", "deliveryAddress",
        "invoicesDueIn", "invoicesDueInType", "currency",
        "displayName",
    },
    "POST /order": {
        "customer", "contact", "attn", "receiverEmail",
        "number", "reference", "ourContactEmployee", "department",
        "orderDate", "project", "invoiceComment", "currency",
        "invoicesDueIn", "invoicesDueInType",
        "deliveryDate", "deliveryAddress", "deliveryComment",
        "isPrioritizeAmountsIncludingVat",
        "orderLines", "markUpOrderLines", "discountPercentage",
    },
    "POST /travelExpense": {
        "project", "employee", "department",
        "travelDetails", "title",
        "isChargeable", "travelAdvance",
        "paymentCurrency", "vatType",
        "perDiemCompensations", "costs",
    },
    "POST /supplier": {
        "name", "organizationNumber", "supplierNumber",
        "email", "invoiceEmail", "phoneNumber", "phoneNumberMobile",
        "description", "language", "isPrivateIndividual",
        "postalAddress", "physicalAddress", "deliveryAddress",
        "bankAccounts", "currency", "ledgerAccount",
        "category1", "category2", "category3",
    },
    "POST /project": {
        "name", "number", "description",
        "projectManager", "department", "mainProject",
        "startDate", "endDate", "customer",
        "isClosed", "isInternal", "isOffer", "isFixedPrice",
        "projectCategory", "deliveryAddress",
        "reference", "externalAccountsNumber",
        "vatType", "fixedprice", "currency",
        "markUpOrderLines", "markUpFeesEarned",
        "isPriceCeiling", "priceCeilingAmount",
        "contact", "attention", "invoiceComment",
        "invoiceDueDate", "invoiceDueDateType",
        "invoiceReceiverEmail", "accessType",
    },
}

# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

def spec_type_to_gemini(prop: dict) -> dict:
    """Convert a spec property to Gemini schema type."""
    typ = prop.get("type", "string")
    enum = prop.get("enum")

    if enum:
        return {"type": "STRING", "enum": enum}

    mapping = {
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "string": "STRING",
        "array": "ARRAY",
        "object": "OBJECT",
    }
    return {"type": mapping.get(typ, "STRING")}


def build_inline_object_schema(schema_name: str, schemas: dict) -> dict:
    """Build a Gemini OBJECT schema from an inline (non-entity) schema."""
    schema = schemas.get(schema_name, {})
    props = schema.get("properties", {})
    out = {}
    for name, field in props.items():
        if field.get("readOnly"):
            continue
        if name in META_FIELDS:
            continue
        ref = field.get("$ref", "")
        if ref and field.get("type") == "object":
            # Nested ref inside inline object -> flatten to _id
            out[f"{name}_id"] = {
                "type": "INTEGER",
                "description": field.get("description", "") or f"ID of {ref}",
            }
        else:
            prop_def = spec_type_to_gemini(field)
            desc = field.get("description", "")
            if desc:
                prop_def["description"] = desc
            out[name] = prop_def
    return {"type": "OBJECT", "properties": out}


# ---------------------------------------------------------------------------
# Tool name generation
# ---------------------------------------------------------------------------

def make_tool_name(path: str, method: str) -> str:
    """Build a clean tool name from path + method."""
    clean = path.replace("/{id}", "").replace("/{travelExpenseId}", "")

    # Handle action endpoints like /:invoice, /:payment, /:send
    action = None
    if "/:" in clean:
        parts = clean.split("/:")
        clean = parts[0]
        action = parts[-1]

    # Handle aggregation endpoints like />withLoginAccess
    agg = None
    if "/>" in clean:
        parts = clean.split("/>")
        clean = parts[0]
        agg = parts[-1]

    # Build resource name from remaining path segments
    segments = [s for s in clean.strip("/").split("/") if s]
    resource = "_".join(segments) if segments else "root"

    if action:
        return f"{action}_{resource}"
    if agg:
        return f"get_{resource}_{agg}" if resource else f"get_{agg}"

    method_prefix = {
        "GET": "search",
        "POST": "post",
        "PUT": "put",
        "DELETE": "delete",
    }
    prefix = method_prefix.get(method, method.lower())

    # For single-resource GET with {id}, use "get" not "search"
    if method == "GET" and "{id}" in path:
        prefix = "get"

    return f"{prefix}_{resource}"


# ---------------------------------------------------------------------------
# Body field processing
# ---------------------------------------------------------------------------

def process_body_fields(
    body_schema: dict,
    method: str,
    path: str,
    schemas: dict,
) -> tuple[dict, list]:
    """
    Extract properties and required list from a request body schema.

    - Skips readOnly fields
    - Skips meta fields (changes, url) always; id/version only for POST
    - Flattens entity $ref objects to _id integer fields
    - Expands inline $ref objects as nested OBJECT params
    - Applies allowlist filtering if configured
    """
    props = body_schema.get("properties", {})
    allowlist = BODY_FIELD_ALLOWLIST.get(f"{method} {path}")
    out_props = {}
    required = []

    for name, field in props.items():
        # Skip meta fields
        if name in META_FIELDS:
            continue
        # For POST, skip id and version (server-assigned)
        if method == "POST" and name in ("id", "version"):
            continue

        # Skip readOnly fields
        if field.get("readOnly"):
            continue

        # Apply allowlist if configured (check original field name, not _id version)
        if allowlist and name not in allowlist:
            continue

        ref = field.get("$ref", "")
        is_required = field.get("required", False)
        description = field.get("description", "")
        typ = field.get("type", "")

        # Handle $ref objects
        if ref and typ == "object":
            if ref in INLINE_SCHEMAS:
                # Inline object: expand its fields as nested OBJECT
                prop_def = build_inline_object_schema(ref, schemas)
                prop_def["description"] = description or f"{ref} details"
                out_props[name] = prop_def
            else:
                # Entity reference: flatten to _id
                param_name = f"{name}_id"
                prop_def = {
                    "type": "INTEGER",
                    "description": description or f"ID of the {ref}",
                }
                out_props[param_name] = prop_def
                if is_required:
                    required.append(param_name)
            continue

        # Handle arrays of $ref
        if typ == "array":
            items = field.get("items", {})
            item_ref = items.get("$ref", "")
            if item_ref:
                prop_def = {
                    "type": "ARRAY",
                    "description": description or f"Array of {item_ref} objects (each needs at minimum an 'id' field, or full object for creation)",
                    "items": {"type": "OBJECT"},
                }
                out_props[name] = prop_def
                if is_required:
                    required.append(name)
                continue
            # Simple typed array
            gemini_item = spec_type_to_gemini(items)
            prop_def = {
                "type": "ARRAY",
                "description": description or f"Array of {items.get('type', 'string')}s",
                "items": gemini_item,
            }
            out_props[name] = prop_def
            if is_required:
                required.append(name)
            continue

        # Standard scalar
        prop_def = spec_type_to_gemini(field)
        if description:
            prop_def["description"] = description
        out_props[name] = prop_def
        if is_required:
            required.append(name)

    return out_props, required


def process_query_params(parameters: list) -> tuple[dict, list]:
    """Extract query/path parameters into Gemini properties."""
    out_props = {}
    required = []

    for param in parameters:
        loc = param.get("in")
        if loc not in ("query", "path"):
            continue

        name = param["name"]

        # Skip paging params (handled by framework)
        if name in PAGING_PARAMS:
            continue

        prop_def = spec_type_to_gemini(param)
        desc = param.get("description", "")
        if desc:
            prop_def["description"] = desc

        out_props[name] = prop_def
        if param.get("required", False):
            required.append(name)

    return out_props, required


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_tools(spec: dict) -> list[dict]:
    endpoints = spec["endpoints"]
    schemas = spec.get("schemas", {})
    tools = []

    for path, method in TARGETS:
        ep_data = endpoints.get(path, {}).get(method)
        if not ep_data:
            print(f"WARNING: {method} {path} not found in spec")
            continue

        tool_name = make_tool_name(path, method)
        summary = ep_data.get("summary", f"{method} {path}")
        description = f"{summary}\n\nAPI: {method} {path}"

        all_props = {}
        all_required = []

        # 1. Process path + query parameters
        parameters = ep_data.get("parameters", [])
        if parameters:
            props, req = process_query_params(parameters)
            all_props.update(props)
            all_required.extend(req)

        # 2. Process request body (POST/PUT with body)
        request_body = ep_data.get("requestBody", {})
        if request_body:
            for content_type, body_schema in request_body.items():
                if isinstance(body_schema, dict) and "properties" in body_schema:
                    props, req = process_body_fields(body_schema, method, path, schemas)
                    all_props.update(props)
                    all_required.extend(req)

        # Build the tool declaration
        tool = {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "OBJECT",
                "properties": all_props,
            },
        }
        if all_required:
            tool["parameters"]["required"] = sorted(set(all_required))

        tools.append(tool)

    return tools


def main():
    with open(SPEC_PATH) as f:
        spec = json.load(f)

    tools = generate_tools(spec)

    # Post-process: patch array item schemas for well-known arrays
    for tool in tools:
        props = tool["parameters"].get("properties", {})
        for field_name, prop in props.items():
            key = (tool["name"], field_name)
            if key in ARRAY_ITEM_SCHEMAS:
                prop["items"] = ARRAY_ITEM_SCHEMAS[key]

    # Post-process: add extra required fields
    for tool in tools:
        extra = EXTRA_REQUIRED.get(tool["name"], [])
        if extra:
            existing = set(tool["parameters"].get("required", []))
            existing.update(extra)
            tool["parameters"]["required"] = sorted(existing)

    # Post-process: apply type overrides
    for tool in tools:
        props = tool["parameters"].get("properties", {})
        for field_name in list(props.keys()):
            key = (tool["name"], field_name)
            if key in TYPE_OVERRIDES:
                props[field_name] = TYPE_OVERRIDES[key]

    output = {"tools": tools, "_meta": {"count": len(tools)}}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(tools)} tool declarations -> {OUTPUT_PATH}")
    print()
    for t in tools:
        props = t["parameters"].get("properties", {})
        req = t["parameters"].get("required", [])
        print(f"  {t['name']:50s} {len(props):2d} params ({len(req)} required)")


if __name__ == "__main__":
    main()
