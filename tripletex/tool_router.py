"""Routes typed tool calls back to HTTP API requests.

Each typed tool (e.g. post_customer, search_employee) maps to a specific
HTTP method + endpoint. This module converts the flat tool params back into
the correct HTTP request shape (query params, body with nested refs).
"""

import logging

log = logging.getLogger("tripletex.router")

# ─── Tool → HTTP mapping ───
# Each entry: tool_name → (method, endpoint_pattern, param_type)
# param_type: "query" = all params go to query string
#             "body"  = all params go to JSON body
#             "body_query" = QUERY_FIELDS go to query, rest to body
#             "path_query" = id goes to path, rest to query
#             "path_body"  = id goes to path, rest to body

TOOL_MAP = {
    # Employee
    "search_employee":      ("GET",    "/employee",           "query"),
    "post_employee":        ("POST",   "/employee",           "body"),
    "put_employee":         ("PUT",    "/employee/{id}",      "path_body"),
    # Department
    "search_department":    ("GET",    "/department",          "query"),
    "post_department":      ("POST",   "/department",          "body"),
    # Customer
    "search_customer":      ("GET",    "/customer",            "query"),
    "post_customer":        ("POST",   "/customer",            "body"),
    "put_customer":         ("PUT",    "/customer/{id}",       "path_body"),
    # Contact
    "search_contact":       ("GET",    "/contact",             "query"),
    "post_contact":         ("POST",   "/contact",             "body"),
    # Product
    "search_product":       ("GET",    "/product",             "query"),
    "post_product":         ("POST",   "/product",             "body"),
    # Order
    "post_order":           ("POST",   "/order",               "body"),
    "invoice_order":        ("PUT",    "/order/{id}/:invoice", "path_query"),
    # Invoice
    "search_invoice":       ("GET",    "/invoice",             "query"),
    "payment_invoice":      ("PUT",    "/invoice/{id}/:payment",        "path_query"),
    "createCreditNote_invoice": ("PUT", "/invoice/{id}/:createCreditNote", "path_query"),
    "send_invoice":         ("PUT",    "/invoice/{id}/:send",  "path_query"),
    "search_invoice_paymentType": ("GET", "/invoice/paymentType", "query"),
    # Supplier
    "search_supplier":      ("GET",    "/supplier",            "query"),
    "post_supplier":        ("POST",   "/supplier",            "body"),
    # Incoming Invoice (supplier invoices)
    "post_incomingInvoice": ("POST",   "/incomingInvoice",     "body_query"),
    # Travel Expense
    "search_travelExpense": ("GET",    "/travelExpense",       "query"),
    "post_travelExpense":   ("POST",   "/travelExpense",       "body"),
    "delete_travelExpense": ("DELETE", "/travelExpense/{id}",  "path_query"),
    "post_travelExpense_cost": ("POST", "/travelExpense/cost", "body"),
    "delete_travelExpense_cost": ("DELETE", "/travelExpense/cost/{id}", "path_query"),
    "search_travelExpense_paymentType": ("GET", "/travelExpense/paymentType", "query"),
    "search_travelExpense_costCategory": ("GET", "/travelExpense/costCategory", "query"),
    # Project
    "search_project":       ("GET",    "/project",             "query"),
    "post_project":         ("POST",   "/project",             "body"),
    # Ledger
    "search_ledger_voucher": ("GET",   "/ledger/voucher",      "query"),
    "post_ledger_voucher":  ("POST",   "/ledger/voucher",      "body_query"),
    "delete_ledger_voucher": ("DELETE", "/ledger/voucher/{id}", "path_query"),
    "search_ledger_account": ("GET",   "/ledger/account",      "query"),
    # Company
    "get_company":          ("GET",    "/company/{id}",        "path_query"),
    "get_company_withLoginAccess": ("GET", "/company/>withLoginAccess", "query"),
    # Entitlement
    "grantEntitlementsByTemplate_employee_entitlement": ("PUT", "/employee/entitlement/:grantEntitlementsByTemplate", "query"),
    # Employment
    "search_employee_employment":      ("GET",    "/employee/employment",           "query"),
    "post_employee_employment":        ("POST",   "/employee/employment",           "body"),
    "put_employee_employment":         ("PUT",    "/employee/employment/{id}",      "path_body"),
    "search_employee_employment_details": ("GET", "/employee/employment/details",   "query"),
    "post_employee_employment_details": ("POST",  "/employee/employment/details",   "body"),
    "put_employee_employment_details": ("PUT",    "/employee/employment/details/{id}", "path_body"),
    "search_employee_employment_occupationCode": ("GET", "/employee/employment/occupationCode", "query"),
    # Accounting Dimensions
    "post_ledger_accountingDimensionName":  ("POST", "/ledger/accountingDimensionName",        "body"),
    "search_ledger_accountingDimensionName": ("GET", "/ledger/accountingDimensionName/search", "query"),
    "post_ledger_accountingDimensionValue": ("POST", "/ledger/accountingDimensionValue",       "body"),
    "search_ledger_accountingDimensionValue": ("GET", "/ledger/accountingDimensionValue/search", "query"),
    # Invoice Reminder
    "createReminder_invoice":          ("PUT",    "/invoice/{id}/:createReminder",  "path_query"),
}

# ─── Query fields for body_query routing ───
# These fields are extracted from args and sent as query params; rest goes to body.
QUERY_FIELDS = {
    "post_ledger_voucher": {"sendToLedger"},
    "post_incomingInvoice": {"sendTo"},
}

# ─── Per-tool search field defaults ───
# Replaces blanket fields="*" with compact field lists.
SEARCH_FIELDS = {
    "search_employee": "id,firstName,lastName,email,employeeNumber,department(id,name)",
    "search_customer": "id,name,organizationNumber,customerNumber,email,supplierNumber",
    "search_product": "id,name,number,priceExcludingVatCurrency,vatType(id)",
    "search_invoice": "id,invoiceNumber,invoiceDate,customer(id,name),amountOutstanding,amountExcludingVatCurrency,isCredited",
    "search_department": "id,name,departmentNumber",
    "search_contact": "id,firstName,lastName,email,customer(id,name)",
    "search_project": "id,name,number,projectManager(id,firstName,lastName),customer(id,name)",
    "search_travelExpense": "id,title,employee(id,firstName,lastName)",
    "search_travelExpense_paymentType": "id,description,displayName",
    "search_travelExpense_costCategory": "id,description,displayName",
    "search_invoice_paymentType": "id,description,displayName,debitAccount(id,number)",
    "search_ledger_voucher": "id,number,date,description",
    "search_ledger_account": "id,number,name",
    "search_supplier": "id,name,supplierNumber,organizationNumber,email",
    "search_employee_employment": "id,employee(id,firstName,lastName),startDate,endDate,employmentId",
    "search_employee_employment_details": "id,employment(id),date,annualSalary,percentageOfFullTimeEquivalent,occupationCode(id,code,nameNO),employmentType,remunerationType,workingHoursScheme",
    "search_employee_employment_occupationCode": "id,code,nameNO",
    "search_ledger_accountingDimensionName": "id,dimensionName,description,dimensionIndex,active",
    "search_ledger_accountingDimensionValue": "id,displayName,dimensionIndex,number,active",
}

# Fields that are flattened refs: tool uses `foo_id` but API wants `{"foo": {"id": N}}`
REF_FIELDS = {
    "department_id": "department",
    "employee_id": "employee",
    "customer_id": "customer",
    "contact_id": "contact",
    "project_id": "project",
    "projectManager_id": "projectManager",
    "ourContact_id": "ourContact",
    "ourContactEmployee_id": "ourContactEmployee",
    "vatType_id": "vatType",
    "currency_id": "currency",
    "costCategory_id": "costCategory",
    "paymentType_id": "paymentType",
    "travelExpense_id": "travelExpense",
    "productUnit_id": "productUnit",
    "supplier_id": "supplier",
    "account_id": "account",
    "projectCategory_id": "projectCategory",
    "paymentCurrency_id": "paymentCurrency",
    "product_id": "product",
    "order_id": "order",
    "accountManager_id": "accountManager",
    "address_id": "address",
    "employeeCategory_id": "employeeCategory",
    "postalAddress_id": "postalAddress",
    "physicalAddress_id": "physicalAddress",
    "deliveryAddress_id": "deliveryAddress",
    "phoneNumberMobileCountry_id": "phoneNumberMobileCountry",
    "discountGroup_id": "discountGroup",
    "resaleProduct_id": "resaleProduct",
    "image_id": "image",
    "mainSupplierProduct_id": "mainSupplierProduct",
    "attn_id": "attn",
    "mainProject_id": "mainProject",
    "attention_id": "attention",
    "voucherType_id": "voucherType",
    "reverseVoucher_id": "reverseVoucher",
    "document_id": "document",
    "attachment_id": "attachment",
    "ediDocument_id": "ediDocument",
    "departmentManager_id": "departmentManager",
    "vendor_id": "vendor",
    "category1_id": "category1",
    "category2_id": "category2",
    "category3_id": "category3",
    "ledgerAccount_id": "ledgerAccount",
    "employment_id": "employment",
    "occupationCode_id": "occupationCode",
    "division_id": "division",
}

# Reverse mapping: nested key → flat key (for canonicalization)
_REF_NAMES = {v: k for k, v in REF_FIELDS.items()}


def _unflatten_refs(params: dict) -> dict:
    """Convert flattened ref fields back to nested objects.
    e.g. department_id=5 → department: {"id": 5}
    """
    result = {}
    for key, value in params.items():
        if key in REF_FIELDS:
            ref_name = REF_FIELDS[key]
            result[ref_name] = {"id": value}
        else:
            result[key] = value
    return result


def _canonicalize_nested_item(item: dict) -> dict:
    """Canonicalize a nested array item (orderLine, posting, etc.).

    Accepts BOTH formats from the model:
      - flat: product_id: 123 → product: {"id": 123}
      - nested: product: {"id": 123} → kept as-is
      - nested without id wrapper: product: 123 → product: {"id": 123}
    """
    result = {}
    for k, v in item.items():
        if k in REF_FIELDS:
            # Flat _id format → convert to nested
            ref_name = REF_FIELDS[k]
            result[ref_name] = {"id": v}
        elif k in _REF_NAMES and isinstance(v, dict) and "id" in v:
            # Already nested format → keep as-is
            result[k] = v
        elif k in _REF_NAMES and isinstance(v, (int, float)):
            # Nested key but raw int → wrap it
            result[k] = {"id": v}
        else:
            result[k] = v
    return result


def _canonicalize_all_arrays(params: dict) -> dict:
    """Canonicalize refs inside ALL array-of-dict values."""
    for key, value in params.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            params[key] = [_canonicalize_nested_item(item) for item in value]
    return params


def route_tool_call(tool_name: str, args: dict) -> tuple:
    """Convert a typed tool call to (method, endpoint, query_params, body).

    Returns (method, endpoint, params_dict, body_dict) ready for HTTP execution.
    Returns (None, None, None, None) if tool_name is unknown.
    """
    if tool_name not in TOOL_MAP:
        return None, None, None, None

    method, endpoint_pattern, param_type = TOOL_MAP[tool_name]
    args = dict(args)  # copy to avoid mutation

    # Extract path parameters (e.g. {id})
    if "{id}" in endpoint_pattern:
        entity_id = args.get("id")
        if entity_id is not None:
            endpoint = endpoint_pattern.replace("{id}", str(entity_id))
            # For path_query: remove id from args (it's only in URL)
            # For path_body: KEEP id in args (Tripletex PUT needs id in body)
            if param_type != "path_body":
                args.pop("id", None)
        else:
            endpoint = endpoint_pattern
    else:
        endpoint = endpoint_pattern

    # Add per-tool search fields (narrowed, not fields="*")
    if param_type == "query" and method == "GET":
        if "fields" not in args:
            args["fields"] = SEARCH_FIELDS.get(tool_name, "*")

    if param_type in ("query", "path_query"):
        return method, endpoint, args, None

    elif param_type == "body_query":
        # Split: known query fields go to params, rest goes to body
        query_keys = QUERY_FIELDS.get(tool_name, set())
        query_params = {}
        body_args = {}
        for k, v in args.items():
            if k in query_keys:
                query_params[k] = v
            else:
                body_args[k] = v
        body = _unflatten_refs(body_args)
        body = _canonicalize_all_arrays(body)
        return method, endpoint, query_params or None, body

    elif param_type in ("body", "path_body"):
        body = _unflatten_refs(args)
        body = _canonicalize_all_arrays(body)
        return method, endpoint, None, body

    else:
        return method, endpoint, args, None
