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
    "put_project":          ("PUT",    "/project/{id}",        "path_body"),
    # Ledger
    "search_ledger_voucher": ("GET",   "/ledger/voucher",      "query"),
    "post_ledger_voucher":  ("POST",   "/ledger/voucher",      "body_query"),
    "delete_ledger_voucher": ("DELETE", "/ledger/voucher/{id}", "path_query"),
    "search_ledger_account": ("GET",   "/ledger/account",      "query"),
    # Company
    "get_company":          ("GET",    "/company/{id}",        "path_query"),
    "get_company_withLoginAccess": ("GET", "/company/withLoginAccess", "query"),
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
    # Timesheet
    "search_activity":                 ("GET",    "/activity",                       "query"),
    "search_timesheet_entry":          ("GET",    "/timesheet/entry",                "query"),
    "post_timesheet_entry":            ("POST",   "/timesheet/entry",                "body"),
    # Salary
    "search_salary_type":              ("GET",    "/salary/type",                    "query"),
    "post_salary_transaction":         ("POST",   "/salary/transaction",             "body"),

    # Supplier Invoice
    "search_supplierInvoice":          ("GET",    "/supplierInvoice",                "query"),
    "addPayment_supplierInvoice":      ("POST",   "/supplierInvoice/{invoiceId}/:addPayment", "path_query"),
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
    "search_employee": "id,firstName,lastName,email,employeeNumber,dateOfBirth,department(id,name)",
    "search_customer": "id,name,organizationNumber,customerNumber,email,phoneNumber,supplierNumber",
    "search_product": "id,name,number,priceExcludingVatCurrency,vatType(id,name)",
    "search_invoice": "id,invoiceNumber,invoiceDate,invoiceDueDate,customer(id,name),currency(id,code),amountOutstanding,amountOutstandingTotal,amount,amountCurrency,amountExcludingVat,amountExcludingVatCurrency,isCredited,isCreditNote,isApproved",
    "search_department": "id,name,departmentNumber",
    "search_contact": "id,firstName,lastName,email,phoneNumberMobile,phoneNumberWork,customer(id,name)",
    "search_project": "id,name,number,version,isFixedPrice,fixedprice,isInternal,projectManager(id,firstName,lastName),customer(id,name),startDate,endDate,description",
    "search_travelExpense": "id,title,state,amount,employee(id,firstName,lastName),travelDetails(departureDate,returnDate)",
    "search_travelExpense_paymentType": "id,description,displayName",
    "search_travelExpense_costCategory": "id,description,displayName",
    "search_invoice_paymentType": "id,description,displayName,debitAccount(id,number)",
    "search_ledger_voucher": "id,number,date,description,year",
    "search_ledger_account": "id,number,name",
    "search_supplier": "id,name,supplierNumber,organizationNumber,email,phoneNumber",
    "search_supplierInvoice": "id,invoiceNumber,invoiceDate,invoiceDueDate,supplier(id,name),amount,amountCurrency,outstandingAmount,currency(id,code),voucher(id)",
    "search_employee_employment": "id,employee(id,firstName,lastName),startDate,endDate,employmentId",
    "search_employee_employment_details": "id,employment(id),date,annualSalary,percentageOfFullTimeEquivalent,occupationCode(id,code,nameNO),employmentType,remunerationType,workingHoursScheme",
    "search_employee_employment_occupationCode": "id,code,nameNO",
    "search_ledger_accountingDimensionName": "id,dimensionName,description,dimensionIndex,active",
    "search_ledger_accountingDimensionValue": "id,displayName,dimensionIndex,number,active",
    "search_activity": "id,name,number,isProjectActivity",
    "search_timesheet_entry": "id,employee(id,firstName,lastName),project(id,name),activity(id,name),date,hours,comment",
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
    "asset_id": "asset",
    "activity_id": "activity",
    "invoice_id": "invoice",
    "voucher_id": "voucher",
    "type_id": "type",
}

# Reverse mapping: nested key → flat key (for canonicalization)
_REF_NAMES = {v: k for k, v in REF_FIELDS.items()}


def _unflatten_refs(params: dict) -> dict:
    """Convert flattened ref fields back to nested objects.
    e.g. department_id=5 → department: {"id": 5}
    Also passes through inline address/ref objects as-is.
    """
    result = {}
    for key, value in params.items():
        if key in REF_FIELDS:
            ref_name = REF_FIELDS[key]
            result[ref_name] = {"id": value}
        elif key in ("postalAddress", "physicalAddress", "deliveryAddress", "address") and isinstance(value, dict):
            # Agent passed inline address object — pass through as-is
            result[key] = value
        else:
            result[key] = value
    return result



# camelCase flat → nested ref mapping (Gemini often emits these in postings/orderLines)
_CAMEL_REF_FIELDS = {
    "accountId": "account",
    "vatTypeId": "vatType",
    "departmentId": "department",
    "productId": "product",
    "customerId": "customer",
    "supplierId": "supplier",
    "employeeId": "employee",
    "projectId": "project",
    "currencyId": "currency",
    "vendorId": "vendor",
    "costCategoryId": "costCategory",
    "paymentTypeId": "paymentType",
    "travelExpenseId": "travelExpense",
    "contactId": "contact",
    "employmentId": "employment",
    "occupationCodeId": "occupationCode",
    "projectManagerId": "projectManager",
    "activityId": "activity",
    "invoiceId": "invoice",
    "voucherId": "voucher",
    "voucherTypeId": "voucherType",
    "productUnitId": "productUnit",
}


def _canonicalize_nested_item(item: dict) -> dict:
    """Canonicalize a nested array item (orderLine, posting, etc.).

    Accepts ALL formats from the model:
      - flat underscore: product_id: 123 → product: {"id": 123}
      - flat camelCase:  accountId: 123  → account: {"id": 123}
      - nested: product: {"id": 123} → kept as-is
      - nested without id wrapper: product: 123 → product: {"id": 123}
    """
    result = {}
    for k, v in item.items():
        if k in REF_FIELDS:
            # Flat _id format → convert to nested
            ref_name = REF_FIELDS[k]
            result[ref_name] = {"id": v}
        elif k in _CAMEL_REF_FIELDS:
            # Flat camelCase format → convert to nested
            ref_name = _CAMEL_REF_FIELDS[k]
            if isinstance(v, dict) and "id" in v:
                result[ref_name] = v
            else:
                result[ref_name] = {"id": v}
            log.info("CANONICALIZE: %s → %s: {id: %s}", k, ref_name, v)
        elif k in _REF_NAMES and isinstance(v, dict) and "id" in v:
            # Already nested format → keep as-is
            result[k] = v
        elif k in _REF_NAMES and isinstance(v, (int, float)):
            # Nested key but raw int → wrap it
            result[k] = {"id": v}
        else:
            result[k] = v
    return result


def _canonicalize_all_arrays(params: dict, tool_name: str = "") -> dict:
    """Canonicalize refs inside ALL array-of-dict values."""
    for key, value in params.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            # incomingInvoice orderLines need FLAT format (accountId, vatTypeId)
            if tool_name == "post_incomingInvoice" and key == "orderLines":
                params[key] = [_flatten_for_incoming_invoice(item) for item in value]
            else:
                params[key] = [_canonicalize_nested_item(item) for item in value]
    return params


def _flatten_for_incoming_invoice(item: dict) -> dict:
    """Ensure incomingInvoice orderLines use FLAT format (accountId, vatTypeId, etc.).

    The incomingInvoice API expects flat integer fields, not nested objects.
    Convert any nested refs back to flat: account:{id:X} → accountId:X
    """
    result = {}
    # Reverse mapping: nested key → flat camelCase key
    _NESTED_TO_FLAT = {
        "account": "accountId",
        "vatType": "vatTypeId",
        "department": "departmentId",
        "product": "productId",
        "customer": "customerId",
        "supplier": "supplierId",
        "project": "projectId",
    }
    for k, v in item.items():
        if k in _NESTED_TO_FLAT and isinstance(v, dict) and "id" in v:
            result[_NESTED_TO_FLAT[k]] = v["id"]
        elif k in REF_FIELDS:
            # underscore format (account_id) → camelCase (accountId)
            ref_name = REF_FIELDS[k]
            if ref_name in _NESTED_TO_FLAT:
                result[_NESTED_TO_FLAT[ref_name]] = v
            else:
                result[k] = v
        else:
            result[k] = v
    return result


def route_tool_call(tool_name: str, args: dict) -> tuple:
    """Convert a typed tool call to (method, endpoint, query_params, body).

    Returns (method, endpoint, params_dict, body_dict) ready for HTTP execution.
    Returns (None, None, None, None) if tool_name is unknown.
    """
    if tool_name not in TOOL_MAP:
        return None, None, None, None

    method, endpoint_pattern, param_type = TOOL_MAP[tool_name]
    args = dict(args)  # copy to avoid mutation

    # Extract path parameters (e.g. {id}, {invoiceId}, {employeeId})
    endpoint = endpoint_pattern
    import re as _re
    for match in _re.findall(r'\{(\w+)\}', endpoint_pattern):
        val = args.get(match)
        if val is not None:
            endpoint = endpoint.replace(f"{{{match}}}", str(val))
            # For path_body: KEEP id in args (Tripletex PUT needs id in body)
            if param_type != "path_body" or match != "id":
                args.pop(match, None)
        elif match == "id":
            pass  # id might not be provided for some endpoints
        # else: leave placeholder — will likely 404 but that's an agent error

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
        body = _canonicalize_all_arrays(body, tool_name=tool_name)
        return method, endpoint, query_params or None, body

    elif param_type in ("body", "path_body"):
        body = _unflatten_refs(args)
        body = _canonicalize_all_arrays(body, tool_name=tool_name)
        return method, endpoint, None, body

    else:
        return method, endpoint, args, None
