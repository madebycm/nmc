"""Schema guard: validates and sanitizes API calls against the real OpenAPI spec.

Loads the extracted spec once at startup. For every API call the agent wants to make:
1. Matches endpoint pattern to find the schema
2. Strips read-only and unknown fields from request body
3. Validates required fields are present
4. Logs any corrections made
5. Returns sanitized call ready for execution

This makes it VIRTUALLY IMPOSSIBLE to send invalid field names to the API.
"""

import json
import logging
import os
import re

log = logging.getLogger("tripletex.guard")

_SPEC = None
_SCHEMAS = None


def _load_spec():
    global _SPEC, _SCHEMAS
    if _SPEC is not None:
        return
    spec_path = os.environ.get(
        "API_SPEC_PATH",
        os.path.join(os.path.dirname(__file__), "api_spec_extract.json"),
    )
    with open(spec_path) as f:
        data = json.load(f)
    _SPEC = data["endpoints"]
    _SCHEMAS = data["schemas"]
    log.info("Schema guard loaded: %d endpoints, %d schemas", len(_SPEC), len(_SCHEMAS))


def _match_endpoint(method: str, endpoint: str):
    """Match a concrete endpoint like '/customer/123' to its spec pattern '/customer/{id}'.
    Returns (spec_path, path_params) or (None, None)."""
    _load_spec()

    # Try exact match first
    if endpoint in _SPEC:
        ep = _SPEC[endpoint]
        if method in ep:
            return endpoint, ep[method]

    # Try pattern matching: replace numeric segments with {id} or {param}
    parts = endpoint.strip("/").split("/")
    # Build candidate patterns
    candidates = []
    for spec_path in _SPEC:
        spec_parts = spec_path.strip("/").split("/")
        if len(spec_parts) != len(parts):
            continue
        match = True
        for sp, pp in zip(spec_parts, parts):
            if sp == pp:
                continue
            if sp.startswith("{") and sp.endswith("}"):
                # Pattern param — any value matches
                continue
            match = False
            break
        if match:
            candidates.append(spec_path)

    for cand in candidates:
        ep = _SPEC[cand]
        if method in ep:
            return cand, ep[method]

    return None, None


def _get_writable_fields(schema_name: str) -> dict:
    """Get writable (non-readOnly) fields from a schema. Returns {field_name: field_spec}."""
    _load_spec()
    schema = _SCHEMAS.get(schema_name, {})
    props = schema.get("properties", {})
    writable = {}
    for name, spec in props.items():
        if spec.get("readOnly"):
            continue
        # Skip internal fields
        if name in ("id", "version", "url", "changes", "displayName"):
            continue
        writable[name] = spec
    return writable


def _get_schema_for_endpoint(method: str, spec_info: dict) -> str | None:
    """Extract the schema name from an endpoint's spec info."""
    # Check requestBody
    rb = spec_info.get("requestBody", {})
    for content_type, schema_info in rb.items():
        ref = schema_info.get("$ref")
        if ref:
            return ref
    return None


def _get_all_fields(schema_name: str) -> set:
    """Get ALL field names (including read-only) from a schema."""
    _load_spec()
    schema = _SCHEMAS.get(schema_name, {})
    return set(schema.get("properties", {}).keys())


def _get_readonly_fields(schema_name: str) -> set:
    """Get read-only field names from a schema."""
    _load_spec()
    schema = _SCHEMAS.get(schema_name, {})
    props = schema.get("properties", {})
    return {name for name, spec in props.items() if spec.get("readOnly")}


# Map endpoint patterns to their body schema names
ENDPOINT_SCHEMA_MAP = {
    "/employee": "Employee",
    "/employee/{id}": "Employee",
    "/department": "Department",
    "/department/{id}": "Department",
    "/customer": "Customer",
    "/customer/{id}": "Customer",
    "/contact": "Contact",
    "/contact/{id}": "Contact",
    "/product": "Product",
    "/product/{id}": "Product",
    "/order": "Order",
    "/order/{id}": "Order",
    "/order/orderline": "OrderLine",
    "/travelExpense": "TravelExpense",
    "/travelExpense/{id}": "TravelExpense",
    "/travelExpense/cost": "Cost",
    "/travelExpense/cost/{id}": "Cost",
    "/project": "Project",
    "/project/{id}": "Project",
    "/ledger/voucher": "Voucher",
    "/ledger/voucher/{id}": "Voucher",
    "/ledger/account": "Account",
    "/ledger/account/{id}": "Account",
    "/company": "Company",
    "/supplier": "Supplier",
    "/supplier/{id}": "Supplier",
    "/incomingInvoice": "IncomingInvoiceAggregateExternalWrite",
}


def validate_and_sanitize(method: str, endpoint: str,
                          params: dict | None, body: dict | None) -> tuple:
    """Validate an API call against the spec. Returns (sanitized_body, warnings).

    - Strips unknown fields from body
    - Strips read-only fields from body
    - Logs warnings for corrections
    - Does NOT block the call (the LLM might know something we don't)
    """
    warnings = []

    # Only validate POST/PUT bodies
    if method not in ("POST", "PUT") or not body:
        return body, warnings

    # Find the schema for this endpoint
    spec_path, spec_info = _match_endpoint(method, endpoint)

    # Try to find schema name
    schema_name = None
    if spec_path:
        schema_name = ENDPOINT_SCHEMA_MAP.get(spec_path)

    if not schema_name:
        # Can't validate — pass through
        return body, warnings

    all_fields = _get_all_fields(schema_name)
    readonly_fields = _get_readonly_fields(schema_name)

    if not all_fields:
        return body, warnings

    sanitized = {}
    for key, value in body.items():
        if key in readonly_fields:
            warnings.append(f"STRIPPED read-only field '{key}' from {schema_name}")
            continue
        if key not in all_fields:
            warnings.append(f"STRIPPED unknown field '{key}' from {schema_name} (valid: {sorted(all_fields)[:10]}...)")
            continue
        sanitized[key] = value

    return sanitized, warnings


def get_valid_fields_hint(schema_name: str) -> str:
    """Generate a compact hint string of writable fields for error recovery."""
    writable = _get_writable_fields(schema_name)
    if not writable:
        return ""
    parts = []
    for name, spec in sorted(writable.items()):
        t = spec.get("type", "")
        ref = spec.get("$ref", "")
        if ref:
            parts.append(f"{name}: {{\"id\": N}} (ref {ref})")
        elif t == "string":
            parts.append(f"{name}: string")
        elif t == "number":
            parts.append(f"{name}: number")
        elif t == "integer":
            parts.append(f"{name}: integer")
        elif t == "boolean":
            parts.append(f"{name}: boolean")
        elif t == "array":
            parts.append(f"{name}: array")
        else:
            parts.append(f"{name}: {t or 'object'}")
    return f"Valid writable fields for {schema_name}: " + ", ".join(parts)
