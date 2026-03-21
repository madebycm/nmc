"""Searchable API catalog from OpenAPI spec + generic validated executor."""

import json
import logging
import os
import re

log = logging.getLogger("tripletex.catalog")

_CATALOG = None


def _load_catalog():
    """Build searchable catalog from api_spec_extract.json."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    spec_path = os.path.join(os.path.dirname(__file__), "api_spec_extract.json")
    with open(spec_path) as f:
        spec = json.load(f)

    catalog = []
    for path, methods in spec["endpoints"].items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue

            summary = details.get("summary", "")
            tags = details.get("tags", [])

            # Extract param names
            params = details.get("parameters", [])
            param_names = [p["name"] for p in params if isinstance(p, dict) and p.get("name")]

            # Extract body field names
            body_fields = []
            body_required = []
            body_enums = {}
            rb = details.get("requestBody", {})
            if rb:
                content = rb.get("application/json; charset=utf-8", rb)
                props = content.get("properties", {})
                for fname, fdef in props.items():
                    if fdef.get("readOnly"):
                        continue
                    if fname in ("changes", "url"):
                        continue
                    body_fields.append(fname)
                    if fdef.get("required"):
                        body_required.append(fname)
                    if "enum" in fdef:
                        body_enums[fname] = fdef["enum"]

            # Extract query param enums
            query_enums = {}
            for p in params:
                if isinstance(p, dict) and "enum" in p:
                    query_enums[p["name"]] = p["enum"]

            # Build search text (lowercase for matching)
            search_text = (
                f"{method} {path} {summary} {' '.join(tags)} "
                f"{' '.join(param_names)} {' '.join(body_fields)}"
            ).lower()

            entry = {
                "method": method.upper(),
                "path": path,
                "summary": summary,
                "tags": tags,
                "query_params": param_names,
                "body_fields": body_fields,
                "body_required": body_required,
                "body_enums": body_enums,
                "query_enums": query_enums,
                "has_body": bool(body_fields),
                "has_path_id": "{id}" in path,
                "_search": search_text,
            }
            catalog.append(entry)

    _CATALOG = catalog
    log.info("Loaded API catalog: %d operations", len(catalog))
    return catalog


def search_spec(query: str, method_filter: str = None, limit: int = 8) -> list:
    """Search the API catalog by keyword matching.

    Returns compact list of matching operations.
    """
    catalog = _load_catalog()
    query_lower = query.lower()
    keywords = query_lower.split()

    scored = []
    for entry in catalog:
        score = 0
        for kw in keywords:
            if kw in entry["_search"]:
                score += 1
            # Bonus for path match
            if kw in entry["path"].lower():
                score += 2

        if method_filter and entry["method"] != method_filter.upper():
            continue

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])

    results = []
    for _, entry in scored[:limit]:
        compact = {
            "method": entry["method"],
            "path": entry["path"],
            "summary": entry["summary"][:120],
        }
        if entry["query_params"]:
            compact["query_params"] = entry["query_params"]
        if entry["body_fields"]:
            compact["body_fields"] = entry["body_fields"]
        if entry["body_required"]:
            compact["required"] = entry["body_required"]
        if entry["body_enums"] or entry["query_enums"]:
            compact["enums"] = {**entry["body_enums"], **entry["query_enums"]}
        results.append(compact)

    return results


def validate_generic_call(
    method: str, path: str, query_params: dict = None, body: dict = None
) -> tuple:
    """Validate a generic API call against the catalog.

    Returns (is_valid: bool, warnings: list[str], cleaned_body: dict|None)
    """
    catalog = _load_catalog()

    # Strip numeric IDs from path for matching
    clean_path = re.sub(r"/\d+", "/{id}", path)

    match = None
    for entry in catalog:
        if entry["method"] == method.upper() and entry["path"] == clean_path:
            match = entry
            break

    if not match:
        return False, [f"Unknown endpoint: {method} {clean_path}"], body

    warnings = []

    # Validate body fields if present
    if body and match["body_fields"]:
        known = set(match["body_fields"]) | {"id", "version"}
        cleaned = {}
        for k, v in body.items():
            if k in known:
                cleaned[k] = v
            else:
                warnings.append(f"Stripped unknown field '{k}' from body")

        # Validate enums
        for field, allowed in match["body_enums"].items():
            if field in cleaned and cleaned[field] not in allowed:
                warnings.append(
                    f"Invalid enum value for '{field}': {cleaned[field]}. "
                    f"Allowed: {allowed}"
                )

        body = cleaned

    # Validate query param enums
    if query_params and match["query_enums"]:
        for field, allowed in match["query_enums"].items():
            if field in query_params and query_params[field] not in allowed:
                warnings.append(
                    f"Invalid query enum for '{field}': {query_params[field]}. "
                    f"Allowed: {allowed}"
                )

    return True, warnings, body
