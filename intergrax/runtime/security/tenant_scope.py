# © Artur Czarnecki. All rights reserved.

"""Security-owned tenant scope normalization and isolation checks for middleware."""

from __future__ import annotations


def normalize_tenant_scope_id(value: str | None) -> str | None:
    """Normalize middleware tenant identity; blank/whitespace → absence (None)."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def tenant_scope_is_valid(
    request_tenant_id: str | None,
    resource_tenant_id: str | None,
    *,
    allow_unscoped: bool,
) -> bool:
    """Evaluate request vs resource tenant scope after normalization."""
    request = normalize_tenant_scope_id(request_tenant_id)
    resource = normalize_tenant_scope_id(resource_tenant_id)
    if request is None and resource is None:
        return allow_unscoped
    if request is None:
        return False
    if resource is None:
        return True
    return request == resource
