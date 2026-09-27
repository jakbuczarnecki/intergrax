# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Marketplace gap Tool handoff reference helpers (S24-GAP-02-P2)."""

from __future__ import annotations

import hashlib
from typing import Final

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

_MARKETPLACE_GAP_HANDOFF_V2_DOMAIN_SEPARATOR: Final = (
    "intergrax.marketplace-gap-handoff.v2"
)
MARKETPLACE_GAP_HANDOFF_V2_PREFIX: Final = "marketplace-gap-handoff:v2:"
MARKETPLACE_DOMAIN_HANDOFF_SCHEME: Final = "handoff://"


class MarketplaceHandoffReferenceError(ValueError):
    """Malformed or unsupported marketplace domain handoff reference."""


def derive_marketplace_gap_tool_handoff_id(
    *,
    tenant_id: str,
    operation_id: str,
) -> str:
    """Deterministic tenant-distinct v2 handoff identity for Marketplace gap Tools."""
    normalized_tenant = require_non_empty_text(tenant_id, label="tenant_id")
    normalized_operation = require_non_empty_text(operation_id, label="operation_id")
    payload = (
        _MARKETPLACE_GAP_HANDOFF_V2_DOMAIN_SEPARATOR
        + "\0"
        + normalized_tenant
        + "\0"
        + normalized_operation
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{MARKETPLACE_GAP_HANDOFF_V2_PREFIX}{digest}"


def marketplace_domain_handoff_reference(handoff_id: str) -> str:
    """Build canonical ``handoff://`` reference for UCA acquisition evidence."""
    normalized = require_non_empty_text(handoff_id, label="handoff_id")
    return f"{MARKETPLACE_DOMAIN_HANDOFF_SCHEME}{normalized}"


def parse_marketplace_domain_handoff_reference(reference: str) -> str:
    """Strict-parse ``handoff://<handoff_id>`` — no metadata or alternate schemes."""
    cleaned = require_non_empty_text(reference, label="domain_handoff_reference")
    if not cleaned.startswith(MARKETPLACE_DOMAIN_HANDOFF_SCHEME):
        raise MarketplaceHandoffReferenceError(
            "domain handoff reference must use handoff:// scheme",
        )
    handoff_id = cleaned[len(MARKETPLACE_DOMAIN_HANDOFF_SCHEME) :]
    if not handoff_id:
        raise MarketplaceHandoffReferenceError("domain handoff reference handoff_id empty")
    return require_non_empty_text(handoff_id, label="handoff_id")


__all__ = [
    "MARKETPLACE_DOMAIN_HANDOFF_SCHEME",
    "MARKETPLACE_GAP_HANDOFF_V2_PREFIX",
    "MarketplaceHandoffReferenceError",
    "derive_marketplace_gap_tool_handoff_id",
    "marketplace_domain_handoff_reference",
    "parse_marketplace_domain_handoff_reference",
]
