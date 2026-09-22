# © Artur Czarnecki. All rights reserved.

"""Deterministic digest projection for integration catalog logical state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from intergrax.contracts.integration_catalog_revision import CatalogRevision
from intergrax.integrations.contracts.base import IntegrationEntry

__all__ = [
    "CatalogRevision",
    "compute_catalog_state_digest",
    "logical_catalog_row",
    "project_target_revision",
]


def logical_catalog_row(entry: IntegrationEntry) -> dict[str, object]:
    """Stable logical projection — excludes factories and runtime handles."""
    return {
        "slug": entry.slug,
        "categories": sorted(category.value for category in entry.categories),
        "status": entry.status.value,
        "env_prefix": entry.env_prefix,
        "description": entry.description,
        "requires_local_container": entry.requires_local_container,
        "contract_specs": sorted(
            {
                "category": spec.category,
                "provider_id": spec.provider_id,
                "integration_kind": spec.integration_kind,
            }
            for spec in entry.contract_specs
        ),
    }


def compute_catalog_state_digest(entries: Mapping[str, IntegrationEntry]) -> str:
    """Deterministic digest from canonical sorted slug → logical row mapping."""
    payload = {
        slug: logical_catalog_row(entry)
        for slug, entry in sorted(entries.items(), key=lambda item: item[0])
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_target_revision(
    current: CatalogRevision,
    candidate_entries: Mapping[str, IntegrationEntry],
) -> CatalogRevision:
    """Target revision after a successful material reload commit."""
    candidate_digest = compute_catalog_state_digest(candidate_entries)
    if candidate_digest == current.state_digest:
        return current
    return CatalogRevision(
        generation=current.generation + 1,
        state_digest=candidate_digest,
    )
