# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared validation for enterprise-private catalog source adapters (Stage 7)."""

from __future__ import annotations

from intergrax.capability_catalog.errors import CapabilityCatalogConfigurationError
from intergrax.contracts.capability_catalog.identity import (
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)

def validate_enterprise_private_source(source: CapabilitySourceIdentity) -> None:
    """Reject sources that are not explicitly enterprise-private catalog instances."""
    if source.source_kind is not CapabilitySourceKind.ENTERPRISE_PRIVATE:
        raise CapabilityCatalogConfigurationError(
            "private catalog source requires source_kind ENTERPRISE_PRIVATE",
        )


def validate_unique_source_logical_records(
    *,
    records: tuple[tuple[str, str | None], ...],
    record_label: str,
) -> None:
    """Fail closed on duplicate logical IDs or contradictory version metadata."""
    seen: dict[str, str | None] = {}
    for logical_id, version_label in records:
        normalized_id = logical_id.strip()
        if not normalized_id:
            raise CapabilityCatalogConfigurationError(
                f"{record_label} logical_id must be non-empty",
            )
        if normalized_id not in seen:
            seen[normalized_id] = version_label
            continue
        if seen[normalized_id] != version_label:
            raise CapabilityCatalogConfigurationError(
                f"conflicting version metadata for {record_label} logical_id "
                f"{normalized_id!r} in private catalog source",
            )
        raise CapabilityCatalogConfigurationError(
            f"duplicate {record_label} logical_id in private catalog source: "
            f"{normalized_id!r}",
        )
