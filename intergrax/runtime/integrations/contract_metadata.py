# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared contract-category helpers for integration projection."""

from __future__ import annotations

from intergrax.runtime.integrations.categories import (
    OBSERVABILITY_BACKEND_CATEGORY,
    OBSERVABILITY_VENDOR_INTEGRATION_KIND,
    PROVIDER_CATEGORY_CONTRACT_REGISTRY,
)
from intergrax.runtime.integrations.contracts import PlatformIntegrationContract


class IntegrationContractMetadataError(ValueError):
    """Base error for contract metadata validation."""


def normalize_contract_identity(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        msg = f"{field_name} must be a non-empty string"
        raise IntegrationContractMetadataError(msg)
    return normalized


def expected_integration_kind_for_category(category: str) -> str:
    normalized = normalize_contract_identity(category, "category")
    if normalized == OBSERVABILITY_BACKEND_CATEGORY:
        return OBSERVABILITY_VENDOR_INTEGRATION_KIND
    return normalized


def contract_for_category(category: str) -> type[PlatformIntegrationContract]:
    normalized = normalize_contract_identity(category, "category")
    try:
        return PROVIDER_CATEGORY_CONTRACT_REGISTRY[normalized]
    except KeyError as exc:
        msg = f"Unknown integration category for contract projection: {normalized!r}"
        raise IntegrationContractMetadataError(msg) from exc


__all__ = [
    "IntegrationContractMetadataError",
    "contract_for_category",
    "expected_integration_kind_for_category",
    "normalize_contract_identity",
]
