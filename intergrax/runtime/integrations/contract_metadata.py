# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared contract-category helpers for integration projection."""

from __future__ import annotations

from typing import TypeAlias

from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.runtime.integrations.categories import (
    OBSERVABILITY_BACKEND_CATEGORY,
    OBSERVABILITY_VENDOR_INTEGRATION_KIND,
    PROVIDER_CATEGORY_CONTRACT_REGISTRY,
)
from intergrax.runtime.integrations.contracts import PlatformIntegrationContract

CategoryIntegrationContract: TypeAlias = (
    type[PlatformIntegrationContract] | type[ExternalWorkIntegration]
)

# Canonical materialized integration instance for any integration category (registry or DI-only).
CategoryIntegrationInstance: TypeAlias = (
    PlatformIntegrationContract | ExternalWorkIntegration
)

# Platform-defined contracts for categories bound via pre-built DI without a catalog provider package.
# Registry-backed categories remain owned by PROVIDER_CATEGORY_CONTRACT_REGISTRY only.
DI_ONLY_CATEGORY_CONTRACT_REGISTRY: dict[str, type[ExternalWorkIntegration]] = {
    "external_work": ExternalWorkIntegration,
}


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


def contract_for_category(category: str) -> CategoryIntegrationContract:
    """Return the canonical platform contract class for an integration category."""
    normalized = normalize_contract_identity(category, "category")
    provider_contract = PROVIDER_CATEGORY_CONTRACT_REGISTRY.get(normalized)
    if provider_contract is not None:
        return provider_contract
    di_only_contract = DI_ONLY_CATEGORY_CONTRACT_REGISTRY.get(normalized)
    if di_only_contract is not None:
        return di_only_contract
    msg = f"Unknown integration category for contract projection: {normalized!r}"
    raise IntegrationContractMetadataError(msg)


__all__ = [
    "CategoryIntegrationContract",
    "CategoryIntegrationInstance",
    "DI_ONLY_CATEGORY_CONTRACT_REGISTRY",
    "IntegrationContractMetadataError",
    "contract_for_category",
    "expected_integration_kind_for_category",
    "normalize_contract_identity",
]
