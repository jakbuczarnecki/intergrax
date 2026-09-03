# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical registration-time contract metadata for integration providers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from intergrax.integrations.core.manifest import IntegrationManifest

IntegrationContractFactory = Callable[..., Any]

# P2-003-B1: typed built-in data/persistence categories require provider-owned specs.
B1_TYPED_CONTRACT_CATEGORIES: frozenset[str] = frozenset(
    {
        "relational_store",
        "document_store",
        "vector_store",
        "key_value_cache",
        "object_storage",
        "graph_store",
    }
)

# Migration-only (provider_id, category) rows outside B1 category gate — removed in P2-003-C.
# Do not add B1 vendors here; B1 fail-closed derives from ``B1_TYPED_CONTRACT_CATEGORIES``.
EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS: frozenset[tuple[str, str]] = frozenset(
    {
        ("openai", "managed_retrieval"),
        ("slack", "conversation_channel"),
        ("slack", "notification_channel"),
        ("langfuse", "observability_backend"),
    }
)


@dataclass(frozen=True, repr=False)
class IntegrationContractSpec:
    """One canonical ``(provider_id, category)`` contract row stored on catalog entries."""

    category: str
    provider_id: str
    integration_kind: str
    contract_class: type[Any]
    integration_class: type[Any]
    contract_factory: IntegrationContractFactory = field(compare=False, repr=False)
    config_class: type[Any] | None = None
    display_name: str = ""
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    security_posture: Any = None
    supports_runtime_binding: bool = True
    supports_health_check: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(str(capability) for capability in self.capabilities))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def declare_integration_contract(
    *,
    category: str,
    provider_id: str,
    integration_class: type[Any],
    contract_factory: IntegrationContractFactory,
    display_name: str,
    config_class: type[Any],
    capabilities: Iterable[str | Any],
    security_posture: Any,
    contract_class: type[Any] | None = None,
    integration_kind: str | None = None,
    supports_runtime_binding: bool = True,
    supports_health_check: bool | None = None,
    metadata: Mapping[str, object] | None = None,
) -> IntegrationContractSpec:
    """Build an explicit provider-owned contract declaration without reflection."""
    from intergrax.runtime.integrations.contract_metadata import (
        contract_for_category,
        expected_integration_kind_for_category,
        normalize_contract_identity,
    )
    from intergrax.runtime.integrations.contracts import PlatformIntegrationCapability

    normalized_category = normalize_contract_identity(category, "category")
    normalized_provider = normalize_contract_identity(provider_id, "provider_id")
    resolved_contract_class = contract_class or contract_for_category(normalized_category)
    resolved_integration_kind = integration_kind or expected_integration_kind_for_category(
        normalized_category,
    )

    if not issubclass(integration_class, resolved_contract_class):
        msg = (
            f"{normalized_provider}: integration_class {integration_class.__name__} "
            f"must subclass {resolved_contract_class.__name__}"
        )
        raise TypeError(msg)
    if not callable(contract_factory):
        msg = f"{normalized_provider}: contract_factory must be callable"
        raise TypeError(msg)

    capability_values = tuple(
        capability.value if isinstance(capability, PlatformIntegrationCapability) else str(capability)
        for capability in capabilities
    )
    health_supported = supports_health_check
    if health_supported is None:
        health_supported = (
            PlatformIntegrationCapability.HEALTH_CHECK.value in capability_values
            and supports_runtime_binding
        )

    return IntegrationContractSpec(
        category=normalized_category,
        provider_id=normalized_provider,
        integration_kind=resolved_integration_kind,
        contract_class=resolved_contract_class,
        integration_class=integration_class,
        contract_factory=contract_factory,
        config_class=config_class,
        display_name=display_name,
        capabilities=capability_values,
        security_posture=security_posture,
        supports_runtime_binding=supports_runtime_binding,
        supports_health_check=health_supported,
        metadata=dict(metadata or {}),
    )


def manifest_category_values(manifest: IntegrationManifest) -> frozenset[str]:
    """Normalize manifest category membership to lowercase slug strings."""
    from intergrax.runtime.integrations.contract_metadata import normalize_contract_identity

    values: set[str] = set()
    for category in manifest.categories:
        raw = category.value if isinstance(category, Enum) else str(category)
        values.add(normalize_contract_identity(raw, "category"))
    return frozenset(values)


def validate_contract_specs_against_manifest(
    manifest: IntegrationManifest,
    specs: Iterable[IntegrationContractSpec],
) -> None:
    """Fail when explicit specs drift from manifest slug/category membership."""
    from intergrax.runtime.integrations.contract_metadata import normalize_contract_identity

    slug = normalize_contract_identity(manifest.slug, "slug")
    allowed_categories = manifest_category_values(manifest)
    for spec in specs:
        validate_contract_spec_identity(slug=slug, spec=spec, observed_provider_id=spec.provider_id)
        if spec.category not in allowed_categories:
            msg = (
                f"Integration {slug!r}: contract spec category {spec.category!r} "
                f"is not declared on manifest categories {sorted(allowed_categories)!r}"
            )
            raise ValueError(msg)


def validate_contract_spec_identity(
    *,
    slug: str,
    spec: IntegrationContractSpec,
    observed_provider_id: str,
) -> None:
    """Fail when typed integration identity drifts from canonical registration metadata."""
    normalized_slug = slug.strip().lower()
    normalized_provider = observed_provider_id.strip().lower()
    normalized_spec = spec.provider_id.strip().lower()
    if normalized_spec != normalized_provider:
        msg = (
            f"Integration contract identity mismatch for slug {normalized_slug!r}: "
            f"registered provider_id={normalized_spec!r}, integration reports {normalized_provider!r}"
        )
        raise ValueError(msg)
    if normalized_spec != normalized_slug:
        msg = (
            f"Integration contract identity mismatch for slug {normalized_slug!r}: "
            f"spec provider_id={normalized_spec!r}"
        )
        raise ValueError(msg)


__all__ = [
    "B1_TYPED_CONTRACT_CATEGORIES",
    "EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS",
    "IntegrationContractFactory",
    "IntegrationContractSpec",
    "declare_integration_contract",
    "manifest_category_values",
    "validate_contract_specs_against_manifest",
    "validate_contract_spec_identity",
]
