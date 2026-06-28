# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared helpers for provider category integration contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import TypeVar

from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationConfig,
    PlatformIntegrationContract,
    derive_platform_integration_id,
)

CategoryIntegrationConfig = PlatformIntegrationConfig

_CONNECT_READ_WRITE_HEALTH: tuple[PlatformIntegrationCapability, ...] = (
    PlatformIntegrationCapability.CONNECT,
    PlatformIntegrationCapability.READ,
    PlatformIntegrationCapability.WRITE,
    PlatformIntegrationCapability.HEALTH_CHECK,
)

_CONNECT_READ_HEALTH: tuple[PlatformIntegrationCapability, ...] = (
    PlatformIntegrationCapability.CONNECT,
    PlatformIntegrationCapability.READ,
    PlatformIntegrationCapability.HEALTH_CHECK,
)

_CONNECT_WRITE_HEALTH: tuple[PlatformIntegrationCapability, ...] = (
    PlatformIntegrationCapability.CONNECT,
    PlatformIntegrationCapability.WRITE,
    PlatformIntegrationCapability.HEALTH_CHECK,
)

_CONNECT_HEALTH: tuple[PlatformIntegrationCapability, ...] = (
    PlatformIntegrationCapability.CONNECT,
    PlatformIntegrationCapability.HEALTH_CHECK,
)

_READ_HEALTH: tuple[PlatformIntegrationCapability, ...] = (
    PlatformIntegrationCapability.READ,
    PlatformIntegrationCapability.HEALTH_CHECK,
)

_T = TypeVar("_T", bound=PlatformIntegrationContract)


def category_for_provider(
    contract_cls: type[_T],
    *,
    provider_id: str,
    integration_kind: str,
    default_capabilities: tuple[PlatformIntegrationCapability, ...],
    capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
    display_name: str | None = None,
    version: str | None = None,
    config: PlatformIntegrationConfig | None = None,
) -> _T:
    """Build a category contract with stable integration_id and constrained integration_kind."""
    return contract_cls(
        integration_id=derive_platform_integration_id(provider_id, integration_kind),
        provider_id=provider_id,
        integration_kind=integration_kind,
        display_name=display_name,
        version=version,
        capabilities=capabilities or default_capabilities,
        config=config or CategoryIntegrationConfig(),
    )
