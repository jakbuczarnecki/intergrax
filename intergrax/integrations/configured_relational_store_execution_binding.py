# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory for execution-bound configured relational ports (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.contracts.configured_relational_store_execution import (
    ConfiguredRelationalStoreExecutionPort,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.execution_bound_configured_relational_store_port import (
    ExecutionBoundConfiguredRelationalStorePort,
)
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationResolution,
)


@runtime_checkable
class ConfiguredRelationalStoreExecutionBindingPort(Protocol):
    """Integrations-owned binder — generic Tools code depends on this protocol only."""

    def create_bound_port(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
        integration_profile: IntegrationProfile | None = None,
        catalog_slug: str | None = None,
    ) -> ConfiguredRelationalStoreExecutionPort: ...


class DefaultConfiguredRelationalStoreExecutionBindingPort(
    ConfiguredRelationalStoreExecutionBindingPort,
):
    def __init__(
        self,
        *,
        resolution: ExecutionBoundIntegrationResolution,
        integration_profile: IntegrationProfile | None = None,
        catalog_slug: str | None = None,
    ) -> None:
        self._resolution = resolution
        self._integration_profile = integration_profile
        self._catalog_slug = catalog_slug

    def create_bound_port(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
        integration_profile: IntegrationProfile | None = None,
        catalog_slug: str | None = None,
    ) -> ConfiguredRelationalStoreExecutionPort:
        return ExecutionBoundConfiguredRelationalStorePort(
            tenant_id=tenant_id,
            execution_id=execution_id,
            adoption=adoption,
            resolution=self._resolution,
            integration_profile=integration_profile or self._integration_profile,
            catalog_slug=catalog_slug or self._catalog_slug,
        )


def build_default_configured_relational_store_execution_binding(
    *,
    resolution: ExecutionBoundIntegrationResolution,
    integration_profile: IntegrationProfile | None = None,
    catalog_slug: str | None = None,
) -> ConfiguredRelationalStoreExecutionBindingPort:
    return DefaultConfiguredRelationalStoreExecutionBindingPort(
        resolution=resolution,
        integration_profile=integration_profile,
        catalog_slug=catalog_slug,
    )


__all__ = [
    "ConfiguredRelationalStoreExecutionBindingPort",
    "DefaultConfiguredRelationalStoreExecutionBindingPort",
    "build_default_configured_relational_store_execution_binding",
]
