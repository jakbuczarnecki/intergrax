# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Configured adoption → invocation wiring resolver projection (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.configured_relational_store_execution_binding import (
    ConfiguredRelationalStoreExecutionBindingPort,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
    ExecutionIntegrationConfigurationAdoptionError,
    ExecutionIntegrationConfigurationAdoptionFailureReason,
)
from intergrax.integrations.invocation_bound_configured_relational_wiring_resolver import (
    InvocationBoundConfiguredRelationalStoreWiringResolver,
)
from intergrax.tools.invocation_wiring import ToolInvocationWiringResolver
from intergrax.tools.providers.database.service import (
    DATABASE_EXECUTE_TOOL_ID,
    DATABASE_QUERY_TOOL_ID,
)


class ConfiguredIntegrationToolInvocationProjectionError(ValueError):
    """Fail-closed projection rejection."""


@runtime_checkable
class ConfiguredIntegrationToolInvocationProjectionPort(Protocol):
    def project(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
        activated_tool_id: str,
    ) -> ToolInvocationWiringResolver: ...


class DefaultConfiguredIntegrationToolInvocationProjectionPort(
    ConfiguredIntegrationToolInvocationProjectionPort,
):
    """Admits RELATIONAL_STORE + database.query/execute only."""

    def __init__(
        self,
        *,
        relational_binding: ConfiguredRelationalStoreExecutionBindingPort,
    ) -> None:
        self._relational_binding = relational_binding

    def project(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
        activated_tool_id: str,
    ) -> ToolInvocationWiringResolver:
        binding = adoption.configured_binding
        if tenant_id != binding.tenant_id:
            raise ConfiguredIntegrationToolInvocationProjectionError(
                "configured_adoption_tenant_mismatch",
            )
        if adoption.integration_category is not IntegrationCategory.RELATIONAL_STORE:
            raise ConfiguredIntegrationToolInvocationProjectionError(
                "configured_adoption_category_unsupported",
            )
        if activated_tool_id not in {
            DATABASE_QUERY_TOOL_ID,
            DATABASE_EXECUTE_TOOL_ID,
        }:
            raise ConfiguredIntegrationToolInvocationProjectionError(
                "configured_adoption_tool_unsupported",
            )
        port = self._relational_binding.create_bound_port(
            tenant_id=tenant_id,
            execution_id=execution_id,
            adoption=adoption,
            catalog_slug=binding.provider_id,
        )
        return InvocationBoundConfiguredRelationalStoreWiringResolver(
            configured_relational_store_execution=port,
        )


__all__ = [
    "ConfiguredIntegrationToolInvocationProjectionError",
    "ConfiguredIntegrationToolInvocationProjectionPort",
    "DefaultConfiguredIntegrationToolInvocationProjectionPort",
]
