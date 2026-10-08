# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Adapter — ExecutionBoundIntegrationResolution as execution-runtime pinning port."""

from __future__ import annotations

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationResolution,
    ExecutionBoundIntegrationResolutionRequest,
)
from intergrax.runtime.execution.execution_integration_configuration_pinning_ports import (
    ExecutionIntegrationConfigurationExecutionPinningPort,
)


class ExecutionBoundIntegrationConfigurationExecutionPinningAdapter(
    ExecutionIntegrationConfigurationExecutionPinningPort,
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

    def pin_configured_adoption_for_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
    ) -> None:
        self._resolution.resolve_and_pin(
            ExecutionBoundIntegrationResolutionRequest(
                tenant_id=tenant_id,
                execution_id=execution_id,
                adoption=adoption,
                integration_profile=self._integration_profile,
                catalog_slug=self._catalog_slug,
            ),
        )


__all__ = ["ExecutionBoundIntegrationConfigurationExecutionPinningAdapter"]
