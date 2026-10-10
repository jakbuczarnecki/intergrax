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
from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    ExecutionIntegrationConfigurationProvenanceRequirementCommitPort,
)
from intergrax.integrations.contracts.execution_integration_configuration_requirement_recovery_staging_source import (
    ExecutionIntegrationConfigurationRequirementRecoveryStagingSourcePort,
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


class ConfiguredRelationalStoreExecutionBindingError(RuntimeError):
    """Fail closed when production requirement-evidence dependencies are missing."""


class DefaultConfiguredRelationalStoreExecutionBindingPort(
    ConfiguredRelationalStoreExecutionBindingPort,
):
    def __init__(
        self,
        *,
        resolution: ExecutionBoundIntegrationResolution,
        integration_profile: IntegrationProfile | None = None,
        catalog_slug: str | None = None,
        requirement_recovery_staging_source: (
            ExecutionIntegrationConfigurationRequirementRecoveryStagingSourcePort | None
        ) = None,
        requirement_commit_port: (
            ExecutionIntegrationConfigurationProvenanceRequirementCommitPort | None
        ) = None,
        require_configured_adopted_requirement_evidence: bool = False,
    ) -> None:
        if require_configured_adopted_requirement_evidence:
            if requirement_recovery_staging_source is None:
                raise ConfiguredRelationalStoreExecutionBindingError(
                    "requirement_recovery_staging_source required for production binding",
                )
            if requirement_commit_port is None:
                raise ConfiguredRelationalStoreExecutionBindingError(
                    "requirement_commit_port required for production binding",
                )
        self._resolution = resolution
        self._integration_profile = integration_profile
        self._catalog_slug = catalog_slug
        self._requirement_recovery_staging_source = requirement_recovery_staging_source
        self._requirement_commit_port = requirement_commit_port
        self._require_configured_adopted_requirement_evidence = (
            require_configured_adopted_requirement_evidence
        )

    def create_bound_port(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
        integration_profile: IntegrationProfile | None = None,
        catalog_slug: str | None = None,
    ) -> ConfiguredRelationalStoreExecutionPort:
        requirement_recovery_staging = None
        if self._requirement_recovery_staging_source is not None:
            requirement_recovery_staging = (
                self._requirement_recovery_staging_source.candidate_staging_for_configured_invocation(
                    tenant_id=tenant_id,
                    execution_id=execution_id,
                )
            )
        elif self._require_configured_adopted_requirement_evidence:
            raise ConfiguredRelationalStoreExecutionBindingError(
                "requirement_recovery_staging_source missing at configured invocation",
            )
        requirement_commit_port = self._requirement_commit_port
        if self._require_configured_adopted_requirement_evidence and requirement_commit_port is None:
            raise ConfiguredRelationalStoreExecutionBindingError(
                "requirement_commit_port missing at configured invocation",
            )
        return ExecutionBoundConfiguredRelationalStorePort(
            tenant_id=tenant_id,
            execution_id=execution_id,
            adoption=adoption,
            resolution=self._resolution,
            integration_profile=integration_profile or self._integration_profile,
            catalog_slug=catalog_slug or self._catalog_slug,
            requirement_recovery_staging=requirement_recovery_staging,
            requirement_commit_port=requirement_commit_port,
            require_configured_adopted_requirement_evidence=(
                self._require_configured_adopted_requirement_evidence
            ),
        )


def build_default_configured_relational_store_execution_binding(
    *,
    resolution: ExecutionBoundIntegrationResolution,
    integration_profile: IntegrationProfile | None = None,
    catalog_slug: str | None = None,
    requirement_recovery_staging_source: (
        ExecutionIntegrationConfigurationRequirementRecoveryStagingSourcePort | None
    ) = None,
    requirement_commit_port: (
        ExecutionIntegrationConfigurationProvenanceRequirementCommitPort | None
    ) = None,
    require_configured_adopted_requirement_evidence: bool = False,
) -> ConfiguredRelationalStoreExecutionBindingPort:
    return DefaultConfiguredRelationalStoreExecutionBindingPort(
        resolution=resolution,
        integration_profile=integration_profile,
        catalog_slug=catalog_slug,
        requirement_recovery_staging_source=requirement_recovery_staging_source,
        requirement_commit_port=requirement_commit_port,
        require_configured_adopted_requirement_evidence=require_configured_adopted_requirement_evidence,
    )


__all__ = [
    "ConfiguredRelationalStoreExecutionBindingError",
    "ConfiguredRelationalStoreExecutionBindingPort",
    "DefaultConfiguredRelationalStoreExecutionBindingPort",
    "build_default_configured_relational_store_execution_binding",
]
